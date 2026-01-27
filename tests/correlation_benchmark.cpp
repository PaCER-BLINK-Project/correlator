#include <vector>
#include <algorithm>
#include <iostream>
#include <locale>

#include <unistd.h>
#include <cstdlib>

#include "common.hpp"
#include "../src/correlation.hpp"
#include "../src/correlation_gpu.hpp"
#include "../src/utils.hpp"

#define WARPS_PER_BLOCK 8
#define WARP_SIZE 64

struct Options {
    unsigned int trials;
    unsigned int channels = 4;
    double itime;
    std::string itime_string;

    void set_trials(const char* value) {
        trials = static_cast<unsigned int>(std::strtoul(value, nullptr, 10));
        has_trials = true;
    }
    void set_channels(const char* value) {
        channels = static_cast<unsigned int>(std::strtoul(value, nullptr, 10));
        has_channels = true;
    }
    void set_itime(const char* value) {
        itime = parse_timespec(value);
        itime_string = std::string(value);
        has_itime = true;
    }

    bool complete() const {
        return has_channels && has_itime && has_trials;
    }

    void usage() {
        std::cerr
            << "Usage: ./correlation_benchmark"
            << " -i <integration time>"
            << " -t <trials>"
            << " -c <channels>\n";
    }
private:
    bool has_trials = false;
    bool has_channels = true;
    bool has_itime = false;
};

int parse_options(int argc, char** argv, Options& opt) {
    int arg;
    while ((arg = getopt(argc, argv, "i:t:c:")) != -1) {
        switch (arg) {
            case 'i': opt.set_itime(optarg); break;
            case 't': opt.set_trials(optarg); break;
            case 'c': opt.set_channels(optarg); break;
            default: opt.usage(); return 1;
        }
    }
    if (!opt.complete()) { opt.usage(); return 1; }
    return 0;
}

template <typename T>
int digits10(T value) {
    static_assert(std::is_integral_v<T>);
    using U = std::make_unsigned_t<T>;
    U v = (value < 0) ? U(-value) : U(value);
    int digits = 1;
    while (v >= 10) {
        v /= 10;
        ++digits;
    }
    return digits;
}

template <typename T>
T mean(const std::vector<T>& vec) {
    T sum = 0.0;
    for (const auto& x : vec) {
        sum += x;
    }
    return sum / vec.size();
}

template <typename T>
std::vector<T>& remove_largest(std::vector<T>& v) {
    if (!v.empty()) {
        auto it = std::max_element(v.begin(), v.end());
        v.erase(it);
    }
    return v;
}

template <typename T>
bool complex_arrays_equal(
    const std::complex<T>* a,
    const std::complex<T>* b,
    size_t length,
    double tol = 0.f,
    size_t max_reported = 10
) {
    double delta;
    size_t n_diffs = 0;
    std::vector<std::pair<size_t, double>> diffs;
    diffs.reserve(std::min(max_reported, length));

    for(size_t i = 0; i < length; i++){
        if (std::abs(a[i]) == 0)
            delta = std::abs(b[i]);
        else
            delta = std::abs(a[i] - b[i]);

        if (delta > tol) {
            n_diffs++;
            if (diffs.size() < max_reported) {
                diffs.push_back({i, delta});
            }
        }
    }

    if (n_diffs > 0) {
        size_t max_width = digits10(length);
        float perc = 100.0 * (n_diffs / static_cast<float>(length));
        printf("Found %zu differing elements (%0.2f%%)\n", n_diffs, perc);
        for (auto& [index, diff] : diffs) {
            printf(" %*zu : %g\n", (int)max_width, index, diff);
        }
        printf("\n");
        return false;
    } else {
        printf("Arrays are equal\n\n");
        return true;
    }
}

struct Info {
    size_t out_size;
    size_t samples_per_interval;
    size_t n_intervals;
    size_t n_antennas;
    size_t n_steps;
    size_t threads_per_block;
    size_t n_out_freq;
    size_t total_wavefronts;
    size_t blocks;
    double itime;
    std::vector<double> times;

    Info(const Voltages& volt, std::vector<double>&& times, const Options& op): times(times) {

        if (times.size() > 2) {
            remove_largest(times);
        }

        itime = op.itime;

        const ObservationInfo& obs {volt.obsInfo};
        const unsigned int n_baselines {((obs.nAntennas + 1) * obs.nAntennas) / 2};

        n_antennas = obs.nAntennas;
        n_steps = volt.nIntegrationSteps;

        const size_t matrix_size = n_baselines * obs.nPolarizations * obs.nPolarizations;
        n_intervals = (obs.nTimesteps + volt.nIntegrationSteps - 1) / volt.nIntegrationSteps;

        n_out_freq = obs.nFrequencies / op.channels;
        out_size = matrix_size * n_out_freq * n_intervals;

        samples_per_interval = volt.nIntegrationSteps * obs.nPolarizations * obs.nAntennas * obs.nFrequencies;

        threads_per_block = WARP_SIZE * WARPS_PER_BLOCK;
        total_wavefronts = static_cast<int>(n_baselines * n_out_freq);
        blocks = (total_wavefronts + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    }

    double mean_time() const {
        return mean(times);
    }

    std::pair<double, double> minmax_time() const {
        auto [mi, ma] = std::minmax_element(times.begin(), times.end());
        return std::pair{*mi, *ma};
    }
};

Info correlate(
    const Voltages& volt,
    const Options& options,
    bool check_correct
) {

    std::vector<double> times;
    times.reserve(options.trials);

    for (int i = 0; i < options.trials; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        auto xcorr_gpu = cross_correlation_gpu(volt, options.channels);
        auto stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = stop - start;
        times.push_back(elapsed.count());

        if (i == options.trials - 1 && check_correct) {
            xcorr_gpu.to_cpu();
            auto xcorr = cross_correlation_cpu(volt, options.channels);
            complex_arrays_equal(xcorr.data(), xcorr_gpu.data(), xcorr.size());
        }
    }

    Info info(volt, std::move(times), options);

    return info;
}

std::string get_data_dir() {
    char* root {std::getenv(ENV_DATA_ROOT_DIR)};
    if(!root){
        std::stringstream ss;
        ss << "'" << ENV_DATA_ROOT_DIR << "' environment variable is not set.";
        throw std::invalid_argument{ss.str()};
    }
    return std::string(root);
}

template <typename T>
void p(const char* k, T v) {
    std::cout << std::left << std::setw(21) << k << ": "
              << std::right << std::setw(16) << v << '\n';
}

int main(int argc, char** argv) {
    Options options;
    if (parse_options(argc, argv, options) != 0) {
        return 1;
    }

    std::string file{get_data_dir() + "/offline_correlator/1240826896_1240827191_ch146.dat"};

    std::cout.imbue(std::locale(""));

    std::cout << "Running correlation on " << file << "\n"
        << " integration time: " << options.itime_string << "\n"
        << " channels:         " << options.channels << "\n"
        << " trials:           " << options.trials << "\n\n";

    unsigned int n_integ_steps =
        static_cast<unsigned int>(options.itime / VCS_OBSERVATION_INFO.timeResolution);

    auto volt = Voltages::from_dat_file(file, VCS_OBSERVATION_INFO, n_integ_steps);


    auto info = correlate(volt, options, true);

    p("mean time (ms)", info.mean_time());
    p("integ. time", info.itime);
    p("integ. steps", info.n_steps);
    p("samples/interval", info.samples_per_interval);
    p("intervals", info.n_intervals);
    p("output", info.out_size);
    p("output freq.", info.n_out_freq);
    p("work items per group", info.threads_per_block);
    p("total wavefronts", info.total_wavefronts);
    p("workgroups", info.blocks);
    std::cout << std::endl;
}