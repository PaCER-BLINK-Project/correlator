#include <vector>
#include <algorithm>
#include <iostream>
#include <locale>

#include <getopt.h>
#include <cstdlib>
#include <functional>

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
    bool check = true;

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

    std::string usage() {
        return "Usage: ./correlation_benchmark"
            " -i <integration time>"
            " -t <trials>"
            " -c <channels>"
            " --no-check";
    }

    static Options parse(int argc, char** argv) {
        static option lopts[] = {
            {"no-check", no_argument, nullptr, 0},
            {nullptr, 0, nullptr, 0}
        };

        int arg;
        int index = 0;

        Options opt;

        while ((arg = getopt_long(argc, argv, "i:t:c:", lopts, &index)) != -1) {
            switch (arg) {
                case 0:
                    if (std::string(lopts[index].name) == "no-check") {
                        opt.check = false;
                    }
                    break;
                case 'i': opt.set_itime(optarg); break;
                case 't': opt.set_trials(optarg); break;
                case 'c': opt.set_channels(optarg); break;
                default: throw std::invalid_argument{opt.usage()};
            }
        }

        if (!opt.complete()) { throw std::invalid_argument{opt.usage()}; }
        return opt;
    }
private:
    bool has_trials = false;
    bool has_channels = true;
    bool has_itime = false;
};


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
    size_t n_baselines;
    size_t n_steps;
    size_t threads_per_block;
    size_t n_out_freq;
    size_t total_wavefronts;
    size_t blocks;
    int warp_size;
    double itime;
    std::vector<double> times;

    Info(const Voltages& volt, std::vector<double>&& times, const Options& op): times(times) {

        if (times.size() > 2) {
            remove_largest(times);
        }

        int device_id;
        gpuGetDevice(&device_id);
        gpuDeviceGetAttribute(&warp_size, gpuDeviceAttributeWarpSize, device_id);

        itime = op.itime;

        const ObservationInfo& obs {volt.obsInfo};
        n_baselines = ((obs.nAntennas + 1) * obs.nAntennas) / 2;

        n_antennas = obs.nAntennas;
        n_steps = volt.nIntegrationSteps;

        const size_t matrix_size = n_baselines * obs.nPolarizations * obs.nPolarizations;
        n_intervals = (obs.nTimesteps + volt.nIntegrationSteps - 1) / volt.nIntegrationSteps;

        n_out_freq = obs.nFrequencies / op.channels;
        out_size = matrix_size * n_out_freq * n_intervals;

        samples_per_interval =
            volt.nIntegrationSteps
            * obs.nPolarizations
            * obs.nAntennas
            * obs.nFrequencies;

        threads_per_block = warp_size * WARPS_PER_BLOCK;
        total_wavefronts = n_baselines * n_out_freq;
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

#define FIELD(label, field) {label, [&]{ std::cout << info.field;}}

void print_info(const Info& info) {
    struct Field { const std::string name; std::function<void()> print; };

    std::vector<Field> fields = {
        FIELD("mean time (ms)", mean_time()),
        FIELD("integ. time (sec)", itime),
        FIELD("integ. steps", n_steps),
        FIELD("intervals", n_intervals),
        FIELD("samples/interval", samples_per_interval),
        FIELD("baselines", n_baselines),
        FIELD("threads/block", threads_per_block),
        FIELD("blocks", blocks),
        FIELD("warps", total_wavefronts),
        FIELD("warp size", warp_size),
        FIELD("output freq.", n_out_freq)
    };

    size_t label_width = 0;
    for (const auto& f : fields) {
        label_width = std::max(label_width, f.name.size());
    }

    for (const auto& f : fields) {
        std::cout << std::left << std::setw(label_width + 1) << f.name << ": ";
        f.print();
        std::cout << '\n';
    }
}

Info correlate( const Voltages& volt, const Options& options) {
    std::vector<double> times;
    times.reserve(options.trials);

    for (int i = 0; i < options.trials; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        auto xcorr_gpu = cross_correlation_gpu(volt, options.channels);
        auto stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = stop - start;
        times.push_back(elapsed.count());
    }

    Info info(volt, std::move(times), options);

    return info;
}

bool check_correctness(const Voltages& volt, const Options& opt) {
    auto xcorr_cpu = cross_correlation_cpu(volt, opt.channels);
    auto xcorr_gpu = cross_correlation_gpu(volt, opt.channels);
    xcorr_gpu.to_cpu();
    return complex_arrays_equal(
        xcorr_gpu.data(),
        xcorr_cpu.data(),
        xcorr_cpu.size()
    );
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

int main(int argc, char** argv) {
    Options options;

    try {
        options = Options::parse(argc, argv);
    } catch (std::invalid_argument& ex) {
        std::cerr << ex.what() << std::endl;
        return 1;
    }

    std::string file{get_data_dir() + "/offline_correlator/1240826896_1240827191_ch146.dat"};

    std::cout.imbue(std::locale(""));

    std::cout << "Running correlation on " << file << "\n"
        << " integration time: " << options.itime_string << "\n"
        << " channels:         " << options.channels << "\n"
        << " trials:           " << options.trials << "\n"
        << " checking correct: " << (options.check ? "true" : "false") << "\n\n";

    unsigned int n_integ_steps =
        static_cast<unsigned int>(options.itime / VCS_OBSERVATION_INFO.timeResolution);

    auto volt = Voltages::from_dat_file(file, VCS_OBSERVATION_INFO, n_integ_steps);

    if (options.check) {
        std::cout << "Checking correctness..." << std::endl;
        if (!check_correctness(volt, options)) {
            return 1;
        }
    }

    auto info = correlate(volt, options);
    print_info(info);
    std::cout << std::endl;
}