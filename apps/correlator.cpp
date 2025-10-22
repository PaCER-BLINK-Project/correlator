#include <iostream>
#include <vector>
#include <string>
#include <unistd.h>
#include <iomanip>
#include <chrono>
#include <stdexcept>
#include <cstring>
#include <sstream>
#include <algorithm>
#include <map>
#include <astroio.hpp>
#include <gpu_macros.hpp>
#include "../src/utils.hpp"
#include "../src/correlation.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif
enum class DataType {MWA, EDA2};

struct ProgramOptions {
    std::vector<std::string> input_files;
    std::string outputDir {"."};
    unsigned int nChannelsToAvg {1};
    double integrationTime {-1.0}; // No default!
    DataType inputDataType {DataType::MWA};
    int n_antennas {128};
    bool legacy_fits {false};
};


// Print the program options used to run of the software.
void print_program_options(const ProgramOptions& opts);
void parse_program_options(int argc, char** argv, ProgramOptions& opts);
void print_help(std::string exec_name);



Visibilities execute_correlation_on_file(const std::string& filename, const ObservationInfo& obs_info, const ProgramOptions& opts){
    unsigned int nIntegrationSteps {static_cast<unsigned int>(opts.integrationTime / obs_info.timeResolution)};
    // auto from_dat_file = (num_available_gpus() > 0) ? Voltages::from_dat_file_gpu : Voltages::from_dat_file;
    auto from_dat_file = Voltages::from_dat_file;
    auto read_voltages = (opts.inputDataType == DataType::MWA) ? \
            from_dat_file : Voltages::from_eda2_file;
    Voltages volt = read_voltages(filename, obs_info, nIntegrationSteps);
    std::cout << "Correlating voltages in " << filename << ".." << std::endl;
    // Start correlation.
    auto tstart = std::chrono::steady_clock::now();
    auto xcorr = cross_correlation(volt, opts.nChannelsToAvg);
    auto tstop = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(tstop - tstart).count();
    std::cout << "Correlation took " << duration << "ms" << std::endl;
    return xcorr;
}



int main(int argc, char **argv){
    if(argc < 2){
        print_help(argv[0]);
        exit(0);
    }
    ProgramOptions opts;
    try {
        parse_program_options(argc, argv, opts);
    } catch (std::invalid_argument ex){
        std::cerr << ex.what() << std::endl;
        exit(1);
    }
    // Create output directory if exists
    if(opts.outputDir != "." && ! dir_exists(opts.outputDir)){
        if(!create_directory(opts.outputDir)){
            std::cerr << "Impossible to create the output directory." << std::endl;
            exit(1);
        }
    }
    print_program_options(opts);

    try {
        // TODO put the following in a function to surround with try/catch
        if(opts.inputDataType == DataType::MWA){
            if(opts.input_files.size() == 1){
                // TODO: handle the case of a single DAT file
                std::string& filename = opts.input_files[0];
                ObservationInfo obs_info = parse_mwa_phase1_dat_file_info(filename);
                obs_info.nAntennas = opts.n_antennas;
                auto xcorr = execute_correlation_on_file(filename, obs_info, opts);
                if(xcorr.on_gpu()) xcorr.to_cpu();
                std::string output_filename {opts.outputDir + "/"};
                if(opts.legacy_fits){
                    output_filename += get_gpubox_fits_filename(0, xcorr.obsInfo);
                    xcorr.to_fits_file(output_filename);
                }else{
                    output_filename += get_mwax_fits_filename(xcorr.obsInfo, 0);
                    xcorr.to_fits_file_mwax(output_filename, 0);
                }
            }else{
                auto observation = parse_mwa_dat_files(opts.input_files);
                for (size_t second_idx {0u}; second_idx < observation.size(); second_idx++) {
                    // The following vector is compute mapping to write GPUBOX files.
                    auto& one_second_data = observation[second_idx];
                    std::vector<unsigned int> coarse_channels;
                    std::vector<Visibilities> vis_vector;
                    #ifdef __GPU__
                    int n_gpus;
                    gpuGetDeviceCount(&n_gpus);
                    #pragma omp parallel for num_threads(n_gpus)
                    #endif
                    for(size_t i = 0; i < one_second_data.size(); i++){
                        #ifdef __GPU__
                        #ifdef _OPENMP
                        gpuSetDevice(omp_get_thread_num());
                        #else
                        gpuSetDevice(0);
                        #endif
                        #endif
                        auto& dat_file = one_second_data[i];
                        std::string& input_filename {dat_file.first};
                        ObservationInfo& obs_info {dat_file.second};
                        obs_info.nAntennas = opts.n_antennas;
                        auto xcorr = execute_correlation_on_file(input_filename, obs_info, opts);
                        if(xcorr.on_gpu()) xcorr.to_cpu();
                        #ifdef __OPENMP
                        #pragma omp critical
                        #endif
                        {
                            coarse_channels.push_back(obs_info.coarseChannel);
                            vis_vector.push_back(std::move(xcorr));
                        }
                    }


                    auto mapping = build_coarse_channel_to_gpubox_mapping(coarse_channels);
                    // Save to disk
                    if(opts.legacy_fits){
                        for(auto vis : vis_vector){
                            auto output_filename =  get_gpubox_fits_filename(mapping.at(vis.obsInfo.coarseChannel), vis.obsInfo);
                            output_filename = std::string {opts.outputDir + "/" + output_filename};
                            vis.to_fits_file(output_filename);
                        }
                    }else{
                        for(auto vis : vis_vector){
                            auto output_filename = get_mwax_fits_filename(vis.obsInfo, second_idx);
                            output_filename = std::string {opts.outputDir + "/" + output_filename};
                            vis.to_fits_file_mwax(output_filename, mapping.at(vis.obsInfo.coarseChannel) - 1);
                        }
                    }
                }
            }
        } else { // EDA 2
            std::string& file_path = opts.input_files[0];
            ObservationInfo obs_info = EDA2_OBSERVATION_INFO;
            auto xcorr = execute_correlation_on_file(file_path, obs_info, opts);
            if(xcorr.on_gpu()) xcorr.to_cpu();
            std::string input_filename {file_path.substr(file_path.find_last_of('/') + 1)};
            std::string output_filename {opts.outputDir + "/" + input_filename + ".fits"};
            xcorr.to_fits_file(output_filename);    
        }
    }catch(std::exception& ex){
        std::cerr << ex.what() << std::endl;
        exit(1);
    }
}



void print_help(std::string exec_name){
    std::cout << "\n" << exec_name << " -t <int time> [-c <cnls>] [-o <outdir>] DATFILE1 [DATFILE2 [DATFILE3 [...]]]\n"
    "\nProgram options:\n-------------\n"
    "\t-t <integration time>: duration of the time interval to integrate over. Accepts a timespec (see below).\n"
    "\t-c <channels to average>: number of contiguous frequency channels to average. Must be >= 1.\n"
    "\t\t Default is 1, that is, no averaging.\n"
    "\t-a <number of antennas>: number of antennas (or better, stations) used for the observation (default: 128).\n"
    "\t-o <output directory>: path to a directory where to save output files. If the directory does\n"
    "\t\t not exist, it will be created. Default is current directory.\n"
    "\t-l output legacy FITS format instead of the newer MWAX FITS format.\n"
    "\t-i [mwa | eda2]: choose which data type is given in input (default: mwa).\n"
    "\t\t is applied.\n"
    "\n"
    "Time specification (timespec)\n-----------------------------\n"
    "A timespec is a convenient way of specifing a time duration. It is made up of a real number\n"
    "followed by a unit of time; for instance '2ms' is a timespec representing 2 milliseconds.\n"
    "Valid unit of times are: 'ms', 'cs', 'ds', and 's'.\n"
    << std::endl;
}


void parse_program_options(int argc, char** argv, ProgramOptions& opts){
    const char *options = "s:t:c:o:i:a:l";
    int current_opt;
    while((current_opt = getopt(argc, argv, options)) != - 1){
        switch(current_opt){
            case 'l': {
                opts.legacy_fits = true;
                break;
            }
            case 't': {
                opts.integrationTime = parse_timespec(optarg);
                if(opts.integrationTime == 0) throw std::invalid_argument("Non positive integration time specified.");
                break;
            }
            case 'a': {
                opts.n_antennas = atoi(optarg);
                if(opts.n_antennas < 1) throw std::invalid_argument("Value for number of antennas must be at least 1.");
                break;
            }
            case 'o' : {
                opts.outputDir = std::string {optarg};
                break;
            }
            case 'c': {
                opts.nChannelsToAvg = atoi(optarg);
                if(opts.nChannelsToAvg < 1) throw std::invalid_argument("Value for number of channels to average must be at least 1.");
                break;
            }
            case 'i': {
                if(!strcmp("eda2", optarg)) opts.inputDataType = DataType::EDA2;
                else if(strcmp("mwa", optarg)){
                    std::stringstream ss;
                    ss << "Unrecognised data type: '" << optarg  << "'.";
                    throw std::invalid_argument(ss.str());
                }
                break;
            }
            default : {
                std::stringstream ss;
                ss << "Unrecognised option: '" << static_cast<char>(optopt) << "'.";
                throw std::invalid_argument(ss.str());
            }
        }
    }
    for(; optind < argc; optind++) opts.input_files.push_back({argv[optind]}); 
    
    // options validation
    if(opts.integrationTime <= 0) throw std::invalid_argument("You must specify a value for the integration time.");
    if(opts.input_files.size() == 0) throw std::invalid_argument("No input file specified.");
    else if(opts.inputDataType == DataType::MWA && opts.input_files.size() % 24 != 0)
        std::cout << "WARNING: number of input voltage files is not a multiple of 24." << std::endl;
}


void print_program_options(const ProgramOptions& opts){
    std::cout << "\nRunning the correlator program with the following options:\n"
    "\t Integration time interval: " << opts.integrationTime << "s\n"
    "\t Number of channels to average: " << opts.nChannelsToAvg << "\n"
    "\t Number of antennas: " << opts.n_antennas << "\n"
    "\t Output directory: " << opts.outputDir << "\n" << std::endl;
}
