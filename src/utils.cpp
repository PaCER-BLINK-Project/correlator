#include <iostream>
#include <fstream>
#include <cstring>

#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>
#include <algorithm>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <astroio.hpp>


#define PARSE_BUFFER_SIZE 1024


void read_data_from_file(std::string filename, char*& data, size_t& file_size) {
    std::ifstream f(filename, std::ios::binary | std::ios::ate);
    if(!f) {
        std::cerr << "read_data_from_file: error opening file." << std::endl;
        data = nullptr;
        file_size = 0;
        return;
    }

    // Get file size (we're at end due to ios::ate)
    file_size = f.tellg();
    f.seekg(0, std::ios::beg);  // Go back to beginning

    // Allocate exact amount needed
    data = new char[file_size];

    // Read entire file in one go
    f.read(data, file_size);

    if(!f) {
        std::cerr << "read_data_from_file: error reading file." << std::endl;
        delete[] data;
        data = nullptr;
        file_size = 0;
    }
}


void read_data_from_file_loop(std::string filename, char*& data, size_t& file_size){
    std::ifstream f;
    f.open(filename, std::ios::binary);
    if(!f){
        std::cerr << "read_data_from_file: error while reading the file." << std::endl;
        data = nullptr;
        return;
    }
    size_t buff_size {4096};
    data = new char[buff_size];
    size_t bytes_read {0};
    const size_t read_size {4096};
    while(f){
        // check if we need to reallocate memory
         if(bytes_read + read_size > buff_size){
            buff_size *= 2;
            char *tmp = new char[buff_size];
            memcpy(tmp, data, bytes_read);
            delete[] data;
            data = tmp;
         }
        f.read(data + bytes_read, read_size);
        bytes_read += f.gcount();
    }
    file_size = bytes_read;
}



double parse_timespec(const char * const spec){
    static char buffer[PARSE_BUFFER_SIZE];
    double result;
    int len = strlen(spec);
    int dots {0};
    if(len == 0) throw std::invalid_argument("Timespec string has zero length.");
    int i = 0;
    while(i < len && (isdigit(spec[i]) || (spec[i] == '.' && dots++ == 0))){
        buffer[i] = spec[i];
        i++;       
    }
    if(i >= PARSE_BUFFER_SIZE - 1) throw std::invalid_argument("Timespec string is too long.");
    buffer[i] = '\0';
    result = static_cast<double>(atof(buffer));
    if(i < len){
        if(!strcmp(spec + i, "ms")) result /= 1000;
        else if(!strcmp(spec + i, "cs")) result /= 100;
        else if(!strcmp(spec + i, "ds")) result /= 10;
        else if(strcmp(spec + i, "s")) throw std::invalid_argument("Invalid timespec string.");
    }else{
        throw std::invalid_argument("Invalid timespec string.");
    }
    return result;
}


bool dir_exists(const std::string& path){
    DIR* dir = opendir(path.c_str());
    if (dir) {
        closedir(dir);
        return true;
    } else {
        return false;
    }
}


bool create_directory(const std::string& path){
    if(!dir_exists(path.c_str())){
        if(mkdir(path.c_str(), 0775) != 0) return false;
        return true;
    }
    return false;
}


/**
 * @brief Index of a coarse channel within the sequence of the twenty-four coarse channels
recorded during an observation. Up until channel 128, the index is the same as
the position following the natural ordering of files. Above that, the order is
reversed because of a hardware bug at the MWA site.
*/
std::map<unsigned int, unsigned int> 
        build_coarse_channel_to_gpubox_mapping(std::vector<unsigned int> coarse_channels){
    if(coarse_channels.size() != 24) throw std::invalid_argument {
        "build_coarse_channel_to_gpubox_mapping: number of coarse channels must be 24." };
    std::map<unsigned int, unsigned int> channel_to_gpubox {};
    std::sort(coarse_channels.begin(), coarse_channels.end());
    size_t n_inverted {0ull};
    for(size_t i {0}; i < coarse_channels.size(); i++){
        unsigned int channel = coarse_channels[i];
        if(channel <= 128u){
            channel_to_gpubox.insert({channel, i + 1ull});
        }else{
            channel_to_gpubox.insert({channel, 24ull - n_inverted++});
        }
    }
    return channel_to_gpubox;
}


std::string unix_to_utcstr(time_t unix_time){
    auto chrono_time = std::chrono::system_clock::from_time_t(unix_time);
    std::stringstream ss;
    ss << std::put_time(std::gmtime(&unix_time), "%Y%m%d%H%M%S");
    return ss.str();
}


/**
 * @brief Construct the filename for for the GPUBOX fits file containing MWA Phase 1 visibilities.
 * @param gpubox_number An integer from 1 to 24 indentifying the correlator server where correlation
 * was run. This is legacy information related to how the Online Correlator is run.
 * @param obs_info Object containing information regarding the observation the visibilities refer to.
 * @return a string containing the appropriate filename for the visibilities file.
*/
std::string get_gpubox_fits_filename(unsigned int gpubox_number, const ObservationInfo& obs_info){
     std::stringstream fits_filename;
    fits_filename << obs_info.id << "_" << unix_to_utcstr(obs_info.startTime) << "_gpubox";
    if(gpubox_number < 10u) fits_filename << "0";
    fits_filename << gpubox_number << "_00.fits";
    return fits_filename.str();
}

// defined in AstroIO. Need to group all these utilities
time_t gps_to_unix(time_t gps);


std::string get_mwax_fits_filename(const ObservationInfo& obs_info, size_t count){
    std::stringstream fits_filename;
    fits_filename << std::setfill('0') << obs_info.id << "_" << unix_to_utcstr(gps_to_unix(std::stoi(obs_info.id))) << \
        "_ch" << std::setw(3) << obs_info.coarseChannel <<  std::setw(0) << "_" \
        << std::setw(3) << count << std::setw(0) << ".fits";
    return fits_filename.str();
}

