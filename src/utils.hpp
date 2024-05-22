#ifndef __UTILS_H__
#define __UTILS_H__

#include <cstddef>
#include <string>
#include <map>
#include <vector>
#include <astroio.hpp>

/**
 * @brief Read data from a given file.
 * 
 * @param filename [IN] path to the file to be read.
 * @param data [OUT] reference to a pointer which will store the location of 
 * an array of bytes representing the file content.
 * @param file_size [OUT] number of bytes read.
 */
void read_data_from_file(std::string filename, char*& data, size_t&file_size);



/**
 * @brief Converts a human readable timespec to the corresponding number of seconds.
 * 
 * A timespec is a sting composed of a integer part `i` followed by a unit of time `u`.
 * Valid units are 'ms', 'cs', 'ds', and 's'. It is a convenient way of expressing an
 * interval of time for humans.
 * 
 * @param spec a string containing a valid timespec.
 * @return A double representing the number of seconds expressed in the timepec.
 */
double parse_timespec(const char * const spec);



/**
 * @brief Checks whether a path corresponds to an existing directory.
 */
bool dir_exists(const std::string& path);



/**
 * @brief Creates a directory at the given path.
 *
 * If a directory already exists, nothing is done.
 */
bool create_directory(const std::string& path);



/**
 * @brief Index of a coarse channel within the sequence of the twenty-four coarse channels
recorded during an observation. Up until channel 128, the index is the same as
the position following the natural ordering of files. Above that, the order is
reversed because of a hardware bug at the MWA site.
*/
std::map<unsigned int, unsigned int> 
        build_coarse_channel_to_gpubox_mapping(std::vector<unsigned int> coarse_channels);


std::string unix_to_utcstr(time_t);


/**
 * @brief Construct the filename for for the GPUBOX fits file containing MWA Phase 1 visibilities.
 * @param gpubox_number An integer from 1 to 24 indentifying the correlator server where correlation
 * was run. This is legacy information related to how the Online Correlator is run.
 * @param obs_info Object containing information regarding the observation the visibilities refer to.
 * @return a string containing the appropriate filename for the visibilities file.
*/
std::string get_gpubox_fits_filename(unsigned int gpubox_number, const ObservationInfo& obs_info);
#endif
