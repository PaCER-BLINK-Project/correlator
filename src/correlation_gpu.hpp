#ifndef __CORRELATION_GPU__
#define __CORRELATION_GPU__ 
#include <astroio.hpp>
/**
 * @brief Perform the cross correlation of voltage data across all the available baselines using
 * a AMD GPU.
 * 
 * @param voltages: A Voltage object containing voltage data.
 * @return Correlation data stored in lower triangular form.
 */
Visibilities cross_correlation_gpu(const Voltages& voltages, unsigned int nChannelsToAvg = 1);
#endif
