#ifndef __CORRELATION_H__
#define __CORRELATION_H__

#include <cstdint>
#include <astroio.hpp>



/**
 * @brief Compute the conjugate cross multiply (ccm) between complex numbers `a` and `b`
 * and add the result to the `res` accumulator.
 * 
 * @tparam T1 
 * @tparam T2 
 * @param a 
 * @param b 
 * @param res 
 */
template <typename T1, typename T2>
void ccm(const std::complex<T1>& a, const std::complex<T1>& b, std::complex<T2>& res){
    res.real(res.real() + static_cast<T2>(a.real()) * b.real() + static_cast<T2>(a.imag()) * b.imag());
    res.imag(res.imag() + static_cast<T2>(a.imag()) * b.real() - static_cast<T2>(a.real()) * b.imag());
}



/**
 * @brief Perform the cross correlation of voltage data across all the available baselines.
 * This function will choose the fastest implementation (CPU or GPU) based on the available
 * resources.
 * 
 * @param voltages: A Voltage object containing voltage data.
 * @return Correlation data stored in lower triangular form.
 */
Visibilities cross_correlation(const Voltages& voltages, unsigned int nChannelsToAvg = 1);



/**
 * @brief Perform the cross correlation of voltage data across all the available baselines.
 * The operation is performed on CPU.
 * 
 * @param voltages: A Voltage object containing voltage data.
 * @return Correlation data stored in lower triangular form.
 */
Visibilities cross_correlation_cpu(const Voltages& voltages, unsigned int nChannelsToAvg = 1);
#endif
