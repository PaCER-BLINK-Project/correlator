#include <cstdint>
#include <iostream>
#include <cmath>
#include <gpu_macros.hpp>
#include <astroio.hpp>
#include <complex>
#include "mycomplex.hpp"


#define WARPS_PER_BLOCK 16
/**
 * @brief Compute the conjugate cross multiply (ccm) between std::complex numbers `a` and `b`
 * and add the result to the `res` accumulator.
 * 
 * @tparam T1 
 * @tparam T2 
 * @param a 
 * @param b 
 * @param res 
 */
template <typename T1, typename T2>
__host__ __device__ void ccm(const Complex<T1>& a, const Complex<T1>& b, Complex<T2>& res){
    res.real += static_cast<T2>(a.real) * b.real + static_cast<T2>(a.imag) * b.imag;
    res.imag += static_cast<T2>(a.imag) * b.real - static_cast<T2>(a.real) * b.imag;
}

/*
Basic idea:
each integration interval is handled by a different grid on a separate stream.

each grid handles an integration interval. In particular, there is a block 
for each frequency. At least initially.

So each block handles cross correlation of all baselines in a frequency channel.

Each thread represents a baseline.

*/
template <typename T>
__global__ void cross_correlation_kernel(const Complex<T> *volt, const ObservationInfo obsInfo, unsigned int nIntegrationSteps, 
        unsigned int nChannelsToAvg, Complex<float> *xcorr){

    const unsigned int n_baselines {(obsInfo.nAntennas / 2) * (obsInfo.nAntennas + 1)};
    const size_t nIntervals {(obsInfo.nTimesteps + nIntegrationSteps - 1) / nIntegrationSteps};
    const unsigned int n_out_frequencies {obsInfo.nFrequencies / nChannelsToAvg};
    const size_t n_warps_needed {n_baselines * n_out_frequencies * nIntervals};
    const unsigned int samplesInFrequency {obsInfo.nAntennas * obsInfo.nPolarizations * nIntegrationSteps};
    const size_t samplesInTimeInterval {samplesInFrequency * obsInfo.nFrequencies};

    const size_t matrixSize {n_baselines * obsInfo.nPolarizations * obsInfo.nPolarizations};
    const size_t samples_in_interval {n_out_frequencies *  matrixSize};
    
    const unsigned int warp_id {threadIdx.x / warpSize};
    const unsigned int lane_id {threadIdx.x % warpSize};
    const unsigned int glb_warp_id {blockIdx.x * WARPS_PER_BLOCK + warp_id};
    const unsigned int n_total_warps {gridDim.x * WARPS_PER_BLOCK};

    for(size_t currWarpPos {glb_warp_id}; currWarpPos < n_warps_needed; currWarpPos += n_total_warps){
        unsigned int baseline { static_cast<unsigned int>(currWarpPos % n_baselines)};
        size_t tmp_freq_index {currWarpPos / n_baselines};
        size_t out_frequency {tmp_freq_index % n_out_frequencies};
        size_t interval {tmp_freq_index / n_out_frequencies};
    
        Complex<float> *currentOutData {xcorr + interval * samples_in_interval + out_frequency * matrixSize};

        unsigned int a1 {static_cast<unsigned int>(-0.5 + sqrt(0.25 + 2*baseline))};
        unsigned int a2 {baseline - ((a1 + 1) * a1)/2};

        const double integrationTime {obsInfo.timeResolution * nIntegrationSteps};
        const Complex<T> *currentChData {volt + interval * samplesInTimeInterval + samplesInFrequency * out_frequency * nChannelsToAvg};
        // for each baseline compute the correlation matrix of its polarization
        for(unsigned int ch {0}; ch < nChannelsToAvg; ch++, currentChData += samplesInFrequency){
            for(unsigned int p1 {0}; p1 < obsInfo.nPolarizations; p1++){
                for(unsigned int p2 {0}; p2 < obsInfo.nPolarizations; p2++){
                    Complex<float> accum {0, 0};
                    // Each thread computes the reduction on its assigned elements independently from the other threads in the
                    // same warp. However, threads access contiguous memory locations, so they are using the cache well. To
                    // to a better job, one would need to "unpack" complex values into two sequences of real and imag values.
                    for(unsigned int step {lane_id}; step < nIntegrationSteps; step+= warpSize){
                        const size_t iA {a1 * nIntegrationSteps * obsInfo.nPolarizations + p1 * nIntegrationSteps + step};
                        const size_t iB {a2 * nIntegrationSteps * obsInfo.nPolarizations + p2 * nIntegrationSteps + step};
                        ccm(currentChData[iA], currentChData[iB], accum);
                    }
                    // now integrate results in accum
                    for(unsigned int i {warpSize/2}; i >= 1; i >>=1) {
                        float up = __gpu_shfl_down(accum.real, i);
                        if(lane_id < i){
                            accum.real += up;
                        }
                    }
                        // now integrate results in accum
                    for(unsigned int i {warpSize/2}; i >= 1; i >>=1) {
                        float up = __gpu_shfl_down(accum.imag, i);
                        if(lane_id < i){
                            accum.imag += up;
                        }
                    }
                    if(lane_id == 0){
                        size_t outIndex { baseline * obsInfo.nPolarizations * obsInfo.nPolarizations + p1*obsInfo.nPolarizations + p2};
                        currentOutData[outIndex] += accum / (float)(integrationTime * nChannelsToAvg);
                    }
                }
            }
        }
    }
}



Visibilities cross_correlation_gpu(const Voltages& voltages, unsigned int nChannelsToAvg){
    std::cout << "Correlation is happening on GPU.." << std::endl;
    if(nChannelsToAvg < 1 || nChannelsToAvg > voltages.obsInfo.nFrequencies)
        throw std::invalid_argument {"NChannelsToAvg is out of range."};
    if(voltages.obsInfo.nTimesteps % voltages.nIntegrationSteps != 0)
        throw std::invalid_argument {"nTimesteps is not an integer multiple of nIntegrationSteps."};
    if(!voltages.on_gpu() && !voltages.pinned()) std::cerr << "'cross_correlation_gpu' warning: CPU memory is not pinned."
        "\nThis will result in poor performance." << std::endl;
    

    // values to compute output size and indexing
    const ObservationInfo& obsInfo {voltages.obsInfo};
    const unsigned int n_baselines {(obsInfo.nAntennas + 1) * (obsInfo.nAntennas / 2)};
    const size_t matrixSize {n_baselines * obsInfo.nPolarizations * obsInfo.nPolarizations};
    const size_t nIntervals {(obsInfo.nTimesteps + voltages.nIntegrationSteps - 1) / voltages.nIntegrationSteps};
    const size_t nOutFrequencies {obsInfo.nFrequencies / nChannelsToAvg};
    const size_t nValuesInTimeInterval {matrixSize * nOutFrequencies};
    const size_t outSize {nValuesInTimeInterval * nIntervals};

    // variables used to compute input index
    const size_t samplesInPol {voltages.nIntegrationSteps};
    const size_t samplesInAntenna {samplesInPol * obsInfo.nPolarizations};
    const size_t samplesInFrequency {samplesInAntenna * obsInfo.nAntennas};
    const size_t samplesInTimeInterval {samplesInFrequency * obsInfo.nFrequencies};

    const float integrationTime {static_cast<float>(obsInfo.timeResolution * voltages.nIntegrationSteps)};
    
    MemoryBuffer<std::complex<float>> dev_xcorr {outSize, true};
    MemoryBuffer<Complex<int8_t>> dev_voltages;
    Complex<int8_t>* dev_voltages_data;
    
    if(!voltages.on_gpu()) {
        dev_voltages.allocate(voltages.size(), true);
        gpuMemcpy(dev_voltages.data(), voltages.data(), sizeof(std::complex<int8_t>) * voltages.size(), gpuMemcpyHostToDevice);
        dev_voltages_data = dev_voltages.data();
    }else{
        dev_voltages_data = reinterpret_cast<Complex<int8_t>*>(const_cast<std::complex<int8_t>*>(voltages.data()));
    }

    gpuMemset(dev_xcorr.data(), 0, sizeof(Complex<float>) * outSize);
    
    // retrieve warp size (32 on NVIDIA, 64 on AMD MI250X)
    int device_id, warp_size;
    gpuGetDevice(&device_id);
    gpuDeviceGetAttribute(&warp_size, gpuDeviceAttributeWarpSize, device_id);
    struct gpuDeviceProp_t props;
    gpuGetDeviceProperties(&props, device_id);
    unsigned int n_blocks = props.multiProcessorCount * 2;
    const int n_threads_per_block {warp_size * WARPS_PER_BLOCK};
    
    cross_correlation_kernel <<< dim3(n_blocks), dim3(n_threads_per_block) >>> (
        dev_voltages_data, obsInfo, voltages.nIntegrationSteps,
            nChannelsToAvg, reinterpret_cast<Complex<float>*>(dev_xcorr.data()));
    gpuDeviceSynchronize();  
    Visibilities result {std::move(dev_xcorr), obsInfo, voltages.nIntegrationSteps, nChannelsToAvg};
    return result;
}



extern "C" int blink_cross_correlation_gpu(const float* voltages, float* visibilities, 
        unsigned int n_antennas, unsigned int n_polarisations,
        unsigned int n_fine_channels, unsigned int n_time_samples, double time_resolution,
        unsigned int n_integrated_samples, unsigned int n_channels_to_avg, unsigned int reset_buffer){
    
    if(n_channels_to_avg < 1 || n_channels_to_avg > n_fine_channels){
        std::cerr << "'n_channels_to_avg' is out of range." << std::endl;
        return 1;
    }
    if(n_time_samples % n_integrated_samples != 0){
        std::cerr << "n_time_samples is not an integer multiple of n_integrated_samples." << std::endl;
        return 1;
    }
    if(!voltages){
        std::cerr << "'voltages' pointer is null." << std::endl;
        return 1;
    }
    if(!visibilities){
        std::cerr << "'visibilities' pointer is null." << std::endl;
        return 1;
    }
    // TODO add checks to make sure pointers are allocated to GPU memory.

    // values to compute output size and indexing
    ObservationInfo obsInfo {};
    obsInfo.nAntennas = n_antennas;
    obsInfo.nPolarizations = n_polarisations;
    obsInfo.nTimesteps = n_time_samples;
    obsInfo.nFrequencies = n_fine_channels;
    obsInfo.timeResolution = time_resolution;

    const unsigned int n_baselines {(obsInfo.nAntennas + 1) * (obsInfo.nAntennas / 2)};
    const size_t matrixSize {n_baselines * obsInfo.nPolarizations * obsInfo.nPolarizations};
    const size_t nIntervals {(obsInfo.nTimesteps + n_integrated_samples - 1) / n_integrated_samples};
    const size_t nOutFrequencies {obsInfo.nFrequencies / n_channels_to_avg};
    const size_t nValuesInTimeInterval {matrixSize * nOutFrequencies};
    const size_t outSize {nValuesInTimeInterval * nIntervals};

    // variables used to compute input index
    const size_t samplesInPol {n_integrated_samples};
    const size_t samplesInAntenna {samplesInPol * obsInfo.nPolarizations};
    const size_t samplesInFrequency {samplesInAntenna * obsInfo.nAntennas};
    const size_t samplesInTimeInterval {samplesInFrequency * obsInfo.nFrequencies};

    const float integrationTime {static_cast<float>(obsInfo.timeResolution * n_integrated_samples)};
    
    Complex<float>* dev_xcorr = reinterpret_cast<Complex<float>*>(visibilities);
    if(reset_buffer)
        gpuMemset(dev_xcorr, 0, sizeof(Complex<float>) * outSize);
    
	const int nStreams {5};
    gpuStream_t *streams {new gpuStream_t[nStreams]};
    for(int i {0}; i < nStreams; i++)
        gpuStreamCreate(streams + i);
    
    // retrieve warp size (32 on NVIDIA, 64 on AMD MI250X)
    int device_id, warp_size;
    gpuGetDevice(&device_id);
    gpuDeviceGetAttribute(&warp_size, gpuDeviceAttributeWarpSize, device_id);
    const int n_threads_per_block {warp_size * WARPS_PER_BLOCK};
    const int n_total_warps {static_cast<int>(n_baselines * nOutFrequencies)}; 
    const int n_blocks {(n_total_warps + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK};

    for(int i {0}; i < nIntervals; i++){
        // if(!voltages.on_gpu()){
        //     gpuMemcpyAsync(dev_voltages.data() + i*samplesInTimeInterval,
        //         voltages.data() + i*samplesInTimeInterval,
        //         sizeof(Complex<int8_t>) * samplesInTimeInterval,
        //         gpuMemcpyHostToDevice, streams[i % nStreams]);
        // }
        cross_correlation_kernel <<< dim3(n_blocks), dim3(n_threads_per_block), 0, streams[i % nStreams] >>> (
            reinterpret_cast<const Complex<float>*>(voltages) + i*samplesInTimeInterval, obsInfo, n_integrated_samples,
            n_channels_to_avg, dev_xcorr + i*nValuesInTimeInterval);
    }
    
    gpuDeviceSynchronize();
    for(int i {0}; i < nStreams; i++)
        gpuStreamDestroy(streams[i]);
    
    return 0;
}