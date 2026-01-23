#include <cstdint>
#include <iostream>
#include <cmath>
#include <gpu_macros.hpp>
#include <astroio.hpp>
#include <complex>
#include <mycomplex.hpp>
#include <hip/hip_runtime.h>


#define WARPS_PER_BLOCK 8

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
__host__ __device__ __forceinline__ void ccm(const Complex<T1>& a, const Complex<T1>& b, Complex<T2>& res){
    T2 ar = static_cast<T2>(a.real);
    T2 ai = static_cast<T2>(a.imag);
    T2 br = static_cast<T2>(b.real);
    T2 bi = static_cast<T2>(b.imag);

    res.real += ar * br + ai * bi;
    res.imag += ai * br - ar * bi;
}

__device__ __forceinline__ int dp4a(int a, int b, int acc) {
    asm volatile(
        "v_dot4_i32_i8 %0, %1, %2, %3\n"
        : "=v"(acc)
        : "v"(a), "v"(b), "v"(acc)
    );
    return acc;
}

__device__ __forceinline__ char4 as_char4(int x) {
    char4 v;
    __builtin_memcpy(&v, &x, sizeof(v));
    return v;
}

template <typename T>
__device__ __forceinline__ int as_int(T v) {
    int x;
    static_assert(sizeof(T) == sizeof(x));
    __builtin_memcpy(&x, &v, sizeof(x));
    return x;
}

__device__ __forceinline__ int load_32bits(const Complex<int8_t>* p) {
    int x;
    __builtin_memcpy(&x, p, sizeof(x));
    return x;
}

__forceinline__ __device__ void ccm_dp4a(int A, int B, int& acc_re, int& acc_im) {
    // unpack 32-bit int into 4 x 8-bit
    // trusting the compiler to optimise away the memcpy
    char4 b = as_char4(B);

    char4 bi;
    bi.x = (signed char)(-b.y);
    bi.y = b.x;
    bi.z = (signed char)(-b.w);
    bi.w = b.z;

    // re-packing into 32-bits
    int Bi = as_int(bi);

    acc_re = dp4a(A, B, acc_re);
    acc_im = dp4a(A, Bi, acc_im);
}

/*
Basic idea:
each integration interval is handled by a different grid on a separate stream.

each grid handles an integration interval. In particular, there is a block
for each frequency. At least initially.

So each block handles cross correlation of all baselines in a frequency channel.

Each thread represents a baseline.

*/
template <int NPOL, typename T>
__global__ void cross_correlation_kernel(
    const Complex<T>* __restrict__ volt,
    const ObservationInfo obs,
    unsigned int n_integration_steps,
    unsigned int n_channels_to_avg,
    Complex<float>* __restrict__ xcorr
) {

    const unsigned int n_baselines {(obs.nAntennas * (obs.nAntennas + 1)) / 2};
    const unsigned int n_total_warps { n_baselines * (obs.nFrequencies / n_channels_to_avg)};
    const unsigned int samples_in_frequency {obs.nAntennas * NPOL * n_integration_steps};
    const size_t matrix_size {n_baselines * NPOL * NPOL};
    const double integration_time {obs.timeResolution * n_integration_steps};

    const unsigned int warp_id {threadIdx.x / warpSize};
    const unsigned int lane_id {threadIdx.x % warpSize};
    const unsigned int glb_warp_id {blockIdx.x * WARPS_PER_BLOCK + warp_id};
    const unsigned int out_frequency {glb_warp_id / n_baselines};
    const unsigned int baseline {glb_warp_id % n_baselines};

    if(glb_warp_id >= n_total_warps) return;

    Complex<float> *out_data {xcorr + out_frequency * matrix_size };

    const unsigned int ant1 {static_cast<unsigned int>(-0.5 + sqrt(0.25 + 2*baseline))};
    const unsigned int ant2 {baseline - ((ant1 + 1) * ant1)/2};

    const size_t base_a = ant1 * n_integration_steps * NPOL;
    const size_t base_b = ant2 * n_integration_steps * NPOL;

    const float norm = 1.f / (float)(integration_time * (double)n_channels_to_avg);

    const Complex<T> *ch_data {volt + samples_in_frequency * out_frequency * n_channels_to_avg};

    // for each baseline compute the correlation matrix of its polarization
    for (unsigned int ch = 0; ch < n_channels_to_avg; ch++, ch_data += samples_in_frequency) {

        #pragma unroll
        for (unsigned int pol_a = 0; pol_a < NPOL; pol_a++) {
            #pragma unroll
            for (unsigned int pol_b = 0; pol_b < NPOL; pol_b++) {

                Complex<float> acc0 {0.f, 0.f};
                Complex<float> acc1 {0.f, 0.f};
                Complex<float> acc2 {0.f, 0.f};
                Complex<float> acc3 {0.f, 0.f};

                unsigned int step = lane_id;
                const size_t A0 = base_a + (size_t)pol_a * n_integration_steps;
                const size_t B0 = base_b + (size_t)pol_b * n_integration_steps;

                // Each thread computes the reduction on its assigned elements independently from the other threads in the
                // same warp. However, threads access contiguous memory locations, so they are using the cache well. To
                // to a better job, one would need to "unpack" complex values into two sequences of real and imag values.
                // unrolled body: process 4 iterations per loop
                for (; step + 3u * warpSize < n_integration_steps; step += 4u * warpSize) {
                    ccm(ch_data[A0 + step],                 ch_data[B0 + step], acc0);
                    ccm(ch_data[A0 + step + warpSize],      ch_data[B0 + step + warpSize], acc1);
                    ccm(ch_data[A0 + step + 2u * warpSize], ch_data[B0 + step + 2u * warpSize], acc2);
                    ccm(ch_data[A0 + step + 3u * warpSize], ch_data[B0 + step + 3u * warpSize], acc3);
                }

                // get any remainder (at most 3 iterations)
                for (; step < n_integration_steps; step += warpSize) {
                    ccm(ch_data[A0 + step], ch_data[B0 + step], acc0);
                }

                Complex<float> acc {
                    acc0.real + acc1.real + acc2.real + acc3.real,
                    acc0.imag + acc1.imag + acc2.imag + acc3.imag
                };

                // now integrate results in accum
                for (unsigned int i = warpSize/2; i >= 1; i >>=1) {
                    float up = __gpu_shfl_down(acc.real, i);
                    if(lane_id < i){
                        acc.real += up;
                    }
                }
                    // now integrate results in accum
                for (unsigned int i = warpSize/2; i >= 1; i >>=1) {
                    float up = __gpu_shfl_down(acc.imag, i);
                    if(lane_id < i){
                        acc.imag += up;
                    }
                }

                if (lane_id == 0) {
                    size_t out_index { baseline * NPOL * NPOL + pol_a*NPOL + pol_b};
                    out_data[out_index].real = fmaf(acc.real, norm, out_data[out_index].real);
                    out_data[out_index].imag = fmaf(acc.imag, norm, out_data[out_index].imag);
                }
            }
        }
    }
}

template <int NPOL>
__global__ void cross_correlation_kernel(
    const Complex<int8_t>* __restrict__ volt,
    const ObservationInfo obs,
    unsigned int n_integration_steps,
    unsigned int n_channels_to_avg,
    Complex<float>* __restrict__ xcorr
) {
    const unsigned int n_baselines {(obs.nAntennas * (obs.nAntennas + 1)) / 2};
    const unsigned int n_total_warps { n_baselines * (obs.nFrequencies / n_channels_to_avg)};
    const unsigned int samples_in_frequency {obs.nAntennas * NPOL * n_integration_steps};
    const size_t matrix_size {n_baselines * NPOL * NPOL};
    const double integration_time {obs.timeResolution * n_integration_steps};

    const unsigned int warp_id {threadIdx.x / warpSize};
    const unsigned int lane_id {threadIdx.x % warpSize};
    const unsigned int glb_warp_id {blockIdx.x * WARPS_PER_BLOCK + warp_id};
    const unsigned int out_frequency {glb_warp_id / n_baselines};
    const unsigned int baseline {glb_warp_id % n_baselines};

    if(glb_warp_id >= n_total_warps) return;

    Complex<float> *out_data {xcorr + out_frequency * matrix_size };

    const unsigned int ant1 {static_cast<unsigned int>(-0.5 + sqrt(0.25 + 2*baseline))};
    const unsigned int ant2 {baseline - ((ant1 + 1) * ant1)/2};

    const size_t base_a = ant1 * n_integration_steps * NPOL;
    const size_t base_b = ant2 * n_integration_steps * NPOL;

    const float norm = 1.f / (float)(integration_time * (double)n_channels_to_avg);

    const Complex<int8_t> *ch_data {volt + samples_in_frequency * out_frequency * n_channels_to_avg};

    // for each baseline compute the correlation matrix of its polarization
    for (unsigned int ch = 0; ch < n_channels_to_avg; ch++, ch_data += samples_in_frequency) {

        #pragma unroll
        for (unsigned int pol_a = 0; pol_a < NPOL; pol_a++) {
            #pragma unroll
            for (unsigned int pol_b = 0; pol_b < NPOL; pol_b++) {

                const size_t A0 = base_a + (size_t)pol_a * n_integration_steps;
                const size_t B0 = base_b + (size_t)pol_b * n_integration_steps;

                unsigned int step = 2u * lane_id;
                int acc_re = 0, acc_im = 0;

                for (; step + 1 < n_integration_steps; step += 2u * warpSize) {
                    int A = load_32bits(&ch_data[A0 + step]);
                    int B = load_32bits(&ch_data[B0 + step]);
                    ccm_dp4a(A, B, acc_re, acc_im);
                }

                // get any remainder
                for (; step < n_integration_steps; step += warpSize) {
                    auto &a = ch_data[A0 + step];
                    auto &b = ch_data[B0 + step];
                    acc_re += int(a.real)*int(b.real) + int(a.imag)*int(b.imag);
                    acc_im += int(a.imag)*int(b.real) - int(a.real)*int(b.imag);
                }

                Complex<float> acc{
                    static_cast<float>(acc_re),
                    static_cast<float>(acc_im)
                };

                // now integrate results in accum
                for (unsigned int i = warpSize/2; i >= 1; i >>=1) {
                    float up = __gpu_shfl_down(acc.real, i);
                    if(lane_id < i){
                        acc.real += up;
                    }
                }
                    // now integrate results in accum
                for (unsigned int i = warpSize/2; i >= 1; i >>=1) {
                    float up = __gpu_shfl_down(acc.imag, i);
                    if(lane_id < i){
                        acc.imag += up;
                    }
                }

                if (lane_id == 0) {
                    size_t out_index { baseline * NPOL * NPOL + pol_a*NPOL + pol_b};
                    out_data[out_index].real = fmaf(acc.real, norm, out_data[out_index].real);
                    out_data[out_index].imag = fmaf(acc.imag, norm, out_data[out_index].imag);
                }
            }
        }
    }
}

Visibilities cross_correlation_gpu(const Voltages& voltages, unsigned int n_channels_to_avg){
    const ObservationInfo& obs_info {voltages.obsInfo};

    if(n_channels_to_avg < 1 || n_channels_to_avg > obs_info.nFrequencies) {
        std::stringstream ss;
        ss << "number of channels to average (" << n_channels_to_avg
           << ") is greater than the number of frequencies (" << obs_info.nFrequencies << ")";
        throw std::invalid_argument(ss.str());
    }
    if(obs_info.nTimesteps % voltages.nIntegrationSteps != 0) {
        std::stringstream ss;
        ss << "number of timesteps (" << obs_info.nTimesteps
           << ") is not an integer multiple of the number of integration steps (" << voltages.nIntegrationSteps << ")";
        throw std::invalid_argument(ss.str());
    }
    if (!voltages.on_gpu() && !voltages.pinned()) {
        std::cerr << "'cross_correlation_gpu' warning: CPU memory is not pinned.\n"
                     "This will result in poor performance." << std::endl;
    }

    if (obs_info.nPolarizations != 2) {
        throw std::invalid_argument {"Expected 2 polarizations per antenna"};
    }

    if(n_channels_to_avg < 1 || n_channels_to_avg > voltages.obsInfo.nFrequencies)
        throw std::invalid_argument {"NChannelsToAvg is out of range."};
    if(voltages.obsInfo.nTimesteps % voltages.nIntegrationSteps != 0)
        throw std::invalid_argument {"nTimesteps is not an integer multiple of nIntegrationSteps."};

    if (!voltages.on_gpu() && !voltages.pinned()) {
        std::cerr << "'cross_correlation_gpu' warning: CPU memory is not pinned.\n"
                     "This will result in poor performance." << std::endl;
    }

    if (voltages.obsInfo.nPolarizations != 2) {
        throw std::invalid_argument {"Expected 2 polarizations per antenna"};
    }


    // values to compute output size and indexing
    const unsigned int n_baselines {((obs_info.nAntennas + 1) * obs_info.nAntennas) / 2};
    const size_t matrixSize {n_baselines * obs_info.nPolarizations * obs_info.nPolarizations};
    const size_t nIntervals {(obs_info.nTimesteps + voltages.nIntegrationSteps - 1) / voltages.nIntegrationSteps};
    const size_t nOutFrequencies {obs_info.nFrequencies / n_channels_to_avg};
    const size_t nValuesInTimeInterval {matrixSize * nOutFrequencies};
    const size_t outSize {nValuesInTimeInterval * nIntervals};

    // variables used to compute input index
    const size_t samplesInPol {voltages.nIntegrationSteps};
    const size_t samplesInAntenna {samplesInPol * obs_info.nPolarizations};
    const size_t samplesInFrequency {samplesInAntenna * obs_info.nAntennas};
    const size_t samplesInTimeInterval {samplesInFrequency * obs_info.nFrequencies};

    const float integrationTime {static_cast<float>(obs_info.timeResolution * voltages.nIntegrationSteps)};

    MemoryBuffer<std::complex<float>> dev_xcorr {outSize, MemoryType::DEVICE};
    MemoryBuffer<Complex<int8_t>> dev_voltages;
    Complex<int8_t>* dev_voltages_data;

    if(!voltages.on_gpu()) {
        dev_voltages.allocate(voltages.size(), MemoryType::DEVICE);
        dev_voltages_data = dev_voltages.data();
    }else{
        dev_voltages_data = reinterpret_cast<Complex<int8_t>*>(const_cast<std::complex<int8_t>*>(voltages.data()));
    }

    gpuMemset(dev_xcorr.data(), 0, sizeof(Complex<float>) * outSize);

	const size_t nStreams = std::min(size_t(5), nIntervals);

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

    for (int i {0}; i < nIntervals; i++) {
        if (!voltages.on_gpu()) {
            gpuMemcpyAsync(dev_voltages.data() + i*samplesInTimeInterval,
                voltages.data() + i*samplesInTimeInterval,
                sizeof(Complex<int8_t>) * samplesInTimeInterval,
                gpuMemcpyHostToDevice, streams[i % nStreams]);
        }
        cross_correlation_kernel<2> <<< dim3(n_blocks), dim3(n_threads_per_block), 0, streams[i % nStreams] >>> (
            dev_voltages_data + i*samplesInTimeInterval,
            obs_info,
            voltages.nIntegrationSteps,
            n_channels_to_avg,
            reinterpret_cast<Complex<float>*>(dev_xcorr.data()) + i*nValuesInTimeInterval);
    }

    gpuDeviceSynchronize();
    for(int i {0}; i < nStreams; i++)
        gpuStreamDestroy(streams[i]);
    delete[] streams;

    Visibilities result {std::move(dev_xcorr), obs_info, voltages.nIntegrationSteps, n_channels_to_avg};
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

    const unsigned int n_baselines {((obsInfo.nAntennas + 1) * obsInfo.nAntennas) / 2};
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
        cross_correlation_kernel<2> <<< dim3(n_blocks), dim3(n_threads_per_block), 0, streams[i % nStreams] >>> (
            reinterpret_cast<const Complex<float>*>(voltages) + i*samplesInTimeInterval, obsInfo, n_integrated_samples,
            n_channels_to_avg, dev_xcorr + i*nValuesInTimeInterval);
    }

    gpuDeviceSynchronize();
    for(int i {0}; i < nStreams; i++)
        gpuStreamDestroy(streams[i]);

    return 0;
}
