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
__host__ __device__ __forceinline__ void ccm(
    const Complex<T1>& a,
    const Complex<T1>& b,
    T2& acc_re, T2& acc_im
) {
    T2 ar = static_cast<T2>(a.real);
    T2 ai = static_cast<T2>(a.imag);
    T2 br = static_cast<T2>(b.real);
    T2 bi = static_cast<T2>(b.imag);

    acc_re += ar * br + ai * bi;
    acc_im += ai * br - ar * bi;
}

/**
 * @brief Reinterpret (type pun) helper.
 *
 * This function uses `__builtin_memcpy` rather than a `reinterpret_cast<>`, a
 * standard pattern for type-punning that preserves strict aliasing rules. The
 * compiler recognises this pattern and optimises away the memcpy.
 *
 * More info: https://en.cppreference.com/w/cpp/language/reinterpret_cast.html#Type_aliasing
 */
template <typename Tout, typename Tin>
__device__ __forceinline__ Tout type_alias(Tin in) {
    Tout out;
    __builtin_memcpy(&out, &in, sizeof(out));
    return out;
}

/**
 * @brief Dot products four 8-bit integers in pairs (each packed into a 32-bit int)
 * and accumulates the result.
 */
__device__ __forceinline__ int dp4a(int a, int b, int acc) {
    #if defined(__HIPCC__)
    return __builtin_amdgcn_sdot4(a, b, acc, false);
    #elif defined(__NVCC__)
    return __dp4a(a, b, acc);
    #else
    char4 a4 = type_alias<char4>(a);
    char4 b4 = type_alias<char4>(b);
    acc += a4.x * b4.x;
    acc += a4.y * b4.y;
    acc += a4.z * b4.z;
    acc += a4.w * b4.w;
    return acc;
    #endif
}

/**
 * @brief Load two contiguous complex samples, type punning through a single
 *  32-bit int.
 */
__device__ __forceinline__ int load_32bits(const Complex<int8_t>* p) {
    int x;
    __builtin_memcpy(&x, p, sizeof(x));
    return x;
}

/**
 * Perform a complex conjugate multiplication on two pairs of complex samples
 *  that have been packed into two 32-bit integers (each sample is 2x 8-bits).
 */
__forceinline__ __device__ void ccm_dp4a(int A, int B, int& acc_re, int& acc_im) {
    // unpack 32-bit int into 4 x 8-bit
    char4 b = type_alias<char4>(B);

    // constructing the conjugate terms
    char4 bi;
    bi.x = (signed char)(-b.y);
    bi.y = b.x;
    bi.z = (signed char)(-b.w);
    bi.w = b.z;

    // re-packing into 32-bits
    int Bi = type_alias<int>(bi);

    acc_re = dp4a(A, B, acc_re);
    acc_im = dp4a(A, Bi, acc_im);
}

/**
 * @brief Innermost loop of the cross correlation kernel, executed by all warps
 * in a lane to accumulate over all integration steps for a given baseline x pol
 * combination.
 *
 * Generic implementation for arbitrary sample type @p T.
 *
 * @param ch_data Pointer to frequency-local, contiguous voltage data.
 * @param A0 Base index for antenna/polarization A.
 * @param B0 Base index for antenna/polarization B.
 * @param lane_id Warp lane index.
 * @param n_integration_steps Number of time steps to accumulate.
 * @return Partial complex correlation accumulated by this lane.
 */
template <typename T>
__device__ __forceinline__ Complex<float> cross_correlation_inner(
    const Complex<T>* __restrict__ ch_data,
    const size_t A0,
    const size_t B0,
    unsigned int lane_id,
    unsigned int n_integration_steps
) {

    Complex<float> acc0 {0.f, 0.f};
    Complex<float> acc1 {0.f, 0.f};
    Complex<float> acc2 {0.f, 0.f};
    Complex<float> acc3 {0.f, 0.f};

    unsigned int step = lane_id;

    for (; step + 3u * warpSize < n_integration_steps; step += 4u * warpSize) {
        ccm(ch_data[A0 + step],                 ch_data[B0 + step],                 acc0.real, acc0.imag);
        ccm(ch_data[A0 + step + warpSize],      ch_data[B0 + step + warpSize],      acc1.real, acc1.imag);
        ccm(ch_data[A0 + step + 2u * warpSize], ch_data[B0 + step + 2u * warpSize], acc2.real, acc2.imag);
        ccm(ch_data[A0 + step + 3u * warpSize], ch_data[B0 + step + 3u * warpSize], acc3.real, acc3.imag);
    }

    // get any remainder (at most 3 iterations)
    for (; step < n_integration_steps; step += warpSize) {
        ccm(ch_data[A0 + step], ch_data[B0 + step], acc0.real, acc0.imag);
    }

    Complex<float> acc {
        acc0.real + acc1.real + acc2.real + acc3.real,
        acc0.imag + acc1.imag + acc2.imag + acc3.imag
    };
    return acc;
}

/**
 * @brief Innermost loop of the cross correlation kernel, executed by all warps
 * in a lane to accumulate over all integration steps for a given baseline x pol
 * combination.
 *
 * Optimised specialisation for `int8_t` samples. Uses packed arithmetic
 * optimizations and is significantly faster than the generic template. Should
 * be preferred whenever int8 voltages are available.
 *
 * @param ch_data Pointer to frequency-local, contiguous voltage data.
 * @param A0 Base index for antenna/polarization A.
 * @param B0 Base index for antenna/polarization B.
 * @param lane_id Warp lane index.
 * @param n_integration_steps Number of time steps to accumulate.
 * @return Partial complex correlation accumulated by this lane.
 */
__device__ __forceinline__ Complex<float> cross_correlation_inner(
    const Complex<int8_t>* __restrict__ ch_data,
    const size_t A0,
    const size_t B0,
    unsigned int lane_id,
    unsigned int n_integration_steps
) {
    unsigned int step = 2u * lane_id;
    int acc_re = 0, acc_im = 0;

    for (; step + 1 < n_integration_steps; step += 2u * warpSize) {
        int A = load_32bits(&ch_data[A0 + step]);
        int B = load_32bits(&ch_data[B0 + step]);
        ccm_dp4a(A, B, acc_re, acc_im);
    }

    // get any remainder
    for (; step < n_integration_steps; step += 2u * warpSize) {
        auto &a = ch_data[A0 + step];
        auto &b = ch_data[B0 + step];
        ccm(a, b, acc_re, acc_im);
    }

    Complex<float> acc{
        static_cast<float>(acc_re),
        static_cast<float>(acc_im)
    };
    return acc;
}

/**
 * Each grid handles an integration interval. In particular, there is a block
 * for each (input) frequency. Each warp handles one output frequency for one
 * baseline.
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

    const Complex<T> *ch_data = volt
        + (size_t)samples_in_frequency * out_frequency * n_channels_to_avg;

    Complex<float> *out_data = xcorr
        + (size_t)out_frequency * matrix_size;

    const unsigned int ant1 {static_cast<unsigned int>(-0.5 + sqrt(0.25 + 2*baseline))};
    const unsigned int ant2 {baseline - ((ant1 + 1) * ant1)/2};

    const size_t base_a = ant1 * n_integration_steps * NPOL;
    const size_t base_b = ant2 * n_integration_steps * NPOL;

    const float norm = 1.f / (float)(integration_time * (double)n_channels_to_avg);

    // for each baseline compute the correlation matrix of its polarization
    for (unsigned int ch = 0; ch < n_channels_to_avg; ch++, ch_data += samples_in_frequency) {

        #pragma unroll
        for (unsigned int pol_a = 0; pol_a < NPOL; pol_a++) {
            #pragma unroll
            for (unsigned int pol_b = 0; pol_b < NPOL; pol_b++) {

                const size_t A0 = base_a + (size_t)pol_a * n_integration_steps;
                const size_t B0 = base_b + (size_t)pol_b * n_integration_steps;

                Complex<float> acc = cross_correlation_inner(
                    ch_data, A0, B0, lane_id, n_integration_steps
                );

                // now integrate results in accum
                for (unsigned int i = warpSize/2; i >= 1; i >>= 1) {
                    float up = __gpu_shfl_down(acc.real, i);
                    if(lane_id < i){
                        acc.real += up;
                    }
                }
                    // now integrate results in accum
                for (unsigned int i = warpSize/2; i >= 1; i >>= 1) {
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

template<>
__global__ void cross_correlation_kernel<2, int8_t>(
    const Complex<int8_t>* __restrict__ volt,
    const ObservationInfo obs,
    unsigned int n_integration_steps,
    unsigned int n_channels_to_avg,
    Complex<float>* __restrict__ xcorr
) {
    constexpr int NPOL = 2;

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

    const Complex<int8_t> *ch_data = volt
        + (size_t)samples_in_frequency * out_frequency * n_channels_to_avg;

    Complex<float> *out_data = xcorr
        + (size_t)out_frequency * matrix_size;

    const unsigned int ant1 {static_cast<unsigned int>(-0.5 + sqrt(0.25 + 2*baseline))};
    const unsigned int ant2 {baseline - ((ant1 + 1) * ant1)/2};

    const size_t base_a = ant1 * n_integration_steps * NPOL;
    const size_t base_b = ant2 * n_integration_steps * NPOL;

    const float norm = 1.f / (float)(integration_time * (double)n_channels_to_avg);

    for (unsigned int ch = 0; ch < n_channels_to_avg; ch++, ch_data += samples_in_frequency) {

        const size_t A00 = base_a + 0u * n_integration_steps;
        const size_t A01 = base_a + 1u * n_integration_steps;
        const size_t B00 = base_b + 0u * n_integration_steps;
        const size_t B01 = base_b + 1u * n_integration_steps;

        unsigned int step = 2u * lane_id;
        int acc00_re = 0, acc00_im = 0;
        int acc01_re = 0, acc01_im = 0;
        int acc10_re = 0, acc10_im = 0;
        int acc11_re = 0, acc11_im = 0;

        for (; step + 1 < n_integration_steps; step += 2u * warpSize) {
            int A0 = load_32bits(&ch_data[A00 + step]);
            int A1 = load_32bits(&ch_data[A01 + step]);
            int B0 = load_32bits(&ch_data[B00 + step]);
            int B1 = load_32bits(&ch_data[B01 + step]);

            ccm_dp4a(A0, B0, acc00_re, acc00_im);
            ccm_dp4a(A0, B1, acc01_re, acc01_im);
            ccm_dp4a(A1, B0, acc10_re, acc10_im);
            ccm_dp4a(A1, B1, acc11_re, acc11_im);
        }

        // get any remainder
        for (; step < n_integration_steps; step += 2u * warpSize) {
            const auto &a0 = ch_data[A00 + step];
            const auto &a1 = ch_data[A01 + step];
            const auto &b0 = ch_data[B00 + step];
            const auto &b1 = ch_data[B01 + step];
            ccm(a0, b0, acc00_re, acc00_im);
            ccm(a0, b1, acc01_re, acc01_im);
            ccm(a1, b0, acc10_re, acc10_im);
            ccm(a1, b1, acc11_re, acc11_im);
        }

        float f00_re = (float)acc00_re, f00_im = (float)acc00_im;
        float f01_re = (float)acc01_re, f01_im = (float)acc01_im;
        float f10_re = (float)acc10_re, f10_im = (float)acc10_im;
        float f11_re = (float)acc11_re, f11_im = (float)acc11_im;

        // warp reduce helper
        auto warp_sum = [&](float v) {
            for (unsigned i = warpSize/2; i >= 1; i >>= 1) {
                v += __gpu_shfl_down(v, i);
            }
            return v;
        };

        f00_re = warp_sum(f00_re); f00_im = warp_sum(f00_im);
        f01_re = warp_sum(f01_re); f01_im = warp_sum(f01_im);
        f10_re = warp_sum(f10_re); f10_im = warp_sum(f10_im);
        f11_re = warp_sum(f11_re); f11_im = warp_sum(f11_im);

        if (lane_id == 0) {
            size_t base = baseline * 4u;
            // 00
            out_data[base + 0].real = fmaf(f00_re, norm, out_data[base + 0].real);
            out_data[base + 0].imag = fmaf(f00_im, norm, out_data[base + 0].imag);
            // 01
            out_data[base + 1].real = fmaf(f01_re, norm, out_data[base + 1].real);
            out_data[base + 1].imag = fmaf(f01_im, norm, out_data[base + 1].imag);
            // 10
            out_data[base + 2].real = fmaf(f10_re, norm, out_data[base + 2].real);
            out_data[base + 2].imag = fmaf(f10_im, norm, out_data[base + 2].imag);
            // 11
            out_data[base + 3].real = fmaf(f11_re, norm, out_data[base + 3].real);
            out_data[base + 3].imag = fmaf(f11_im, norm, out_data[base + 3].imag);
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

    const int n_threads_per_block = warp_size * WARPS_PER_BLOCK;
    const int n_total_warps = n_baselines * nOutFrequencies;
    // ceil(n_total_warps / WARPS_PER_BLOCK)
    const int n_blocks = (n_total_warps + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    for (int i {0}; i < nIntervals; i++) {
        const gpuStream_t stream = streams[i % nStreams];
        if (!voltages.on_gpu()) {
            const size_t off = i*samplesInTimeInterval;
            const size_t count = samplesInTimeInterval;
            gpuMemcpyAsync(
                dev_voltages.data() + off,
                voltages.data() + off,
                sizeof(Complex<int8_t>) * count,
                gpuMemcpyHostToDevice,
                stream
            );
        }
        cross_correlation_kernel<2><<<dim3(n_blocks), dim3(n_threads_per_block), 0, stream>>> (
            dev_voltages_data + i*samplesInTimeInterval,
            obs_info,
            voltages.nIntegrationSteps,
            n_channels_to_avg,
            reinterpret_cast<Complex<float>*>(dev_xcorr.data())
        );
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
        cross_correlation_kernel<2> <<< dim3(n_blocks), dim3(n_threads_per_block), 0, streams[i % nStreams] >>> (
            reinterpret_cast<const Complex<float>*>(voltages) + i*samplesInTimeInterval, obsInfo, n_integrated_samples,
            n_channels_to_avg, dev_xcorr + i*nValuesInTimeInterval);
    }

    gpuDeviceSynchronize();
    for(int i {0}; i < nStreams; i++)
        gpuStreamDestroy(streams[i]);
    delete[] streams;

    return 0;
}
