#include <cstdint>
#include <iostream>
#include <cmath>
#include <gpu_macros.hpp>
#include <astroio.hpp>
#include <complex>
#include <mycomplex.hpp>

#include "gpu_helpers.hpp"

#define WARPS_PER_BLOCK 8
#define N_POLARIZATIONS 2


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
        ccm(ch_data[A0 + step],                 ch_data[B0 + step],                 acc0);
        ccm(ch_data[A0 + step + warpSize],      ch_data[B0 + step + warpSize],      acc1);
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
    Complex<int> acc{0, 0};

    for (; step + 1 < n_integration_steps; step += 2u * warpSize) {
        int A = load_32bits(&ch_data[A0 + step]);
        int B = load_32bits(&ch_data[B0 + step]);
        ccm_dp4a(A, B, acc);
    }

    // get any remainder
    for (; step < n_integration_steps; step += 2u * warpSize) {
        auto &a = ch_data[A0 + step];
        auto &b = ch_data[B0 + step];
        ccm(a, b, acc);
    }

    Complex<float> facc{
        static_cast<float>(acc.real),
        static_cast<float>(acc.imag)
    };
    return facc;
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

                acc = warp_sum(acc);

                if (lane_id == 0) {
                    size_t out_index = baseline*NPOL*NPOL + pol_a*NPOL + pol_b;
                    out_data[out_index].real = fmaf(acc.real, norm, out_data[out_index].real);
                    out_data[out_index].imag = fmaf(acc.imag, norm, out_data[out_index].imag);
                }
            }
        }
    }
}

/**
 * @brief Specialisation of cross correlation kernel for 2 polarizations and 8-bit
 * integer samples.
 *
 * Fuses the polarization loops to eliminate branching and increase memory ops
 * per loop. Also uses the same packed arithmetic optimisations as the generic
 * kernel.
 */
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

        // base indicies for each antenna (A and B) and each polarization (0 and 1)
        const size_t base_A0 = base_a + 0u * n_integration_steps;
        const size_t base_A1 = base_a + 1u * n_integration_steps;
        const size_t base_B0 = base_b + 0u * n_integration_steps;
        const size_t base_B1 = base_b + 1u * n_integration_steps;

        unsigned int step = 2u * lane_id;

        // accumulators for each ant x pol combo
        Complex<int> acc00{0, 0}; // for A0 x B0
        Complex<int> acc01{0, 0}; // for A0 x B1
        Complex<int> acc10{0, 0}; // for A1 x B0
        Complex<int> acc11{0, 0}; // for A1 x B1

        // each warp loads 2 x warpSize samples (2 x 16 x warpSize bits) at a time
        for (; step + 1 < n_integration_steps; step += 2u * warpSize) {
            int A0 = load_32bits(&ch_data[base_A0 + step]);
            int A1 = load_32bits(&ch_data[base_A1 + step]);
            int B0 = load_32bits(&ch_data[base_B0 + step]);
            int B1 = load_32bits(&ch_data[base_B1 + step]);

            ccm_dp4a(A0, B0, acc00);
            ccm_dp4a(A0, B1, acc01);
            ccm_dp4a(A1, B0, acc10);
            ccm_dp4a(A1, B1, acc11);
        }

        // get any remainder
        for (; step < n_integration_steps; step += 2u * warpSize) {
            const auto &a0 = ch_data[base_A0 + step];
            const auto &a1 = ch_data[base_A1 + step];
            const auto &b0 = ch_data[base_B0 + step];
            const auto &b1 = ch_data[base_B1 + step];
            ccm(a0, b0, acc00);
            ccm(a0, b1, acc01);
            ccm(a1, b0, acc10);
            ccm(a1, b1, acc11);
        }

        Complex<float> f00 {(float)acc00.real, (float)acc00.imag};
        Complex<float> f01 {(float)acc01.real, (float)acc01.imag};
        Complex<float> f10 {(float)acc10.real, (float)acc10.imag};
        Complex<float> f11 {(float)acc11.real, (float)acc11.imag};

        f00 = warp_sum(f00);
        f01 = warp_sum(f01);
        f10 = warp_sum(f10);
        f11 = warp_sum(f11);

        if (lane_id == 0) {
            size_t base = baseline * NPOL * NPOL;
            // 00
            out_data[base + 0].real = fmaf(f00.real, norm, out_data[base + 0].real);
            out_data[base + 0].imag = fmaf(f00.imag, norm, out_data[base + 0].imag);
            // 01
            out_data[base + 1].real = fmaf(f01.real, norm, out_data[base + 1].real);
            out_data[base + 1].imag = fmaf(f01.imag, norm, out_data[base + 1].imag);
            // 10
            out_data[base + 2].real = fmaf(f10.real, norm, out_data[base + 2].real);
            out_data[base + 2].imag = fmaf(f10.imag, norm, out_data[base + 2].imag);
            // 11
            out_data[base + 3].real = fmaf(f11.real, norm, out_data[base + 3].real);
            out_data[base + 3].imag = fmaf(f11.imag, norm, out_data[base + 3].imag);
        }
    }
}

Visibilities cross_correlation_gpu(const Voltages& voltages, unsigned int n_channels_to_avg) {
    const ObservationInfo& obs_info {voltages.obsInfo};

    if(n_channels_to_avg < 1 || n_channels_to_avg > obs_info.nFrequencies) {
        std::stringstream ss;
        ss << "number of channels to average (" << n_channels_to_avg << ")"
           << " is greater than the number of frequencies (" << obs_info.nFrequencies << ")";
        throw std::invalid_argument(ss.str());
    }
    if(obs_info.nTimesteps % voltages.nIntegrationSteps != 0) {
        std::stringstream ss;
        ss << "number of timesteps (" << obs_info.nTimesteps << ")"
           << " is not an integer multiple of the number of integration steps (" << voltages.nIntegrationSteps << ")";
        throw std::invalid_argument(ss.str());
    }
    if (!voltages.on_gpu() && !voltages.pinned()) {
        std::cerr << "'cross_correlation_gpu' warning: CPU memory is not pinned.\n"
                     "This will result in poor performance." << std::endl;
    }

    if (obs_info.nPolarizations != N_POLARIZATIONS) {
        std::stringstream ss;
        ss << "expected " << N_POLARIZATIONS << " polarizations per antenna"
           << " (found " << obs_info.nPolarizations << ")";
        throw std::invalid_argument(ss.str());
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

	const size_t nStreams = std::min(5ul, nIntervals);

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
            reinterpret_cast<Complex<float>*>(dev_xcorr.data()) + i*nValuesInTimeInterval
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

	const size_t nStreams = std::min(5ul, nIntervals);

    gpuStream_t *streams {new gpuStream_t[nStreams]};

    for(size_t i = 0; i < nStreams; i++)
        gpuStreamCreate(streams + i);

    // retrieve warp size (32 on NVIDIA, 64 on AMD MI250X)
    int device_id, warp_size;
    gpuGetDevice(&device_id);
    gpuDeviceGetAttribute(&warp_size, gpuDeviceAttributeWarpSize, device_id);
    const int n_threads_per_block {warp_size * WARPS_PER_BLOCK};
    const int n_total_warps {static_cast<int>(n_baselines * nOutFrequencies)};
    const int n_blocks {(n_total_warps + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK};

    for(int i {0}; i < nIntervals; i++){
        cross_correlation_kernel<2> <<<dim3(n_blocks), dim3(n_threads_per_block), 0, streams[i % nStreams]>>> (
            reinterpret_cast<const Complex<float>*>(voltages) + i*samplesInTimeInterval, obsInfo, n_integrated_samples,
            n_channels_to_avg, dev_xcorr + i*nValuesInTimeInterval);
    }

    gpuDeviceSynchronize();
    for(int i {0}; i < nStreams; i++)
        gpuStreamDestroy(streams[i]);
    delete[] streams;

    return 0;
}
