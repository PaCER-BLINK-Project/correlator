#include <cstdint>
#include <iostream>
#include <cmath>
#include "correlation.hpp"
#ifdef __GPU__
#include "correlation_gpu.hpp"
#endif


Visibilities cross_correlation(const Voltages& voltages, unsigned int nChannelsToAvg){
    #ifdef __GPU__
    if(gpu_support() && num_available_gpus() > 0)
        return cross_correlation_gpu(voltages, nChannelsToAvg);
    else
        return cross_correlation_cpu(voltages, nChannelsToAvg);
    #else
    return cross_correlation_cpu(voltages, nChannelsToAvg);
    #endif
}


/**
 * @brief Perform the cross correlation of voltage data across all the available baselines.
 * 
 * @param voltages: A Voltage object containing voltage data.
 * @return Correlation data stored in lower triangular form.
 */
Visibilities cross_correlation_cpu(const Voltages& voltages, unsigned int nChannelsToAvg){
    std::cout << "Correlation is happening on CPU.." << std::endl;
    if(voltages.on_gpu()) throw std::invalid_argument {"Voltages are allocated in GPU memory."};
    if(nChannelsToAvg < 1 || nChannelsToAvg > voltages.obsInfo.nFrequencies)
        throw std::invalid_argument {"NChannelsToAvg is out of range."};
    if(voltages.obsInfo.nTimesteps % voltages.nIntegrationSteps != 0)
        throw std::invalid_argument {"nTimesteps is not an integer multiple of nIntegrationSteps."};

    // values to compute output size and indexing
    const ObservationInfo& obsInfo {voltages.obsInfo};
    const unsigned int n_baselines {((obsInfo.nAntennas + 1) * obsInfo.nAntennas) / 2};
    const size_t matrixSize {n_baselines * obsInfo.nPolarizations * obsInfo.nPolarizations};
    const size_t nIntervals {(obsInfo.nTimesteps + voltages.nIntegrationSteps - 1) / voltages.nIntegrationSteps};
    const size_t nOutFrequencies {obsInfo.nFrequencies / nChannelsToAvg};
    const size_t nValuesInTimeInterval {matrixSize * nOutFrequencies};
    const size_t outSize {nValuesInTimeInterval * nIntervals};

    MemoryBuffer<std::complex<float>> xcorr {outSize};
    memset((void*)xcorr.data(), 0, sizeof(std::complex<float>) *outSize);
    // variables used to compute input index
    const size_t samplesInPol {voltages.nIntegrationSteps};
    const size_t samplesInAntenna {samplesInPol * obsInfo.nPolarizations};
    const size_t samplesInFrequency {samplesInAntenna * obsInfo.nAntennas};
    const size_t samplesInTimeInterval {samplesInFrequency * obsInfo.nFrequencies};

    const double integrationTime {obsInfo.timeResolution * voltages.nIntegrationSteps};

    #pragma omp parallel for schedule(static) if(nIntervals > 10)
    for(unsigned int interval = 0; interval < nIntervals; interval++){
        #pragma omp parallel for schedule(static) if(nIntervals <= 10)
        for(unsigned int baseline = 0; baseline < n_baselines; baseline++){
            for(unsigned int ch {0}; ch < obsInfo.nFrequencies; ch++){
                unsigned int avgCh {ch / nChannelsToAvg};            
                // For more info about the following formula, check
                // https://stackoverflow.com/questions/40950460/how-to-convert-triangular-matrix-indexes-in-to-row-column-coordinates
                unsigned int a1 {static_cast<unsigned int>(-0.5 + std::sqrt(0.25 + 2*baseline))};
                unsigned int a2 {baseline - ((a1 + 1) * a1)/2};
                // for each baseline compute the correlation matrix of its polarization
                for(unsigned int p1 {0}; p1 < obsInfo.nPolarizations; p1++){
                    for(unsigned int p2 {0}; p2 < obsInfo.nPolarizations; p2++){
                        std::complex<int64_t> accum {0, 0};
                        for(unsigned int step {0}; step < voltages.nIntegrationSteps; step++){
                            const size_t iA {interval * samplesInTimeInterval + ch * samplesInFrequency + a1 * samplesInAntenna + p1 * samplesInPol + step};
                            const size_t iB {interval * samplesInTimeInterval + ch * samplesInFrequency + a2 * samplesInAntenna + p2 * samplesInPol + step};
                            ccm(voltages[iA], voltages[iB], accum);
                        }
                        size_t outIndex {interval * nValuesInTimeInterval + avgCh * matrixSize + baseline * obsInfo.nPolarizations * obsInfo.nPolarizations
                            + p1*obsInfo.nPolarizations + p2};
                        xcorr[outIndex] += std::complex<float> { static_cast<float>(accum.real() / (integrationTime * nChannelsToAvg)),
                            static_cast<float>(accum.imag() / (integrationTime * nChannelsToAvg))};
                    }
                }
            }
        }
    }
    return {std::move(xcorr), obsInfo, voltages.nIntegrationSteps, nChannelsToAvg};
}
