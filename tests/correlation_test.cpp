#include <exception>
#include <string>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <stdexcept>

#include <astroio.hpp>
#include "../src/utils.hpp"
#include "../src/correlation.hpp"
#include "common.hpp"

#include <cstdio>

std::string dataRootDir;



void test_complex_conjugate_multiply(){
    
    std::complex<int> a, b, res {0, 0};
    // subtest 1: multiply a number by its conjugate
    a.real(8);
    a.imag(3);

    b.real(8);
    b.imag(3);
    ccm(a, b, res);
    if(res.real() != 73 || res.imag() != 0){
        throw TestFailed("test_complex_conjugate_multiply failed at subtest 1.");
    }

    // subtest 2: multiply a number by another conjugate
    res = std::complex<int> {0, 0};
    b.real(5);
    b.imag(1);

    // TODO: fix this test.
    std::cout << "'test_complex_conjugate_multiply' passed." << std::endl;
}



template <typename T>
bool complex_vectors_equal(const std::complex<T>* a, const std::complex<T>* b, size_t length){
    double delta;
    const double TOL {0};
    for(size_t i {0}; i < length; i++){
        if (std::abs(a[i]) == 0) 
            delta = std::abs(b[i]);
        else 
            delta = std::abs(a[i] - b[i]);
        
        if (delta > TOL) {
            std::cout << "Elements at position " << i << " differs (delta = " << delta <<"): " << "a[i] = " << a[i] << ", b[i] = " << b[i] << std::endl;
            return false;
        }
    }
    return true;
}



void test_correlation_with_xgpu_data(){
    char *inputData, *outputData;
    size_t insize, outsize;
    read_data_from_file(dataRootDir + "/xGPU/input_array_128_128_128_100.bin", inputData, insize);
    read_data_from_file(dataRootDir + "/xGPU/output_matrix_128_128_128_100.bin", outputData, outsize);
    
    ObservationInfo obsInfo {VCS_OBSERVATION_INFO};
    obsInfo.nTimesteps = 100; // xgpu processes 1/10th of the total 1hour VCS observation at a time.
    auto voltages = Voltages::from_memory((int8_t*) inputData, insize, obsInfo, 100);
    auto xcorr = cross_correlation_cpu(voltages);
    
    const std::complex<float>* a {reinterpret_cast<std::complex<float>*>(outputData)};
    const std::complex<float>* b {xcorr.data()};
    // xGPU does not compute the time average and does not average channels, so we need to scale back
    // the correlator result.
    const float factor {static_cast<float>(obsInfo.timeResolution * voltages.nIntegrationSteps)};
     for(size_t i {0}; i < xcorr.size(); i++){
        if(a[i] != (b[i] * factor)){
            std::cout << "Elements at position " << i << " differs: " << "a[i] = " << a[i] << ", b[i] = " << b[i] << std::endl;
            throw TestFailed("test_corrrelation_with_xgpu_data failed.");
        }
     }

    delete[] inputData;
    delete[] outputData;
    std::cout << "'test_correlation_with_xgpu_data' passed." << std::endl;
}


void test_correlation_with_eda2_data(){
    auto volt = Voltages::from_eda2_file(dataRootDir + "/eda2/channel_cont_20220118_41581_0_binary.bin", EDA2_OBSERVATION_INFO, 262144);
    auto xcorr = cross_correlation_cpu(volt, 1);
    // TODO improve this test
    std::cout << "'test_correlation_with_eda2_data' passed." << std::endl;
}



void test_correlation_with_offline_correlator_data(){
    auto volt = Voltages::from_dat_file(dataRootDir + "/offline_correlator/1240826896_1240827191_ch146.dat", VCS_OBSERVATION_INFO, 1000);
    auto v1 = cross_correlation_cpu(volt, 32);
    auto v2 = Visibilities::from_fits_file(dataRootDir + "/offline_correlator/1313388760_20110815061242_gpubox20_00.fits");
    
    if (!complex_vectors_equal(v1.data(), v2.data(), v1.size())){
        throw TestFailed("test_corrrelation_with_offline_correlator_data failed.");
    }
    std::cout << "'test_correlation_with_offline_correlator_data' passed." << std::endl;
}



void test_correlation_bad_input() {
    auto volt = Voltages::from_dat_file(dataRootDir + "/offline_correlator/1240826896_1240827191_ch146.dat", VCS_OBSERVATION_INFO, 1000);
    bool badParamCaught {false};

    try {
        Visibilities vis = cross_correlation_cpu(volt, 0);
    } catch (std::invalid_argument& ex){
        badParamCaught = true;
    }
    if(!badParamCaught) throw TestFailed("test_correlation_bad_input: didn't check for meaningful channel averaging parameter.");
    badParamCaught = false;

    try {
        Visibilities vis = cross_correlation_cpu(volt, 512);
    } catch (std::invalid_argument& ex) {
        badParamCaught = true;
    }
    if(!badParamCaught) throw TestFailed("test_correlation_bad_input: didn't check for nIntegrationSteps to be a integer multiple of nTimesteps.");
    std::cout << "'test_correlation_with_bad_input' passed." << std::endl;
}



#ifdef __GPU__
#include "../src/correlation_gpu.hpp"
void test_correlation_gpu(){
    // auto start_all = std::chrono::high_resolution_clock::now();
    auto volt = Voltages::from_dat_file(dataRootDir + "/offline_correlator/1240826896_1240827191_ch146.dat", VCS_OBSERVATION_INFO, 100);
    auto xcorr_cpu = cross_correlation_cpu(volt, 32);
    //auto start = std::chrono::high_resolution_clock::now();
    auto xcorr_gpu = cross_correlation_gpu(volt, 32);
    xcorr_gpu.to_cpu();
    // auto stop = std::chrono::high_resolution_clock::now();
    // std::cout << "corr execution time (ms): " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << std::endl;
    // xcorr_gpu.to_fits_file("xcorr_gpu.fits");
    if (!complex_vectors_equal(xcorr_cpu.data(), xcorr_gpu.data(), xcorr_gpu.size())){
       throw TestFailed("test_corrrelation_gpu failed.");
    }
    // auto stop_all = std::chrono::high_resolution_clock::now();
    // std::cout << "all execution time (ms): " << std::chrono::duration_cast<std::chrono::milliseconds>(stop_all - start_all).count() << std::endl;
    std::cout << "'test_correlation_gpu' passed." << std::endl;
}
#endif



int main(void){
    char *pathToData {std::getenv(ENV_DATA_ROOT_DIR)};
    if(!pathToData){
        std::cerr << "'" << ENV_DATA_ROOT_DIR << "' environment variable is not set." << std::endl;
        return -1;
    }
    dataRootDir = std::string {pathToData};

    try{
        test_complex_conjugate_multiply();
        test_correlation_with_xgpu_data();
        test_correlation_with_offline_correlator_data();
        test_correlation_with_eda2_data();
        test_correlation_bad_input();
        #ifdef __GPU__
        test_correlation_gpu();
        #endif
    } catch (std::exception& ex){
        std::cerr << ex.what() << std::endl;
        return 1;
    }
    
    std::cout << "All tests passed." << std::endl;
    return 0;
}
