#ifndef __BLINK_CORRELATOR_H__
#define __BLINK_CORRELATOR_H__
#ifdef __cplusplus
extern "C" {  
#endif  

int blink_cross_correlation_gpu(const float* voltages, float* visibilities, 
        unsigned int n_antennas, unsigned int n_polarisations,
        unsigned int n_fine_channels, unsigned int n_time_samples, double time_resolution,
        unsigned int n_integrated_samples, unsigned int n_channels_to_avg, unsigned int reset_visibilities);

#ifdef __cplusplus  
} // extern "C"  
#endif
#endif