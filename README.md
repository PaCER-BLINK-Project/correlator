# Correlator

This is Cristian's implementation of a correlation library and a correlator application.

## Dependencies

You will need the following third party libraries to compile the project:

- BLINK AstroIO, part of the BLINK project.

Other dependencies are optional:

- OpenMP
- HIP/ROCm for AMD GPU acceleration.
- CUDA for NVIDIA GPU acceleration.

## Compiling

The compilation process is handled by CMake. To build a CPU only version, do the following:

```
mkdir build; cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make 
```

A CUDA enabled build can be obtained by simply adding the `-DUSE_CUDA=ON` argument to `cmake`.

To compile the code with HIP support, you will need to specify `-DUSE_HIP=ON -DCMAKE_CXX_COMPILER=hipcc`. 

To run tests, execute `make test`.

Available CMake flags are:

- `USE_OPENMP` (default: `ON`): run the correlation function in parallel on multiple cores.
- `USE_HIP` (default: `OFF`): build the library using HIP to enable AMD GPU support.
- `USE_CUDA` (default: `OFF`): build the library using CUDA to enable NVIDIA GPU support.

## Applications

The library comes with a CLI,  `blink_correlator`. Currently it only supports MWA's `.dat` data format.

Here is an example. To correlate one second worth of MWA data at full bandwith, at one second resolution and channel averaging factor of 4, the command to use is 

```
blink_correlator -t 1s -c 4 1276619416_1276619418_*.dat  
``` 
