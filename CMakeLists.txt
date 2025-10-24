cmake_minimum_required(VERSION 3.10.0)
project(correlation)
include(CTest)
set(CMAKE_CXX_STANDARD 17)

option(USE_OPENMP "Compile with OpenMP enabled." ON)
option(USE_CUDA "Compile the code with NVIDIA GPU support." OFF)
option(USE_HIP "Compile the code with AMD GPU support." OFF)

if(USE_CUDA)
enable_language(CUDA CXX)
endif()


# External dependencies
set(BLINK_TEST_DATADIR $ENV{BLINK_TEST_DATADIR})
if(NOT BLINK_TEST_DATADIR)
message(FATAL_ERROR "The 'BLINK_TEST_DATADIR' environment variable was not set and is needed for testing.")
endif()


find_library(BLINK_ASTROIO_LIB blink_astroio HINTS ENV LD_LIBRARY_PATH)
if( NOT BLINK_ASTROIO_LIB )
    message(FATAL_ERROR "BLINK_ASTROIO_LIB not found.")
endif()


file(GLOB correlation_sources 
    "src/correlation.cpp"
    "src/utils.cpp"
)

file(GLOB correlation_headers
    "src/correlation.hpp"
    "src/utils.hpp"
    "src/correlation.h"
)

file(GLOB apps "apps/*.cpp")
file(GLOB tests "tests/*.cpp")

if(USE_HIP OR USE_CUDA)
    file(GLOB gpu_sources "src/correlation_gpu.cpp")
    file(GLOB gpu_headers "src/correlation_gpu.hpp")
    list(APPEND correlation_sources ${gpu_sources})
    list(APPEND correlation_headers ${gpu_headers})
endif()


if(USE_CUDA)
set_source_files_properties( ${correlation_sources} ${correlation_headers} ${apps} ${tests} PROPERTIES LANGUAGE CUDA)
add_compile_options("--expt-relaxed-constexpr")
endif()


add_library(correlation SHARED ${correlation_sources})
set_target_properties(correlation PROPERTIES PUBLIC_HEADER "${correlation_headers}")
target_link_libraries(correlation ${BLINK_ASTROIO_LIB})


install(TARGETS correlation
    LIBRARY DESTINATION "lib"
    PUBLIC_HEADER DESTINATION "include"
)


# APPS
add_executable(blink-correlator apps/correlator.cpp)
target_link_libraries(blink-correlator correlation stdc++fs)
install(TARGETS blink-correlator DESTINATION "bin")

if(USE_OPENMP)
    find_package(OpenMP REQUIRED)
    target_link_libraries(correlation OpenMP::OpenMP_CXX)
    target_link_libraries(blink-correlator OpenMP::OpenMP_CXX)
    
endif()

# TESTS 
add_executable(correlation_test tests/correlation_test.cpp)
target_link_libraries(correlation_test correlation)
add_test(NAME correlation_test COMMAND correlation_test)

add_executable(utils_test tests/utils_test.cpp)
target_link_libraries(utils_test correlation)
add_test(NAME utils_test COMMAND utils_test)

