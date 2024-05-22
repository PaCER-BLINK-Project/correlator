#!/bin/bash -e
# First, you need to source the bash library
module load bash-utils
source "${BASH_UTILS_DIR}/build_utils.sh"

get_commit_hash

PROGRAM_NAME=blink_correlator
PROGRAM_VERSION=no-template #master #-${COMMIT_HASH:0:7}

 
# the following function sets up the installation path according to the
# cluster the script is running on and the first argument given. The argument
# can be:
# - "group": install the software in the group wide directory
# - "user": install the software only for the current user
# - "test": install the software in the current working directory 
process_build_script_input user


# load all the modules required for the program to compile and run.
# the following command also adds those module names in the modulefile
# that this script will generate.
echo "Loading required modules ..."
module reset 
module_load blink_test_data/devel blink_astroio/master rocm/5.4.3

# cmake is only required at build time, so we use the normal module load
module load cmake/3.24.3
# build your software..
echo "Building the software.."

[ -d build ] || mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} -DCMAKE_CXX_COMPILER=hipcc -DUSE_HIP=ON  -DCMAKE_CXX_FLAGS=--offload-arch=gfx90a -DCMAKE_C_COMPILER=hipcc -DCMAKE_BUILD_TYPE=Release
make VERBOSE=1

# Install the software
make test
# make install

# echo "Creating the modulefile.."
# create_modulefile

echo "Done."


