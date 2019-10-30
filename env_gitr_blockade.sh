module purge
module load gcc
module load mpich
module load cmake
module load netcdf
module load vim

ncxx=/lore/gopan/install/build-netcdfcxx431/install
#export NETCDF_PREFIX=$ncxx
export NetCDF_PREFIX=$ncxx
export CMAKE_PREFIX_PATH=$ncxx:$CMAKE_PREFIX_PATH
cuda=/usr/local/cuda-10.1
export THRUST_INCLUDE_DIR=$cuda/include
export PATH=$cuda/bin:$PATH
export LD_LIBRARY_PATH=$cuda/lib64:$LD_LIBRARY_PATH

