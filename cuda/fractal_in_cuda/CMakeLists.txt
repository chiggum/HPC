cmake_minimum_required (VERSION 2.6)
project(fractal)

include_directories(/usr/local/include)
include_directories(/usr/include)
include_directories(/usr/local/cuda-6.5/include)
link_directories(/usr/local/lib)
link_directories(/usr/lib)
link_directories(/usr/lib64)
link_directories(/usr/local/cuda-6.5/lib)
link_directories(/usr/local/cuda-6.5/lib64)

set(CMAKE_CXX_COMPILER /usr/local/cuda-6.5/bin/nvcc)
add_executable(fractal fractals.cu)