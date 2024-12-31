cmake_minimum_required(VERSION 3.18)
project(dynamic_slice LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/include)

#FIND_PACKAGE(CUDA REQUIRED)
#INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(/usr/local/cuda/include)
LINK_DIRECTORIES(/usr/local/cuda/lib64)

INCLUDE_DIRECTORIES(/home/xiaoying/code/dep/TensorRT-10.6.0.26/include)
LINK_DIRECTORIES(/home/xiaoying/code/dep/TensorRT-10.6.0.26/lib)

add_executable(dynamic_slice main.cpp)
target_link_libraries(dynamic_slice cudart)
target_link_libraries(dynamic_slice nvinfer)
target_link_libraries(dynamic_slice nvinfer_plugin)
target_link_libraries(dynamic_slice pthread)
