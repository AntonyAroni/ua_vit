cmake_minimum_required(VERSION 3.18)
project(ia_vit LANGUAGES CXX CUDA)

# Set standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA path explicitly
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.6/bin/nvcc")
set(CUDAToolkit_ROOT "/usr/local/cuda-12.6")

# Find CUDA (modern way)
find_package(CUDAToolkit REQUIRED)

# Set architecture for GTX 1650 (compute capability 7.5)
set(CMAKE_CUDA_ARCHITECTURES "75")

# Source files
set(SOURCES
        main.cpp
        vit_transformer.cpp
        data_loader.cpp
        cuda_kernels.cu
)

# Main executable
add_executable(${PROJECT_NAME} ${SOURCES})

# CUDA properties
set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Link CUDA libraries
target_link_libraries(${PROJECT_NAME} PRIVATE
        CUDA::cudart
        CUDA::cublas
        CUDA::curand
)

# Optional cuDNN support
find_package(CUDNN QUIET)
if(CUDNN_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudnn)
    target_compile_definitions(${PROJECT_NAME} PRIVATE USE_CUDNN)
    message(STATUS "cuDNN found: v${CUDNN_VERSION}")
else()
    message(STATUS "cuDNN not found")
endif()

# Compiler options
target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -fopenmp>
        $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        --expt-extended-lambda
        --default-stream per-thread
        --use_fast_math
        -Xcompiler=-fopenmp
        >
)

# Debug/Release configurations
target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:-g -O0>
        $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G -g -O0>
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-O3 -DNDEBUG -march=native>
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-O3 -DNDEBUG>
)

# OpenMP linking
target_link_libraries(${PROJECT_NAME} PRIVATE
        $<$<CXX_COMPILER_ID:GNU>:gomp>
        $<$<CXX_COMPILER_ID:Clang>:omp>
)

# Data download target
add_custom_target(download_data
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/data
        COMMAND wget -nc -P ${CMAKE_SOURCE_DIR}/data https://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
        COMMAND wget -nc -P ${CMAKE_SOURCE_DIR}/data https://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
        COMMAND wget -nc -P ${CMAKE_SOURCE_DIR}/data https://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
        COMMAND wget -nc -P ${CMAKE_SOURCE_DIR}/data https://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
        COMMAND gunzip -f ${CMAKE_SOURCE_DIR}/data/*.gz
        COMMENT "Downloading Fashion-MNIST dataset"
)

# Install
install(TARGETS ${PROJECT_NAME} DESTINATION bin)