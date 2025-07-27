cmake_minimum_required(VERSION 3.18)
project(ia_vit LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)
enable_language(CUDA)

# Find cuBLAS and cuRAND (cuDNN is optional)
find_library(CUBLAS_LIBRARIES cublas HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64 ${CUDA_TOOLKIT_ROOT_DIR}/lib)
find_library(CURAND_LIBRARIES curand HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64 ${CUDA_TOOLKIT_ROOT_DIR}/lib)
find_library(CUDNN_LIBRARIES cudnn HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64 ${CUDA_TOOLKIT_ROOT_DIR}/lib)

# Check if cuDNN is available
if(CUDNN_LIBRARIES)
    message(STATUS "cuDNN found: ${CUDNN_LIBRARIES}")
    set(USE_CUDNN ON)
else()
    message(STATUS "cuDNN not found, using custom implementations")
    set(USE_CUDNN OFF)
endif()

# Include directories
include_directories(${CUDA_INCLUDE_DIRS})

# CUDA compilation flags optimized for GTX 1650 (compute capability 7.5)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_75,code=sm_75")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_75,code=compute_75")

# Optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")

# Enable fast math for CUDA
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")

# Source files
set(SOURCES
    main.cpp
    vit_transformer.cpp
    data_loader.cpp
)

# CUDA kernel files
set(CUDA_SOURCES
    cuda_kernels.cu
)

# Create executable
add_executable(${PROJECT_NAME} ${SOURCES} ${CUDA_SOURCES})

# Set properties for CUDA files
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# Link libraries
target_link_libraries(${PROJECT_NAME} 
    ${CUDA_LIBRARIES}
    ${CUBLAS_LIBRARIES}
    ${CURAND_LIBRARIES}
)

# Link cuDNN if available
if(USE_CUDNN)
    target_link_libraries(${PROJECT_NAME} ${CUDNN_LIBRARIES})
    target_compile_definitions(${PROJECT_NAME} PRIVATE USE_CUDNN)
endif()

# Compiler-specific options
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(${PROJECT_NAME} PRIVATE 
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -fopenmp>
    )
    target_link_libraries(${PROJECT_NAME} gomp)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(${PROJECT_NAME} PRIVATE 
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -fopenmp>
    )
    target_link_libraries(${PROJECT_NAME} omp)
endif()

# CUDA-specific options
target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        --expt-extended-lambda
        --default-stream per-thread
        -Xcompiler=-fopenmp
    >
)

# Debug and Release configurations
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-g -O0>
        $<$<COMPILE_LANGUAGE:CUDA>:-g -G -O0>
    )
else()
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3 -DNDEBUG>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 -DNDEBUG>
    )
endif()

# Custom target for downloading Fashion-MNIST data
add_custom_target(download_data
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/data
    COMMAND wget -nc -P ${CMAKE_BINARY_DIR}/data http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz || true
    COMMAND wget -nc -P ${CMAKE_BINARY_DIR}/data http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz || true
    COMMAND wget -nc -P ${CMAKE_BINARY_DIR}/data http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz || true
    COMMAND wget -nc -P ${CMAKE_BINARY_DIR}/data http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz || true
    COMMAND gunzip -f ${CMAKE_BINARY_DIR}/data/*.gz || true
    COMMENT "Downloading and extracting Fashion-MNIST dataset"
)

# Create CUDA test executable
add_executable(test_cuda test_cuda.cu)
set_property(TARGET test_cuda PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(test_cuda ${CUDA_LIBRARIES})

# Create quick test executable
add_executable(quick_test quick_test.cpp vit_transformer.cpp data_loader.cpp ${CUDA_SOURCES})
set_property(TARGET quick_test PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET quick_test PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(quick_test ${CUDA_LIBRARIES} ${CUBLAS_LIBRARIES} ${CURAND_LIBRARIES})
if(USE_CUDNN)
    target_link_libraries(quick_test ${CUDNN_LIBRARIES})
    target_compile_definitions(quick_test PRIVATE USE_CUDNN)
endif()

# Create data loader test executable
add_executable(test_data_loader test_data_loader.cpp data_loader.cpp)
target_link_libraries(test_data_loader ${CUDA_LIBRARIES})

# Create debug training executable
add_executable(debug_training debug_training.cpp data_loader.cpp)
target_link_libraries(debug_training ${CUDA_LIBRARIES})

# Create simple ViT test executable
add_executable(simple_vit simple_vit.cpp data_loader.cpp)
target_link_libraries(simple_vit ${CUDA_LIBRARIES} ${CUBLAS_LIBRARIES})

# Create hybrid ViT executable
add_executable(hybrid_vit hybrid_main.cpp hybrid_vit.cpp data_loader.cpp hybrid_kernels.cu)
set_property(TARGET hybrid_vit PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET hybrid_vit PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(hybrid_vit ${CUDA_LIBRARIES} ${CUBLAS_LIBRARIES} ${CURAND_LIBRARIES})
if(USE_CUDNN)
    target_link_libraries(hybrid_vit ${CUDNN_LIBRARIES})
    target_compile_definitions(hybrid_vit PRIVATE USE_CUDNN)
endif()

# Install target
install(TARGETS ${PROJECT_NAME} DESTINATION bin)

# Print configuration information
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA version: ${CUDA_VERSION}")
message(STATUS "CUDA libraries: ${CUDA_LIBRARIES}")
message(STATUS "cuBLAS libraries: ${CUBLAS_LIBRARIES}")
message(STATUS "cuDNN libraries: ${CUDNN_LIBRARIES}")
