#include <cuda_runtime.h>
#include <iostream>

__global__ void testKernel() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Thread %d running on GPU!\n", idx);
}

int main() {
    // Check CUDA device properties
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    std::cout << "Number of CUDA devices: " << deviceCount << std::endl;
    
    if (deviceCount == 0) {
        std::cout << "No CUDA devices found!" << std::endl;
        return -1;
    }
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        std::cout << "\n=== Device " << i << " ===" << std::endl;
        std::cout << "Name: " << prop.name << std::endl;
        std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "Total Global Memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
        std::cout << "Shared Memory per Block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
        std::cout << "Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "Max Threads per MP: " << prop.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "Number of MPs: " << prop.multiProcessorCount << std::endl;
        std::cout << "Memory Clock Rate: " << prop.memoryClockRate / 1000 << " MHz" << std::endl;
        std::cout << "Memory Bus Width: " << prop.memoryBusWidth << " bits" << std::endl;
        std::cout << "Peak Memory Bandwidth: " << 2.0 * prop.memoryClockRate * (prop.memoryBusWidth/8) / 1.0e6 << " GB/s" << std::endl;
    }
    
    // Test simple kernel launch
    std::cout << "\n=== Testing Kernel Launch ===" << std::endl;
    testKernel<<<1, 8>>>();
    
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cout << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }
    
    std::cout << "Kernel executed successfully!" << std::endl;
    
    // Test memory allocation
    std::cout << "\n=== Testing Memory Allocation ===" << std::endl;
    
    size_t test_size = 1024 * 1024 * sizeof(float);  // 1M floats = 4MB
    float *d_test;
    
    err = cudaMalloc(&d_test, test_size);
    if (err != cudaSuccess) {
        std::cout << "Memory allocation failed: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }
    
    std::cout << "Successfully allocated " << test_size / (1024*1024) << " MB on GPU" << std::endl;
    
    cudaFree(d_test);
    
    std::cout << "\n=== CUDA Setup Test PASSED ===" << std::endl;
    return 0;
}