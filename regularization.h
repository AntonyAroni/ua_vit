#pragma once

#include <cuda_runtime.h>

extern "C" {
    // Weight decay kernel
    void launch_weight_decay(float* weights, float* gradients, float weight_decay, int size);
    
    // Dropout kernels
    void launch_dropout_forward(const float* input, float* output, float* mask, 
                               float dropout_rate, int size, int seed);
    void launch_dropout_backward(const float* grad_output, const float* mask, 
                               float* grad_input, int size);
}