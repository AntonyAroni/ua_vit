#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

#define BLOCK_SIZE 256
#define TILE_SIZE 16

// GELU activation function
__device__ float gelu_activation(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

// Patch embedding kernel - simplified and robust
__global__ void patch_embed_kernel(const float* input, float* output, 
                                 const float* weight, const float* bias,
                                 int batch_size, int img_size, int patch_size, 
                                 int embed_dim) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int patches_per_side = img_size / patch_size;
    int num_patches = patches_per_side * patches_per_side;
    int patch_pixels = patch_size * patch_size;
    
    if (tid < batch_size * num_patches * embed_dim) {
        int batch_idx = tid / (num_patches * embed_dim);
        int patch_idx = (tid % (num_patches * embed_dim)) / embed_dim;
        int embed_idx = tid % embed_dim;
        
        // Calculate patch position
        int patch_row = patch_idx / patches_per_side;
        int patch_col = patch_idx % patches_per_side;
        
        // Linear projection of flattened patch
        float result = bias[embed_idx];
        
        for (int i = 0; i < patch_size; i++) {
            for (int j = 0; j < patch_size; j++) {
                int img_row = patch_row * patch_size + i;
                int img_col = patch_col * patch_size + j;
                int pixel_idx = batch_idx * img_size * img_size + img_row * img_size + img_col;
                int weight_idx = embed_idx * patch_pixels + i * patch_size + j;
                
                result += input[pixel_idx] * weight[weight_idx];
            }
        }
        
        output[tid] = result;
    }
}

// Add positional embeddings and class token
__global__ void add_pos_embed_kernel(float* tokens, const float* pos_embed,
                                   const float* cls_token, int batch_size,
                                   int seq_len, int embed_dim) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < batch_size * seq_len * embed_dim) {
        int batch_idx = tid / (seq_len * embed_dim);
        int seq_idx = (tid % (seq_len * embed_dim)) / embed_dim;
        int embed_idx = tid % embed_dim;
        
        if (seq_idx == 0) {
            // First position is class token
            tokens[tid] = cls_token[embed_idx] + pos_embed[embed_idx];
        } else {
            // Add positional embedding to existing token
            int pos_idx = seq_idx * embed_dim + embed_idx;
            tokens[tid] += pos_embed[pos_idx];
        }
    }
}

// Layer normalization kernel
__global__ void layer_norm_kernel(const float* input, float* output,
                                const float* weight, const float* bias,
                                int batch_size, int seq_len, int embed_dim) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < batch_size * seq_len) {
        int batch_idx = tid / seq_len;
        int seq_idx = tid % seq_len;
        int base_idx = batch_idx * seq_len * embed_dim + seq_idx * embed_dim;
        
        // Compute mean
        float mean = 0.0f;
        for (int i = 0; i < embed_dim; i++) {
            mean += input[base_idx + i];
        }
        mean /= embed_dim;
        
        // Compute variance
        float variance = 0.0f;
        for (int i = 0; i < embed_dim; i++) {
            float diff = input[base_idx + i] - mean;
            variance += diff * diff;
        }
        variance /= embed_dim;
        
        // Normalize
        float inv_std = rsqrtf(variance + 1e-6f);
        for (int i = 0; i < embed_dim; i++) {
            float normalized = (input[base_idx + i] - mean) * inv_std;
            output[base_idx + i] = normalized * weight[i] + bias[i];
        }
    }
}

// GELU activation kernel
__global__ void gelu_kernel(const float* input, float* output, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < size) {
        output[tid] = gelu_activation(input[tid]);
    }
}

// Softmax kernel for classification
__global__ void softmax_kernel(const float* input, float* output,
                             int batch_size, int num_classes) {
    int batch_idx = blockIdx.x;
    
    if (batch_idx < batch_size) {
        int base_idx = batch_idx * num_classes;
        
        // Find maximum for numerical stability
        float max_val = input[base_idx];
        for (int i = 1; i < num_classes; i++) {
            max_val = fmaxf(max_val, input[base_idx + i]);
        }
        
        // Compute exp and sum
        float sum_exp = 0.0f;
        for (int i = 0; i < num_classes; i++) {
            float exp_val = expf(input[base_idx + i] - max_val);
            output[base_idx + i] = exp_val;
            sum_exp += exp_val;
        }
        
        // Normalize
        for (int i = 0; i < num_classes; i++) {
            output[base_idx + i] /= sum_exp;
        }
    }
}

// Cross entropy loss kernel
__global__ void cross_entropy_kernel(const float* predictions, const int* targets,
                                   float* loss, int batch_size, int num_classes) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < batch_size) {
        int target = targets[tid];
        int pred_idx = tid * num_classes + target;
        
        // Compute softmax-cross entropy loss
        float max_pred = predictions[tid * num_classes];
        for (int i = 1; i < num_classes; i++) {
            max_pred = fmaxf(max_pred, predictions[tid * num_classes + i]);
        }
        
        float sum_exp = 0.0f;
        for (int i = 0; i < num_classes; i++) {
            sum_exp += expf(predictions[tid * num_classes + i] - max_pred);
        }
        
        float log_prob = predictions[pred_idx] - max_pred - logf(sum_exp);
        loss[tid] = -log_prob;
    }
}

// Launcher functions
extern "C" {
    void launch_patch_embed_kernel(const float* input, float* output, 
                                 const float* weight, const float* bias,
                                 int batch_size, int img_size, int patch_size, 
                                 int embed_dim) {
        int patches_per_side = img_size / patch_size;
        int num_patches = patches_per_side * patches_per_side;
        int total_elements = batch_size * num_patches * embed_dim;
        
        int blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
        patch_embed_kernel<<<blocks, BLOCK_SIZE>>>(
            input, output, weight, bias, batch_size, img_size, patch_size, embed_dim
        );
        cudaDeviceSynchronize();
    }
    
    void launch_add_pos_embed_kernel(float* tokens, const float* pos_embed,
                                   const float* cls_token, int batch_size,
                                   int seq_len, int embed_dim) {
        int total_elements = batch_size * seq_len * embed_dim;
        int blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
        
        add_pos_embed_kernel<<<blocks, BLOCK_SIZE>>>(
            tokens, pos_embed, cls_token, batch_size, seq_len, embed_dim
        );
        cudaDeviceSynchronize();
    }
    
    void launch_layer_norm_kernel(const float* input, float* output,
                                const float* weight, const float* bias,
                                int batch_size, int seq_len, int embed_dim) {
        int blocks = (batch_size * seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
        
        layer_norm_kernel<<<blocks, BLOCK_SIZE>>>(
            input, output, weight, bias, batch_size, seq_len, embed_dim
        );
        cudaDeviceSynchronize();
    }
    
    void launch_gelu_kernel(const float* input, float* output, int size) {
        int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        
        gelu_kernel<<<blocks, BLOCK_SIZE>>>(input, output, size);
        cudaDeviceSynchronize();
    }
    
    void launch_softmax_kernel(const float* input, float* output,
                             int batch_size, int num_classes) {
        softmax_kernel<<<batch_size, 1>>>(input, output, batch_size, num_classes);
        cudaDeviceSynchronize();
    }
    
    void launch_cross_entropy_kernel(const float* predictions, const int* targets,
                                   float* loss, int batch_size, int num_classes) {
        int blocks = (batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        
        cross_entropy_kernel<<<blocks, BLOCK_SIZE>>>(
            predictions, targets, loss, batch_size, num_classes
        );
        cudaDeviceSynchronize();
    }
}