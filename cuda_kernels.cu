#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand_kernel.h>
#include <math.h>

#define BLOCK_SIZE 256
#define TILE_SIZE 16

__device__ float gelu_activation(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

__device__ float gelu_derivative(float x) {
    float tanh_arg = 0.7978845608f * (x + 0.044715f * x * x * x);
    float tanh_val = tanhf(tanh_arg);
    float sech2 = 1.0f - tanh_val * tanh_val;
    return 0.5f * (1.0f + tanh_val) + 0.5f * x * sech2 * 0.7978845608f * (1.0f + 3.0f * 0.044715f * x * x);
}

__global__ void patch_embedding_kernel(float* input, float* output, float* weight, 
                                     float* bias, int batch_size, int img_size, 
                                     int patch_size, int embed_dim) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int patches_per_side = img_size / patch_size;
    int num_patches = patches_per_side * patches_per_side;
    int total_elements = batch_size * num_patches * embed_dim;
    
    if (tid < total_elements) {
        int batch_idx = tid / (num_patches * embed_dim);
        int patch_idx = (tid % (num_patches * embed_dim)) / embed_dim;
        int embed_idx = tid % embed_dim;
        
        int patch_row = patch_idx / patches_per_side;
        int patch_col = patch_idx % patches_per_side;
        
        float sum = 0.0f;
        
        // Convolution-like operation for patch projection
        for (int pr = 0; pr < patch_size; pr++) {
            for (int pc = 0; pc < patch_size; pc++) {
                int img_row = patch_row * patch_size + pr;
                int img_col = patch_col * patch_size + pc;
                int input_idx = batch_idx * img_size * img_size + img_row * img_size + img_col;
                int weight_idx = embed_idx * patch_size * patch_size + pr * patch_size + pc;
                
                sum += input[input_idx] * weight[weight_idx];
            }
        }
        
        output[tid] = sum + bias[embed_idx];
    }
}

__global__ void add_positional_embedding_kernel(float* input, float* pos_embed, 
                                               float* cls_token, int batch_size, 
                                               int seq_len, int embed_dim) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * seq_len * embed_dim;
    
    if (tid < total_elements) {
        int batch_idx = tid / (seq_len * embed_dim);
        int seq_idx = (tid % (seq_len * embed_dim)) / embed_dim;
        int embed_idx = tid % embed_dim;
        
        if (seq_idx == 0) {
            // Add class token
            input[tid] = cls_token[embed_idx] + pos_embed[embed_idx];
        } else {
            // Add positional embedding
            int pos_idx = seq_idx * embed_dim + embed_idx;
            input[tid] += pos_embed[pos_idx];
        }
    }
}

__global__ void multi_head_attention_qkv_kernel(float* input, float* qkv_weight, 
                                               float* qkv_bias, float* q_out, 
                                               float* k_out, float* v_out,
                                               int batch_size, int seq_len, 
                                               int embed_dim, int num_heads) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int head_dim = embed_dim / num_heads;
    int total_elements = batch_size * seq_len * embed_dim;
    
    if (tid < total_elements) {
        int batch_idx = tid / (seq_len * embed_dim);
        int seq_idx = (tid % (seq_len * embed_dim)) / embed_dim;
        int embed_idx = tid % embed_dim;
        
        // Compute Q, K, V
        float q_val = 0.0f, k_val = 0.0f, v_val = 0.0f;
        
        for (int i = 0; i < embed_dim; i++) {
            int input_idx = batch_idx * seq_len * embed_dim + seq_idx * embed_dim + i;
            
            // Q projection
            q_val += input[input_idx] * qkv_weight[embed_idx * embed_dim + i];
            // K projection  
            k_val += input[input_idx] * qkv_weight[(embed_dim + embed_idx) * embed_dim + i];
            // V projection
            v_val += input[input_idx] * qkv_weight[(2 * embed_dim + embed_idx) * embed_dim + i];
        }
        
        q_out[tid] = q_val + qkv_bias[embed_idx];
        k_out[tid] = k_val + qkv_bias[embed_dim + embed_idx];
        v_out[tid] = v_val + qkv_bias[2 * embed_dim + embed_idx];
    }
}

__global__ void scaled_dot_product_attention_kernel(float* q, float* k, float* v, 
                                                   float* output, int batch_size, 
                                                   int num_heads, int seq_len, 
                                                   int head_dim) {
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int seq_i = threadIdx.x;
    int seq_j = threadIdx.y;
    
    if (batch_idx < batch_size && head_idx < num_heads && 
        seq_i < seq_len && seq_j < seq_len) {
        
        float scale = 1.0f / sqrtf((float)head_dim);
        
        // Compute attention score
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            int q_idx = batch_idx * num_heads * seq_len * head_dim + 
                       head_idx * seq_len * head_dim + seq_i * head_dim + d;
            int k_idx = batch_idx * num_heads * seq_len * head_dim + 
                       head_idx * seq_len * head_dim + seq_j * head_dim + d;
            score += q[q_idx] * k[k_idx];
        }
        score *= scale;
        
        // Store in shared memory for softmax
        __shared__ float scores[TILE_SIZE][TILE_SIZE];
        if (seq_i < TILE_SIZE && seq_j < TILE_SIZE) {
            scores[seq_i][seq_j] = score;
        }
        __syncthreads();
        
        // Softmax across seq_j dimension
        if (seq_i < TILE_SIZE && seq_j == 0) {
            float max_score = scores[seq_i][0];
            for (int j = 1; j < min(seq_len, TILE_SIZE); j++) {
                max_score = fmaxf(max_score, scores[seq_i][j]);
            }
            
            float sum_exp = 0.0f;
            for (int j = 0; j < min(seq_len, TILE_SIZE); j++) {
                scores[seq_i][j] = expf(scores[seq_i][j] - max_score);
                sum_exp += scores[seq_i][j];
            }
            
            for (int j = 0; j < min(seq_len, TILE_SIZE); j++) {
                scores[seq_i][j] /= sum_exp;
            }
        }
        __syncthreads();
        
        // Apply attention to values
        if (seq_i < seq_len && seq_j < head_dim) {
            float result = 0.0f;
            for (int s = 0; s < seq_len; s++) {
                int v_idx = batch_idx * num_heads * seq_len * head_dim + 
                           head_idx * seq_len * head_dim + s * head_dim + seq_j;
                if (s < TILE_SIZE) {
                    result += scores[seq_i][s] * v[v_idx];
                }
            }
            
            int out_idx = batch_idx * num_heads * seq_len * head_dim + 
                         head_idx * seq_len * head_dim + seq_i * head_dim + seq_j;
            output[out_idx] = result;
        }
    }
}

__global__ void layer_norm_kernel(float* input, float* weight, float* bias, 
                                float* output, int batch_size, int seq_len, 
                                int embed_dim) {
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
        
        float std_inv = rsqrtf(variance + 1e-6f);
        
        // Normalize and scale
        for (int i = 0; i < embed_dim; i++) {
            float normalized = (input[base_idx + i] - mean) * std_inv;
            output[base_idx + i] = normalized * weight[i] + bias[i];
        }
    }
}

__global__ void mlp_forward_kernel(float* input, float* weight1, float* bias1,
                                 float* weight2, float* bias2, float* output,
                                 int batch_size, int seq_len, int embed_dim,
                                 int hidden_dim) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * seq_len * embed_dim;
    
    if (tid < total_elements) {
        int batch_idx = tid / (seq_len * embed_dim);
        int seq_idx = (tid % (seq_len * embed_dim)) / embed_dim;
        int embed_idx = tid % embed_dim;
        
        int base_input_idx = batch_idx * seq_len * embed_dim + seq_idx * embed_dim;
        int base_hidden_idx = batch_idx * seq_len * hidden_dim + seq_idx * hidden_dim;
        
        // First linear layer with GELU activation
        for (int h = 0; h < hidden_dim; h++) {
            float hidden_val = bias1[h];
            for (int e = 0; e < embed_dim; e++) {
                hidden_val += input[base_input_idx + e] * weight1[h * embed_dim + e];
            }
            
            // Apply GELU activation and store temporarily
            __shared__ float hidden_cache[BLOCK_SIZE];
            if (threadIdx.x < hidden_dim) {
                hidden_cache[threadIdx.x] = gelu_activation(hidden_val);
            }
            __syncthreads();
            
            // Second linear layer
            if (embed_idx == 0) {  // Only compute once per sequence position
                for (int e = 0; e < embed_dim; e++) {
                    float out_val = bias2[e];
                    for (int h2 = 0; h2 < hidden_dim; h2++) {
                        out_val += hidden_cache[h2] * weight2[e * hidden_dim + h2];
                    }
                    output[base_input_idx + e] = out_val;
                }
            }
        }
    }
}

__global__ void gelu_kernel(float* input, float* output, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < size) {
        output[tid] = gelu_activation(input[tid]);
    }
}

__global__ void softmax_kernel(float* input, float* output, int batch_size, 
                             int seq_len) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < batch_size * seq_len) {
        int batch_idx = tid / seq_len;
        int seq_idx = tid % seq_len;
        int base_idx = batch_idx * seq_len;
        
        // Find max for numerical stability
        float max_val = input[base_idx];
        for (int i = 1; i < seq_len; i++) {
            max_val = fmaxf(max_val, input[base_idx + i]);
        }
        
        // Compute exp and sum
        float sum_exp = 0.0f;
        for (int i = 0; i < seq_len; i++) {
            float exp_val = expf(input[base_idx + i] - max_val);
            output[base_idx + i] = exp_val;
            sum_exp += exp_val;
        }
        
        // Normalize
        for (int i = 0; i < seq_len; i++) {
            output[base_idx + i] /= sum_exp;
        }
    }
}

__global__ void cross_entropy_loss_kernel(float* predictions, int* targets, 
                                        float* loss, int batch_size, 
                                        int num_classes) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < batch_size) {
        int target = targets[tid];
        int pred_idx = tid * num_classes + target;
        
        // Compute softmax for numerical stability
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

// Backward pass kernels
__global__ void cross_entropy_backward_kernel(float* predictions, int* targets,
                                            float* grad_output, int batch_size,
                                            int num_classes) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < batch_size * num_classes) {
        int batch_idx = tid / num_classes;
        int class_idx = tid % num_classes;
        int target = targets[batch_idx];
        
        // Softmax probabilities
        float max_pred = predictions[batch_idx * num_classes];
        for (int i = 1; i < num_classes; i++) {
            max_pred = fmaxf(max_pred, predictions[batch_idx * num_classes + i]);
        }
        
        float sum_exp = 0.0f;
        for (int i = 0; i < num_classes; i++) {
            sum_exp += expf(predictions[batch_idx * num_classes + i] - max_pred);
        }
        
        float prob = expf(predictions[tid] - max_pred) / sum_exp;
        grad_output[tid] = prob - (class_idx == target ? 1.0f : 0.0f);
        grad_output[tid] /= batch_size;  // Average over batch
    }
}

// Launcher functions
extern "C" {
    void launch_patch_embedding_kernel(float* input, float* output, float* weight, 
                                     float* bias, int batch_size, int img_size, 
                                     int patch_size, int embed_dim) {
        int patches_per_side = img_size / patch_size;
        int num_patches = patches_per_side * patches_per_side;
        int total_elements = batch_size * num_patches * embed_dim;
        
        int blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
        patch_embedding_kernel<<<blocks, BLOCK_SIZE>>>(
            input, output, weight, bias, batch_size, img_size, patch_size, embed_dim
        );
        cudaDeviceSynchronize();
    }
    
    void launch_add_positional_embedding_kernel(float* input, float* pos_embed, 
                                               float* cls_token, int batch_size, 
                                               int seq_len, int embed_dim) {
        int total_elements = batch_size * seq_len * embed_dim;
        int blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
        
        add_positional_embedding_kernel<<<blocks, BLOCK_SIZE>>>(
            input, pos_embed, cls_token, batch_size, seq_len, embed_dim
        );
        cudaDeviceSynchronize();
    }
    
    void launch_multi_head_attention_kernel(float* input, float* qkv_weight, 
                                          float* qkv_bias, float* output, 
                                          int batch_size, int seq_len, 
                                          int embed_dim, int num_heads) {
        // This would involve multiple kernel launches for QKV computation,
        // attention computation, and output projection
        // Implementation details omitted for brevity
    }
    
    void launch_layer_norm_kernel(float* input, float* weight, float* bias, 
                                float* output, int batch_size, int seq_len, 
                                int embed_dim) {
        int blocks = (batch_size * seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
        layer_norm_kernel<<<blocks, BLOCK_SIZE>>>(
            input, weight, bias, output, batch_size, seq_len, embed_dim
        );
        cudaDeviceSynchronize();
    }
    
    void launch_gelu_kernel(float* input, float* output, int size) {
        int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        gelu_kernel<<<blocks, BLOCK_SIZE>>>(input, output, size);
        cudaDeviceSynchronize();
    }
    
    void launch_softmax_kernel(float* input, float* output, int batch_size, 
                             int seq_len) {
        int blocks = (batch_size * seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
        softmax_kernel<<<blocks, BLOCK_SIZE>>>(input, output, batch_size, seq_len);
        cudaDeviceSynchronize();
    }
    
    void launch_cross_entropy_loss_kernel(float* predictions, int* targets, 
                                        float* loss, int batch_size, 
                                        int num_classes) {
        int blocks = (batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        cross_entropy_loss_kernel<<<blocks, BLOCK_SIZE>>>(
            predictions, targets, loss, batch_size, num_classes
        );
        cudaDeviceSynchronize();
    }
}