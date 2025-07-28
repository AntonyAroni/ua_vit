#ifndef VIT_TRANSFORMER_H
#define VIT_TRANSFORMER_H

#include <cuda_runtime.h>
#include <cublas_v2.h>
#ifdef USE_CUDNN
#include <cudnn.h>
#endif
#include <memory>
#include <vector>

class Matrix {
public:
    float* data;
    int rows, cols;
    bool on_device;
    
    Matrix(int r, int c, bool device = true);
    ~Matrix();
    
    void copyToDevice();
    void copyToHost();
    void zero();
    void random(float mean = 0.0f, float std = 0.02f);
    
    Matrix& operator=(const Matrix& other);
    Matrix(const Matrix& other);
};

class VisionTransformer {
private:
    // Model parameters
    int img_size;
    int patch_size;
    int in_channels;
    int num_classes;
    int embed_dim;
    int depth;
    int num_heads;
    int num_patches;
    
    // CUDA handles
    cublasHandle_t cublas_handle;
#ifdef USE_CUDNN
    cudnnHandle_t cudnn_handle;
#endif
    
    // Layer weights and biases
    std::unique_ptr<Matrix> patch_proj_weight;
    std::unique_ptr<Matrix> patch_proj_bias;
    std::unique_ptr<Matrix> pos_embed;
    std::unique_ptr<Matrix> cls_token;
    
    // Transformer blocks
    std::vector<std::unique_ptr<Matrix>> qkv_weights;
    std::vector<std::unique_ptr<Matrix>> qkv_bias;
    std::vector<std::unique_ptr<Matrix>> proj_weights;
    std::vector<std::unique_ptr<Matrix>> proj_bias;
    std::vector<std::unique_ptr<Matrix>> norm1_weight;
    std::vector<std::unique_ptr<Matrix>> norm1_bias;
    std::vector<std::unique_ptr<Matrix>> mlp_fc1_weight;
    std::vector<std::unique_ptr<Matrix>> mlp_fc1_bias;
    std::vector<std::unique_ptr<Matrix>> mlp_fc2_weight;
    std::vector<std::unique_ptr<Matrix>> mlp_fc2_bias;
    std::vector<std::unique_ptr<Matrix>> norm2_weight;
    std::vector<std::unique_ptr<Matrix>> norm2_bias;
    
    // Final classifier
    std::unique_ptr<Matrix> head_weight;
    std::unique_ptr<Matrix> head_bias;
    std::unique_ptr<Matrix> final_norm_weight;
    std::unique_ptr<Matrix> final_norm_bias;
    
    // Workspace matrices for forward/backward
    std::unique_ptr<Matrix> patches;
    std::unique_ptr<Matrix> embedded_patches;
    std::unique_ptr<Matrix> transformer_input;
    std::vector<std::unique_ptr<Matrix>> attention_scores;
    std::vector<std::unique_ptr<Matrix>> attention_weights;
    std::vector<std::unique_ptr<Matrix>> attention_output;
    std::vector<std::unique_ptr<Matrix>> mlp_hidden;
    std::vector<std::unique_ptr<Matrix>> block_output;
    std::unique_ptr<Matrix> final_output;
    std::unique_ptr<Matrix> logits;
    
    // QKV matrices
    std::vector<std::unique_ptr<Matrix>> q_matrices;
    std::vector<std::unique_ptr<Matrix>> k_matrices;
    std::vector<std::unique_ptr<Matrix>> v_matrices;
    
    // Dropout masks
    std::vector<std::unique_ptr<Matrix>> dropout_masks;
    
    // Gradient workspace
    std::vector<std::unique_ptr<Matrix>> grad_activations;
    
    // Gradients
    std::unique_ptr<Matrix> grad_patch_proj_weight;
    std::unique_ptr<Matrix> grad_patch_proj_bias;
    std::vector<std::unique_ptr<Matrix>> grad_qkv_weights;
    std::vector<std::unique_ptr<Matrix>> grad_proj_weights;
    std::vector<std::unique_ptr<Matrix>> grad_mlp_fc1_weight;
    std::vector<std::unique_ptr<Matrix>> grad_mlp_fc2_weight;
    std::unique_ptr<Matrix> grad_head_weight;
    std::unique_ptr<Matrix> grad_head_bias;
    
    // Regularization parameters
    float dropout_rate;
    float weight_decay;
    
public:
    VisionTransformer(int img_size = 28, int patch_size = 4, int in_channels = 1,
                     int num_classes = 10, int embed_dim = 192, int depth = 12, 
                     int num_heads = 8);
    ~VisionTransformer();
    
    void initialize_weights();
    std::unique_ptr<Matrix> forward(const Matrix& input);
    void backward(const Matrix& grad_output, const Matrix& input);
    void update_weights(float learning_rate);
    
    float compute_loss(const Matrix& predictions, const std::vector<int>& targets);
    float compute_accuracy(const Matrix& predictions, const std::vector<int>& targets);
};

// CUDA kernel declarations
extern "C" {
    void launch_patch_embedding_kernel(float* input, float* output, float* weight, 
                                     float* bias, int batch_size, int img_size, 
                                     int patch_size, int embed_dim);
    
    void launch_add_positional_embedding_kernel(float* input, float* pos_embed, 
                                               float* cls_token, int batch_size, 
                                               int seq_len, int embed_dim);
    
    void launch_multi_head_attention_kernel(float* input, float* qkv_weight, 
                                          float* qkv_bias, float* output, 
                                          int batch_size, int seq_len, 
                                          int embed_dim, int num_heads);
    
    void launch_layer_norm_kernel(float* input, float* weight, float* bias, 
                                float* output, int batch_size, int seq_len, 
                                int embed_dim);
    
    void launch_gelu_kernel(float* input, float* output, int size);
    
    void launch_softmax_kernel(float* input, float* output, int batch_size, 
                             int seq_len);
    
    void launch_cross_entropy_loss_kernel(float* predictions, int* targets, 
                                        float* loss, int batch_size, 
                                        int num_classes);
    
    void launch_cross_entropy_backward_kernel(float* predictions, int* targets,
                                            float* grad_output, int batch_size,
                                            int num_classes);
    
    // Backward pass kernels
    void launch_compute_gradients_kernel(float* grad_output, float* activations,
                                       float* grad_weights, int batch_size,
                                       int input_dim, int output_dim);
    
    void launch_backward_linear_kernel(float* grad_output, float* weights,
                                     float* grad_input, int batch_size,
                                     int input_dim, int output_dim);
    
    void launch_gelu_backward_kernel(float* grad_output, float* input,
                                   float* grad_input, int size);
    
    // Attention kernels
    void launch_qkv_projection_kernel(float* input, float* qkv_weight, float* qkv_bias,
                                    float* q_out, float* k_out, float* v_out,
                                    int batch_size, int seq_len, int embed_dim);
    
    void launch_attention_scores_kernel(float* q, float* k, float* scores,
                                      int batch_size, int num_heads, int seq_len, int head_dim);
    
    void launch_softmax_attention_kernel(float* scores, float* weights,
                                       int batch_size, int num_heads, int seq_len);
}

#endif