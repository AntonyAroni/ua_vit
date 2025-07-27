#ifndef HYBRID_VIT_H
#define HYBRID_VIT_H

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <memory>

// Simple matrix class for GPU operations
class GPUMatrix {
public:
    float* data;
    int rows, cols;
    bool on_device;
    
    GPUMatrix(int r, int c, bool device = true);
    ~GPUMatrix();
    
    void copyFromHost(const float* host_data);
    void copyToHost(float* host_data) const;
    void zero();
    void random(float mean = 0.0f, float std = 0.02f);
    void print(int max_rows = 5, int max_cols = 5);
    
    int size() const { return rows * cols; }
};

// Hybrid Vision Transformer - simplified but functional
class HybridVisionTransformer {
private:
    // Model hyperparameters
    int img_size;
    int patch_size;
    int num_patches;
    int embed_dim;
    int num_heads;
    int num_layers;
    int num_classes;
    int mlp_dim;
    
    // CUDA handle
    cublasHandle_t cublas_handle;
    
    // Model parameters
    std::unique_ptr<GPUMatrix> patch_embedding_weight;
    std::unique_ptr<GPUMatrix> patch_embedding_bias;
    std::unique_ptr<GPUMatrix> positional_embeddings;
    std::unique_ptr<GPUMatrix> cls_token;
    
    // Transformer layers
    std::vector<std::unique_ptr<GPUMatrix>> attn_qkv_weights;
    std::vector<std::unique_ptr<GPUMatrix>> attn_qkv_bias;
    std::vector<std::unique_ptr<GPUMatrix>> attn_proj_weights;
    std::vector<std::unique_ptr<GPUMatrix>> attn_proj_bias;
    
    std::vector<std::unique_ptr<GPUMatrix>> norm1_weight;
    std::vector<std::unique_ptr<GPUMatrix>> norm1_bias;
    std::vector<std::unique_ptr<GPUMatrix>> norm2_weight;
    std::vector<std::unique_ptr<GPUMatrix>> norm2_bias;
    
    std::vector<std::unique_ptr<GPUMatrix>> mlp_fc1_weight;
    std::vector<std::unique_ptr<GPUMatrix>> mlp_fc1_bias;
    std::vector<std::unique_ptr<GPUMatrix>> mlp_fc2_weight;
    std::vector<std::unique_ptr<GPUMatrix>> mlp_fc2_bias;
    
    // Classification head
    std::unique_ptr<GPUMatrix> head_weight;
    std::unique_ptr<GPUMatrix> head_bias;
    std::unique_ptr<GPUMatrix> final_norm_weight;
    std::unique_ptr<GPUMatrix> final_norm_bias;
    
    // Workspace matrices
    std::unique_ptr<GPUMatrix> patches_flat;
    std::unique_ptr<GPUMatrix> embedded_patches;
    std::unique_ptr<GPUMatrix> transformer_input;
    std::vector<std::unique_ptr<GPUMatrix>> layer_outputs;
    std::unique_ptr<GPUMatrix> final_output;
    std::unique_ptr<GPUMatrix> logits;
    
public:
    HybridVisionTransformer(int img_sz = 28, int patch_sz = 4, int embed_d = 128,
                           int n_heads = 4, int n_layers = 6, int n_classes = 10);
    ~HybridVisionTransformer();
    
    void initialize_weights();
    
    // Main forward pass
    std::unique_ptr<GPUMatrix> forward(const GPUMatrix& input);
    
    // Loss and accuracy computation
    float compute_loss(const GPUMatrix& predictions, const std::vector<int>& targets);
    float compute_accuracy(const GPUMatrix& predictions, const std::vector<int>& targets);
    
    // Simple training update (gradient descent)
    void update_weights(float learning_rate);
    
    // Helper functions
    void patch_embedding(const GPUMatrix& input, GPUMatrix& output);
    void add_positional_embeddings(GPUMatrix& tokens);
    void transformer_layer(const GPUMatrix& input, GPUMatrix& output, int layer_idx);
    void multi_head_attention(const GPUMatrix& input, GPUMatrix& output, int layer_idx);
    void feed_forward(const GPUMatrix& input, GPUMatrix& output, int layer_idx);
    void layer_norm(const GPUMatrix& input, GPUMatrix& output, 
                   const GPUMatrix& weight, const GPUMatrix& bias);
    void classification_head(const GPUMatrix& input, GPUMatrix& output);
    
    // Utility functions
    void print_model_info();
    int get_parameter_count();
};

// CUDA kernel declarations
extern "C" {
    void launch_patch_embed_kernel(const float* input, float* output, 
                                 const float* weight, const float* bias,
                                 int batch_size, int img_size, int patch_size, 
                                 int embed_dim);
    
    void launch_add_pos_embed_kernel(float* tokens, const float* pos_embed,
                                   const float* cls_token, int batch_size,
                                   int seq_len, int embed_dim);
    
    void launch_layer_norm_kernel(const float* input, float* output,
                                const float* weight, const float* bias,
                                int batch_size, int seq_len, int embed_dim);
    
    void launch_gelu_kernel(const float* input, float* output, int size);
    
    void launch_softmax_kernel(const float* input, float* output,
                             int batch_size, int seq_len);
    
    void launch_cross_entropy_kernel(const float* predictions, const int* targets,
                                   float* loss, int batch_size, int num_classes);
}

#endif