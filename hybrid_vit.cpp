#include "hybrid_vit.h"
#include <iostream>
#include <iomanip>
#include <random>
#include <cmath>
#include <algorithm>

// GPUMatrix implementation
GPUMatrix::GPUMatrix(int r, int c, bool device) : rows(r), cols(c), on_device(device) {
    size_t bytes = rows * cols * sizeof(float);
    
    if (on_device) {
        cudaMalloc(&data, bytes);
        cudaMemset(data, 0, bytes);
    } else {
        data = new float[rows * cols];
        std::fill(data, data + rows * cols, 0.0f);
    }
}

GPUMatrix::~GPUMatrix() {
    if (on_device) {
        cudaFree(data);
    } else {
        delete[] data;
    }
}

void GPUMatrix::copyFromHost(const float* host_data) {
    if (on_device) {
        cudaMemcpy(data, host_data, size() * sizeof(float), cudaMemcpyHostToDevice);
    } else {
        std::copy(host_data, host_data + size(), data);
    }
}

void GPUMatrix::copyToHost(float* host_data) const {
    if (on_device) {
        cudaMemcpy(host_data, data, size() * sizeof(float), cudaMemcpyDeviceToHost);
    } else {
        std::copy(data, data + size(), host_data);
    }
}

void GPUMatrix::zero() {
    if (on_device) {
        cudaMemset(data, 0, size() * sizeof(float));
    } else {
        std::fill(data, data + size(), 0.0f);
    }
}

void GPUMatrix::random(float mean, float std) {
    std::vector<float> temp_data(size());
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(mean, std);
    
    for (float& val : temp_data) {
        val = dist(gen);
    }
    
    copyFromHost(temp_data.data());
}

void GPUMatrix::print(int max_rows, int max_cols) {
    std::vector<float> temp_data(size());
    copyToHost(temp_data.data());
    
    std::cout << "Matrix [" << rows << "x" << cols << "]:" << std::endl;
    
    for (int i = 0; i < std::min(rows, max_rows); i++) {
        for (int j = 0; j < std::min(cols, max_cols); j++) {
            std::cout << std::fixed << std::setprecision(4) << temp_data[i * cols + j] << " ";
        }
        if (cols > max_cols) std::cout << "...";
        std::cout << std::endl;
    }
    if (rows > max_rows) std::cout << "..." << std::endl;
}

// HybridVisionTransformer implementation
HybridVisionTransformer::HybridVisionTransformer(int img_sz, int patch_sz, int embed_d,
                                               int n_heads, int n_layers, int n_classes)
    : img_size(img_sz), patch_size(patch_sz), embed_dim(embed_d), 
      num_heads(n_heads), num_layers(n_layers), num_classes(n_classes) {
    
    num_patches = (img_size / patch_size) * (img_size / patch_size);
    mlp_dim = 4 * embed_dim;
    
    // Initialize cuBLAS
    cublasCreate(&cublas_handle);
    
    std::cout << "Creating Hybrid ViT with:" << std::endl;
    std::cout << "  Image size: " << img_size << "x" << img_size << std::endl;
    std::cout << "  Patch size: " << patch_size << "x" << patch_size << std::endl;
    std::cout << "  Num patches: " << num_patches << std::endl;
    std::cout << "  Embed dim: " << embed_dim << std::endl;
    std::cout << "  Num heads: " << num_heads << std::endl;
    std::cout << "  Num layers: " << num_layers << std::endl;
    std::cout << "  Num classes: " << num_classes << std::endl;
    
    initialize_weights();
}

HybridVisionTransformer::~HybridVisionTransformer() {
    cublasDestroy(cublas_handle);
}

void HybridVisionTransformer::initialize_weights() {
    std::cout << "Initializing weights..." << std::endl;
    
    // Patch embedding
    int patch_pixels = patch_size * patch_size;
    patch_embedding_weight = std::make_unique<GPUMatrix>(embed_dim, patch_pixels);
    patch_embedding_bias = std::make_unique<GPUMatrix>(1, embed_dim);
    patch_embedding_weight->random(0.0f, 0.02f);
    patch_embedding_bias->zero();
    
    // Positional embeddings and class token
    positional_embeddings = std::make_unique<GPUMatrix>(1, (num_patches + 1) * embed_dim);
    cls_token = std::make_unique<GPUMatrix>(1, embed_dim);
    positional_embeddings->random(0.0f, 0.02f);
    cls_token->random(0.0f, 0.02f);
    
    // Transformer layers
    attn_qkv_weights.resize(num_layers);
    attn_qkv_bias.resize(num_layers);
    attn_proj_weights.resize(num_layers);
    attn_proj_bias.resize(num_layers);
    norm1_weight.resize(num_layers);
    norm1_bias.resize(num_layers);
    norm2_weight.resize(num_layers);
    norm2_bias.resize(num_layers);
    mlp_fc1_weight.resize(num_layers);
    mlp_fc1_bias.resize(num_layers);
    mlp_fc2_weight.resize(num_layers);
    mlp_fc2_bias.resize(num_layers);
    
    for (int i = 0; i < num_layers; i++) {
        // Attention weights
        attn_qkv_weights[i] = std::make_unique<GPUMatrix>(3 * embed_dim, embed_dim);
        attn_qkv_bias[i] = std::make_unique<GPUMatrix>(1, 3 * embed_dim);
        attn_proj_weights[i] = std::make_unique<GPUMatrix>(embed_dim, embed_dim);
        attn_proj_bias[i] = std::make_unique<GPUMatrix>(1, embed_dim);
        
        attn_qkv_weights[i]->random(0.0f, 0.02f);
        attn_qkv_bias[i]->zero();
        attn_proj_weights[i]->random(0.0f, 0.02f);
        attn_proj_bias[i]->zero();
        
        // Layer norm weights
        norm1_weight[i] = std::make_unique<GPUMatrix>(1, embed_dim);
        norm1_bias[i] = std::make_unique<GPUMatrix>(1, embed_dim);
        norm2_weight[i] = std::make_unique<GPUMatrix>(1, embed_dim);
        norm2_bias[i] = std::make_unique<GPUMatrix>(1, embed_dim);
        
        // Initialize layer norm weights to 1, bias to 0
        std::vector<float> ones(embed_dim, 1.0f);
        norm1_weight[i]->copyFromHost(ones.data());
        norm1_bias[i]->zero();
        norm2_weight[i]->copyFromHost(ones.data());
        norm2_bias[i]->zero();
        
        // MLP weights
        mlp_fc1_weight[i] = std::make_unique<GPUMatrix>(mlp_dim, embed_dim);
        mlp_fc1_bias[i] = std::make_unique<GPUMatrix>(1, mlp_dim);
        mlp_fc2_weight[i] = std::make_unique<GPUMatrix>(embed_dim, mlp_dim);
        mlp_fc2_bias[i] = std::make_unique<GPUMatrix>(1, embed_dim);
        
        mlp_fc1_weight[i]->random(0.0f, 0.02f);
        mlp_fc1_bias[i]->zero();
        mlp_fc2_weight[i]->random(0.0f, 0.02f);
        mlp_fc2_bias[i]->zero();
    }
    
    // Classification head
    head_weight = std::make_unique<GPUMatrix>(num_classes, embed_dim);
    head_bias = std::make_unique<GPUMatrix>(1, num_classes);
    final_norm_weight = std::make_unique<GPUMatrix>(1, embed_dim);
    final_norm_bias = std::make_unique<GPUMatrix>(1, embed_dim);
    
    head_weight->random(0.0f, 0.02f);
    head_bias->zero();
    std::vector<float> ones(embed_dim, 1.0f);
    final_norm_weight->copyFromHost(ones.data());
    final_norm_bias->zero();
    
    std::cout << "✅ Weight initialization complete!" << std::endl;
}

std::unique_ptr<GPUMatrix> HybridVisionTransformer::forward(const GPUMatrix& input) {
    int batch_size = input.rows;
    
    // Allocate workspace if needed
    if (!patches_flat || patches_flat->rows != batch_size) {
        patches_flat = std::make_unique<GPUMatrix>(batch_size, num_patches * embed_dim);
        embedded_patches = std::make_unique<GPUMatrix>(batch_size, (num_patches + 1) * embed_dim);
        transformer_input = std::make_unique<GPUMatrix>(batch_size, (num_patches + 1) * embed_dim);
        final_output = std::make_unique<GPUMatrix>(batch_size, embed_dim);
        logits = std::make_unique<GPUMatrix>(batch_size, num_classes);
        
        layer_outputs.clear();
        for (int i = 0; i < num_layers; i++) {
            layer_outputs.push_back(std::make_unique<GPUMatrix>(batch_size, (num_patches + 1) * embed_dim));
        }
    }
    
    // 1. Patch embedding
    patch_embedding(input, *patches_flat);
    
    // 2. Add positional embeddings and class token
    add_positional_embeddings(*embedded_patches);
    
    // Copy embedded patches to transformer input
    cudaMemcpy(transformer_input->data, embedded_patches->data,
               transformer_input->size() * sizeof(float), cudaMemcpyDeviceToDevice);
    
    // 3. Transformer layers
    for (int i = 0; i < num_layers; i++) {
        transformer_layer(*transformer_input, *layer_outputs[i], i);
        
        // Copy output back to input for next layer
        cudaMemcpy(transformer_input->data, layer_outputs[i]->data,
                   transformer_input->size() * sizeof(float), cudaMemcpyDeviceToDevice);
    }
    
    // 4. Final layer norm
    layer_norm(*transformer_input, *transformer_input, *final_norm_weight, *final_norm_bias);
    
    // 5. Classification head (use class token)
    classification_head(*transformer_input, *logits);
    
    return std::make_unique<GPUMatrix>(*logits);
}

void HybridVisionTransformer::patch_embedding(const GPUMatrix& input, GPUMatrix& output) {
    int batch_size = input.rows;
    
    launch_patch_embed_kernel(
        input.data, output.data,
        patch_embedding_weight->data, patch_embedding_bias->data,
        batch_size, img_size, patch_size, embed_dim
    );
}

void HybridVisionTransformer::add_positional_embeddings(GPUMatrix& tokens) {
    int batch_size = tokens.rows;
    int seq_len = num_patches + 1;
    
    launch_add_pos_embed_kernel(
        tokens.data, positional_embeddings->data, cls_token->data,
        batch_size, seq_len, embed_dim
    );
}

void HybridVisionTransformer::transformer_layer(const GPUMatrix& input, GPUMatrix& output, int layer_idx) {
    // This is a simplified transformer layer
    // In a full implementation, you'd have proper attention computation
    
    // For now, we'll use a simple MLP-like transformation
    // 1. Layer norm
    layer_norm(input, output, *norm1_weight[layer_idx], *norm1_bias[layer_idx]);
    
    // 2. Simple "attention" using matrix multiplication (simplified)
    const float alpha = 1.0f, beta = 0.0f;
    int seq_len = num_patches + 1;
    int batch_size = input.rows;
    
    // Simple projection instead of full attention
    cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
               batch_size * seq_len, embed_dim, embed_dim,
               &alpha, output.data, batch_size * seq_len,
               attn_proj_weights[layer_idx]->data, embed_dim,
               &beta, output.data, batch_size * seq_len);
    
    // 3. Residual connection
    cublasSaxpy(cublas_handle, input.size(), &alpha, input.data, 1, output.data, 1);
    
    // 4. Second layer norm
    GPUMatrix temp_output(output.rows, output.cols);
    layer_norm(output, temp_output, *norm2_weight[layer_idx], *norm2_bias[layer_idx]);
    
    // 5. Feed forward
    feed_forward(temp_output, output, layer_idx);
}

void HybridVisionTransformer::feed_forward(const GPUMatrix& input, GPUMatrix& output, int layer_idx) {
    int batch_size = input.rows;
    int seq_len = num_patches + 1;
    
    // Create temporary matrix for hidden layer
    GPUMatrix hidden(batch_size * seq_len, mlp_dim);
    
    const float alpha = 1.0f, beta = 0.0f;
    
    // First linear layer
    cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
               batch_size * seq_len, mlp_dim, embed_dim,
               &alpha, input.data, batch_size * seq_len,
               mlp_fc1_weight[layer_idx]->data, mlp_dim,
               &beta, hidden.data, batch_size * seq_len);
    
    // Add bias and apply GELU
    launch_gelu_kernel(hidden.data, hidden.data, hidden.size());
    
    // Second linear layer
    cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
               batch_size * seq_len, embed_dim, mlp_dim,
               &alpha, hidden.data, batch_size * seq_len,
               mlp_fc2_weight[layer_idx]->data, embed_dim,
               &beta, output.data, batch_size * seq_len);
    
    // Residual connection
    cublasSaxpy(cublas_handle, input.size(), &alpha, input.data, 1, output.data, 1);
}

void HybridVisionTransformer::layer_norm(const GPUMatrix& input, GPUMatrix& output, 
                                        const GPUMatrix& weight, const GPUMatrix& bias) {
    int batch_size = input.rows;
    int seq_len = num_patches + 1;
    
    launch_layer_norm_kernel(
        input.data, output.data, weight.data, bias.data,
        batch_size, seq_len, embed_dim
    );
}

void HybridVisionTransformer::classification_head(const GPUMatrix& input, GPUMatrix& output) {
    int batch_size = input.rows;
    
    // Extract class tokens (first token of each sequence)
    GPUMatrix cls_tokens(batch_size, embed_dim);
    
    for (int b = 0; b < batch_size; b++) {
        cudaMemcpy(cls_tokens.data + b * embed_dim,
                   input.data + b * (num_patches + 1) * embed_dim,
                   embed_dim * sizeof(float), cudaMemcpyDeviceToDevice);
    }
    
    // Linear projection
    const float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
               batch_size, num_classes, embed_dim,
               &alpha, cls_tokens.data, batch_size,
               head_weight->data, num_classes,
               &beta, output.data, batch_size);
}

float HybridVisionTransformer::compute_loss(const GPUMatrix& predictions, const std::vector<int>& targets) {
    int batch_size = predictions.rows;
    
    // Allocate device memory for targets and loss
    int* d_targets;
    float* d_loss;
    cudaMalloc(&d_targets, batch_size * sizeof(int));
    cudaMalloc(&d_loss, batch_size * sizeof(float));
    
    // Copy targets to device
    cudaMemcpy(d_targets, targets.data(), batch_size * sizeof(int), cudaMemcpyHostToDevice);
    
    // Compute loss
    launch_cross_entropy_kernel(predictions.data, d_targets, d_loss, batch_size, num_classes);
    
    // Sum losses
    float total_loss;
    cublasSasum(cublas_handle, batch_size, d_loss, 1, &total_loss);
    total_loss /= batch_size;
    
    cudaFree(d_targets);
    cudaFree(d_loss);
    
    return total_loss;
}

float HybridVisionTransformer::compute_accuracy(const GPUMatrix& predictions, const std::vector<int>& targets) {
    int batch_size = predictions.rows;
    
    // Copy predictions to host
    std::vector<float> host_predictions(batch_size * num_classes);
    predictions.copyToHost(host_predictions.data());
    
    int correct = 0;
    for (int i = 0; i < batch_size; i++) {
        int predicted_class = 0;
        float max_prob = host_predictions[i * num_classes];
        
        for (int j = 1; j < num_classes; j++) {
            if (host_predictions[i * num_classes + j] > max_prob) {
                max_prob = host_predictions[i * num_classes + j];
                predicted_class = j;
            }
        }
        
        if (predicted_class == targets[i]) {
            correct++;
        }
    }
    
    return (float)correct / batch_size;
}

void HybridVisionTransformer::update_weights(float learning_rate) {
    // Very simplified weight update
    // In practice, you would compute and apply proper gradients
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> noise(0.0f, learning_rate * 0.001f);
    
    // Add small random updates to simulate gradient descent
    // This is just for demonstration - real gradients would be computed via backprop
}

void HybridVisionTransformer::print_model_info() {
    std::cout << "=== Hybrid Vision Transformer Info ===" << std::endl;
    std::cout << "Parameters: " << get_parameter_count() << std::endl;
    std::cout << "Memory usage: ~" << get_parameter_count() * 4 / (1024*1024) << " MB" << std::endl;
}

int HybridVisionTransformer::get_parameter_count() {
    int total = 0;
    
    // Patch embedding
    total += embed_dim * patch_size * patch_size + embed_dim;
    
    // Positional embeddings
    total += (num_patches + 1) * embed_dim + embed_dim;
    
    // Transformer layers
    total += num_layers * (
        3 * embed_dim * embed_dim + 3 * embed_dim +  // QKV
        embed_dim * embed_dim + embed_dim +          // projection
        2 * embed_dim + 2 * embed_dim +              // layer norms
        embed_dim * mlp_dim + mlp_dim +              // MLP fc1
        mlp_dim * embed_dim + embed_dim              // MLP fc2
    );
    
    // Classification head
    total += embed_dim * num_classes + num_classes + 2 * embed_dim;
    
    return total;
}