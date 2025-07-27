#include "vit_transformer.h"
#include <iostream>
#include <random>
#include <algorithm>
#include <cmath>

// Matrix implementation
Matrix::Matrix(int r, int c, bool device) : rows(r), cols(c), on_device(device) {
    size_t size = rows * cols * sizeof(float);
    
    if (on_device) {
        cudaMalloc(&data, size);
        cudaMemset(data, 0, size);
    } else {
        data = new float[rows * cols];
        std::fill(data, data + rows * cols, 0.0f);
    }
}

Matrix::~Matrix() {
    if (on_device) {
        cudaFree(data);
    } else {
        delete[] data;
    }
}

void Matrix::copyToDevice() {
    if (!on_device) {
        float* device_data;
        size_t size = rows * cols * sizeof(float);
        cudaMalloc(&device_data, size);
        cudaMemcpy(device_data, data, size, cudaMemcpyHostToDevice);
        delete[] data;
        data = device_data;
        on_device = true;
    }
}

void Matrix::copyToHost() {
    if (on_device) {
        float* host_data = new float[rows * cols];
        size_t size = rows * cols * sizeof(float);
        cudaMemcpy(host_data, data, size, cudaMemcpyDeviceToHost);
        cudaFree(data);
        data = host_data;
        on_device = false;
    }
}

void Matrix::zero() {
    size_t size = rows * cols * sizeof(float);
    if (on_device) {
        cudaMemset(data, 0, size);
    } else {
        std::fill(data, data + rows * cols, 0.0f);
    }
}

void Matrix::random(float mean, float std) {
    if (on_device) {
        // Create temporary host array for initialization
        float* temp = new float[rows * cols];
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(mean, std);
        
        for (int i = 0; i < rows * cols; i++) {
            temp[i] = dist(gen);
        }
        
        cudaMemcpy(data, temp, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
        delete[] temp;
    } else {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(mean, std);
        
        for (int i = 0; i < rows * cols; i++) {
            data[i] = dist(gen);
        }
    }
}

Matrix& Matrix::operator=(const Matrix& other) {
    if (this != &other) {
        if (on_device) {
            cudaFree(data);
        } else {
            delete[] data;
        }
        
        rows = other.rows;
        cols = other.cols;
        on_device = other.on_device;
        
        size_t size = rows * cols * sizeof(float);
        if (on_device) {
            cudaMalloc(&data, size);
            cudaMemcpy(data, other.data, size, cudaMemcpyDeviceToDevice);
        } else {
            data = new float[rows * cols];
            std::copy(other.data, other.data + rows * cols, data);
        }
    }
    return *this;
}

Matrix::Matrix(const Matrix& other) : rows(other.rows), cols(other.cols), on_device(other.on_device) {
    size_t size = rows * cols * sizeof(float);
    if (on_device) {
        cudaMalloc(&data, size);
        cudaMemcpy(data, other.data, size, cudaMemcpyDeviceToDevice);
    } else {
        data = new float[rows * cols];
        std::copy(other.data, other.data + rows * cols, data);
    }
}

// VisionTransformer implementation
VisionTransformer::VisionTransformer(int img_size, int patch_size, int in_channels,
                                   int num_classes, int embed_dim, int depth, 
                                   int num_heads)
    : img_size(img_size), patch_size(patch_size), in_channels(in_channels),
      num_classes(num_classes), embed_dim(embed_dim), depth(depth), 
      num_heads(num_heads) {
    
    num_patches = (img_size / patch_size) * (img_size / patch_size);
    
    // Initialize CUDA handles
    cublasCreate(&cublas_handle);
#ifdef USE_CUDNN
    cudnnCreate(&cudnn_handle);
#endif
    
    // Initialize weights and workspace matrices
    initialize_weights();
    
    // Allocate workspace matrices
    patches = std::make_unique<Matrix>(1, num_patches * patch_size * patch_size, true);
    embedded_patches = std::make_unique<Matrix>(1, (num_patches + 1) * embed_dim, true);
    transformer_input = std::make_unique<Matrix>(1, (num_patches + 1) * embed_dim, true);
    final_output = std::make_unique<Matrix>(1, embed_dim, true);
    logits = std::make_unique<Matrix>(1, num_classes, true);
    
    // Allocate per-layer workspace
    attention_scores.resize(depth);
    attention_weights.resize(depth);
    attention_output.resize(depth);
    mlp_hidden.resize(depth);
    block_output.resize(depth);
    
    for (int i = 0; i < depth; i++) {
        attention_scores[i] = std::make_unique<Matrix>(num_heads, (num_patches + 1) * (num_patches + 1), true);
        attention_weights[i] = std::make_unique<Matrix>(num_heads, (num_patches + 1) * (num_patches + 1), true);
        attention_output[i] = std::make_unique<Matrix>(1, (num_patches + 1) * embed_dim, true);
        mlp_hidden[i] = std::make_unique<Matrix>(1, (num_patches + 1) * embed_dim * 4, true);
        block_output[i] = std::make_unique<Matrix>(1, (num_patches + 1) * embed_dim, true);
    }
}

VisionTransformer::~VisionTransformer() {
    cublasDestroy(cublas_handle);
#ifdef USE_CUDNN
    cudnnDestroy(cudnn_handle);
#endif
}

void VisionTransformer::initialize_weights() {
    // Patch projection
    patch_proj_weight = std::make_unique<Matrix>(embed_dim, patch_size * patch_size * in_channels, true);
    patch_proj_bias = std::make_unique<Matrix>(1, embed_dim, true);
    patch_proj_weight->random(0.0f, 0.02f);
    patch_proj_bias->zero();
    
    // Positional embeddings and class token
    pos_embed = std::make_unique<Matrix>(1, (num_patches + 1) * embed_dim, true);
    cls_token = std::make_unique<Matrix>(1, embed_dim, true);
    pos_embed->random(0.0f, 0.02f);
    cls_token->random(0.0f, 0.02f);
    
    // Transformer blocks
    qkv_weights.resize(depth);
    qkv_bias.resize(depth);
    proj_weights.resize(depth);
    proj_bias.resize(depth);
    norm1_weight.resize(depth);
    norm1_bias.resize(depth);
    mlp_fc1_weight.resize(depth);
    mlp_fc1_bias.resize(depth);
    mlp_fc2_weight.resize(depth);
    mlp_fc2_bias.resize(depth);
    norm2_weight.resize(depth);
    norm2_bias.resize(depth);
    
    for (int i = 0; i < depth; i++) {
        // Multi-head attention
        qkv_weights[i] = std::make_unique<Matrix>(3 * embed_dim, embed_dim, true);
        qkv_bias[i] = std::make_unique<Matrix>(1, 3 * embed_dim, true);
        proj_weights[i] = std::make_unique<Matrix>(embed_dim, embed_dim, true);
        proj_bias[i] = std::make_unique<Matrix>(1, embed_dim, true);
        
        qkv_weights[i]->random(0.0f, 0.02f);
        qkv_bias[i]->zero();
        proj_weights[i]->random(0.0f, 0.02f);
        proj_bias[i]->zero();
        
        // Layer normalization 1
        norm1_weight[i] = std::make_unique<Matrix>(1, embed_dim, true);
        norm1_bias[i] = std::make_unique<Matrix>(1, embed_dim, true);
        norm1_weight[i]->random(1.0f, 0.0f);  // Initialize to 1
        norm1_bias[i]->zero();
        
        // MLP
        int hidden_dim = 4 * embed_dim;
        mlp_fc1_weight[i] = std::make_unique<Matrix>(hidden_dim, embed_dim, true);
        mlp_fc1_bias[i] = std::make_unique<Matrix>(1, hidden_dim, true);
        mlp_fc2_weight[i] = std::make_unique<Matrix>(embed_dim, hidden_dim, true);
        mlp_fc2_bias[i] = std::make_unique<Matrix>(1, embed_dim, true);
        
        mlp_fc1_weight[i]->random(0.0f, 0.02f);
        mlp_fc1_bias[i]->zero();
        mlp_fc2_weight[i]->random(0.0f, 0.02f);
        mlp_fc2_bias[i]->zero();
        
        // Layer normalization 2
        norm2_weight[i] = std::make_unique<Matrix>(1, embed_dim, true);
        norm2_bias[i] = std::make_unique<Matrix>(1, embed_dim, true);
        norm2_weight[i]->random(1.0f, 0.0f);  // Initialize to 1
        norm2_bias[i]->zero();
    }
    
    // Final classifier
    head_weight = std::make_unique<Matrix>(num_classes, embed_dim, true);
    head_bias = std::make_unique<Matrix>(1, num_classes, true);
    final_norm_weight = std::make_unique<Matrix>(1, embed_dim, true);
    final_norm_bias = std::make_unique<Matrix>(1, embed_dim, true);
    
    head_weight->random(0.0f, 0.02f);
    head_bias->zero();
    final_norm_weight->random(1.0f, 0.0f);  // Initialize to 1
    final_norm_bias->zero();
    
    // Initialize gradients
    grad_patch_proj_weight = std::make_unique<Matrix>(embed_dim, patch_size * patch_size * in_channels, true);
    grad_patch_proj_bias = std::make_unique<Matrix>(1, embed_dim, true);
    grad_head_weight = std::make_unique<Matrix>(num_classes, embed_dim, true);
    
    grad_qkv_weights.resize(depth);
    grad_proj_weights.resize(depth);
    grad_mlp_fc1_weight.resize(depth);
    grad_mlp_fc2_weight.resize(depth);
    
    for (int i = 0; i < depth; i++) {
        grad_qkv_weights[i] = std::make_unique<Matrix>(3 * embed_dim, embed_dim, true);
        grad_proj_weights[i] = std::make_unique<Matrix>(embed_dim, embed_dim, true);
        grad_mlp_fc1_weight[i] = std::make_unique<Matrix>(4 * embed_dim, embed_dim, true);
        grad_mlp_fc2_weight[i] = std::make_unique<Matrix>(embed_dim, 4 * embed_dim, true);
    }
}

std::unique_ptr<Matrix> VisionTransformer::forward(const Matrix& input) {
    int batch_size = input.rows;
    
    // 1. Patch embedding
    launch_patch_embedding_kernel(
        (float*)input.data, embedded_patches->data, 
        patch_proj_weight->data, patch_proj_bias->data,
        batch_size, img_size, patch_size, embed_dim
    );
    
    // 2. Add positional embeddings and class token
    launch_add_positional_embedding_kernel(
        embedded_patches->data, pos_embed->data, cls_token->data,
        batch_size, num_patches + 1, embed_dim
    );
    
    // Copy embedded patches to transformer input
    cudaMemcpy(transformer_input->data, embedded_patches->data,
               batch_size * (num_patches + 1) * embed_dim * sizeof(float),
               cudaMemcpyDeviceToDevice);
    
    // 3. Transformer encoder blocks
    for (int layer = 0; layer < depth; layer++) {
        // Layer normalization 1
        launch_layer_norm_kernel(
            transformer_input->data, norm1_weight[layer]->data, 
            norm1_bias[layer]->data, transformer_input->data,
            batch_size, num_patches + 1, embed_dim
        );
        
        // Multi-head attention
        launch_multi_head_attention_kernel(
            transformer_input->data, qkv_weights[layer]->data,
            qkv_bias[layer]->data, attention_output[layer]->data,
            batch_size, num_patches + 1, embed_dim, num_heads
        );
        
        // Residual connection 1
        const float alpha = 1.0f, beta = 1.0f;
        cublasSaxpy(cublas_handle, batch_size * (num_patches + 1) * embed_dim,
                   &alpha, attention_output[layer]->data, 1,
                   transformer_input->data, 1);
        
        // Layer normalization 2
        launch_layer_norm_kernel(
            transformer_input->data, norm2_weight[layer]->data,
            norm2_bias[layer]->data, transformer_input->data,
            batch_size, num_patches + 1, embed_dim
        );
        
        // MLP
        int hidden_dim = 4 * embed_dim;
        
        // First linear layer
        cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                   hidden_dim, batch_size * (num_patches + 1), embed_dim,
                   &alpha, mlp_fc1_weight[layer]->data, hidden_dim,
                   transformer_input->data, embed_dim,
                   &beta, mlp_hidden[layer]->data, hidden_dim);
        
        // Add bias and apply GELU
        launch_gelu_kernel(mlp_hidden[layer]->data, mlp_hidden[layer]->data,
                          batch_size * (num_patches + 1) * hidden_dim);
        
        // Second linear layer
        cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                   embed_dim, batch_size * (num_patches + 1), hidden_dim,
                   &alpha, mlp_fc2_weight[layer]->data, embed_dim,
                   mlp_hidden[layer]->data, hidden_dim,
                   &beta, block_output[layer]->data, embed_dim);
        
        // Residual connection 2
        cublasSaxpy(cublas_handle, batch_size * (num_patches + 1) * embed_dim,
                   &alpha, block_output[layer]->data, 1,
                   transformer_input->data, 1);
    }
    
    // 4. Final layer normalization
    launch_layer_norm_kernel(
        transformer_input->data, final_norm_weight->data,
        final_norm_bias->data, transformer_input->data,
        batch_size, num_patches + 1, embed_dim
    );
    
    // 5. Extract class token and classify
    cudaMemcpy(final_output->data, transformer_input->data,
               batch_size * embed_dim * sizeof(float),
               cudaMemcpyDeviceToDevice);
    
    // Final linear layer
    const float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
               num_classes, batch_size, embed_dim,
               &alpha, head_weight->data, num_classes,
               final_output->data, embed_dim,
               &beta, logits->data, num_classes);
    
    return std::make_unique<Matrix>(*logits);
}

void VisionTransformer::backward(const Matrix& grad_output, const Matrix& input) {
    // This is a simplified backward pass
    // In practice, you would need to implement the full backward computation
    // for each layer, computing gradients w.r.t. weights and inputs
    
    int batch_size = input.rows;
    
    // Compute gradients for final layer
    const float alpha = 1.0f / batch_size, beta = 0.0f;
    
    // Gradient w.r.t. head weights
    cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
               num_classes, embed_dim, batch_size,
               &alpha, grad_output.data, num_classes,
               final_output->data, embed_dim,
               &beta, grad_head_weight->data, num_classes);
    
    // Continue backward pass through transformer blocks...
    // This would involve computing gradients for each layer in reverse order
}

void VisionTransformer::update_weights(float learning_rate) {
    // Update weights using computed gradients
    const float alpha = -learning_rate;
    
    // Update head weights
    cublasSaxpy(cublas_handle, num_classes * embed_dim,
               &alpha, grad_head_weight->data, 1,
               head_weight->data, 1);
    
    // Update other weights...
    for (int i = 0; i < depth; i++) {
        cublasSaxpy(cublas_handle, 3 * embed_dim * embed_dim,
                   &alpha, grad_qkv_weights[i]->data, 1,
                   qkv_weights[i]->data, 1);
        
        cublasSaxpy(cublas_handle, embed_dim * embed_dim,
                   &alpha, grad_proj_weights[i]->data, 1,
                   proj_weights[i]->data, 1);
        
        cublasSaxpy(cublas_handle, 4 * embed_dim * embed_dim,
                   &alpha, grad_mlp_fc1_weight[i]->data, 1,
                   mlp_fc1_weight[i]->data, 1);
        
        cublasSaxpy(cublas_handle, embed_dim * 4 * embed_dim,
                   &alpha, grad_mlp_fc2_weight[i]->data, 1,
                   mlp_fc2_weight[i]->data, 1);
    }
}

float VisionTransformer::compute_loss(const Matrix& predictions, const std::vector<int>& targets) {
    int batch_size = predictions.rows;
    
    // Allocate device memory for targets and loss
    int* d_targets;
    float* d_loss;
    cudaMalloc(&d_targets, batch_size * sizeof(int));
    cudaMalloc(&d_loss, batch_size * sizeof(float));
    
    // Copy targets to device
    cudaMemcpy(d_targets, targets.data(), batch_size * sizeof(int), cudaMemcpyHostToDevice);
    
    // Compute cross-entropy loss
    launch_cross_entropy_loss_kernel(predictions.data, d_targets, d_loss, 
                                   batch_size, num_classes);
    
    // Sum losses
    float total_loss;
    cublasSasum(cublas_handle, batch_size, d_loss, 1, &total_loss);
    total_loss /= batch_size;
    
    cudaFree(d_targets);
    cudaFree(d_loss);
    
    return total_loss;
}

float VisionTransformer::compute_accuracy(const Matrix& predictions, const std::vector<int>& targets) {
    int batch_size = predictions.rows;
    
    // Copy predictions to host
    float* host_predictions = new float[batch_size * num_classes];
    cudaMemcpy(host_predictions, predictions.data, 
               batch_size * num_classes * sizeof(float), cudaMemcpyDeviceToHost);
    
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
    
    delete[] host_predictions;
    return (float)correct / batch_size;
}