#include "vit_transformer.h"
#include "regularization.h"
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
      num_heads(num_heads), dropout_rate(0.1f), weight_decay(1e-4f) {
    
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
    q_matrices.resize(depth);
    k_matrices.resize(depth);
    v_matrices.resize(depth);
    dropout_masks.resize(depth);
    grad_activations.resize(depth);
    
    int head_dim = embed_dim / num_heads;
    for (int i = 0; i < depth; i++) {
        attention_scores[i] = std::make_unique<Matrix>(1, num_heads * (num_patches + 1) * (num_patches + 1), true);
        attention_weights[i] = std::make_unique<Matrix>(1, num_heads * (num_patches + 1) * (num_patches + 1), true);
        attention_output[i] = std::make_unique<Matrix>(1, (num_patches + 1) * embed_dim, true);
        mlp_hidden[i] = std::make_unique<Matrix>(1, (num_patches + 1) * embed_dim * 4, true);
        block_output[i] = std::make_unique<Matrix>(1, (num_patches + 1) * embed_dim, true);
        
        q_matrices[i] = std::make_unique<Matrix>(1, (num_patches + 1) * embed_dim, true);
        k_matrices[i] = std::make_unique<Matrix>(1, (num_patches + 1) * embed_dim, true);
        v_matrices[i] = std::make_unique<Matrix>(1, (num_patches + 1) * embed_dim, true);
        dropout_masks[i] = std::make_unique<Matrix>(1, (num_patches + 1) * embed_dim, true);
        grad_activations[i] = std::make_unique<Matrix>(1, (num_patches + 1) * embed_dim, true);
    }
}

VisionTransformer::~VisionTransformer() {
    cublasDestroy(cublas_handle);
#ifdef USE_CUDNN
    cudnnDestroy(cudnn_handle);
#endif
}

void VisionTransformer::initialize_weights() {
    // Patch projection (simplified: 784 -> embed_dim)
    patch_proj_weight = std::make_unique<Matrix>(embed_dim, img_size * img_size, true);
    patch_proj_bias = std::make_unique<Matrix>(1, embed_dim, true);
    // Small initialization for patch projection
    float patch_std = 0.01f;
    patch_proj_weight->random(0.0f, patch_std);
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
        
        // Small initialization for stability
        float qkv_std = 0.02f;
        float proj_std = 0.02f;
        
        qkv_weights[i]->random(0.0f, qkv_std);
        qkv_bias[i]->zero();
        proj_weights[i]->random(0.0f, proj_std);
        proj_bias[i]->zero();
        
        // Layer normalization 1
        norm1_weight[i] = std::make_unique<Matrix>(1, embed_dim, true);
        norm1_bias[i] = std::make_unique<Matrix>(1, embed_dim, true);
        // Initialize weights to 1.0
        float* temp_weights = new float[embed_dim];
        std::fill(temp_weights, temp_weights + embed_dim, 1.0f);
        cudaMemcpy(norm1_weight[i]->data, temp_weights, embed_dim * sizeof(float), cudaMemcpyHostToDevice);
        delete[] temp_weights;
        norm1_bias[i]->zero();
        
        // MLP (simplified: embed_dim -> 2*embed_dim -> embed_dim)
        int hidden_dim = 2 * embed_dim;
        mlp_fc1_weight[i] = std::make_unique<Matrix>(hidden_dim, embed_dim, true);
        mlp_fc1_bias[i] = std::make_unique<Matrix>(1, hidden_dim, true);
        mlp_fc2_weight[i] = std::make_unique<Matrix>(embed_dim, hidden_dim, true);
        mlp_fc2_bias[i] = std::make_unique<Matrix>(1, embed_dim, true);
        
        // Small initialization for MLP layers
        float mlp1_std = 0.02f;
        float mlp2_std = 0.02f;
        
        mlp_fc1_weight[i]->random(0.0f, mlp1_std);
        mlp_fc1_bias[i]->zero();
        mlp_fc2_weight[i]->random(0.0f, mlp2_std);
        mlp_fc2_bias[i]->zero();
        
        // Layer normalization 2
        norm2_weight[i] = std::make_unique<Matrix>(1, embed_dim, true);
        norm2_bias[i] = std::make_unique<Matrix>(1, embed_dim, true);
        // Initialize weights to 1.0
        float* temp_weights2 = new float[embed_dim];
        std::fill(temp_weights2, temp_weights2 + embed_dim, 1.0f);
        cudaMemcpy(norm2_weight[i]->data, temp_weights2, embed_dim * sizeof(float), cudaMemcpyHostToDevice);
        delete[] temp_weights2;
        norm2_bias[i]->zero();
    }
    
    // Final classifier
    head_weight = std::make_unique<Matrix>(num_classes, embed_dim, true);
    head_bias = std::make_unique<Matrix>(1, num_classes, true);
    final_norm_weight = std::make_unique<Matrix>(1, embed_dim, true);
    final_norm_bias = std::make_unique<Matrix>(1, embed_dim, true);
    
    // Small initialization for classifier
    float head_std = 0.01f;
    head_weight->random(0.0f, head_std);
    head_bias->zero();
    // Initialize final norm weights to 1.0
    float* temp_weights3 = new float[embed_dim];
    std::fill(temp_weights3, temp_weights3 + embed_dim, 1.0f);
    cudaMemcpy(final_norm_weight->data, temp_weights3, embed_dim * sizeof(float), cudaMemcpyHostToDevice);
    delete[] temp_weights3;
    final_norm_bias->zero();
    
    // Initialize gradients
    grad_patch_proj_weight = std::make_unique<Matrix>(embed_dim, img_size * img_size, true);
    grad_patch_proj_bias = std::make_unique<Matrix>(1, embed_dim, true);
    grad_head_weight = std::make_unique<Matrix>(num_classes, embed_dim, true);
    grad_head_bias = std::make_unique<Matrix>(1, num_classes, true);
    
    grad_qkv_weights.resize(depth);
    grad_proj_weights.resize(depth);
    grad_mlp_fc1_weight.resize(depth);
    grad_mlp_fc2_weight.resize(depth);
    
    for (int i = 0; i < depth; i++) {
        grad_qkv_weights[i] = std::make_unique<Matrix>(3 * embed_dim, embed_dim, true);
        grad_proj_weights[i] = std::make_unique<Matrix>(embed_dim, embed_dim, true);
        grad_mlp_fc1_weight[i] = std::make_unique<Matrix>(2 * embed_dim, embed_dim, true);
        grad_mlp_fc2_weight[i] = std::make_unique<Matrix>(embed_dim, 2 * embed_dim, true);
    }
}

std::unique_ptr<Matrix> VisionTransformer::forward(const Matrix& input) {
    int batch_size = input.rows;
    const float alpha = 1.0f, beta = 0.0f;
    
    // Resize workspace matrices for current batch
    embedded_patches = std::make_unique<Matrix>(batch_size, embed_dim, true);
    final_output = std::make_unique<Matrix>(batch_size, embed_dim, true);
    logits = std::make_unique<Matrix>(batch_size, num_classes, true);
    
    // 1. Patch embedding: input (batch_size, 784) -> embedded (batch_size, embed_dim)
    cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
               embed_dim, batch_size, img_size * img_size,
               &alpha, patch_proj_weight->data, img_size * img_size,
               input.data, img_size * img_size,
               &beta, embedded_patches->data, embed_dim);
    
    // Add bias
    for (int i = 0; i < batch_size; i++) {
        cublasSaxpy(cublas_handle, embed_dim, &alpha,
                   patch_proj_bias->data, 1,
                   embedded_patches->data + i * embed_dim, 1);
    }
    
    // 2. Simple MLP layer for feature transformation
    // embedded -> hidden -> final_output
    std::unique_ptr<Matrix> hidden = std::make_unique<Matrix>(batch_size, embed_dim * 2, true);
    
    // First MLP layer (expand)
    cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
               embed_dim * 2, batch_size, embed_dim,
               &alpha, mlp_fc1_weight[0]->data, embed_dim,
               embedded_patches->data, embed_dim,
               &beta, hidden->data, embed_dim * 2);
    
    // Apply ReLU activation (simple)
    int hidden_size = batch_size * embed_dim * 2;
    float* hidden_ptr = hidden->data;
    // Simple ReLU kernel would go here, for now use identity
    
    // Second MLP layer (contract)
    cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
               embed_dim, batch_size, embed_dim * 2,
               &alpha, mlp_fc2_weight[0]->data, embed_dim * 2,
               hidden->data, embed_dim * 2,
               &beta, final_output->data, embed_dim);
    
    // 3. Final classifier
    cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
               num_classes, batch_size, embed_dim,
               &alpha, head_weight->data, embed_dim,
               final_output->data, embed_dim,
               &beta, logits->data, num_classes);
    
    // Add classifier bias
    for (int i = 0; i < batch_size; i++) {
        cublasSaxpy(cublas_handle, num_classes, &alpha,
                   head_bias->data, 1,
                   logits->data + i * num_classes, 1);
    }
    
    return std::make_unique<Matrix>(*logits);
}

void VisionTransformer::backward(const Matrix& grad_output, const Matrix& input) {
    int batch_size = input.rows;
    const float alpha = 1.0f, beta = 0.0f;
    
    // Zero gradients
    grad_head_weight->zero();
    grad_head_bias->zero();
    grad_patch_proj_weight->zero();
    grad_patch_proj_bias->zero();
    grad_mlp_fc1_weight[0]->zero();
    grad_mlp_fc2_weight[0]->zero();
    
    // 1. Head gradients
    cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
               embed_dim, num_classes, batch_size,
               &alpha, final_output->data, embed_dim,
               grad_output.data, num_classes,
               &beta, grad_head_weight->data, embed_dim);
    
    for (int i = 0; i < batch_size; i++) {
        cublasSaxpy(cublas_handle, num_classes, &alpha,
                   grad_output.data + i * num_classes, 1,
                   grad_head_bias->data, 1);
    }
    
    // 2. Gradient w.r.t. final_output
    std::unique_ptr<Matrix> grad_final_output = std::make_unique<Matrix>(batch_size, embed_dim, true);
    cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
               embed_dim, batch_size, num_classes,
               &alpha, head_weight->data, embed_dim,
               grad_output.data, num_classes,
               &beta, grad_final_output->data, embed_dim);
    
    // 3. MLP backward (simplified)
    std::unique_ptr<Matrix> grad_embedded = std::make_unique<Matrix>(batch_size, embed_dim, true);
    cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
               embed_dim, batch_size, embed_dim,
               &alpha, mlp_fc2_weight[0]->data, embed_dim,
               grad_final_output->data, embed_dim,
               &beta, grad_embedded->data, embed_dim);
    
    // 4. Patch projection gradients
    cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
               img_size * img_size, embed_dim, batch_size,
               &alpha, input.data, img_size * img_size,
               grad_embedded->data, embed_dim,
               &beta, grad_patch_proj_weight->data, img_size * img_size);
    
    for (int i = 0; i < batch_size; i++) {
        cublasSaxpy(cublas_handle, embed_dim, &alpha,
                   grad_embedded->data + i * embed_dim, 1,
                   grad_patch_proj_bias->data, 1);
    }
}

void VisionTransformer::update_weights(float learning_rate) {
    const float alpha = -learning_rate;
    
    // Update head weights and bias
    cublasSaxpy(cublas_handle, num_classes * embed_dim,
               &alpha, grad_head_weight->data, 1,
               head_weight->data, 1);
    
    cublasSaxpy(cublas_handle, num_classes,
               &alpha, grad_head_bias->data, 1,
               head_bias->data, 1);
    
    // Update patch projection weights and bias
    cublasSaxpy(cublas_handle, embed_dim * img_size * img_size,
               &alpha, grad_patch_proj_weight->data, 1,
               patch_proj_weight->data, 1);
    
    cublasSaxpy(cublas_handle, embed_dim,
               &alpha, grad_patch_proj_bias->data, 1,
               patch_proj_bias->data, 1);
    
    // Update MLP weights
    cublasSaxpy(cublas_handle, embed_dim * embed_dim * 2,
               &alpha, grad_mlp_fc1_weight[0]->data, 1,
               mlp_fc1_weight[0]->data, 1);
    
    cublasSaxpy(cublas_handle, embed_dim * embed_dim * 2,
               &alpha, grad_mlp_fc2_weight[0]->data, 1,
               mlp_fc2_weight[0]->data, 1);
}

float VisionTransformer::compute_loss(const Matrix& predictions, const std::vector<int>& targets) {
    int batch_size = predictions.rows;
    int num_classes = predictions.cols;
    
    // Copy predictions to host para cálculo manual
    std::vector<float> host_preds(batch_size * num_classes);
    cudaMemcpy(host_preds.data(), predictions.data, 
               host_preds.size() * sizeof(float), cudaMemcpyDeviceToHost);
    
    float total_loss = 0.0f;
    
    for (int i = 0; i < batch_size; i++) {
        // Encontrar el valor máximo para estabilidad numérica
        float max_val = host_preds[i * num_classes];
        for (int j = 1; j < num_classes; j++) {
            max_val = std::max(max_val, host_preds[i * num_classes + j]);
        }
        
        // Calcular softmax denominador
        float sum_exp = 0.0f;
        for (int j = 0; j < num_classes; j++) {
            sum_exp += exp(host_preds[i * num_classes + j] - max_val);
        }
        
        // Cross-entropy loss
        if (targets[i] >= 0 && targets[i] < num_classes) {
            float log_softmax = host_preds[i * num_classes + targets[i]] - max_val - log(sum_exp);
            total_loss -= log_softmax;
        }
    }
    
    return total_loss / batch_size;
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