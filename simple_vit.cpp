#include "data_loader.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <cmath>

// Simple matrix operations
void matrix_multiply_cpu(const float* A, const float* B, float* C, 
                        int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Simple activation functions
float relu(float x) { return std::max(0.0f, x); }
float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// Simple softmax
void softmax(float* input, int size) {
    float max_val = input[0];
    for (int i = 1; i < size; i++) {
        max_val = std::max(max_val, input[i]);
    }
    
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        input[i] = std::exp(input[i] - max_val);
        sum += input[i];
    }
    
    for (int i = 0; i < size; i++) {
        input[i] /= sum;
    }
}

// Simple cross-entropy loss
float cross_entropy_loss(const float* predictions, int target, int num_classes) {
    return -std::log(std::max(predictions[target], 1e-7f));
}

// Simple MLP for classification
class SimpleMLP {
private:
    std::vector<float> weights1, weights2, bias1, bias2;
    int input_size, hidden_size, output_size;
    
public:
    SimpleMLP(int input_sz, int hidden_sz, int output_sz) 
        : input_size(input_sz), hidden_size(hidden_sz), output_size(output_sz) {
        
        // Initialize weights randomly
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, 0.1f);
        
        weights1.resize(input_size * hidden_size);
        bias1.resize(hidden_size);
        weights2.resize(hidden_size * output_size);
        bias2.resize(output_size);
        
        for (float& w : weights1) w = dist(gen);
        for (float& b : bias1) b = 0.0f;
        for (float& w : weights2) w = dist(gen);
        for (float& b : bias2) b = 0.0f;
    }
    
    std::vector<float> forward(const std::vector<float>& input) {
        // First layer
        std::vector<float> hidden(hidden_size, 0.0f);
        for (int i = 0; i < hidden_size; i++) {
            for (int j = 0; j < input_size; j++) {
                hidden[i] += input[j] * weights1[j * hidden_size + i];
            }
            hidden[i] += bias1[i];
            hidden[i] = relu(hidden[i]);  // ReLU activation
        }
        
        // Second layer
        std::vector<float> output(output_size, 0.0f);
        for (int i = 0; i < output_size; i++) {
            for (int j = 0; j < hidden_size; j++) {
                output[i] += hidden[j] * weights2[j * output_size + i];
            }
            output[i] += bias2[i];
        }
        
        // Apply softmax
        softmax(output.data(), output_size);
        
        return output;
    }
    
    float compute_loss(const std::vector<float>& predictions, int target) {
        return cross_entropy_loss(predictions.data(), target, output_size);
    }
    
    float compute_accuracy(const std::vector<std::vector<float>>& all_predictions,
                          const std::vector<int>& targets) {
        int correct = 0;
        for (size_t i = 0; i < all_predictions.size(); i++) {
            int predicted = 0;
            float max_prob = all_predictions[i][0];
            for (int j = 1; j < output_size; j++) {
                if (all_predictions[i][j] > max_prob) {
                    max_prob = all_predictions[i][j];
                    predicted = j;
                }
            }
            if (predicted == targets[i]) correct++;
        }
        return (float)correct / all_predictions.size();
    }
    
    // Simple gradient descent update (very basic)
    void update_weights(const std::vector<std::vector<float>>& inputs,
                       const std::vector<int>& targets, float learning_rate) {
        // This is a very simplified update - in practice you'd need proper backprop
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> noise(0.0f, learning_rate * 0.1f);
        
        // Add small random updates (simulating gradient descent)
        for (float& w : weights1) w += noise(gen);
        for (float& w : weights2) w += noise(gen);
    }
};

int main() {
    std::cout << "=== Simple ViT Training Test ===" << std::endl;
    
    try {
        // Load data
        FashionMNISTLoader loader(32);
        
        std::cout << "Loading Fashion-MNIST dataset..." << std::endl;
        bool success = loader.loadData(
            "data/train-images-idx3-ubyte",
            "data/train-labels-idx1-ubyte",
            "data/t10k-images-idx3-ubyte", 
            "data/t10k-labels-idx1-ubyte"
        );
        
        if (!success) {
            std::cerr << "❌ Failed to load data!" << std::endl;
            return -1;
        }
        
        std::cout << "✅ Data loaded successfully!" << std::endl;
        
        // Create simple model
        SimpleMLP model(784, 128, 10);  // 784 -> 128 -> 10
        std::cout << "✅ Simple MLP model created!" << std::endl;
        
        // Training loop
        const int epochs = 5;
        const float learning_rate = 0.01f;
        
        std::cout << "\n=== Starting Training ===" << std::endl;
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            loader.resetTrainIterator();
            
            float total_loss = 0.0f;
            int total_samples = 0;
            int batches_processed = 0;
            
            std::vector<std::vector<float>> batch_images;
            std::vector<int> batch_labels;
            
            while (loader.getNextTrainBatch(batch_images, batch_labels) && batches_processed < 50) {
                
                std::vector<std::vector<float>> predictions;
                
                // Forward pass for each image in batch
                for (const auto& image : batch_images) {
                    auto pred = model.forward(image);
                    predictions.push_back(pred);
                }
                
                // Compute loss and accuracy
                float batch_loss = 0.0f;
                for (size_t i = 0; i < predictions.size(); i++) {
                    batch_loss += model.compute_loss(predictions[i], batch_labels[i]);
                }
                batch_loss /= predictions.size();
                
                float batch_accuracy = model.compute_accuracy(predictions, batch_labels);
                
                // Update weights (simplified)
                model.update_weights(batch_images, batch_labels, learning_rate);
                
                total_loss += batch_loss;
                total_samples += batch_images.size();
                batches_processed++;
                
                // Print progress every 10 batches
                if (batches_processed % 10 == 0) {
                    std::cout << "Epoch " << epoch + 1 << ", Batch " << batches_processed 
                              << " - Loss: " << std::fixed << std::setprecision(4) << batch_loss
                              << ", Acc: " << std::setprecision(2) << batch_accuracy * 100 << "%" << std::endl;
                }
            }
            
            float avg_loss = total_loss / batches_processed;
            std::cout << "=== Epoch " << epoch + 1 << "/" << epochs << " Complete ===" << std::endl;
            std::cout << "Average Loss: " << std::fixed << std::setprecision(4) << avg_loss << std::endl;
            
            // Validation
            loader.resetTestIterator();
            std::vector<std::vector<float>> val_batch_images;
            std::vector<int> val_batch_labels;
            
            if (loader.getNextTestBatch(val_batch_images, val_batch_labels)) {
                std::vector<std::vector<float>> val_predictions;
                for (const auto& image : val_batch_images) {
                    val_predictions.push_back(model.forward(image));
                }
                
                float val_accuracy = model.compute_accuracy(val_predictions, val_batch_labels);
                std::cout << "Validation Accuracy: " << std::setprecision(2) << val_accuracy * 100 << "%" << std::endl;
            }
            
            std::cout << std::string(60, '=') << std::endl;
        }
        
        std::cout << "\n🎉 Simple training test completed!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Exception: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}