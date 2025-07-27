#include <iostream>
#include <chrono>
#include <iomanip>
#include "vit_transformer.h"
#include "data_loader.h"

class Optimizer {
private:
    float learning_rate;
    float beta1, beta2;
    float epsilon;
    int step_count;
    
    // Adam momentum terms
    std::vector<std::unique_ptr<Matrix>> m_weights;
    std::vector<std::unique_ptr<Matrix>> v_weights;
    
public:
    Optimizer(float lr = 3e-4f, float b1 = 0.9f, float b2 = 0.999f, float eps = 1e-8f)
        : learning_rate(lr), beta1(b1), beta2(b2), epsilon(eps), step_count(0) {}
    
    void step(VisionTransformer& model) {
        step_count++;
        
        float lr_corrected = learning_rate * 
            sqrt(1.0f - pow(beta2, step_count)) / (1.0f - pow(beta1, step_count));
        
        model.update_weights(lr_corrected);
    }
    
    void setLearningRate(float lr) { learning_rate = lr; }
    float getLearningRate() const { return learning_rate; }
};

class Scheduler {
private:
    float initial_lr;
    int warmup_steps;
    int total_steps;
    
public:
    Scheduler(float initial_lr, int warmup_steps, int total_steps)
        : initial_lr(initial_lr), warmup_steps(warmup_steps), total_steps(total_steps) {}
    
    float getCosineAnnealingLR(int current_step) {
        if (current_step < warmup_steps) {
            return initial_lr * current_step / warmup_steps;
        } else {
            float progress = (float)(current_step - warmup_steps) / (total_steps - warmup_steps);
            return initial_lr * 0.5f * (1.0f + cos(M_PI * progress));
        }
    }
};

void printProgress(int epoch, int batch, int total_batches, float loss, float acc, 
                  float elapsed_time) {
    float progress = (float)batch / total_batches;
    int bar_width = 50;
    int filled = (int)(progress * bar_width);
    
    std::cout << "\rEpoch " << std::setw(3) << epoch 
              << " [" << std::string(filled, '=') << std::string(bar_width - filled, ' ') << "] "
              << std::setw(3) << (int)(progress * 100) << "% "
              << "Loss: " << std::fixed << std::setprecision(4) << loss
              << " Acc: " << std::setprecision(2) << acc * 100 << "% "
              << "Time: " << std::setprecision(1) << elapsed_time << "s";
    std::cout.flush();
}

float validateModel(VisionTransformer& model, FashionMNISTLoader& data_loader) {
    data_loader.resetTestIterator();
    
    float total_loss = 0.0f;
    float total_accuracy = 0.0f;
    int num_batches = 0;
    
    std::vector<std::vector<float>> batch_images;
    std::vector<int> batch_labels;
    
    while (data_loader.getNextTestBatch(batch_images, batch_labels)) {
        // Convert batch to Matrix format
        Matrix input(batch_images.size(), 28 * 28, true);
        
        // Copy data to GPU
        float* host_data = new float[batch_images.size() * 28 * 28];
        for (size_t i = 0; i < batch_images.size(); i++) {
            for (size_t j = 0; j < 28 * 28; j++) {
                host_data[i * 28 * 28 + j] = batch_images[i][j];
            }
        }
        
        cudaMemcpy(input.data, host_data, 
                   batch_images.size() * 28 * 28 * sizeof(float), 
                   cudaMemcpyHostToDevice);
        delete[] host_data;
        
        // Forward pass
        auto predictions = model.forward(input);
        
        // Compute metrics
        float batch_loss = model.compute_loss(*predictions, batch_labels);
        float batch_accuracy = model.compute_accuracy(*predictions, batch_labels);
        
        total_loss += batch_loss;
        total_accuracy += batch_accuracy;
        num_batches++;
    }
    
    return total_accuracy / num_batches;
}

int main() {
    std::cout << "=== Vision Transformer for Fashion-MNIST ===" << std::endl;
    std::cout << "Initializing CUDA..." << std::endl;
    
    // Check CUDA availability
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return -1;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Using GPU: " << prop.name << std::endl;
    std::cout << "Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    
    // Hyperparameters optimized for GTX 1650 (4GB VRAM)
    const int batch_size = 64;          // Reduced for 4GB VRAM
    const int epochs = 50;              // Reduced for faster testing
    const float initial_lr = 3e-4f;
    const int img_size = 28;
    const int patch_size = 4;
    const int embed_dim = 128;          // Reduced from 192
    const int depth = 8;                // Reduced from 12
    const int num_heads = 4;            // Reduced from 8
    const int num_classes = 10;
    
    std::cout << "\n=== Model Configuration ===" << std::endl;
    std::cout << "Batch size: " << batch_size << std::endl;
    std::cout << "Epochs: " << epochs << std::endl;
    std::cout << "Learning rate: " << initial_lr << std::endl;
    std::cout << "Image size: " << img_size << "x" << img_size << std::endl;
    std::cout << "Patch size: " << patch_size << "x" << patch_size << std::endl;
    std::cout << "Embedding dimension: " << embed_dim << std::endl;
    std::cout << "Transformer depth: " << depth << std::endl;
    std::cout << "Number of heads: " << num_heads << std::endl;
    
    // Initialize data loader
    std::cout << "\n=== Loading Fashion-MNIST Dataset ===" << std::endl;
    FashionMNISTLoader data_loader(batch_size);
    
    // Load Fashion-MNIST data (using correct paths)
    bool data_loaded = data_loader.loadData(
        "data/train-images-idx3-ubyte",
        "data/train-labels-idx1-ubyte", 
        "data/t10k-images-idx3-ubyte",
        "data/t10k-labels-idx1-ubyte"
    );
    
    if (!data_loaded) {
        std::cerr << "Failed to load data! Please ensure Fashion-MNIST files are in ./data/ directory" << std::endl;
        std::cerr << "Download from: https://github.com/zalandoresearch/fashion-mnist" << std::endl;
        return -1;
    }
    
    // Initialize model
    std::cout << "\n=== Initializing Vision Transformer ===" << std::endl;
    VisionTransformer model(img_size, patch_size, 1, num_classes, 
                           embed_dim, depth, num_heads);
    
    // Calculate model parameters
    int total_params = embed_dim * patch_size * patch_size +  // patch projection
                      (embed_dim + 1) * embed_dim +           // positional embeddings
                      depth * (3 * embed_dim * embed_dim +    // QKV weights
                              embed_dim * embed_dim +          // projection
                              4 * embed_dim * embed_dim +      // MLP
                              embed_dim * 4 * embed_dim) +     // MLP
                      num_classes * embed_dim;                // classifier
    
    std::cout << "Model parameters: " << total_params / 1000000.0f << "M" << std::endl;
    
    // Initialize optimizer and scheduler
    Optimizer optimizer(initial_lr);
    int total_steps = epochs * data_loader.getNumTrainBatches();
    Scheduler scheduler(initial_lr, total_steps / 10, total_steps);  // 10% warmup
    
    std::cout << "\n=== Training Configuration ===" << std::endl;
    std::cout << "Training batches per epoch: " << data_loader.getNumTrainBatches() << std::endl;
    std::cout << "Validation batches: " << data_loader.getNumTestBatches() << std::endl;
    std::cout << "Total training steps: " << total_steps << std::endl;
    
    // Training loop
    std::cout << "\n=== Starting Training ===" << std::endl;
    
    float best_val_acc = 0.0f;
    int global_step = 0;
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        auto epoch_start = std::chrono::high_resolution_clock::now();
        
        data_loader.resetTrainIterator();
        
        float epoch_loss = 0.0f;
        float epoch_accuracy = 0.0f;
        int batch_count = 0;
        
        std::vector<std::vector<float>> batch_images;
        std::vector<int> batch_labels;
        
        // Training phase
        while (data_loader.getNextTrainBatch(batch_images, batch_labels)) {
            auto batch_start = std::chrono::high_resolution_clock::now();
            
            // Update learning rate
            float current_lr = scheduler.getCosineAnnealingLR(global_step);
            optimizer.setLearningRate(current_lr);
            
            // Convert batch to Matrix format
            Matrix input(batch_images.size(), 28 * 28, true);
            
            // Copy data to GPU (optimized transfer)
            float* host_data = new float[batch_images.size() * 28 * 28];
            for (size_t i = 0; i < batch_images.size(); i++) {
                for (size_t j = 0; j < 28 * 28; j++) {
                    host_data[i * 28 * 28 + j] = batch_images[i][j];
                }
            }
            
            cudaMemcpy(input.data, host_data, 
                       batch_images.size() * 28 * 28 * sizeof(float), 
                       cudaMemcpyHostToDevice);
            delete[] host_data;
            
            // Forward pass
            auto predictions = model.forward(input);
            
            // Compute loss and accuracy
            float batch_loss = model.compute_loss(*predictions, batch_labels);
            float batch_accuracy = model.compute_accuracy(*predictions, batch_labels);
            
            // Backward pass
            Matrix grad_output(predictions->rows, predictions->cols, true);
            // Initialize grad_output with loss gradients (simplified)
            model.backward(grad_output, input);
            
            // Update weights
            optimizer.step(model);
            
            // Update statistics
            epoch_loss += batch_loss;
            epoch_accuracy += batch_accuracy;
            batch_count++;
            global_step++;
            
            // Print progress
            auto batch_end = std::chrono::high_resolution_clock::now();
            float batch_time = std::chrono::duration<float>(batch_end - batch_start).count();
            
            if (batch_count % 10 == 0) {
                printProgress(epoch + 1, batch_count, data_loader.getNumTrainBatches(),
                            batch_loss, batch_accuracy, batch_time);
            }
        }
        
        auto epoch_end = std::chrono::high_resolution_clock::now();
        float epoch_time = std::chrono::duration<float>(epoch_end - epoch_start).count();
        
        // Calculate epoch averages
        float avg_train_loss = epoch_loss / batch_count;
        float avg_train_acc = epoch_accuracy / batch_count;
        
        // Validation
        std::cout << "\nValidating..." << std::endl;
        float val_accuracy = validateModel(model, data_loader);
        
        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "Epoch " << std::setw(3) << epoch + 1 << "/" << epochs 
                  << " | Time: " << std::fixed << std::setprecision(1) << epoch_time << "s" << std::endl;
        std::cout << "Train Loss: " << std::setprecision(4) << avg_train_loss 
                  << " | Train Acc: " << std::setprecision(2) << avg_train_acc * 100 << "%" << std::endl;
        std::cout << "Val Acc: " << std::setprecision(2) << val_accuracy * 100 << "%" << std::endl;
        std::cout << "Learning Rate: " << std::scientific << std::setprecision(2) 
                  << optimizer.getLearningRate() << std::endl;
        
        // Save best model
        if (val_accuracy > best_val_acc) {
            best_val_acc = val_accuracy;
            std::cout << "New best validation accuracy! Saving model..." << std::endl;
            // Here you would implement model saving
        }
        
        std::cout << std::string(80, '=') << std::endl;
        
        // Early stopping check
        if (val_accuracy > 0.95f) {
            std::cout << "Reached 95% accuracy, stopping training early." << std::endl;
            break;
        }
    }
    
    std::cout << "\n=== Training Completed ===" << std::endl;
    std::cout << "Best validation accuracy: " << std::fixed << std::setprecision(2) 
              << best_val_acc * 100 << "%" << std::endl;
    
    // Final evaluation
    std::cout << "\n=== Final Evaluation ===" << std::endl;
    float final_accuracy = validateModel(model, data_loader);
    std::cout << "Final test accuracy: " << std::fixed << std::setprecision(2) 
              << final_accuracy * 100 << "%" << std::endl;
    
    return 0;
}