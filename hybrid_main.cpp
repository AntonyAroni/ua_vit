#include "hybrid_vit.h"
#include "data_loader.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>

class SimpleOptimizer {
private:
    float learning_rate;
    float decay_rate;
    int step_count;
    
public:
    SimpleOptimizer(float lr = 0.001f, float decay = 0.95f) 
        : learning_rate(lr), decay_rate(decay), step_count(0) {}
    
    void step(HybridVisionTransformer& model) {
        step_count++;
        
        // Simple learning rate decay
        float current_lr = learning_rate * pow(decay_rate, step_count / 100.0f);
        model.update_weights(current_lr);
    }
    
    float get_learning_rate() const { 
        return learning_rate * pow(decay_rate, step_count / 100.0f);
    }
};

void print_progress_bar(int current, int total, float loss, float acc, float time) {
    float progress = (float)current / total;
    int bar_width = 40;
    int pos = bar_width * progress;
    
    std::cout << "\r[";
    for (int i = 0; i < bar_width; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << "% ";
    std::cout << "Loss: " << std::fixed << std::setprecision(4) << loss;
    std::cout << " Acc: " << std::setprecision(2) << acc * 100 << "%";
    std::cout << " Time: " << std::setprecision(1) << time << "s";
    std::cout.flush();
}

float validate_model(HybridVisionTransformer& model, FashionMNISTLoader& data_loader) {
    data_loader.resetTestIterator();
    
    float total_accuracy = 0.0f;
    int num_batches = 0;
    
    std::vector<std::vector<float>> batch_images;
    std::vector<int> batch_labels;
    
    while (data_loader.getNextTestBatch(batch_images, batch_labels) && num_batches < 20) {
        // Convert to GPU matrix
        GPUMatrix input(batch_images.size(), 784);
        std::vector<float> flat_batch;
        
        for (const auto& image : batch_images) {
            flat_batch.insert(flat_batch.end(), image.begin(), image.end());
        }
        
        input.copyFromHost(flat_batch.data());
        
        // Forward pass
        auto predictions = model.forward(input);
        
        // Compute accuracy
        float batch_accuracy = model.compute_accuracy(*predictions, batch_labels);
        total_accuracy += batch_accuracy;
        num_batches++;
    }
    
    return total_accuracy / num_batches;
}

int main() {
    std::cout << "=== Hybrid Vision Transformer Training ===" << std::endl;
    
    // Check CUDA
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cerr << "❌ No CUDA devices found!" << std::endl;
        return -1;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "✅ Using GPU: " << prop.name << std::endl;
    std::cout << "✅ Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    
    try {
        // Hyperparameters optimized for GTX 1650
        const int batch_size = 32;      // Smaller batch for 4GB VRAM
        const int epochs = 20;          // Fewer epochs for testing
        const float initial_lr = 0.001f;
        
        std::cout << "\n=== Configuration ===" << std::endl;
        std::cout << "Batch size: " << batch_size << std::endl;
        std::cout << "Epochs: " << epochs << std::endl;
        std::cout << "Learning rate: " << initial_lr << std::endl;
        
        // Load data
        std::cout << "\n=== Loading Data ===" << std::endl;
        FashionMNISTLoader data_loader(batch_size);
        
        bool data_loaded = data_loader.loadData(
            "data/train-images-idx3-ubyte",
            "data/train-labels-idx1-ubyte",
            "data/t10k-images-idx3-ubyte", 
            "data/t10k-labels-idx1-ubyte"
        );
        
        if (!data_loaded) {
            std::cerr << "❌ Failed to load Fashion-MNIST data!" << std::endl;
            return -1;
        }
        
        std::cout << "✅ Data loaded successfully!" << std::endl;
        std::cout << "Training samples: " << data_loader.getTrainSize() << std::endl;
        std::cout << "Test samples: " << data_loader.getTestSize() << std::endl;
        
        // Create model
        std::cout << "\n=== Creating Model ===" << std::endl;
        HybridVisionTransformer model(
            28,    // img_size
            4,     // patch_size  
            128,   // embed_dim (reduced for GTX 1650)
            4,     // num_heads (reduced)
            6,     // num_layers (reduced)
            10     // num_classes
        );
        
        model.print_model_info();
        
        // Create optimizer
        SimpleOptimizer optimizer(initial_lr, 0.98f);
        
        std::cout << "\n=== Starting Training ===" << std::endl;
        
        float best_val_acc = 0.0f;
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            auto epoch_start = std::chrono::high_resolution_clock::now();
            
            data_loader.resetTrainIterator();
            
            float epoch_loss = 0.0f;
            float epoch_accuracy = 0.0f;
            int batch_count = 0;
            const int max_batches = 100;  // Limit batches for faster testing
            
            std::vector<std::vector<float>> batch_images;
            std::vector<int> batch_labels;
            
            while (data_loader.getNextTrainBatch(batch_images, batch_labels) && 
                   batch_count < max_batches) {
                
                auto batch_start = std::chrono::high_resolution_clock::now();
                
                // Convert batch to GPU matrix
                GPUMatrix input(batch_images.size(), 784);
                std::vector<float> flat_batch;
                
                for (const auto& image : batch_images) {
                    flat_batch.insert(flat_batch.end(), image.begin(), image.end());
                }
                
                input.copyFromHost(flat_batch.data());
                
                // Forward pass
                auto predictions = model.forward(input);
                
                // Compute loss and accuracy
                float batch_loss = model.compute_loss(*predictions, batch_labels);
                float batch_accuracy = model.compute_accuracy(*predictions, batch_labels);
                
                // Update weights
                optimizer.step(model);
                
                // Accumulate statistics
                epoch_loss += batch_loss;
                epoch_accuracy += batch_accuracy;
                batch_count++;
                
                // Print progress
                auto batch_end = std::chrono::high_resolution_clock::now();
                float batch_time = std::chrono::duration<float>(batch_end - batch_start).count();
                
                if (batch_count % 10 == 0 || batch_count == 1) {
                    print_progress_bar(batch_count, max_batches, batch_loss, batch_accuracy, batch_time);
                }
            }
            
            auto epoch_end = std::chrono::high_resolution_clock::now();
            float epoch_time = std::chrono::duration<float>(epoch_end - epoch_start).count();
            
            // Calculate averages
            float avg_loss = epoch_loss / batch_count;
            float avg_accuracy = epoch_accuracy / batch_count;
            
            std::cout << std::endl; // New line after progress bar
            
            // Validation
            std::cout << "Validating..." << std::endl;
            float val_accuracy = validate_model(model, data_loader);
            
            // Print epoch summary
            std::cout << "\n" << std::string(70, '=') << std::endl;
            std::cout << "Epoch " << std::setw(2) << epoch + 1 << "/" << epochs;
            std::cout << " | Time: " << std::fixed << std::setprecision(1) << epoch_time << "s" << std::endl;
            std::cout << "Train Loss: " << std::setprecision(4) << avg_loss;
            std::cout << " | Train Acc: " << std::setprecision(2) << avg_accuracy * 100 << "%" << std::endl;
            std::cout << "Val Acc: " << std::setprecision(2) << val_accuracy * 100 << "%";
            std::cout << " | LR: " << std::scientific << std::setprecision(2) << optimizer.get_learning_rate() << std::endl;
            
            // Save best model
            if (val_accuracy > best_val_acc) {
                best_val_acc = val_accuracy;
                std::cout << "🎉 New best validation accuracy!" << std::endl;
            }
            
            std::cout << std::string(70, '=') << std::endl;
            
            // Early stopping
            if (val_accuracy > 0.85f) {
                std::cout << "🎯 Reached 85% accuracy, stopping early!" << std::endl;
                break;
            }
        }
        
        std::cout << "\n🎉 Training Complete!" << std::endl;
        std::cout << "Best validation accuracy: " << std::fixed << std::setprecision(2) 
                  << best_val_acc * 100 << "%" << std::endl;
        
        // Final test
        std::cout << "\n=== Final Evaluation ===" << std::endl;
        float final_accuracy = validate_model(model, data_loader);
        std::cout << "Final test accuracy: " << std::setprecision(2) 
                  << final_accuracy * 100 << "%" << std::endl;
        
        std::cout << "\n✨ Hybrid Vision Transformer training completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Exception: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}