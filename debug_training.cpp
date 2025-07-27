#include "data_loader.h"
#include <iostream>
#include <random>

// Simple test to debug the training pipeline
int main() {
    std::cout << "=== Debugging Training Pipeline ===" << std::endl;
    
    try {
        // Load a small batch of data
        FashionMNISTLoader loader(4);  // Very small batch
        
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
        
        // Get a batch
        std::vector<std::vector<float>> batch_images;
        std::vector<int> batch_labels;
        
        bool got_batch = loader.getNextTrainBatch(batch_images, batch_labels);
        if (!got_batch) {
            std::cerr << "❌ Failed to get batch!" << std::endl;
            return -1;
        }
        
        std::cout << "✅ Got batch with " << batch_images.size() << " images" << std::endl;
        
        // Analyze the data
        std::cout << "\n=== Data Analysis ===" << std::endl;
        std::cout << "Batch size: " << batch_images.size() << std::endl;
        std::cout << "Image size: " << batch_images[0].size() << " pixels" << std::endl;
        
        // Check labels
        std::cout << "Labels: ";
        for (int label : batch_labels) {
            std::cout << label << " ";
        }
        std::cout << std::endl;
        
        // Check label distribution
        std::vector<int> label_count(10, 0);
        for (int label : batch_labels) {
            if (label >= 0 && label < 10) {
                label_count[label]++;
            } else {
                std::cout << "❌ Invalid label: " << label << std::endl;
            }
        }
        
        std::cout << "Label distribution: ";
        for (int i = 0; i < 10; i++) {
            if (label_count[i] > 0) {
                std::cout << i << ":" << label_count[i] << " ";
            }
        }
        std::cout << std::endl;
        
        // Check pixel values
        std::cout << "\n=== Pixel Analysis ===" << std::endl;
        const auto& first_image = batch_images[0];
        
        float min_val = first_image[0];
        float max_val = first_image[0];
        float sum = 0.0f;
        
        for (float pixel : first_image) {
            min_val = std::min(min_val, pixel);
            max_val = std::max(max_val, pixel);
            sum += pixel;
        }
        
        float mean = sum / first_image.size();
        std::cout << "Pixel range: [" << min_val << ", " << max_val << "]" << std::endl;
        std::cout << "Pixel mean: " << mean << std::endl;
        
        // Check if data is normalized properly
        if (min_val >= -5.0f && max_val <= 5.0f) {
            std::cout << "✅ Data appears to be normalized" << std::endl;
        } else {
            std::cout << "⚠️  Data normalization might be off" << std::endl;
        }
        
        // Test random predictions
        std::cout << "\n=== Testing Random Predictions ===" << std::endl;
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        
        // Generate random "predictions" (like what a untrained model would output)
        std::vector<std::vector<float>> fake_predictions(batch_images.size());
        for (int i = 0; i < batch_images.size(); i++) {
            fake_predictions[i].resize(10);  // 10 classes
            for (int j = 0; j < 10; j++) {
                fake_predictions[i][j] = dis(gen);
            }
        }
        
        // Test accuracy calculation
        int correct = 0;
        for (int i = 0; i < batch_images.size(); i++) {
            int predicted_class = 0;
            float max_prob = fake_predictions[i][0];
            
            for (int j = 1; j < 10; j++) {
                if (fake_predictions[i][j] > max_prob) {
                    max_prob = fake_predictions[i][j];
                    predicted_class = j;
                }
            }
            
            if (predicted_class == batch_labels[i]) {
                correct++;
            }
            
            std::cout << "Sample " << i << ": True=" << batch_labels[i] 
                      << ", Pred=" << predicted_class 
                      << " (" << (predicted_class == batch_labels[i] ? "✅" : "❌") << ")" << std::endl;
        }
        
        float accuracy = (float)correct / batch_images.size();
        std::cout << "Random accuracy: " << accuracy * 100 << "%" << std::endl;
        std::cout << "Expected random accuracy: ~10%" << std::endl;
        
        std::cout << "\n🎉 Pipeline debugging complete!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Exception: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}