#include "data_loader.h"
#include <iostream>

int main() {
    std::cout << "=== Testing Fashion-MNIST Data Loader ===" << std::endl;
    
    try {
        // Create data loader
        FashionMNISTLoader loader(32);  // Small batch for testing
        
        // Load data
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
        std::cout << "Training samples: " << loader.getTrainSize() << std::endl;
        std::cout << "Test samples: " << loader.getTestSize() << std::endl;
        std::cout << "Training batches: " << loader.getNumTrainBatches() << std::endl;
        std::cout << "Test batches: " << loader.getNumTestBatches() << std::endl;
        
        // Test getting a batch
        std::vector<std::vector<float>> batch_images;
        std::vector<int> batch_labels;
        
        std::cout << "\nTesting batch loading..." << std::endl;
        bool got_batch = loader.getNextTrainBatch(batch_images, batch_labels);
        
        if (got_batch) {
            std::cout << "✅ Got training batch!" << std::endl;
            std::cout << "Batch size: " << batch_images.size() << std::endl;
            std::cout << "Image size: " << batch_images[0].size() << " pixels" << std::endl;
            std::cout << "First few labels: ";
            for (int i = 0; i < std::min(10, (int)batch_labels.size()); i++) {
                std::cout << batch_labels[i] << " ";
            }
            std::cout << std::endl;
            
            // Check pixel values
            float min_val = batch_images[0][0], max_val = batch_images[0][0];
            for (const auto& pixel : batch_images[0]) {
                min_val = std::min(min_val, pixel);
                max_val = std::max(max_val, pixel);
            }
            std::cout << "Pixel range: [" << min_val << ", " << max_val << "]" << std::endl;
            
        } else {
            std::cerr << "❌ Failed to get training batch!" << std::endl;
            return -1;
        }
        
        // Test validation batch
        loader.resetTestIterator();
        bool got_test_batch = loader.getNextTestBatch(batch_images, batch_labels);
        
        if (got_test_batch) {
            std::cout << "✅ Got test batch!" << std::endl;
            std::cout << "Test batch size: " << batch_images.size() << std::endl;
        } else {
            std::cerr << "❌ Failed to get test batch!" << std::endl;
            return -1;
        }
        
        std::cout << "\n🎉 All data loader tests passed!" << std::endl;
        std::cout << "Your Fashion-MNIST dataset is ready for training!" << std::endl;
        
        // Fashion-MNIST class names
        std::vector<std::string> class_names = {
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
        };
        
        std::cout << "\nFashion-MNIST Classes:" << std::endl;
        for (int i = 0; i < class_names.size(); i++) {
            std::cout << i << ": " << class_names[i] << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Exception: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}