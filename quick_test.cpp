#include "vit_transformer.h"
#include <iostream>

int main() {
    std::cout << "=== Quick ViT Test for GTX 1650 ===" << std::endl;
    
    try {
        // Test Matrix creation and operations
        std::cout << "Testing Matrix class..." << std::endl;
        Matrix test_matrix(32, 784, true);  // 32 images of 28x28
        test_matrix.random(0.0f, 0.1f);
        std::cout << "✓ Matrix creation and initialization successful" << std::endl;
        
        // Test ViT creation with smaller parameters for GTX 1650
        std::cout << "Testing ViT model creation..." << std::endl;
        VisionTransformer model(
            28,    // img_size
            4,     // patch_size  
            1,     // in_channels
            10,    // num_classes
            96,    // embed_dim (reduced from 192 for GTX 1650)
            6,     // depth (reduced from 12 for GTX 1650)
            4      // num_heads (reduced from 8)
        );
        std::cout << "✓ ViT model creation successful" << std::endl;
        
        // Test forward pass with small batch
        std::cout << "Testing forward pass..." << std::endl;
        Matrix input(4, 784, true);  // Small batch of 4 images
        input.random(0.0f, 1.0f);
        
        auto output = model.forward(input);
        std::cout << "✓ Forward pass successful" << std::endl;
        std::cout << "Output shape: " << output->rows << "x" << output->cols << std::endl;
        
        // Test accuracy computation
        std::vector<int> dummy_targets = {0, 1, 2, 3};
        float accuracy = model.compute_accuracy(*output, dummy_targets);
        std::cout << "✓ Accuracy computation successful: " << accuracy << std::endl;
        
        std::cout << "\n🎉 All tests passed! Your GTX 1650 is ready for ViT training!" << std::endl;
        std::cout << "\nRecommended settings for GTX 1650:" << std::endl;
        std::cout << "- Batch size: 32-64" << std::endl;
        std::cout << "- Embed dim: 96-128" << std::endl;
        std::cout << "- Depth: 6-8 layers" << std::endl;
        std::cout << "- Num heads: 4-6" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}