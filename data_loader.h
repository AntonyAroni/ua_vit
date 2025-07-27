#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <vector>
#include <string>
#include <memory>

class FashionMNISTLoader {
private:
    std::vector<std::vector<float>> train_images;
    std::vector<int> train_labels;
    std::vector<std::vector<float>> test_images;
    std::vector<int> test_labels;
    
    int current_train_idx;
    int current_test_idx;
    int batch_size;
    
    // Helper functions
    int reverseInt(int i);
    void normalizeImages(std::vector<std::vector<float>>& images);
    void shuffleTrainData();
    
public:
    FashionMNISTLoader(int batch_size = 32);
    ~FashionMNISTLoader();
    
    bool loadData(const std::string& train_images_path,
                  const std::string& train_labels_path,
                  const std::string& test_images_path,
                  const std::string& test_labels_path);
    
    bool getNextTrainBatch(std::vector<std::vector<float>>& batch_images,
                          std::vector<int>& batch_labels);
    
    bool getNextTestBatch(std::vector<std::vector<float>>& batch_images,
                         std::vector<int>& batch_labels);
    
    void resetTrainIterator();
    void resetTestIterator();
    
    int getTrainSize() const { return train_images.size(); }
    int getTestSize() const { return test_images.size(); }
    int getBatchSize() const { return batch_size; }
    int getNumTrainBatches() const { return (train_images.size() + batch_size - 1) / batch_size; }
    int getNumTestBatches() const { return (test_images.size() + batch_size - 1) / batch_size; }
    
    // Data augmentation functions
    void applyRandomHorizontalFlip(std::vector<float>& image, float probability = 0.5f);
    void applyRandomRotation(std::vector<float>& image, float max_angle = 10.0f);
    void applyNoise(std::vector<float>& image, float noise_std = 0.01f);
};

class GPUDataLoader {
private:
    float* d_train_images;
    int* d_train_labels;
    float* d_test_images;
    int* d_test_labels;
    
    float* d_batch_images;
    int* d_batch_labels;
    
    int train_size;
    int test_size;
    int batch_size;
    int image_size;
    
public:
    GPUDataLoader(int batch_size, int image_size = 28);
    ~GPUDataLoader();
    
    bool loadDataToGPU(const FashionMNISTLoader& cpu_loader);
    
    float* getTrainImagesBatch(int batch_idx);
    int* getTrainLabelsBatch(int batch_idx);
    float* getTestImagesBatch(int batch_idx);
    int* getTestLabelsBatch(int batch_idx);
    
    int getTrainSize() const { return train_size; }
    int getTestSize() const { return test_size; }
    int getBatchSize() const { return batch_size; }
    int getNumTrainBatches() const { return (train_size + batch_size - 1) / batch_size; }
    int getNumTestBatches() const { return (test_size + batch_size - 1) / batch_size; }
};

#endif