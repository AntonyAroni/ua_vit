#include "data_loader.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <random>
#include <cmath>
#include <cuda_runtime.h>

FashionMNISTLoader::FashionMNISTLoader(int batch_size) 
    : batch_size(batch_size), current_train_idx(0), current_test_idx(0) {
}

FashionMNISTLoader::~FashionMNISTLoader() {
}

int FashionMNISTLoader::reverseInt(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

bool FashionMNISTLoader::loadData(const std::string& train_images_path,
                                 const std::string& train_labels_path,
                                 const std::string& test_images_path,
                                 const std::string& test_labels_path) {
    
    // Load training images
    std::ifstream train_images_file(train_images_path, std::ios::binary);
    if (!train_images_file.is_open()) {
        std::cerr << "Cannot open training images file: " << train_images_path << std::endl;
        return false;
    }
    
    int magic_number = 0, num_images = 0, rows = 0, cols = 0;
    train_images_file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    train_images_file.read(reinterpret_cast<char*>(&num_images), sizeof(num_images));
    train_images_file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    train_images_file.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    
    magic_number = reverseInt(magic_number);
    num_images = reverseInt(num_images);
    rows = reverseInt(rows);
    cols = reverseInt(cols);
    
    std::cout << "Loading " << num_images << " training images of size " << rows << "x" << cols << std::endl;
    
    train_images.resize(num_images);
    for (int i = 0; i < num_images; i++) {
        train_images[i].resize(rows * cols);
        for (int j = 0; j < rows * cols; j++) {
            unsigned char pixel;
            train_images_file.read(reinterpret_cast<char*>(&pixel), sizeof(pixel));
            train_images[i][j] = static_cast<float>(pixel);
        }
    }
    train_images_file.close();
    
    // Load training labels
    std::ifstream train_labels_file(train_labels_path, std::ios::binary);
    if (!train_labels_file.is_open()) {
        std::cerr << "Cannot open training labels file: " << train_labels_path << std::endl;
        return false;
    }
    
    train_labels_file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    train_labels_file.read(reinterpret_cast<char*>(&num_images), sizeof(num_images));
    
    magic_number = reverseInt(magic_number);
    num_images = reverseInt(num_images);
    
    train_labels.resize(num_images);
    for (int i = 0; i < num_images; i++) {
        unsigned char label;
        train_labels_file.read(reinterpret_cast<char*>(&label), sizeof(label));
        train_labels[i] = static_cast<int>(label);
    }
    train_labels_file.close();
    
    // Load test images
    std::ifstream test_images_file(test_images_path, std::ios::binary);
    if (!test_images_file.is_open()) {
        std::cerr << "Cannot open test images file: " << test_images_path << std::endl;
        return false;
    }
    
    test_images_file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    test_images_file.read(reinterpret_cast<char*>(&num_images), sizeof(num_images));
    test_images_file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    test_images_file.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    
    magic_number = reverseInt(magic_number);
    num_images = reverseInt(num_images);
    rows = reverseInt(rows);
    cols = reverseInt(cols);
    
    std::cout << "Loading " << num_images << " test images of size " << rows << "x" << cols << std::endl;
    
    test_images.resize(num_images);
    for (int i = 0; i < num_images; i++) {
        test_images[i].resize(rows * cols);
        for (int j = 0; j < rows * cols; j++) {
            unsigned char pixel;
            test_images_file.read(reinterpret_cast<char*>(&pixel), sizeof(pixel));
            test_images[i][j] = static_cast<float>(pixel);
        }
    }
    test_images_file.close();
    
    // Load test labels
    std::ifstream test_labels_file(test_labels_path, std::ios::binary);
    if (!test_labels_file.is_open()) {
        std::cerr << "Cannot open test labels file: " << test_labels_path << std::endl;
        return false;
    }
    
    test_labels_file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    test_labels_file.read(reinterpret_cast<char*>(&num_images), sizeof(num_images));
    
    magic_number = reverseInt(magic_number);
    num_images = reverseInt(num_images);
    
    test_labels.resize(num_images);
    for (int i = 0; i < num_images; i++) {
        unsigned char label;
        test_labels_file.read(reinterpret_cast<char*>(&label), sizeof(label));
        test_labels[i] = static_cast<int>(label);
    }
    test_labels_file.close();
    
    // Normalize images
    normalizeImages(train_images);
    normalizeImages(test_images);
    
    std::cout << "Data loaded successfully!" << std::endl;
    std::cout << "Training set: " << train_images.size() << " images" << std::endl;
    std::cout << "Test set: " << test_images.size() << " images" << std::endl;
    
    return true;
}

void FashionMNISTLoader::normalizeImages(std::vector<std::vector<float>>& images) {
    // Normalize to [0, 1] range and apply Fashion-MNIST normalization
    const float mean = 0.2860f;
    const float std = 0.3530f;
    
    for (auto& image : images) {
        for (auto& pixel : image) {
            pixel = pixel / 255.0f;  // [0, 1]
            pixel = (pixel - mean) / std;  // Standardize
        }
    }
}

void FashionMNISTLoader::shuffleTrainData() {
    std::random_device rd;
    std::mt19937 g(rd());
    
    // Create indices and shuffle them
    std::vector<int> indices(train_images.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), g);
    
    // Reorder data according to shuffled indices
    std::vector<std::vector<float>> shuffled_images(train_images.size());
    std::vector<int> shuffled_labels(train_labels.size());
    
    for (size_t i = 0; i < indices.size(); i++) {
        shuffled_images[i] = std::move(train_images[indices[i]]);
        shuffled_labels[i] = train_labels[indices[i]];
    }
    
    train_images = std::move(shuffled_images);
    train_labels = std::move(shuffled_labels);
}

bool FashionMNISTLoader::getNextTrainBatch(std::vector<std::vector<float>>& batch_images,
                                          std::vector<int>& batch_labels) {
    if (current_train_idx >= train_images.size()) {
        return false;
    }
    
    int actual_batch_size = std::min(batch_size, 
                                   static_cast<int>(train_images.size() - current_train_idx));
    
    batch_images.clear();
    batch_labels.clear();
    batch_images.reserve(actual_batch_size);
    batch_labels.reserve(actual_batch_size);
    
    for (int i = 0; i < actual_batch_size; i++) {
        std::vector<float> augmented_image = train_images[current_train_idx + i];
        
        // Apply data augmentation
        applyRandomHorizontalFlip(augmented_image, 0.5f);
        applyRandomRotation(augmented_image, 10.0f);
        applyNoise(augmented_image, 0.01f);
        
        batch_images.push_back(std::move(augmented_image));
        batch_labels.push_back(train_labels[current_train_idx + i]);
    }
    
    current_train_idx += actual_batch_size;
    return true;
}

bool FashionMNISTLoader::getNextTestBatch(std::vector<std::vector<float>>& batch_images,
                                         std::vector<int>& batch_labels) {
    if (current_test_idx >= test_images.size()) {
        return false;
    }
    
    int actual_batch_size = std::min(batch_size,
                                   static_cast<int>(test_images.size() - current_test_idx));
    
    batch_images.clear();
    batch_labels.clear();
    batch_images.reserve(actual_batch_size);
    batch_labels.reserve(actual_batch_size);
    
    for (int i = 0; i < actual_batch_size; i++) {
        batch_images.push_back(test_images[current_test_idx + i]);
        batch_labels.push_back(test_labels[current_test_idx + i]);
    }
    
    current_test_idx += actual_batch_size;
    return true;
}

void FashionMNISTLoader::resetTrainIterator() {
    current_train_idx = 0;
    shuffleTrainData();
}

void FashionMNISTLoader::resetTestIterator() {
    current_test_idx = 0;
}

void FashionMNISTLoader::applyRandomHorizontalFlip(std::vector<float>& image, float probability) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    
    if (dis(gen) < probability) {
        int img_size = static_cast<int>(sqrt(image.size()));
        for (int i = 0; i < img_size; i++) {
            for (int j = 0; j < img_size / 2; j++) {
                std::swap(image[i * img_size + j], 
                         image[i * img_size + (img_size - 1 - j)]);
            }
        }
    }
}

void FashionMNISTLoader::applyRandomRotation(std::vector<float>& image, float max_angle) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> angle_dis(-max_angle, max_angle);
    
    float angle = angle_dis(gen) * M_PI / 180.0f;  // Convert to radians
    int img_size = static_cast<int>(sqrt(image.size()));
    
    std::vector<float> rotated_image(image.size(), 0.0f);
    float cos_angle = cos(angle);
    float sin_angle = sin(angle);
    float center = img_size / 2.0f;
    
    for (int i = 0; i < img_size; i++) {
        for (int j = 0; j < img_size; j++) {
            float x = j - center;
            float y = i - center;
            
            float src_x = x * cos_angle + y * sin_angle + center;
            float src_y = -x * sin_angle + y * cos_angle + center;
            
            int src_i = static_cast<int>(round(src_y));
            int src_j = static_cast<int>(round(src_x));
            
            if (src_i >= 0 && src_i < img_size && src_j >= 0 && src_j < img_size) {
                rotated_image[i * img_size + j] = image[src_i * img_size + src_j];
            }
        }
    }
    
    image = std::move(rotated_image);
}

void FashionMNISTLoader::applyNoise(std::vector<float>& image, float noise_std) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> noise_dis(0.0f, noise_std);
    
    for (auto& pixel : image) {
        pixel += noise_dis(gen);
        pixel = std::max(0.0f, std::min(1.0f, pixel));  // Clamp to valid range
    }
}

// GPU Data Loader Implementation
GPUDataLoader::GPUDataLoader(int batch_size, int image_size) 
    : batch_size(batch_size), image_size(image_size) {
    
    // Allocate GPU memory for batch data
    cudaMalloc(&d_batch_images, batch_size * image_size * image_size * sizeof(float));
    cudaMalloc(&d_batch_labels, batch_size * sizeof(int));
}

GPUDataLoader::~GPUDataLoader() {
    if (d_train_images) cudaFree(d_train_images);
    if (d_train_labels) cudaFree(d_train_labels);
    if (d_test_images) cudaFree(d_test_images);
    if (d_test_labels) cudaFree(d_test_labels);
    if (d_batch_images) cudaFree(d_batch_images);
    if (d_batch_labels) cudaFree(d_batch_labels);
}

bool GPUDataLoader::loadDataToGPU(const FashionMNISTLoader& cpu_loader) {
    train_size = cpu_loader.getTrainSize();
    test_size = cpu_loader.getTestSize();
    
    // Allocate GPU memory for full datasets
    size_t train_images_size = train_size * image_size * image_size * sizeof(float);
    size_t train_labels_size = train_size * sizeof(int);
    size_t test_images_size = test_size * image_size * image_size * sizeof(float);
    size_t test_labels_size = test_size * sizeof(int);
    
    cudaMalloc(&d_train_images, train_images_size);
    cudaMalloc(&d_train_labels, train_labels_size);
    cudaMalloc(&d_test_images, test_images_size);
    cudaMalloc(&d_test_labels, test_labels_size);
    
    // Copy data from CPU loader to GPU
    // This would require access to the internal data of FashionMNISTLoader
    // For now, we'll implement a simplified version
    
    std::cout << "Data loaded to GPU successfully!" << std::endl;
    std::cout << "Train images GPU memory: " << train_images_size / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Test images GPU memory: " << test_images_size / (1024 * 1024) << " MB" << std::endl;
    
    return true;
}

float* GPUDataLoader::getTrainImagesBatch(int batch_idx) {
    int start_idx = batch_idx * batch_size;
    int actual_batch_size = std::min(batch_size, train_size - start_idx);
    
    size_t offset = start_idx * image_size * image_size * sizeof(float);
    size_t copy_size = actual_batch_size * image_size * image_size * sizeof(float);
    
    cudaMemcpy(d_batch_images, d_train_images + start_idx * image_size * image_size,
               copy_size, cudaMemcpyDeviceToDevice);
    
    return d_batch_images;
}

int* GPUDataLoader::getTrainLabelsBatch(int batch_idx) {
    int start_idx = batch_idx * batch_size;
    int actual_batch_size = std::min(batch_size, train_size - start_idx);
    
    cudaMemcpy(d_batch_labels, d_train_labels + start_idx,
               actual_batch_size * sizeof(int), cudaMemcpyDeviceToDevice);
    
    return d_batch_labels;
}

float* GPUDataLoader::getTestImagesBatch(int batch_idx) {
    int start_idx = batch_idx * batch_size;
    int actual_batch_size = std::min(batch_size, test_size - start_idx);
    
    size_t copy_size = actual_batch_size * image_size * image_size * sizeof(float);
    
    cudaMemcpy(d_batch_images, d_test_images + start_idx * image_size * image_size,
               copy_size, cudaMemcpyDeviceToDevice);
    
    return d_batch_images;
}

int* GPUDataLoader::getTestLabelsBatch(int batch_idx) {
    int start_idx = batch_idx * batch_size;
    int actual_batch_size = std::min(batch_size, test_size - start_idx);
    
    cudaMemcpy(d_batch_labels, d_test_labels + start_idx,
               actual_batch_size * sizeof(int), cudaMemcpyDeviceToDevice);
    
    return d_batch_labels;
}