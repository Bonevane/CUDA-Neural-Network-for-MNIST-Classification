#ifndef UTILS_CUH
#define UTILS_CUH

#include <fstream>
#include <iostream>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <random>
#include <string>

// Error checking macros
#define CHECK_CUDA(call)                                                           \
    if ((call) != cudaSuccess)                                                     \
    {                                                                              \
        std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE);                                                        \
    }

#define CHECK_CUBLAS(call)                                                           \
    if ((call) != CUBLAS_STATUS_SUCCESS)                                             \
    {                                                                                \
        std::cerr << "cuBLAS Error at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE);                                                          \
    }

// CUDA kernels
__global__ void relu_activation(float *input, int size);
__global__ void relu_backward(const float *output, const float *grad_output, float *grad_input, int size);
__global__ void compute_cross_entropy_grad(const float *probs, const float *labels, float *grad, int total_size);

// CPU functions
void softmax(const float *logits, float *probs, int output_dim, int batch_size);

// Weight management
bool save_weights(const std::string& filename,
                  const std::vector<float>& input_weights,
                  const std::vector<float>& output_weights,
                  int input_dim, int hidden_dim, int output_dim);

bool load_weights(const std::string& filename,
                  std::vector<float>& input_weights,
                  std::vector<float>& output_weights,
                  int expected_input_dim, int expected_hidden_dim, int expected_output_dim);

// Evaluation functions
float evaluate_model(cublasHandle_t handle, 
                    const std::vector<float>& h_input_weights_col,
                    const std::vector<float>& h_output_weights_col,
                    const std::vector<float>& test_inputs,
                    const std::vector<float>& test_labels,
                    int num_test_samples, int batch_size, int input_dim, 
                    int hidden_dim, int output_dim, int threads);

void predict_single_image(cublasHandle_t handle,
                         const std::vector<float>& h_input_weights_col,
                         const std::vector<float>& h_output_weights_col,
                         const std::vector<float>& image_data,
                         int input_dim, int hidden_dim, int output_dim, int threads);

// Data loading
bool load_training_data(const std::string& images_file, const std::string& labels_file,
                       std::vector<float>& inputs, std::vector<float>& labels,
                       int num_samples, int input_dim, int output_dim);

bool load_test_data(const std::string& images_file, const std::string& labels_file,
                   std::vector<float>& inputs, std::vector<float>& labels,
                   int num_samples, int input_dim, int output_dim);

// Weight initialization
void initialize_weights(std::vector<float>& input_weights, std::vector<float>& output_weights,
                       std::vector<float>& input_weights_col, std::vector<float>& output_weights_col,
                       int input_dim, int hidden_dim, int output_dim);

#endif // UTILS_CUH