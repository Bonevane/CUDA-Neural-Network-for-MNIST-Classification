#include "utils.cuh"

// === CUDA kernels ===

// Kernel for ReLU activation. Self explanatory.
__global__ void relu_activation(float *input, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        input[idx] = fmaxf(0.0f, input[idx]);
    }
}

// Kernel for ReLU backward pass. Sets gradients to 0 where input was <= 0.
__global__ void relu_backward(const float *output, const float *grad_output, float *grad_input, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        grad_input[idx] = output[idx] > 0.0f ? grad_output[idx] : 0.0f;
    }
}

// Kernel for computing cross-entropy gradient. Computes dL/dz = p - y.
// where p is the predicted probabilities, y is the true labels.
__global__ void compute_cross_entropy_grad(const float *probs, const float *labels, float *grad, int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size)
        grad[idx] = probs[idx] - labels[idx];
}

// CPU softmax function. 
void softmax(const float *logits, float *probs, int output_dim, int batch_size)
{
    for (int i = 0; i < batch_size; ++i)
    {
        float max_logit = logits[i * output_dim];
        for (int j = 1; j < output_dim; ++j)
        {
            if (logits[i * output_dim + j] > max_logit)
            {
                max_logit = logits[i * output_dim + j];
            }
        }
        float sum = 0.0f;
        for (int j = 0; j < output_dim; ++j)
        {
            probs[i * output_dim + j] = expf(logits[i * output_dim + j] - max_logit);
            sum += probs[i * output_dim + j];
        }
        for (int j = 0; j < output_dim; ++j)
        {
            probs[i * output_dim + j] /= sum;
        }
    }
}

// Weight management functions
bool save_weights(const std::string &filename,
                  const std::vector<float> &input_weights,
                  const std::vector<float> &output_weights,
                  int input_dim, int hidden_dim, int output_dim)
{
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open file " << filename << " for writing.\n";
        return false;
    }

    // Write header with dimensions for validation
    file.write(reinterpret_cast<const char *>(&input_dim), sizeof(int));
    file.write(reinterpret_cast<const char *>(&hidden_dim), sizeof(int));
    file.write(reinterpret_cast<const char *>(&output_dim), sizeof(int));

    // Write input weights
    size_t input_weights_size = input_weights.size();
    file.write(reinterpret_cast<const char *>(&input_weights_size), sizeof(size_t));
    file.write(reinterpret_cast<const char *>(input_weights.data()),
               input_weights_size * sizeof(float));

    // Write output weights
    size_t output_weights_size = output_weights.size();
    file.write(reinterpret_cast<const char *>(&output_weights_size), sizeof(size_t));
    file.write(reinterpret_cast<const char *>(output_weights.data()),
               output_weights_size * sizeof(float));

    file.close();

    if (file.fail())
    {
        std::cerr << "Error: Failed to write weights to " << filename << ".\n";
        return false;
    }

    std::cout << "Weights saved successfully to " << filename << "\n";
    return true;
}

bool load_weights(const std::string &filename,
                  std::vector<float> &input_weights,
                  std::vector<float> &output_weights,
                  int expected_input_dim, int expected_hidden_dim, int expected_output_dim)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open file " << filename << " for reading.\n";
        return false;
    }

    // Read and validate header
    int input_dim, hidden_dim, output_dim;
    file.read(reinterpret_cast<char *>(&input_dim), sizeof(int));
    file.read(reinterpret_cast<char *>(&hidden_dim), sizeof(int));
    file.read(reinterpret_cast<char *>(&output_dim), sizeof(int));

    if (input_dim != expected_input_dim || hidden_dim != expected_hidden_dim ||
        output_dim != expected_output_dim)
    {
        std::cerr << "Error: Weight dimensions mismatch!\n";
        std::cerr << "Expected: " << expected_input_dim << "x" << expected_hidden_dim
                  << "x" << expected_output_dim << "\n";
        std::cerr << "Found: " << input_dim << "x" << hidden_dim << "x" << output_dim << "\n";
        file.close();
        return false;
    }

    // Read input weights
    size_t input_weights_size;
    file.read(reinterpret_cast<char *>(&input_weights_size), sizeof(size_t));

    if (input_weights_size != expected_input_dim * expected_hidden_dim)
    {
        std::cerr << "Error: Input weights size mismatch!\n";
        file.close();
        return false;
    }

    input_weights.resize(input_weights_size);
    file.read(reinterpret_cast<char *>(input_weights.data()),
              input_weights_size * sizeof(float));

    // Read output weights
    size_t output_weights_size;
    file.read(reinterpret_cast<char *>(&output_weights_size), sizeof(size_t));

    if (output_weights_size != expected_hidden_dim * expected_output_dim)
    {
        std::cerr << "Error: Output weights size mismatch!\n";
        file.close();
        return false;
    }

    output_weights.resize(output_weights_size);
    file.read(reinterpret_cast<char *>(output_weights.data()),
              output_weights_size * sizeof(float));

    file.close();

    if (file.fail())
    {
        std::cerr << "Error: Failed to read weights from " << filename << ".\n";
        return false;
    }

    std::cout << "Weights loaded successfully from " << filename << "\n";
    return true;
}

// Evaluation function
float evaluate_model(cublasHandle_t handle,
                     const std::vector<float> &h_input_weights_col,
                     const std::vector<float> &h_output_weights_col,
                     const std::vector<float> &test_inputs,
                     const std::vector<float> &test_labels,
                     int num_test_samples, int batch_size, int input_dim,
                     int hidden_dim, int output_dim, int threads)
{
    // Allocate device (GPU) memory for testing
    float *d_test_input, *d_test_weights1, *d_test_output, *d_test_weights2;
    float *d_test_logits, *d_test_labels;

    CHECK_CUDA(cudaMalloc(&d_test_input, batch_size * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_test_weights1, input_dim * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_test_output, batch_size * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_test_weights2, hidden_dim * output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_test_logits, batch_size * output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_test_labels, batch_size * output_dim * sizeof(float)));

    // Upload trained weights to GPU
    CHECK_CUDA(cudaMemcpy(d_test_weights1, h_input_weights_col.data(),
                          h_input_weights_col.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_test_weights2, h_output_weights_col.data(),
                          h_output_weights_col.size() * sizeof(float), cudaMemcpyHostToDevice));

    // Host buffers for batch processing
    std::vector<float> h_test_batch_input(batch_size * input_dim);
    std::vector<float> h_test_batch_labels(batch_size * output_dim);
    std::vector<float> h_test_logits(batch_size * output_dim);
    std::vector<float> h_test_probs(batch_size * output_dim);

    float alpha = 1.0f, beta = 0.0f;
    int correct_predictions = 0;
    float total_loss = 0.0f;
    int total_processed = 0;

    for (int i = 0; i < num_test_samples; i += batch_size)
    {
        int current_batch = std::min(batch_size, num_test_samples - i);

        // Copy test batch
        std::copy(test_inputs.begin() + i * input_dim,
                  test_inputs.begin() + (i + current_batch) * input_dim,
                  h_test_batch_input.begin());
        std::copy(test_labels.begin() + i * output_dim,
                  test_labels.begin() + (i + current_batch) * output_dim,
                  h_test_batch_labels.begin());

        // Upload test data
        CHECK_CUDA(cudaMemcpy(d_test_input, h_test_batch_input.data(),
                              current_batch * input_dim * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_test_labels, h_test_batch_labels.data(),
                              current_batch * output_dim * sizeof(float), cudaMemcpyHostToDevice));

        // Forward pass
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 hidden_dim, current_batch, input_dim,
                                 &alpha,
                                 d_test_weights1, hidden_dim,
                                 d_test_input, input_dim,
                                 &beta,
                                 d_test_output, hidden_dim));

        int blocks = (current_batch * hidden_dim + threads - 1) / threads;
        relu_activation<<<blocks, threads>>>(d_test_output, current_batch * hidden_dim);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 output_dim, current_batch, hidden_dim,
                                 &alpha,
                                 d_test_weights2, output_dim,
                                 d_test_output, hidden_dim,
                                 &beta,
                                 d_test_logits, output_dim));

        // Get predictions
        CHECK_CUDA(cudaMemcpy(h_test_logits.data(), d_test_logits,
                              current_batch * output_dim * sizeof(float), cudaMemcpyDeviceToHost));

        // Apply softmax and compute accuracy
        softmax(h_test_logits.data(), h_test_probs.data(), output_dim, current_batch);

        // Calculate accuracy and loss for this batch
        for (int sample = 0; sample < current_batch; ++sample)
        {
            // Find predicted class (highest probability)
            int predicted_class = 0;
            float max_prob = h_test_probs[sample * output_dim];
            for (int j = 1; j < output_dim; ++j)
            {
                if (h_test_probs[sample * output_dim + j] > max_prob)
                {
                    max_prob = h_test_probs[sample * output_dim + j];
                    predicted_class = j;
                }
            }

            // Find true class
            int true_class = 0;
            for (int j = 0; j < output_dim; ++j)
            {
                if (h_test_batch_labels[sample * output_dim + j] == 1.0f)
                {
                    true_class = j;
                    break;
                }
            }

            // Check if prediction is correct
            if (predicted_class == true_class)
                correct_predictions++;

            // Add to loss
            total_loss += -logf(fmaxf(h_test_probs[sample * output_dim + true_class], 1e-8f));
            total_processed++;
        }
    }

    // Cleanup
    cudaFree(d_test_input);
    cudaFree(d_test_weights1);
    cudaFree(d_test_output);
    cudaFree(d_test_weights2);
    cudaFree(d_test_logits);
    cudaFree(d_test_labels);

    float accuracy = (float)correct_predictions / total_processed;
    float avg_loss = total_loss / total_processed;

    std::cout << "Test Results:\n";
    std::cout << "  Accuracy: " << accuracy * 100.0f << "% (" << correct_predictions
              << "/" << total_processed << ")\n";
    std::cout << "  Average Loss: " << avg_loss << "\n";

    return accuracy;
}

// Single image prediction
void predict_single_image(cublasHandle_t handle,
                          const std::vector<float> &h_input_weights_col,
                          const std::vector<float> &h_output_weights_col,
                          const std::vector<float> &image_data,
                          int input_dim, int hidden_dim, int output_dim, int threads)
{
    // Allocate device memory for single prediction
    float *d_input, *d_weights1, *d_output, *d_weights2, *d_logits;

    CHECK_CUDA(cudaMalloc(&d_input, input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_weights1, input_dim * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_weights2, hidden_dim * output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_logits, output_dim * sizeof(float)));

    // Upload data
    CHECK_CUDA(cudaMemcpy(d_input, image_data.data(), input_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_weights1, h_input_weights_col.data(),
                          h_input_weights_col.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_weights2, h_output_weights_col.data(),
                          h_output_weights_col.size() * sizeof(float), cudaMemcpyHostToDevice));

    float alpha = 1.0f, beta = 0.0f;

    // Forward pass
    CHECK_CUBLAS(cublasSgemv(handle, CUBLAS_OP_N, hidden_dim, input_dim,
                             &alpha, d_weights1, hidden_dim, d_input, 1,
                             &beta, d_output, 1));

    int blocks = (hidden_dim + threads - 1) / threads;
    relu_activation<<<blocks, threads>>>(d_output, hidden_dim);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUBLAS(cublasSgemv(handle, CUBLAS_OP_N, output_dim, hidden_dim,
                             &alpha, d_weights2, output_dim, d_output, 1,
                             &beta, d_logits, 1));

    // Get results
    std::vector<float> logits(output_dim);
    std::vector<float> probs(output_dim);
    CHECK_CUDA(cudaMemcpy(logits.data(), d_logits, output_dim * sizeof(float), cudaMemcpyDeviceToHost));

    // Apply softmax
    softmax(logits.data(), probs.data(), output_dim, 1);

    // Find prediction
    int predicted_class = 0;
    float max_prob = probs[0];
    for (int i = 1; i < output_dim; ++i)
    {
        if (probs[i] > max_prob)
        {
            max_prob = probs[i];
            predicted_class = i;
        }
    }

    std::cout << "Prediction: " << predicted_class << " (confidence: " << max_prob * 100.0f << "%)\n";
    std::cout << "All probabilities: ";
    for (int i = 0; i < output_dim; ++i)
        std::cout << i << ":" << probs[i] * 100.0f << "% ";
    std::cout << "\n";

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_weights1);
    cudaFree(d_output);
    cudaFree(d_weights2);
    cudaFree(d_logits);
}

// Data loading functions
bool load_training_data(const std::string &images_file, const std::string &labels_file,
                        std::vector<float> &inputs, std::vector<float> &labels,
                        int num_samples, int input_dim, int output_dim)
{
    FILE *f_images = fopen(images_file.c_str(), "rb");
    FILE *f_labels = fopen(labels_file.c_str(), "rb");

    if (!f_images || !f_labels)
    {
        std::cerr << "Failed to open training files: " << images_file << " or " << labels_file << "\n";
        if (f_images)
            fclose(f_images);
        if (f_labels)
            fclose(f_labels);
        return false;
    }

    // Verify file size
    fseek(f_images, 0, SEEK_END);
    size_t image_file_size = ftell(f_images);
    fseek(f_images, 0, SEEK_SET);

    if (image_file_size != num_samples * input_dim * sizeof(float))
    {
        std::cerr << "Invalid " << images_file << " size: " << image_file_size
                  << ", expected: " << num_samples * input_dim * sizeof(float) << "\n";
        fclose(f_images);
        fclose(f_labels);
        return false;
    }

    inputs.resize(num_samples * input_dim);
    labels.resize(num_samples * output_dim);

    if (fread(inputs.data(), sizeof(float), inputs.size(), f_images) != inputs.size() ||
        fread(labels.data(), sizeof(float), labels.size(), f_labels) != labels.size())
    {
        std::cerr << "Error reading training data.\n";
        fclose(f_images);
        fclose(f_labels);
        return false;
    }

    fclose(f_images);
    fclose(f_labels);
    return true;
}

bool load_test_data(const std::string &images_file, const std::string &labels_file,
                    std::vector<float> &inputs, std::vector<float> &labels,
                    int num_samples, int input_dim, int output_dim)
{
    return load_training_data(images_file, labels_file, inputs, labels, num_samples, input_dim, output_dim);
}

// Weight initialization
void initialize_weights(std::vector<float> &input_weights, std::vector<float> &output_weights,
                        std::vector<float> &input_weights_col, std::vector<float> &output_weights_col,
                        int input_dim, int hidden_dim, int output_dim)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);

    float std_input = sqrtf(2.0f / input_dim);
    float std_hidden = sqrtf(2.0f / hidden_dim);

    // Initialize weights in row-major format
    input_weights.resize(input_dim * hidden_dim);
    output_weights.resize(hidden_dim * output_dim);

    for (int i = 0; i < input_dim; ++i)
    {
        for (int j = 0; j < hidden_dim; ++j)
        {
            input_weights[i * hidden_dim + j] = dist(gen) * std_input;
        }
    }
    for (int i = 0; i < hidden_dim; ++i)
    {
        for (int j = 0; j < output_dim; ++j)
        {
            output_weights[i * output_dim + j] = dist(gen) * std_hidden;
        }
    }

    // Convert to column-major format for cuBLAS
    input_weights_col.resize(input_dim * hidden_dim);
    output_weights_col.resize(hidden_dim * output_dim);

    for (int i = 0; i < input_dim; ++i)
    {
        for (int j = 0; j < hidden_dim; ++j)
        {
            input_weights_col[j * input_dim + i] = input_weights[i * hidden_dim + j];
        }
    }

    for (int i = 0; i < hidden_dim; ++i)
    {
        for (int j = 0; j < output_dim; ++j)
        {
            output_weights_col[j * hidden_dim + i] = output_weights[i * output_dim + j];
        }
    }
}