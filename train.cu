#include "utils.cuh"

int main()
{
    const int batch_size = 64;
    const int input_dim = 784;
    const int hidden_dim = 128;
    const int output_dim = 10;
    const float learning_rate = 0.005f; // Works well enough. Could be tuned further
    const float momentum = 0.9f;
    const float clip_norm = 1.0f;
    const int epochs = 20; // Works well enough. Could be tuned further
    const int num_samples = 60000;
    const int threads = 256;

    std::cout << "=== MNIST Neural Network Training ===\n";

    // Loading training data
    std::vector<float> h_all_inputs, h_all_labels;
    if (!load_training_data("train_images.bin", "train_labels.bin",
                            h_all_inputs, h_all_labels, num_samples, input_dim, output_dim))
    {
        return 1;
    }

    std::cout << "Training data loaded successfully.\n";

    // Loading / initializing weights
    std::vector<float> h_input_weights, h_output_weights;
    std::vector<float> h_input_weights_col, h_output_weights_col;
    std::vector<float> h_input_weights_velocity(input_dim * hidden_dim, 0.0f);
    std::vector<float> h_output_weights_velocity(hidden_dim * output_dim, 0.0f);

    bool weights_loaded = false;

    std::cout << "Checking for existing weights file 'trained_weights.bin'...\n";
    if (load_weights("trained_weights.bin", h_input_weights_col, h_output_weights_col,
                     input_dim, hidden_dim, output_dim))
    {
        weights_loaded = true;
        std::cout << "Loaded pre-trained weights. Skipping training...\n";
    }
    else
    {
        std::cout << "No existing weights found. Initializing new weights...\n";
        initialize_weights(h_input_weights, h_output_weights, h_input_weights_col, h_output_weights_col,
                           input_dim, hidden_dim, output_dim);
    }

    if (!weights_loaded)
    {
        std::cout << "\n=== Starting Training ===\n";

        // Batch Buffers for the host
        std::vector<float> h_input(batch_size * input_dim);
        std::vector<float> h_labels(batch_size * output_dim);
        std::vector<float> h_output(batch_size * hidden_dim);
        std::vector<float> h_logits(batch_size * output_dim);
        std::vector<float> h_probs(batch_size * output_dim);
        std::vector<float> h_output_grad(batch_size * hidden_dim);
        std::vector<float> h_relu_back(batch_size * hidden_dim);
        std::vector<float> h_dlogits(batch_size * output_dim);

        // Allocating device (GPU) memory
        float *d_input, *d_input_weights, *d_output, *d_output_weights;
        float *d_logits, *d_labels, *d_softmax, *d_dlogits;
        float *d_output_weights_grad, *d_output_grad, *d_relu_back, *d_input_weights_grad;
        float *d_input_weights_velocity, *d_output_weights_velocity;

        CHECK_CUDA(cudaMalloc(&d_input, batch_size * input_dim * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_input_weights, input_dim * hidden_dim * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_output, batch_size * hidden_dim * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_output_weights, hidden_dim * output_dim * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_logits, batch_size * output_dim * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_labels, batch_size * output_dim * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_softmax, batch_size * output_dim * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_dlogits, batch_size * output_dim * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_output_weights_grad, output_dim * hidden_dim * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_output_grad, batch_size * hidden_dim * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_relu_back, batch_size * hidden_dim * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_input_weights_grad, hidden_dim * input_dim * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_input_weights_velocity, hidden_dim * input_dim * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_output_weights_velocity, output_dim * hidden_dim * sizeof(float)));
        CHECK_CUDA(cudaMemset(d_input_weights_velocity, 0, hidden_dim * input_dim * sizeof(float)));
        CHECK_CUDA(cudaMemset(d_output_weights_velocity, 0, output_dim * hidden_dim * sizeof(float)));

        cublasHandle_t handle;
        CHECK_CUBLAS(cublasCreate(&handle));

        float alpha = 1.0f, beta = 0.0f;
        float neg_lr = -learning_rate;

        // Training loop
        for (int epoch = 0; epoch < epochs; ++epoch)
        {
            float total_loss = 0.0f;

            for (int i = 0; i < num_samples; i += batch_size)
            {
                int current_batch = std::min(batch_size, num_samples - i);

                // Copying batch data
                std::copy(h_all_inputs.begin() + i * input_dim,
                          h_all_inputs.begin() + (i + current_batch) * input_dim,
                          h_input.begin());
                std::copy(h_all_labels.begin() + i * output_dim,
                          h_all_labels.begin() + (i + current_batch) * output_dim,
                          h_labels.begin());

                // Uploading data to GPU
                CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), current_batch * input_dim * sizeof(float), cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaMemcpy(d_labels, h_labels.data(), current_batch * output_dim * sizeof(float), cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaMemcpy(d_input_weights, h_input_weights_col.data(), h_input_weights_col.size() * sizeof(float), cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaMemcpy(d_output_weights, h_output_weights_col.data(), h_output_weights_col.size() * sizeof(float), cudaMemcpyHostToDevice));

                // Forward pass
                CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                         hidden_dim, current_batch, input_dim,
                                         &alpha,
                                         d_input_weights, hidden_dim,
                                         d_input, input_dim,
                                         &beta,
                                         d_output, hidden_dim));

                int blocks = (current_batch * hidden_dim + threads - 1) / threads;
                relu_activation<<<blocks, threads>>>(d_output, current_batch * hidden_dim);
                CHECK_CUDA(cudaDeviceSynchronize());

                CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                         output_dim, current_batch, hidden_dim,
                                         &alpha,
                                         d_output_weights, output_dim,
                                         d_output, hidden_dim,
                                         &beta,
                                         d_logits, output_dim));

                CHECK_CUDA(cudaMemcpy(h_logits.data(), d_logits, current_batch * output_dim * sizeof(float), cudaMemcpyDeviceToHost));
                softmax(h_logits.data(), h_probs.data(), output_dim, current_batch);
                CHECK_CUDA(cudaMemcpy(d_softmax, h_probs.data(), current_batch * output_dim * sizeof(float), cudaMemcpyHostToDevice));

                // Compute loss
                float batch_loss = 0.0f;
                for (int j = 0; j < current_batch * output_dim; ++j)
                    if (h_labels[j] == 1.0f)
                        batch_loss += -logf(fmaxf(h_probs[j], 1e-8f));
                batch_loss /= current_batch;
                total_loss += batch_loss;

                // Backward pass
                int grad_blocks = (current_batch * output_dim + threads - 1) / threads;
                compute_cross_entropy_grad<<<grad_blocks, threads>>>(d_softmax, d_labels, d_dlogits, current_batch * output_dim);
                CHECK_CUDA(cudaDeviceSynchronize());

                // Compute output weights gradient
                CHECK_CUDA(cudaMemset(d_output_weights_grad, 0, output_dim * hidden_dim * sizeof(float)));
                CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                                         output_dim, hidden_dim, current_batch,
                                         &alpha,
                                         d_dlogits, output_dim,
                                         d_output, hidden_dim,
                                         &beta,
                                         d_output_weights_grad, output_dim));

                // Scale gradient by batch size
                float inv_batch_size = 1.0f / current_batch;
                CHECK_CUBLAS(cublasSscal(handle, output_dim * hidden_dim, &inv_batch_size, d_output_weights_grad, 1));

                // Gradient clipping for output weights
                float raw_norm;
                CHECK_CUBLAS(cublasSnrm2(handle, output_dim * hidden_dim, d_output_weights_grad, 1, &raw_norm));
                float norm = raw_norm;
                if (norm > clip_norm || !std::isfinite(norm))
                {
                    float scale = clip_norm / (norm + 1e-8f);
                    CHECK_CUBLAS(cublasSscal(handle, output_dim * hidden_dim, &scale, d_output_weights_grad, 1));
                }

                // Update output weights with momentum
                CHECK_CUBLAS(cublasSscal(handle, output_dim * hidden_dim, &momentum, d_output_weights_velocity, 1));
                CHECK_CUBLAS(cublasSaxpy(handle, output_dim * hidden_dim, &neg_lr, d_output_weights_grad, 1, d_output_weights_velocity, 1));
                CHECK_CUBLAS(cublasSaxpy(handle, output_dim * hidden_dim, &alpha, d_output_weights_velocity, 1, d_output_weights, 1));

                // Compute hidden layer gradient
                CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                         hidden_dim, current_batch, output_dim,
                                         &alpha,
                                         d_output_weights, output_dim,
                                         d_dlogits, output_dim,
                                         &beta,
                                         d_output_grad, hidden_dim));

                int hidden_blocks = (current_batch * hidden_dim + threads - 1) / threads;
                relu_backward<<<hidden_blocks, threads>>>(d_output, d_output_grad, d_relu_back, current_batch * hidden_dim);
                CHECK_CUDA(cudaDeviceSynchronize());

                // Compute input weights gradient
                CHECK_CUDA(cudaMemset(d_input_weights_grad, 0, hidden_dim * input_dim * sizeof(float)));
                CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                                         hidden_dim, input_dim, current_batch,
                                         &alpha,
                                         d_relu_back, hidden_dim,
                                         d_input, input_dim,
                                         &beta,
                                         d_input_weights_grad, hidden_dim));

                // Scale gradient by batch size
                CHECK_CUBLAS(cublasSscal(handle, hidden_dim * input_dim, &inv_batch_size, d_input_weights_grad, 1));

                // Gradient clipping for input weights
                CHECK_CUBLAS(cublasSnrm2(handle, hidden_dim * input_dim, d_input_weights_grad, 1, &raw_norm));
                norm = raw_norm;
                if (norm > clip_norm || !std::isfinite(norm))
                {
                    float scale = clip_norm / (norm + 1e-8f);
                    CHECK_CUBLAS(cublasSscal(handle, hidden_dim * input_dim, &scale, d_input_weights_grad, 1));
                }

                // Update input weights with momentum
                CHECK_CUBLAS(cublasSscal(handle, hidden_dim * input_dim, &momentum, d_input_weights_velocity, 1));
                CHECK_CUBLAS(cublasSaxpy(handle, hidden_dim * input_dim, &neg_lr, d_input_weights_grad, 1, d_input_weights_velocity, 1));
                CHECK_CUBLAS(cublasSaxpy(handle, hidden_dim * input_dim, &alpha, d_input_weights_velocity, 1, d_input_weights, 1));

                // Copy updated weights back
                CHECK_CUDA(cudaMemcpy(h_input_weights_col.data(), d_input_weights,
                                      h_input_weights_col.size() * sizeof(float), cudaMemcpyDeviceToHost));
                CHECK_CUDA(cudaMemcpy(h_output_weights_col.data(), d_output_weights,
                                      h_output_weights_col.size() * sizeof(float), cudaMemcpyDeviceToHost));
            }

            std::cout << "Epoch " << (epoch + 1) << "/" << epochs
                      << " - Training Loss: " << (total_loss / (num_samples / batch_size)) << "\n";
        }

        // Save weights after training
        std::cout << "\n=== Saving Trained Weights ===\n";
        if (save_weights("trained_weights.bin", h_input_weights_col, h_output_weights_col,
                         input_dim, hidden_dim, output_dim))
        {
            std::cout << "Model weights saved to 'trained_weights.bin'\n";
        }

        // Cleanup
        cudaFree(d_input);
        cudaFree(d_input_weights);
        cudaFree(d_output);
        cudaFree(d_output_weights);
        cudaFree(d_logits);
        cudaFree(d_labels);
        cudaFree(d_softmax);
        cudaFree(d_dlogits);
        cudaFree(d_output_weights_grad);
        cudaFree(d_output_grad);
        cudaFree(d_relu_back);
        cudaFree(d_input_weights_grad);
        cudaFree(d_input_weights_velocity);
        cudaFree(d_output_weights_velocity);
        cublasDestroy(handle);
    }

    std::cout << "\nTraining completed successfully!\n";
    return 0;
}