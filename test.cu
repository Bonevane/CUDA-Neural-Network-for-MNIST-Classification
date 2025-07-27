#include "utils.cuh"

int main()
{
    const int batch_size = 64;
    const int input_dim = 784;
    const int hidden_dim = 128;
    const int output_dim = 10;
    const int threads = 256;
    const int num_test_samples = 10000; // This is the size of the MNIST test set

    std::cout << "=== MNIST Neural Network Testing (Real Test Data) ===\n";

    // Load trained weights
    std::vector<float> h_input_weights_col, h_output_weights_col;

    if (!load_weights("trained_weights.bin", h_input_weights_col, h_output_weights_col,
                      input_dim, hidden_dim, output_dim))
    {
        std::cerr << "Failed to load weights. Run training first!\n";
        return 1;
    }

    // Load real test data
    std::vector<float> h_test_inputs, h_test_labels;
    if (!load_test_data("test_images.bin", "test_labels.bin",
                        h_test_inputs, h_test_labels, num_test_samples, input_dim, output_dim))
    {
        std::cerr << "Failed to load test data. Run 'python create_mnist_data.py' first!\n";
        return 1;
    }

    std::cout << "Real test data loaded successfully (" << num_test_samples << " samples).\n";

    // Initialize cuBLAS
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // Evaluate on full test set
    std::cout << "\n=== Evaluating on Real Test Data ===\n";
    float test_accuracy = evaluate_model(handle, h_input_weights_col, h_output_weights_col,
                                         h_test_inputs, h_test_labels,
                                         num_test_samples, batch_size, input_dim,
                                         hidden_dim, output_dim, threads);

    // Test individual image predictions
    std::cout << "\n=== Testing Individual Images ===\n";
    for (int i = 0; i < std::min(10, num_test_samples); ++i)
    {
        std::vector<float> single_image(h_test_inputs.begin() + i * input_dim,
                                        h_test_inputs.begin() + (i + 1) * input_dim);

        // Find true label
        int true_label = 0;
        for (int j = 0; j < output_dim; ++j)
        {
            if (h_test_labels[i * output_dim + j] == 1.0f)
            {
                true_label = j;
                break;
            }
        }

        std::cout << "Test image " << i << " (true label: " << true_label << "): ";
        predict_single_image(handle, h_input_weights_col, h_output_weights_col,
                             single_image, input_dim, hidden_dim, output_dim, threads);
    }

    // Also test on a small subset of training data for comparison
    std::cout << "\n=== Comparing with Training Data Subset ===\n";
    std::vector<float> h_train_inputs, h_train_labels;
    if (load_training_data("train_images.bin", "train_labels.bin",
                           h_train_inputs, h_train_labels, 60000, input_dim, output_dim))
    {
        // Test on first 1000 training samples
        std::vector<float> train_subset_inputs(h_train_inputs.begin(),
                                               h_train_inputs.begin() + 1000 * input_dim);
        std::vector<float> train_subset_labels(h_train_labels.begin(),
                                               h_train_labels.begin() + 1000 * output_dim);

        float train_accuracy = evaluate_model(handle, h_input_weights_col, h_output_weights_col,
                                              train_subset_inputs, train_subset_labels,
                                              1000, batch_size, input_dim,
                                              hidden_dim, output_dim, threads);

        std::cout << "\n=== Summary ===\n";
        std::cout << "Test Accuracy (unseen data): " << test_accuracy * 100.0f << "%\n";
        std::cout << "Train Accuracy (subset): " << train_accuracy * 100.0f << "%\n";

        float overfitting = train_accuracy - test_accuracy;
        if (overfitting > 0.05f)
        {
            std::cout << "Warning: Possible overfitting detected (gap: " << overfitting * 100.0f << "%)\n";
        }
        else
        {
            std::cout << "Good generalization (gap: " << overfitting * 100.0f << "%)\n";
        }
    }

    cublasDestroy(handle);
    return 0;
}