# CUDA Neural Network for MNIST Classification

A high-performance neural network implementation using CUDA and cuBLAS for MNIST digit classification. This project demonstrates GPU-accelerated deep learning and achieves 97.5% accuracy on the MNIST test set.

## Architecture

- **Input Layer**: 784 neurons (28×28 flattened MNIST images)
- **Hidden Layer**: 128 neurons with ReLU activation
- **Output Layer**: 10 neurons with Softmax activation (digit classes 0-9)
- **Optimizer**: SGD with momentum
- **Loss Function**: Cross-entropy loss

## Project Structure

```
├── utils.cuh              # Header file with utility function declarations
├── utils.cu               # Utility functions implementation
├── train.cu               # Training code
├── test.cu                # Testing code
├── create_mnist_data.py   # Python script to generate binary data files from MNIST
├── Makefile              # Build config
└── README.md             # Read this.
```

## 🔧 Dependencies

### System Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit (tested with CUDA 11.0+)
- cuBLAS library
- GCC/G++ compiler
- Python 3.x (for data preparation)

Note: This project was created in Coursera's Lab Environment.

### Python Dependencies

```bash
pip install torch torchvision numpy
```

## Getting Started

### 1. Prepare the MNIST Dataset

```bash
python create_mnist_dataset.py
```

This creates:

- train_images.bin (60,000 × 784 float32)
- train_labels.bin (60,000 × 10 float32)
- test_images.bin (10,000 × 784 float32)
- test_labels.bin (10,000 × 10 float32)

### 2. Build the Project

```bash
make clean build
```

This compiles:

- `utils.o` - Utility functions object file
- `train.exe` - Training executable
- `test.exe` - Testing executable

### 3. Train the Model

```bash
./train.exe
```

### 4. Test the Model

```bash
./test.exe
```

## Model Features

### CUDA Kernels

- **ReLU Activation**: GPU-accelerated forward and backward pass
- **Cross-entropy Gradient**: Parallel gradient computation

### Training Features

- **Batch Processing**: Configurable batch size (default: 64)
- **Momentum**: SGD with momentum (default: 0.9)
- **Gradient Clipping**: Prevents gradient explosion (norm: 1.0)
- **Weight Persistence**: Automatic save/load of trained weights
- **He Initialization**: Proper weight initialization for ReLU networks

### Performance Optimizations

- **cuBLAS Integration**: Optimized matrix operations
- **Memory Management**: Efficient GPU memory allocation
- **Column-major Storage**: cuBLAS-compatible weight layout

## Configuration

Key parameters in `train.cu`:

```cpp
const int batch_size = 64;        // Training batch size
const int hidden_dim = 128;       // Hidden layer neurons
const float learning_rate = 0.005f; // Learning rate
const float momentum = 0.9f;      // Momentum coefficient
const int epochs = 20;             // Training epochs
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. Feel free to modify and extend for your own learning and research.

## Contributing

Suggestions for improvements:

- Add more activation functions
- Implement different optimizers (Adam, RMSprop)
- Add regularization techniques (dropout, weight decay)
- Support for different architectures
- Visualization tools for training progress
