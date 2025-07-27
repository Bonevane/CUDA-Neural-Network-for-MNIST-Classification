import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

def save_mnist_data():
    # Load MNIST training set
    mnist_train = datasets.MNIST(
        root='mnist_data',
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )

    # Load MNIST test set
    mnist_test = datasets.MNIST(
        root='mnist_data',
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )

    # Process training data
    train_images = mnist_train.data.numpy().reshape(-1, 28 * 28).astype(np.float32) / 255.0  # (60000, 784)
    train_labels_tensor = torch.tensor(mnist_train.targets)
    train_labels = F.one_hot(train_labels_tensor, num_classes=10).numpy().astype(np.float32)  # (60000, 10)

    # Process test data
    test_images = mnist_test.data.numpy().reshape(-1, 28 * 28).astype(np.float32) / 255.0  # (10000, 784)
    test_labels_tensor = torch.tensor(mnist_test.targets)
    test_labels = F.one_hot(test_labels_tensor, num_classes=10).numpy().astype(np.float32)  # (10000, 10)

    # Save training data
    print("Saving train_images.bin:", train_images.shape, train_images.dtype)
    train_images.tofile("train_images.bin")

    print("Saving train_labels.bin:", train_labels.shape, train_labels.dtype)
    train_labels.tofile("train_labels.bin")

    # Save test data
    print("Saving test_images.bin:", test_images.shape, test_images.dtype)
    test_images.tofile("test_images.bin")

    print("Saving test_labels.bin:", test_labels.shape, test_labels.dtype)
    test_labels.tofile("test_labels.bin")

    # Print some statistics
    print(f"\nData statistics:")
    print(f"Training set: {train_images.shape[0]} samples")
    print(f"Test set: {test_images.shape[0]} samples")
    print(f"Image dimensions: {train_images.shape[1]} pixels (28x28 flattened)")
    print(f"Number of classes: {train_labels.shape[1]}")
    
    print(f"\nPixel value ranges:")
    print(f"Train images: [{train_images.min():.3f}, {train_images.max():.3f}]")
    print(f"Test images: [{test_images.min():.3f}, {test_images.max():.3f}]")
    
    print(f"\nSample labels (first 10 training samples):")
    for i in range(10):
        label_idx = np.argmax(train_labels[i])
        print(f"Sample {i}: class {label_idx}")

if __name__ == "__main__":
    save_mnist_data()