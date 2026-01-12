"""
Example usage of load_embeddings.py
"""
from dataset import get_imagenet_train_embeddings, get_imagenet_test_embeddings
from dataset import get_satellite_train_embeddings, get_satellite_test_embeddings
import torch
import matplotlib.pyplot as plt
from tabicl import TabICLClassifier
import time

percentages = [2, 5, 10, 20, 60, 100]
accuracies = []
fit_times = []

# Load test set once (fixed for all trials)
# embeddings_test, labels_test = get_imagenet_test_embeddings(root_dir="embeddings/imagewoof2", train_folder="train", test_folder="val")

embeddings_test, labels_test = get_satellite_test_embeddings(train_percent = 80)

print(f"Test set shape: {embeddings_test.shape}")
print(f"Test set labels shape: {labels_test.shape}")
for percent in percentages:
    # embeddings_train, labels_train = get_imagenet_train_embeddings(root_dir="embeddings/imagewoof2", train_folder="train", test_folder="val", percent=percent)
    embeddings_train, labels_train = get_satellite_train_embeddings(train_percent = 80, percent = percent)
    print(f"Train set shape: {embeddings_train.shape}")
    print(f"Train set labels shape: {labels_train.shape}")
    clf = TabICLClassifier(device="cuda")
    start_time = time.time()
    clf.fit(embeddings_train, labels_train)
    outputs = clf.predict(embeddings_test)   # in-context learning happens here
    end_time = time.time()
    fit_time = end_time - start_time
    fit_times.append(fit_time)
    # Measure accuracy between outputs and labels_test (both are lists)
    correct = sum([o == t for o, t in zip(outputs, labels_test)])
    accuracy = correct / len(labels_test)
    accuracies.append(accuracy)
    print(f" Accuracy: {accuracy:.4f}  |  Fit time: {fit_time:.4f} s")

# Plotting accuracy
plt.figure(figsize=(7, 4))
plt.plot(percentages, accuracies, marker='o')
plt.xlabel('Percentage of train samples')
plt.ylabel('Test Accuracy')
plt.title('TabICL Test Accuracy vs. percentage of train samples')
plt.grid(True)
plt.tight_layout()
plt.savefig("plot.png")

# Plotting fit times vs. percentage of train samples
plt.figure(figsize=(7, 4))
plt.plot(percentages, fit_times, marker='o', color='orange')
plt.xlabel('Percentage of train samples')
plt.ylabel('Fit Time (s)')
plt.title('TabICL Fit Time vs. percentage of train samples')
plt.grid(True)
plt.tight_layout()
plt.savefig("plot_time.png")
