"""
Example usage of load_embeddings.py
"""
from dataset import get_imagenet_test_embeddings
import torch
import matplotlib.pyplot as plt
from tabicl import TabICLClassifier
import time
import numpy as np

# Load test set once (fixed for all trials)
embeddings_test, labels_test = get_imagenet_test_embeddings(root_dir="embeddings/imagenette2", train_folder="train", test_folder="val")
print(f"Test set shape: {embeddings_test.shape}")
print(f"Test set labels shape: {labels_test.shape}")

icl = np.load("data.npy", allow_pickle=True).item()
embeddings_train = icl["image"]
labels_train = icl["labels"]

clf = TabICLClassifier(device="cuda")
start_time = time.time()
clf.fit(embeddings_train, labels_train)
outputs = clf.predict(embeddings_test)   # in-context learning happens here
end_time = time.time()
fit_time = end_time - start_time
# Measure accuracy between outputs and labels_test (both are lists)
correct = sum([o == t for o, t in zip(outputs, labels_test)])
accuracy = correct / len(labels_test)
print(f" Accuracy: {accuracy:.4f}  |  Fit time: {fit_time:.4f} s")
