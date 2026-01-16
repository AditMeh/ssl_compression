import os
import numpy as np
import torch


def get_imagenet_train_embeddings(percent=100, root_dir="embeddings/imagenet", train_folder="train", test_folder="test", seed=42):
    """
    Returns a tuple (embeddings, labels) for ImageNet train split, using a percentage of all embeddings after shuffling.
    
    Args:
        percent (float): Percentage of all embeddings to load (1-100), shuffled together (no class-wise stratification)
        root_dir (str): Path to embeddings root directory (e.g., "embeddings/imagenet")
        train_folder (str): Name of the folder containing training data (default: "train")
        test_folder (str): Name of the folder containing test data (default: "test")
        seed (int): Random seed for shuffling (default: 42)
        
    Returns:
        embeddings_tensor: shape (N, D), dtype=torch.float32
        labels_tensor: shape (N, ), dtype=torch.long
        
    Example:
        If your structure is embeddings/imagenet/training/class1/... and embeddings/imagenet/validation/class1/...,
        use train_folder="training" and test_folder="validation"
    """
    split_dir = os.path.join(root_dir, train_folder)
    class_names = sorted([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])

    # Load all embeddings from all classes
    all_embeddings = []
    all_labels = []
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(split_dir, class_name)
        emb_files = sorted([f for f in os.listdir(class_dir) if f.endswith('.npy')])
        
        for f in emb_files:
            emb_path = os.path.join(class_dir, f)
            emb = np.load(emb_path)
            all_embeddings.append(emb)
            all_labels.append(class_idx)
    
    # Convert to numpy arrays
    all_embeddings = np.stack(all_embeddings)  # (N, D)
    all_labels = np.array(all_labels)          # (N,)
    
    # Shuffle with fixed seed
    rng = np.random.RandomState(seed)
    indices = np.arange(len(all_embeddings))
    rng.shuffle(indices)
    all_embeddings = all_embeddings[indices]
    all_labels = all_labels[indices]
    
    # Take percentage of shuffled data
    n_total = len(all_embeddings)
    n_use = int(np.ceil((percent / 100.0) * n_total))
    embeddings = all_embeddings[:n_use]
    labels = all_labels[:n_use]
    
    embeddings_tensor = torch.from_numpy(embeddings).float()
    labels_tensor = torch.from_numpy(labels).long()
    return embeddings_tensor, labels_tensor


def get_imagenet_test_embeddings(root_dir="embeddings/imagenet", train_folder="train", test_folder="test"):
    """
    Returns (embeddings, labels) stacked for the full test set (all embeddings).
    
    Args:
        root_dir (str): Path to embeddings root directory (e.g., "embeddings/imagenet")
        train_folder (str): Name of the folder containing training data (default: "train")
        test_folder (str): Name of the folder containing test data (default: "test")
        
    Returns:
        embeddings_tensor: shape (N, D), dtype=torch.float32
        labels_tensor: shape (N, ), dtype=torch.long
    """
    split_dir = os.path.join(root_dir, test_folder)
    class_names = sorted([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])
    
    embeddings = []
    labels = []
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(split_dir, class_name)
        emb_files = sorted([f for f in os.listdir(class_dir) if f.endswith('.npy')])
        
        for f in emb_files:
            emb_path = os.path.join(class_dir, f)
            emb = np.load(emb_path)
            embeddings.append(emb)
            labels.append(class_idx)
    
    embeddings = np.stack(embeddings)  # (N, D)
    labels = np.array(labels)          # (N,)
    
    embeddings_tensor = torch.from_numpy(embeddings).float()
    labels_tensor = torch.from_numpy(labels).long()
    return embeddings_tensor, labels_tensor
