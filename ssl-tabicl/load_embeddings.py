import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import argparse
from typing import Tuple, Optional, List

def load_embeddings(
    embeddings_folder: str,
    num_per_class: int,
    classes: Optional[List[str]] = None,
    shuffle: bool = True,
    seed: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load embeddings from a folder and return concatenated matrix with labels.
    
    Args:
        embeddings_folder: Path to folder containing class subfolders (e.g., 'train_embeddings')
        num_per_class: Number of embeddings to load from each class. If -1, loads all embeddings for each class.
        classes: List of class names to load. If None, loads all classes found.
        shuffle: Whether to randomly shuffle embeddings before selecting
        seed: Random seed for reproducibility
    
    Returns:
        embeddings: (N, D) tensor where N = num_per_class * num_classes (or total if num_per_class=-1), D = 384
        labels: (N,) tensor with class labels (0-indexed)
    """
    if seed is not None:
        np.random.seed(seed)
    
    embeddings_path = Path(embeddings_folder)
    if not embeddings_path.exists():
        raise ValueError(f"Embeddings folder not found: {embeddings_folder}")
    
    # Get all class folders
    class_dirs = sorted([d for d in embeddings_path.iterdir() if d.is_dir()])
    
    if classes is not None:
        # Filter to only requested classes
        class_dirs = [d for d in class_dirs if d.name in classes]
        if len(class_dirs) != len(classes):
            found = [d.name for d in class_dirs]
            missing = set(classes) - set(found)
            raise ValueError(f"Classes not found: {missing}")
    
    num_classes = len(class_dirs)
    class_names = [d.name for d in class_dirs]
    
    if num_per_class == -1:
        print(f"Loading all embeddings per class from {num_classes} classes")
    else:
        print(f"Loading {num_per_class} embeddings per class from {num_classes} classes")
    print(f"Classes: {class_names}")
    
    all_embeddings = []
    all_labels = []
    
    for class_idx, class_dir in enumerate(tqdm(class_dirs, desc="Loading classes")):
        # Get all .npy files in this class folder
        embedding_files = sorted(class_dir.glob('*.npy'))
        
        if len(embedding_files) == 0:
            raise ValueError(f"No embeddings found in class folder: {class_dir}")
        
        # Shuffle if requested
        if shuffle:
            indices = np.random.permutation(len(embedding_files))
            embedding_files = [embedding_files[i] for i in indices]
        
         # Load up to num_per_class embeddings (or all if num_per_class == -1)
        if num_per_class == -1:
            num_to_load = len(embedding_files)
            selected_files = embedding_files
        else:
            num_to_load = min(num_per_class, len(embedding_files))
            selected_files = embedding_files[:num_to_load]
            
            if num_to_load < num_per_class:
                print(f"Warning: Only {num_to_load} embeddings available for class '{class_dir.name}', "
                      f"requested {num_per_class}")
        
        # Load embeddings
        class_embeddings = []
        for emb_file in tqdm(selected_files, desc=f"Loading {class_dir.name}", leave=False):
            embedding = np.load(emb_file)
            class_embeddings.append(embedding)
        
        # Stack embeddings for this class
        class_embeddings = np.stack(class_embeddings)  # Shape: (num_to_load, 384)
        
        # Create labels for this class
        class_labels = np.full(num_to_load, class_idx, dtype=np.int64)
        
        all_embeddings.append(class_embeddings)
        all_labels.append(class_labels)
    
    # Concatenate all classes
    embeddings = np.concatenate(all_embeddings, axis=0)  # Shape: (N, 384)
    labels = np.concatenate(all_labels, axis=0)  # Shape: (N,)
    
    # Convert to torch tensors
    embeddings_tensor = torch.from_numpy(embeddings).float()
    labels_tensor = torch.from_numpy(labels).long()
    
    print(f"\nLoaded embeddings shape: {embeddings_tensor.shape}")
    print(f"Labels shape: {labels_tensor.shape}")
    print(f"Total samples: {len(labels_tensor)}")
    
    return embeddings_tensor, labels_tensor


