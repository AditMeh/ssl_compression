import numpy as np
from pathlib import Path
import argparse
import random

def load_embeddings(embeddings_dir):
    """Load all .npy files grouped by class folder."""
    embeddings_dir = Path(embeddings_dir)
    class_embeddings = {}
    
    # Get all class folders (assuming train/val structure)
    train_dir = embeddings_dir / "train"
    if not train_dir.exists():
        train_dir = embeddings_dir  # If no train/val split, use root
    
    for class_folder in sorted(train_dir.iterdir()):
        if class_folder.is_dir():
            embeddings = []
            for npy_file in sorted(class_folder.glob("*.npy")):
                emb = np.load(npy_file)
                embeddings.append(emb)
            if embeddings:
                class_embeddings[class_folder.name] = np.array(embeddings)
    
    return class_embeddings

def compute_centroid_representatives(class_embeddings):
    """Compute class representatives as the sample closest to the mean."""
    representatives = []
    labels = []
    class_names = sorted(class_embeddings.keys())
    
    for class_idx, class_name in enumerate(class_names):
        embeddings = class_embeddings[class_name]  # (N, D)
        mean_emb = embeddings.mean(axis=0)  # (D,)
        
        # Find the sample closest to the mean (L2 distance)
        distances = np.linalg.norm(embeddings - mean_emb, axis=1)  # (N,)
        closest_idx = np.argmin(distances)
        rep = embeddings[closest_idx]  # (D,)
        
        representatives.append(rep)
        labels.append(class_idx)
    
    return np.array(representatives), np.array(labels)

def compute_random_representatives(class_embeddings):
    """Compute class representatives by randomly sampling one per class."""
    representatives = []
    labels = []
    class_names = sorted(class_embeddings.keys())
    
    for class_idx, class_name in enumerate(class_names):
        embeddings = class_embeddings[class_name]  # (N, D)
        rep = random.choice(embeddings)  # (D,)
        
        representatives.append(rep)
        labels.append(class_idx)
    
    return np.array(representatives), np.array(labels)

def compute_shuffled_percent(class_embeddings, percent=100, seed=42):
    """Compute ICL samples as a random percentage of the shuffled train set."""
    # Load all embeddings from all classes
    all_embeddings = []
    all_labels = []
    class_names = sorted(class_embeddings.keys())
    
    for class_idx, class_name in enumerate(class_names):
        embeddings = class_embeddings[class_name]  # (N, D)
        for emb in embeddings:
            all_embeddings.append(emb)
            all_labels.append(class_idx)
    
    # Convert to numpy arrays
    all_embeddings = np.stack(all_embeddings)  # (N, D)
    all_labels = np.array(all_labels)  # (N,)
    
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
    
    return embeddings, labels

def main():
    parser = argparse.ArgumentParser(description="Create baseline class representatives from embeddings")
    parser.add_argument("embeddings_dir", type=str, help="Path to embeddings directory")
    parser.add_argument("--output", "-o", type=str, default="baselines.npy", help="Output .npy file")
    parser.add_argument("--method", type=str, choices=["centroid", "random", "shuffled_percent"], default="centroid",
                       help="Method to compute representatives: centroid, random, or shuffled_percent")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for random sampling")
    parser.add_argument("--percent", type=float, default=100.0, 
                       help="Percentage of shuffled train set to use (for shuffled_percent method, default: 100.0)")
    args = parser.parse_args()
    
    if args.method in ["random", "shuffled_percent"]:
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    # Load data
    class_embeddings = load_embeddings(args.embeddings_dir)
    print(f"Loaded {len(class_embeddings)} classes")
    
    # Compute representatives
    if args.method == "centroid":
        representatives, labels = compute_centroid_representatives(class_embeddings)
    elif args.method == "random":
        representatives, labels = compute_random_representatives(class_embeddings)
    elif args.method == "shuffled_percent":
        representatives, labels = compute_shuffled_percent(class_embeddings, percent=args.percent, seed=args.seed)
    else:
        raise ValueError(f"Unknown method: {args.method}")
    
    print(f"Computed representatives: shape {representatives.shape}")
    
    # Save
    output_dict = {"image": representatives, "labels": labels}
    np.save(args.output, output_dict)
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()

