#!/usr/bin/env python

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class TensorFolderDataset(Dataset):
    """Dataset that loads tensors, structured like ImageFolder.
    Folders are labels, files inside folders are samples (tensors loaded via torch.load).
    Can be reused for both train and val splits.
    """
    def __init__(self, data_path, classes=None, class_to_idx=None):
        self.data_path = Path(data_path)
        self.samples = []
        
        # Find all class folders
        if not self.data_path.is_dir():
            raise ValueError(f"Data path {data_path} is not a directory")
        
        # If classes are provided (from another split), use them
        if classes is not None and class_to_idx is not None:
            self.classes = classes
            self.class_to_idx = class_to_idx
        else:
            # Otherwise, discover classes from directory structure
            self.classes = []
            self.class_to_idx = {}
        
        # Get all subdirectories (classes)
        for class_name in sorted(os.listdir(self.data_path)):
            class_path = self.data_path / class_name
            if class_path.is_dir():
                # Use existing class_idx if classes were provided, otherwise create new
                if class_name in self.class_to_idx:
                    class_idx = self.class_to_idx[class_name]
                else:
                    class_idx = len(self.classes)
                    self.classes.append(class_name)
                    self.class_to_idx[class_name] = class_idx
                
                # Get all tensor files in this class folder
                for file_name in sorted(os.listdir(class_path)):
                    if file_name.endswith('.pt'):
                        file_path = class_path / file_name
                        self.samples.append((file_path, class_idx))
        
        print(f"Found {len(self.classes)} classes with {len(self.samples)} total samples")
        if len(self.samples) > 0:
            print(f"Classes: {self.classes}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        tensor = torch.load(file_path, map_location='cpu', weights_only=False)
        
        # Ensure tensor is 1D (embedding vector)
        if len(tensor.shape) > 1:
            tensor = tensor.flatten()
        
        return tensor.float(), label


class MLPProbe(nn.Module):
    """3-layer MLP for linear probing"""
    def __init__(self, input_dim=512, hidden_dim=512, num_classes=10):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        return self.mlp(x)


def evaluate(model, dataloader, device):
    """Evaluate model on validation set"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for embeddings, labels in dataloader:
            embeddings = embeddings.to(device)
            labels = labels.to(device)
            
            outputs = model(embeddings)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy


def main():
    parser = argparse.ArgumentParser(description='Linear probing with 3-layer MLP')
    parser.add_argument('--data-path', type=str, default='/home/aditmeh/mae/linprobe_imagenette_tensors',
                        help='path to tensor dataset folder')
    parser.add_argument('--batch-size', type=int, default=64, help='batch size')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--hidden-dim', type=int, default=512, help='hidden dimension for MLP')
    parser.add_argument('--input-dim', type=int, default=512, help='input embedding dimension')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use (default: 0)')
    parser.add_argument('--num-workers', type=int, default=4, help='number of data loading workers')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load datasets - first train to discover classes, then val uses same class mapping
    train_path = Path(args.data_path) / 'train'
    val_path = Path(args.data_path) / 'val'
    
    print(f"=> Loading training data from {train_path}")
    dataset_train = TensorFolderDataset(train_path)
    
    print(f"=> Loading validation data from {val_path}")
    # Reuse classes and class_to_idx from train dataset
    dataset_val = TensorFolderDataset(val_path, classes=dataset_train.classes, class_to_idx=dataset_train.class_to_idx)
    
    num_classes = len(dataset_train.classes)
    print(f"=> Number of classes: {num_classes}")
    
    # Check input dimension from first sample
    sample_tensor, _ = dataset_train[0]
    actual_input_dim = sample_tensor.shape[0]
    if args.input_dim != actual_input_dim:
        print(f"Warning: specified input_dim={args.input_dim} but actual tensor dim={actual_input_dim}")
        print(f"Using actual dimension: {actual_input_dim}")
        args.input_dim = actual_input_dim
    
    # Create dataloaders
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model
    model = MLPProbe(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_classes=num_classes
    )
    model = model.to(device)
    print(f"=> Model: {model}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    print(f"=> Starting training for {args.epochs} epochs")
    print(f"=> Batch size: {args.batch_size}, LR: {args.lr}")
    
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(dataloader_train, desc=f'Epoch {epoch+1}/{args.epochs}')
        for embeddings, labels in pbar:
            embeddings = embeddings.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * train_correct / train_total:.2f}%'
            })
        
        train_acc = 100 * train_correct / train_total
        avg_loss = train_loss / len(dataloader_train)
        
        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%")
        
        # Validation every 5 epochs
        if (epoch + 1) % 5 == 0:
            val_acc = evaluate(model, dataloader_val, device)
            print(f"Epoch {epoch+1}/{args.epochs} - Val Acc: {val_acc:.2f}%")
            
            if val_acc > best_acc:
                best_acc = val_acc
                print(f"New best validation accuracy: {best_acc:.2f}%")
    
    # Final evaluation
    print("\n=> Final evaluation on validation set")
    final_val_acc = evaluate(model, dataloader_val, device)
    print(f"Final validation accuracy: {final_val_acc:.2f}%")
    print(f"Best validation accuracy: {best_acc:.2f}%")


if __name__ == '__main__':
    main()

