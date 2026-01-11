#!/usr/bin/env python

import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from queue import Queue
import threading

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main_pretrain_singlegpu_embeddings import MaskedAutoencoderViT_NoPatchEmbed


class TensorFileDataset(Dataset):
    """Dataset that loads all .pt tensor files recursively from a directory"""
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.files = []
        
        # Find all .pt files recursively
        for pt_file in self.data_path.rglob("*.pt"):
            self.files.append(pt_file)
        
        self.files = sorted(self.files)
        print(f"Found {len(self.files)} tensor files")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_path = self.files[idx]
        tensor = torch.load(file_path, map_location='cpu', weights_only=False)
        
        # Ensure shape is (patches, dim) - if batch dimension exists, take first
        if len(tensor.shape) == 3:  # (B, patches, dim)
            tensor = tensor[0]
        elif len(tensor.shape) != 2:
            raise ValueError(f"Expected 2D (patches, dim) or 3D (B, patches, dim), got {tensor.shape} at {file_path}")
        
        return tensor.float(), file_path


def collate_fn(batch):
    """Collate function that stacks tensors and collects paths"""
    batch_tensors = []
    paths = []
    
    for tensor, file_path in batch:
        if tensor is not None:
            batch_tensors.append(tensor)
            paths.append(file_path)
    
    if len(batch_tensors) == 0:
        return None, []
    
    # Stack to get (B, patches, dim)
    batch_tensors = torch.stack(batch_tensors)
    return batch_tensors, paths


def extract_encoder_features(model, embeddings, device):
    """
    Extract encoder features from embeddings using MAE encoder.
    Returns only the cls token (after transformer blocks).
    """
    model.eval()
    with torch.no_grad():
        # embeddings shape: (B, num_patches, embed_dim)
        embeddings = embeddings.to(device)
        
        # Use forward_encoder with mask_ratio=0 to get all features (no masking)
        x, _, _ = model.forward_encoder(embeddings, mask_ratio=0.0)
        # x shape: (B, num_patches+1, embed_dim) - includes cls token at position 0
        
        # Return only cls token (position 0)
        return x[:, 0, :].cpu()  # (B, embed_dim)


def main():
    parser = argparse.ArgumentParser(description='Extract features from tensor files using trained MAE model')
    parser.add_argument('checkpoint', type=str, help='path to MAE checkpoint')
    parser.add_argument('dataset_dir', type=str, help='path to dataset folder (will process all .pt files recursively)')
    parser.add_argument('--output-dir', type=str, required=True, help='output directory (will clone folder structure)')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use (default: 0)')
    parser.add_argument('--batch-size', type=int, default=8, help='batch size for processing (default: 8)')
    parser.add_argument('--num-workers', type=int, default=4, help='number of data loading workers (default: 4)')
    parser.add_argument('--embed-dim', type=int, default=768, help='embedding dimension (default: 768)')
    parser.add_argument('--num-patches', type=int, default=196, help='number of patches (default: 196)')
    
    args = parser.parse_args()
    
    # Set GPU
    torch.cuda.set_device(args.gpu)
    device = torch.device(f'cuda:{args.gpu}')
    
    # Load checkpoint
    print(f"=> Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=f'cuda:{args.gpu}', weights_only=False)
    
    # Extract model state dict
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'model_without_ddp' in checkpoint:
        state_dict = checkpoint['model_without_ddp']
    else:
        state_dict = checkpoint
    
    # Create model
    print(f"=> Creating model with embed_dim={args.embed_dim}, num_patches={args.num_patches}")
    model = MaskedAutoencoderViT_NoPatchEmbed(
        embed_dim=args.embed_dim,
        num_patches=args.num_patches
    )
    
    # Load weights
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model = model.to(device)
    print("=> Model loaded successfully")
    
    # Create dataset
    dataset_path = Path(args.dataset_dir)
    output_path = Path(args.output_dir)
    
    print(f"=> Loading tensor files from {args.dataset_dir}")
    dataset = TensorFileDataset(args.dataset_dir)
    print(f"=> Found {len(dataset)} tensor files")
    
    if len(dataset) == 0:
        print(f"=> No tensor files found in {args.dataset_dir}")
        return
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # Setup progress bar and save queue
    pbar = tqdm(total=len(dataset), desc="Processing tensors")
    
    save_queue = Queue()
    def save_worker():
        while True:
            item = save_queue.get()
            if item is None:
                break
            tensor, path = item
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(tensor, path, _use_new_zipfile_serialization=True)
            save_queue.task_done()
            pbar.update(1)
    
    save_thread = threading.Thread(target=save_worker, daemon=True)
    save_thread.start()
    
    # Process batches
    print("=> Processing tensors...")
    with torch.no_grad():
        for embeddings, file_paths in dataloader:
            # Extract encoder features (only cls token)
            cls_tokens = extract_encoder_features(model, embeddings, device)
            # cls_tokens shape: (B, embed_dim) - only cls token
            
            # Save each sample's cls token preserving directory structure
            for i, input_file_path in enumerate(file_paths):
                # Get relative path from dataset directory
                rel_path = input_file_path.relative_to(dataset_path)
                
                # Create output path preserving directory structure
                output_file_path = output_path / rel_path
                
                # Save only cls token (embed_dim,)
                save_queue.put((cls_tokens[i], output_file_path))
    
    # Wait for all saves to complete
    save_queue.join()
    save_queue.put(None)
    save_thread.join()
    
    pbar.close()
    print(f"=> Done! Processed tensors saved to {args.output_dir}")
    print(f"=> Output feature shape: ({args.embed_dim},) - cls token only")


if __name__ == '__main__':
    main()
