#!/usr/bin/env python

import argparse
import os
from pathlib import Path
from PIL import Image

import torch
import torchvision.transforms as transforms
import torchvision.models as torchvision_models
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from queue import Queue
import threading
import vits
import resnet_torch

torchvision_model_names = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

model_names = ['vit_small', 'vit_base', 'vit_conv_small', 'vit_conv_base'] + torchvision_model_names

parser = argparse.ArgumentParser(description='Extract features from dataset folder (converts image files to feature tensors)')
parser.add_argument('checkpoint', type=str, help='path to checkpoint')
parser.add_argument('dataset_dir', type=str, help='path to dataset folder (will process all image files recursively)')
parser.add_argument('--output-dir', type=str, default=None, help='output directory for features (default: dataset_dir + "_features")')
parser.add_argument('--gpu', default=0, type=int, help='GPU id to use (default: 0)')
parser.add_argument('--batch-size', type=int, default=8, help='batch size for processing (default: 8)')
parser.add_argument('--num-workers', type=int, default=4, help='number of data loading workers (default: 4)')
parser.add_argument('--final-layer-planes', default=512, type=int,
                    help='number of planes in final ResNet layer (default: 512, standard ResNet50 uses 512)')

def chunk_image(image, chunk_size=32):
    """Chunk a 224x224 image into 32x32 patches using unfold"""
    # image shape: [3, 224, 224]
    # Unfold along height dimension: [C, H//chunk_size, W, chunk_size]
    patches_h = image.unfold(1, chunk_size, chunk_size)  # [3, 7, 224, 32]
    # Unfold along width dimension: [C, H//chunk_size, W//chunk_size, chunk_size, chunk_size]
    patches = patches_h.unfold(2, chunk_size, chunk_size)  # [3, 7, 7, 32, 32]
    # Reshape to [C, num_patches, chunk_size, chunk_size] and permute to [num_patches, C, chunk_size, chunk_size]
    num_patches_h = patches.shape[1]
    num_patches_w = patches.shape[2]
    patches = patches.contiguous().view(image.shape[0], num_patches_h * num_patches_w, chunk_size, chunk_size)
    patches = patches.permute(1, 0, 2, 3)  # [49, 3, 32, 32]
    return patches

class ImageDataset(Dataset):
    """Dataset that loads images and returns patches"""
    def __init__(self, image_files):
        self.image_files = image_files
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image)  # [3, 224, 224]
            patches = chunk_image(image_tensor)  # [49, 3, 32, 32]
            return patches, image_path
        except Exception as e:
            print(f"=> Error loading {image_path}: {e}")
            # Return None to be filtered out in collate_fn
            return None, image_path

def collate_fn(batch):
    """Collate function that stacks patches and collects paths"""
    batch_patches = []
    paths = []
    
    for patches, image_path in batch:
        if patches is not None:
            batch_patches.append(patches)
            paths.append(image_path)
    
    if len(batch_patches) == 0:
        return None, []
    
    # Stack to get (B, P, C, H, W) where B=batch_size, P=49, C=3, H=32, W=32
    batch_patches = torch.stack(batch_patches)  # [B, 49, 3, 32, 32]
    return batch_patches, paths

def main():
    args = parser.parse_args()
    
    # Set GPU
    torch.cuda.set_device(args.gpu)
    device = torch.device(f'cuda:{args.gpu}')
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=f'cuda:{args.gpu}')
    arch = checkpoint['arch']
    state_dict = checkpoint['state_dict']
    
    # Create backbone encoder (without projectors/predictors)
    print(f"=> creating backbone '{arch}'")
    if arch.startswith('vit'):
        backbone = vits.__dict__[arch]()
        # Remove head (projector will be in state dict, we ignore it)
        backbone.head = torch.nn.Identity()
        linear_keyword = 'head'
    else:
        if arch == 'resnet50':
            backbone = resnet_torch.resnet50(zero_init_residual=True, final_layer_planes=args.final_layer_planes)
        else:
            backbone = torchvision_models.__dict__[arch](zero_init_residual=True)
        # Remove fc (projector will be in state dict, we ignore it)
        backbone.fc = torch.nn.Identity()
        linear_keyword = 'fc'
    
    # Filter state dict to only include momentum_encoder backbone keys (exclude projector/predictor)
    backbone_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('momentum_encoder.') and not k.startswith(f'momentum_encoder.{linear_keyword}'):
            # Remove 'momentum_encoder.' prefix
            new_key = k[len('momentum_encoder.'):]
            backbone_state_dict[new_key] = v
    
    # Load backbone weights
    backbone.load_state_dict(backbone_state_dict, strict=True)
    backbone.eval()
    backbone = backbone.to(device)
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = args.dataset_dir + "_features"
    
    # Find all image files recursively (common image formats)
    dataset_path = Path(args.dataset_dir)
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}
    image_files = []
    for ext in image_extensions:
        image_files.extend(dataset_path.rglob(f"*{ext}"))
        image_files.extend(dataset_path.rglob(f"*{ext.upper()}"))
    
    if len(image_files) == 0:
        print(f"=> No image files found in {args.dataset_dir}")
        return
    
    print(f"=> Found {len(image_files)} image files")
    
    # Create dataset and dataloader
    dataset = ImageDataset(image_files)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    pbar = tqdm(total=len(image_files), desc="Extracting features")

    save_queue = Queue()
    def save_worker():
        while True:
            item = save_queue.get()
            if item is None:
                break
            tensor, path = item
            torch.save(tensor, path, _use_new_zipfile_serialization=True)
            save_queue.task_done()
            pbar.update(1)

    save_thread = threading.Thread(target=save_worker, daemon=True)
    save_thread.start()
    
    # Process images in batches
    with torch.no_grad():
        for batch_patches, paths in dataloader:
            if batch_patches is None or len(paths) == 0:
                continue
            
            # batch_patches shape: [B, P, C, H, W] = [B, 49, 3, 32, 32]
            B, P, C, H, W = batch_patches.shape
            
            # Reshape to combine batch and patch dimensions: [B*P, C, H, W]
            patches_flat = batch_patches.view(B * P, C, H, W).to(device)
            
            # Run encoder on all patches at once
            features_flat = backbone(patches_flat)  # [B*P, D]
            
            # Reshape back to [B, P, D]
            features = features_flat.view(B, P, -1).cpu()  # [B, P, D]
            
            # Save each image's features
            for i, image_path in enumerate(paths):
                rel_path = Path(image_path).relative_to(dataset_path)
                output_path = Path(args.output_dir) / rel_path.with_suffix('.pt')
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Save features for this image: [P, D] = [49, D]
                
                save_queue.put((features[i], output_path))
                # torch.save(features[i].cpu(), output_path, _use_new_zipfile_serialization=True)
            
        save_queue.join()
        save_queue.put(None)
        save_thread.join()
    
    pbar.close()
    
    print(f"=> Done! Features saved to {args.output_dir}")

if __name__ == '__main__':
    main()

