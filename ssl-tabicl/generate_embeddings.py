from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
import argparse
import sys

# Load model once
print("Loading DINOv2 model...")
model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
model.eval()  # Set to evaluation mode

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"Using device: {device}")

# Set up augmentation pipeline (matching imagenet_susbset.py) + ImageNet norm
res = 224
crop_res = 224
crop_mode = "center"  # Can be "center" or "random"
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = transforms.Compose(
    [
        transforms.Resize(res),
        (
            transforms.CenterCrop(crop_res)
            if crop_mode == "center"
            else transforms.RandomCrop(crop_res)
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
)

def process_image(image_path, transform, model, device):
    """Process a single image and return its embedding."""
    try:
        image = Image.open(image_path).convert('RGB')
        # Apply transforms
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        image_tensor = image_tensor.to(device)
        
        with torch.no_grad():
            # DINOv2 models from torch.hub return CLS token directly as [batch, dim]
            outputs = model(image_tensor)
            # Convert to numpy and squeeze batch dimension: [1, 768] -> [768]
            embedding = outputs.cpu().numpy().squeeze(0)
        return embedding
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def process_folder(input_folder, output_base_folder, transform, model, device):
    """Recursively process all images in a folder and save embeddings with identical structure.
    
    For example, if input_folder is 'foldera/folderb', it will create 'embeddings/folderb/'
    with the same structure as folderb (not including foldera in the output path).
    """
    input_path = Path(input_folder).resolve()  # Resolve to absolute path
    root_folder_name = input_path.name  # Extract just the folder name (e.g., 'folderb' from 'foldera/folderb')
    output_base = Path(output_base_folder)
    output_path = output_base / root_folder_name  # e.g., 'embeddings/folderb'
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # Recursively find all image files
    image_files = []
    for img_file in sorted(input_path.rglob('*')):
        if img_file.is_file() and img_file.suffix.lower() in image_extensions:
            # Get relative path from input folder
            relative_path = img_file.relative_to(input_path)
            # Create corresponding output path with .npy extension
            output_file = output_path / relative_path.with_suffix('.npy')
            # Create parent directories
            output_file.parent.mkdir(parents=True, exist_ok=True)
            image_files.append((img_file, output_file))
    
    # Process all images with progress bar
    for img_path, output_path_npy in tqdm(image_files, desc=f"Processing {root_folder_name}"):
        embedding = process_image(img_path, transform, model, device)
        if embedding is not None:
            np.save(output_path_npy, embedding)

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Generate embeddings for images in a folder structure')
parser.add_argument('input_folder', type=str, help='Path to the folder containing images to process')
parser.add_argument('--output', '-o', type=str, default='./embeddings', 
                    help='Base output directory for embeddings (default: ./embeddings)')
args = parser.parse_args()

# Validate input folder
input_path = Path(args.input_folder).resolve()
if not input_path.exists():
    print(f"Error: Input folder '{input_path}' does not exist.")
    sys.exit(1)
if not input_path.is_dir():
    print(f"Error: '{input_path}' is not a directory.")
    sys.exit(1)

# Set up output directory
output_base = Path(args.output)
output_base.mkdir(parents=True, exist_ok=True)

# Process the folder
print(f"\nProcessing folder: {input_path}")
process_folder(
    input_path,
    output_base,
    transform,
    model,
    device
)

print(f"\nDone! Embeddings saved in {output_base / input_path.name}/ with identical folder structure.")

