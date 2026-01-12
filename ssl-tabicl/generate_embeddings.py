from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
import argparse
import sys

# Load model and processor once
print("Loading DINOv2 model and processor...")
processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small', use_fast=True)
model = AutoModel.from_pretrained('facebook/dinov2-small')
model.eval()  # Set to evaluation mode

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"Using device: {device}")

def process_image(image_path, processor, model, device):
    """Process a single image and return its embedding."""
    try:
        image = Image.open(image_path).convert('RGB')
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            last_hidden_states = outputs.last_hidden_state[..., 0, :]
            # Convert to numpy and squeeze batch dimension: [1, 384] -> [384]
            embedding = last_hidden_states.cpu().numpy().squeeze(0)
        return embedding
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def process_folder(input_folder, output_base_folder, processor, model, device):
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
        embedding = process_image(img_path, processor, model, device)
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
    processor,
    model,
    device
)

print(f"\nDone! Embeddings saved in {output_base / input_path.name}/ with identical folder structure.")

