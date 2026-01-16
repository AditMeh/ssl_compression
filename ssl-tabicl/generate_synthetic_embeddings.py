from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
import argparse
import sys
import torch.nn.functional as F
from torchvision.transforms.functional import InterpolationMode

# Load model once
print("Loading DINOv2 model...")
model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
model.eval()  # Set to evaluation mode

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device).eval()
print(f"Using device: {device}")

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
transform = transforms.Compose(
    [
        transforms.Normalize(mean=mean, std=std),
        transforms.CenterCrop(224)
    ]
)
def create_npy_file(pth_file):
    with torch.no_grad():
        data = torch.load(pth_file, map_location="cuda")
        images = data["images"]  # shape (B, C, H, W), values in [0,1]
        labels = data["labels"]  # shape (B,)
        images = transform(images)
        print(images.shape)

        images = model(images)
        images = images.cpu().numpy()
        labels = labels.cpu().numpy()

        save_dict = {"image": images, "labels": labels}
        # Always save as synthetic_samples.npy in the same directory as the input file
        output_path = str(Path(pth_file).with_name("synthetic_samples.npy"))
        np.save(output_path, save_dict)
        print(f"Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate embeddings for images in a folder structure')
    parser.add_argument('pth_file', type=str, help='Path to pth file')

    args = parser.parse_args()
    create_npy_file(args.pth_file)