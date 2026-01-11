import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from lucidrain_vit import ViT
from torch.optim import Adam
from accelerate import Accelerator
import wandb

parser = argparse.ArgumentParser(description='Train ViT on feature maps')
parser.add_argument('--num-classes', type=int, required=True, help='number of classes')
parser.add_argument('--root-dir', type=str, required=True, help='root directory containing train/val subdirectories')
parser.add_argument('--batch-size', type=int, default=64, help='batch size (default: 64)')
parser.add_argument('--feature-dim', type=int, default=2048, help='feature dim of the per-patch representation (default: 2048)')
parser.add_argument('--run-name', type=str, default='Base_Imagenette_probe_vit_imagenette', help='WandB run name')
args = parser.parse_args()

BATCH_SIZE = args.batch_size
LEARNING_RATE = 3e-4
EPOCHS = 100
NUM_PATCHES = 49
FEATURE_DIM = args.feature_dim
NUM_CLASSES = args.num_classes
TRACK_EXPERIMENT_ONLINE = False
ROOT_DIR = args.root_dir

class FeatureMapDataset(Dataset):
    """Loads pre-computed feature maps from .pt files"""
    def __init__(self, root_dir, class_to_idx=None):
        self.samples = []
        self.classes = []
        
        if os.path.isdir(root_dir):
            # Get class directories
            class_dirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
            
            # Use provided class_to_idx if available, otherwise create new one
            if class_to_idx is None:
                self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_dirs)}
            else:
                self.class_to_idx = class_to_idx
            
            self.classes = list(self.class_to_idx.keys())
            
            # Load samples
            for class_name in class_dirs:
                if class_name in self.class_to_idx:
                    class_dir = os.path.join(root_dir, class_name)
                    files = [f for f in sorted(os.listdir(class_dir)) if f.endswith('.pt')]
                    for fname in files:
                        self.samples.append((os.path.join(class_dir, fname), self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, label = self.samples[idx]
        feature_map = torch.load(filepath)
        return feature_map, label

# Create train dataset
train_dataset = FeatureMapDataset(root_dir=os.path.join(ROOT_DIR, 'train'))
# Create val dataset with same class mapping as train
val_dataset = FeatureMapDataset(root_dir=os.path.join(ROOT_DIR, 'val'), class_to_idx=train_dataset.class_to_idx)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

vit = ViT(
    num_patches=NUM_PATCHES,
    num_classes=NUM_CLASSES,
    dim=FEATURE_DIM,
    depth=1,
    heads=1,
    dim_head=64,
    mlp_dim=128 * 4,
)
print(vit)
optim = Adam(vit.parameters(), lr=LEARNING_RATE)

accelerator = Accelerator()
vit, optim, train_dataloader, val_dataloader = accelerator.prepare(vit, optim, train_dataloader, val_dataloader)


def evaluate(val_dataloader, model):
    """Evaluate model on validation set"""
    model.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0
    
    with torch.no_grad():
        for feature_maps, labels in val_dataloader:
            logits = model(feature_maps)
            loss = F.cross_entropy(logits, labels)
            
            predictions = logits.argmax(dim=1)
            correct = (predictions == labels).sum().item()
            
            total_correct += correct
            total_samples += labels.size(0)
            total_loss += loss.item() * labels.size(0)
    
    accuracy = total_correct / total_samples
    avg_loss = total_loss / total_samples
    model.train()
    return accuracy, avg_loss

wandb.init(
    project='mocov3_singlegpu',
    name=args.run_name,
)
step = 0
for epoch in range(EPOCHS):
    # Training loop
    for feature_maps, labels in train_dataloader:
        logits = vit(feature_maps)
        loss = F.cross_entropy(logits, labels)
        
        # Calculate accuracy
        predictions = logits.argmax(dim=1)
        accuracy = (predictions == labels).float().mean()
        
        wandb.log({'train_loss': loss, 'train_accuracy': accuracy}, step=step)
        accelerator.print(f'epoch {epoch}, train_loss: {loss.item():.3f}, train_accuracy: {accuracy.item():.3f}')
        accelerator.backward(loss)
        optim.step()
        optim.zero_grad()
        step += 1
    
    # Evaluate on validation set every 5 epochs
    if (epoch + 1) % 5 == 0:
        val_accuracy, val_loss = evaluate(val_dataloader, vit)
        # Log validation metrics without committing (will be committed with next training step)
        wandb.log({'val_loss': val_loss, 'val_accuracy': val_accuracy}, commit=False)
        accelerator.print(f'epoch {epoch}, val_loss: {val_loss:.3f}, val_accuracy: {val_accuracy:.3f}')
