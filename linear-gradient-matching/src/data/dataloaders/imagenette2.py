from typing import Literal

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import Tensor
from torch.utils.data import DataLoader

from .base import BaseRealDataset


class ImageNette2(BaseRealDataset):

    def __init__(
        self,
        split: str = "train",
        res=256,
        crop_res: int = 256,
        crop_mode: Literal["center", "random"] = "center",
        data_root: str = "data/datasets",
    ):

        super().__init__()

        self.num_classes = 10

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        self.transform = transforms.Compose(
            [
                transforms.Resize(res),
                (
                    transforms.CenterCrop(crop_res)
                    if crop_mode == "center"
                    else transforms.RandomCrop(crop_res)
                ),
                transforms.ToTensor(),
            ]
        )

        self.mean = torch.tensor(mean, device="cuda").reshape(1, 3, 1, 1)
        self.std = torch.tensor(std, device="cuda").reshape(1, 3, 1, 1)

        self.data_dir = f"{data_root}/imagenette2"
        if split == "train":
            traindir = f"{self.data_dir}/train"
            self.ds = datasets.ImageFolder(traindir, transform=self.transform)
        elif split == "val" or split == "test":
            valdir = f"{self.data_dir}/val"
            self.ds = datasets.ImageFolder(valdir, transform=self.transform)
        else:
            raise ValueError(f"Unknown split: {split}. Must be 'train' or 'val'/'test'")

        self.class_names = self.ds.classes

    def __getitem__(self, index):

        image, label = self.ds.__getitem__(index)
        return image, label

    def __len__(self):
        return len(self.ds)

    def get_single_class(self, cls: int) -> Tensor:

        # Get the class name
        class_name = self.class_names[cls]
        
        # Create a temporary dataset with the same transform
        temp_ds = datasets.ImageFolder(
            f"{self.data_dir}/train",
            transform=self.transform
        )

        # Filter by class
        class_idx = temp_ds.class_to_idx[class_name]
        samples = [s for s in temp_ds.samples if s[1] == class_idx]

        num_samples = len(samples)

        loader = DataLoader(
            torch.utils.data.Subset(temp_ds, [temp_ds.samples.index(s) for s in samples]),
            batch_size=64,
            num_workers=8
        )
        images = []
        labels = []
        print(f"Loading all {num_samples} images for class {cls} ({class_name})...")
        for x, y in loader:
            images.append(x)
            labels.append(y)
        images = torch.cat(images)
        labels = torch.cat(labels)
        print("Done.")

        return images
