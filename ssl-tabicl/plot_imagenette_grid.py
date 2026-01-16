import os
import random
from pathlib import Path

from PIL import Image
import matplotlib.pyplot as plt


def sample_one_image_per_class(root_dir: str):
    """
    Walk imagenette2 directory and sample one random image path per class.

    Expects structure like:
      root_dir/
        train/
          class0/
            *.JPEG / *.png / ...
          class1/
            ...
    Falls back to using root_dir directly if there is no train/ subfolder.
    """
    root = Path(os.path.expanduser(root_dir))

    train_dir = root / "train"
    if train_dir.exists():
        base = train_dir
    else:
        base = root

    class_to_image = {}
    for class_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        # Collect image files in this class directory
        image_files = [
            p
            for p in class_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ]
        if not image_files:
            continue
        class_to_image[class_dir.name] = random.choice(image_files)

    return class_to_image


def save_grid(class_to_image, out_path: str, n_rows: int = 2, n_cols: int = 5):
    """Save a tightly packed grid of resized images (no labels/axes)."""
    classes = sorted(class_to_image.keys())
    # Limit to n_rows * n_cols classes if more are present
    max_imgs = n_rows * n_cols
    classes = classes[:max_imgs]

    # Each image will be 224x224; choose figure size to roughly match that
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.24, n_rows * 2.24))
    axes = axes.flatten()

    for ax, cls in zip(axes, classes):
        img_path = class_to_image[cls]
        img = Image.open(img_path).convert("RGB")
        img = img.resize((224, 224), Image.BICUBIC)
        ax.imshow(img)
        ax.axis("off")

    # Turn off any unused axes
    for ax in axes[len(classes) :]:
        ax.axis("off")

    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Sample one random image per class from imagenette2 and save a 2x5 grid."
    )
    parser.add_argument(
        "--root",
        type=str,
        default="~/scratch/imagenette2",
        help="Root directory of imagenette2 (default: ~/scratch/imagenette2)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="imagenette2_grid.png",
        help="Output PNG path (default: imagenette2_grid.png)",
    )
    args = parser.parse_args()

    class_to_image = sample_one_image_per_class(args.root)
    if not class_to_image:
        raise RuntimeError(f"No images found under {args.root}")

    save_grid(class_to_image, args.output, n_rows=2, n_cols=5)
    print(f"Saved grid with {min(len(class_to_image), 10)} classes to {args.output}")


if __name__ == "__main__":
    main()


