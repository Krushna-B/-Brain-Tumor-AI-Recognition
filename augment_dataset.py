

import argparse
import os
from pathlib import Path
import random
import shutil
from typing import Tuple, List

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from torchvision import transforms



def build_augment_pipeline(img_size: int):
    
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomApply([
            transforms.RandomAffine(
                degrees=25,          # rotation ±25°
                translate=(0.05, 0.05),  # shift ±5%
                scale=(0.9, 1.1),   # zoom 0.9x–1.1x
                shear=10            # shear ±10°
            )
        ], p=0.9),
        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=0.15,    
                contrast=0.15       
            )
        ], p=0.9),
        transforms.RandomHorizontalFlip(p=0.5),
      
    ])


def is_image(p: Path) -> bool:
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def gather_images(input_dir: Path) -> Tuple[List[Path], List[int], List[str]]:
    """
    Scans input_dir expecting class subfolders (e.g., yes/, no/).
    Returns: (files, labels, class_names)
    """
    classes = []
    files = []
    labels = []
    class_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
    for cls_idx, cls_dir in enumerate(sorted(class_dirs, key=lambda p: p.name)):
        classes.append(cls_dir.name)
        for f in sorted(cls_dir.rglob("*")):
            if is_image(f):
                files.append(f)
                labels.append(cls_idx)
    return files, labels, classes


def stratified_split(files: List[Path], labels: List[int], test_ratio: float, seed: int):
    """
    Simple stratified split by shuffling within each class.
    """
    random.seed(seed)
    by_class = {}
    for f, y in zip(files, labels):
        by_class.setdefault(y, []).append(f)
    train_files, train_labels, test_files, test_labels = [], [], [], []
    for y, flist in by_class.items():
        random.shuffle(flist)
        n = len(flist)
        n_test = max(1, int(round(n * test_ratio)))
        test_split = flist[:n_test]
        train_split = flist[n_test:]
        test_files.extend(test_split)
        test_labels.extend([y] * len(test_split))
        train_files.extend(train_split)
        train_labels.extend([y] * len(train_split))
    return train_files, train_labels, test_files, test_labels


def save_image(img: Image.Image, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ext = out_path.suffix.lower()
    fmt = "PNG" if ext == ".png" else "JPEG"
    if fmt == "JPEG" and img.mode != "RGB":
        img = img.convert("RGB")
    img.save(out_path, fmt, quality=95)


def main():
    ap = argparse.ArgumentParser(description="Offline data augmentation to materialize train/test folders.")
    ap.add_argument("--input", type=Path, required=True,
                    help="Input dataset root with class subfolders (e.g., yes/ and no/).")
    ap.add_argument("--output", type=Path, required=True,
                    help="Output root. Will create train/ and test/ with class subfolders.")
    ap.add_argument("--test-ratio", type=float, default=0.2,
                    help="Fraction of images for test split (default: 0.2).")
    ap.add_argument("--img-size", type=int, default=128,
                    help="Resize images to this square size (default: 128).")
    ap.add_argument("--copies", type=int, default=2,
                    help="Number of augmented copies per training image (default: 2).")
    ap.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    args = ap.parse_args()

    input_dir: Path = args.input
    output_dir: Path = args.output
    test_ratio: float = args.test_ratio
    img_size: int = args.img_size
    copies: int = args.copies
    seed: int = args.seed

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    train_root = output_dir 
    test_root = output_dir 

    # Build transforms
    aug = build_augment_pipeline(img_size)
    resize_only = transforms.Resize((img_size, img_size))


    files, labels, classes = gather_images(input_dir)
    if len(files) == 0:
        raise RuntimeError(
            f"No images in {input_dir}. Expected: {input_dir}/yes/*.jpg and {input_dir}/no/*.jpg etc."
        )

   
    for split_root in [train_root, test_root]:
        for cls_name in classes:
            (split_root / cls_name).mkdir(parents=True, exist_ok=True)

    # Split
    train_files, train_labels, test_files, test_labels = stratified_split(files, labels, test_ratio, seed)

  
    print(f"[Test] Writing {len(test_files)} images...")
    for src, y in zip(test_files, test_labels):
        cls_name = classes[y]
        out_path = test_root / cls_name / (src.stem + ".jpg")
        try:
            with Image.open(src) as im:
                im = im.convert("RGB")
                im = resize_only(im)
                save_image(im, out_path)
        except Exception as e:
            print(f"  ! Skipping {src} due to error: {e}")

   
    print(f"[Train] Writing {len(train_files)} originals + {copies} aug copies each...")
    processed = 0
    for src, y in zip(train_files, train_labels):
        cls_name = classes[y]
        base = src.stem

        try:
            with Image.open(src) as im:
                im = im.convert("RGB")
                im = resize_only(im)
                save_image(im, train_root / cls_name / f"{base}_orig.jpg")
        except Exception as e:
            print(f"  ! Skipping original {src}: {e}")
            continue


        try:
            with Image.open(src) as im:
                im = im.convert("RGB")
                for i in range(copies):
                    out = aug(im)  # PIL -> PIL
                    save_image(out, train_root / cls_name / f"{base}_aug{i+1}.jpg")
        except Exception as e:
            print(f"  ! Augmentation failed for {src}: {e}")
            continue

        processed += 1
        if processed % 100 == 0:
            print(f"  processed {processed} training images...")

    print("Done.")
    print(f"Output:\n  {train_root}/<class>/<file>.jpg\n  {test_root}/<class>/<file>.jpg")
    print("You can now use torchvision.datasets.ImageFolder on train/ and test/.")


if __name__ == "__main__":
    main()