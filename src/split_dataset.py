import os
import shutil
import random
import argparse
from pathlib import Path

def split_dataset(source_dir, dest_dir="d:/AI-Based Pothole Detection System/dataset", split_ratio=(0.7, 0.2, 0.1)):
    """
    Splits images and YOLO labels into train, val, and test sets.
    """
    # Define source paths
    src_images_dir = Path(source_dir) / "images"
    src_labels_dir = Path(source_dir) / "labels-YOLO"

    if not src_images_dir.exists() or not src_labels_dir.exists():
        print(f"Error: Could not find 'images' or 'labels-YOLO' inside {source_dir}")
        print("Please ensure the path is correct.")
        return

    # Create destination directories if they don't exist
    dest_path = Path(dest_dir)
    for split in ["train", "val", "test"]:
        (dest_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (dest_path / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Get all image files
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    images = [f for f in os.listdir(src_images_dir) if Path(f).suffix.lower() in valid_extensions]
    
    # Shuffle for randomness
    random.seed(42)
    random.shuffle(images)

    total_images = len(images)
    print(f"Found {total_images} images.")

    # Calculate splits
    train_end = int(total_images * split_ratio[0])
    val_end = train_end + int(total_images * split_ratio[1])

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    def copy_files(file_list, split_name):
        copied = 0
        missing_labels = 0
        for img_name in file_list:
            # Paths
            img_src = src_images_dir / img_name
            label_name = Path(img_name).stem + ".txt"
            label_src = src_labels_dir / label_name

            # Check if label exists, if not, skip or create empty
            if not label_src.exists():
                missing_labels += 1
                continue

            # Copy image
            shutil.copy(img_src, dest_path / "images" / split_name / img_name)
            # Copy label
            shutil.copy(label_src, dest_path / "labels" / split_name / label_name)
            copied += 1
            
        print(f"Copied {copied} files to {split_name} (Missing labels: {missing_labels})")

    print("\nStarting copying process...")
    copy_files(train_images, "train")
    copy_files(val_images, "val")
    copy_files(test_images, "test")
    
    print("\nDataset split complete!")
    print(f"Check your dataset folder at: {dest_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into train/val/test")
    parser.add_argument("source_dir", type=str, help="Path to the downloaded dataset folder")
    args = parser.parse_args()
    
    split_dataset(args.source_dir)
