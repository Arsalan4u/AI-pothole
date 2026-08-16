import cv2
import os
import random
from pathlib import Path
import matplotlib.pyplot as plt

def verify_dataset(dataset_dir="d:/AI-Based Pothole Detection System/dataset", output_dir="d:/AI-Based Pothole Detection System/outputs/reports"):
    """
    Picks a random image from the training set, reads its YOLO labels, 
    draws the bounding boxes, and saves the output to verify dataset integrity.
    """
    train_images_dir = Path(dataset_dir) / "images" / "train"
    train_labels_dir = Path(dataset_dir) / "labels" / "train"
    
    if not train_images_dir.exists():
        print(f"Error: {train_images_dir} does not exist.")
        return

    # Get a random image
    images = list(train_images_dir.glob("*.jpg")) + list(train_images_dir.glob("*.png")) + list(train_images_dir.glob("*.jpeg"))
    if not images:
        print("No images found in training set!")
        return
        
    random_image_path = random.choice(images)
    label_path = train_labels_dir / (random_image_path.stem + ".txt")
    
    # Read the image
    img = cv2.imread(str(random_image_path))
    if img is None:
        print(f"Failed to read image {random_image_path}")
        return
        
    h, w, _ = img.shape
    
    # Read labels if they exist
    if label_path.exists():
        with open(label_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id, x_center, y_center, width, height = map(float, parts[:5])
                
                # Convert YOLO format (normalized) back to pixel coordinates
                x_center, y_center = int(x_center * w), int(y_center * h)
                width, height = int(width * w), int(height * h)
                
                x_min = int(x_center - width / 2)
                y_min = int(y_center - height / 2)
                x_max = int(x_center + width / 2)
                y_max = int(y_center + height / 2)
                
                # Draw bounding box
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(img, f"Class {int(class_id)}", (x_min, y_min - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Save the output
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / "dataset_verification.jpg"
    cv2.imwrite(str(output_path), img)
    print(f"Dataset verification successful! Verified image saved to: {output_path}")

if __name__ == "__main__":
    verify_dataset()
