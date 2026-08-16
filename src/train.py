import os
import argparse
from pathlib import Path
import torch
from ultralytics import YOLO
import shutil

def train_model(data_yaml="d:/AI-Based Pothole Detection System/dataset.yaml", 
                epochs=50, imgsz=640, batch=16, model_type="yolov8n.pt"):
    """
    Trains the YOLO model on the pothole dataset.
    """
    print("="*50)
    print(" AI-Based Pothole Detection - Training Pipeline")
    print("="*50)

    # 1. Device selection (GPU if available, else CPU)
    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"[*] Using device: {'GPU (CUDA)' if device == '0' else 'CPU'}")
    print(f"[*] Pretrained Model: {model_type}")
    print(f"[*] Epochs: {epochs} | Batch Size: {batch} | Image Size: {imgsz}")
    print("="*50)
    
    # 2. Load the pretrained YOLO model (Transfer Learning)
    # Using a nano (n) or small (s) variant is best for student hardware
    model = YOLO(model_type)
    
    # 3. Train the model
    # The results will be saved inside the 'runs/pothole_detection' folder
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project="d:/AI-Based Pothole Detection System/runs",
        name="pothole_detection",
        exist_ok=True, # Overwrite if same name exists to avoid clutter
        save=True,     # Save checkpoints
        save_period=10, # Save a checkpoint every 10 epochs
        plots=True     # Automatically generate training graphs
    )
    
    # 4. Save the best model
    best_model_src = Path("d:/AI-Based Pothole Detection System/runs/pothole_detection/weights/best.pt")
    target_model_path = Path("d:/AI-Based Pothole Detection System/models/best.pt")
    
    if best_model_src.exists():
        target_model_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(best_model_src, target_model_path)
        print("\n" + "="*50)
        print(f"[*] Training successful!")
        print(f"[*] Best model saved to: {target_model_path}")
        print(f"[*] Training graphs saved in: d:/AI-Based Pothole Detection System/runs/pothole_detection/")
        print("="*50)
    else:
        print("[!] Warning: best.pt not found. Training might have encountered an issue.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO model for Pothole Detection")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Pretrained model to use (e.g. yolov8n.pt, yolov8s.pt)")
    
    args = parser.parse_args()
    
    train_model(epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, model_type=args.model)
