import os
from pathlib import Path
from ultralytics import YOLO
import shutil
import pandas as pd

def evaluate_model(model_path="d:/AI-Based Pothole Detection System/models/best.pt", 
                   data_yaml="d:/AI-Based Pothole Detection System/dataset.yaml",
                   report_dir="d:/AI-Based Pothole Detection System/outputs/reports"):
    """
    Evaluates the trained YOLO model on the test dataset and generates a report.
    """
    print("="*50)
    print(" AI-Based Pothole Detection - Model Evaluation")
    print("="*50)

    model_file = Path(model_path)
    if not model_file.exists():
        print(f"[!] Error: Model not found at {model_path}")
        print("Please train the model first using src/train.py")
        return

    print(f"[*] Loading trained model: {model_path}")
    model = YOLO(model_path)

    print(f"[*] Starting evaluation on test dataset...")
    
    # Run validation
    # YOLO handles the test set evaluation when split='test' is provided.
    metrics = model.val(data=data_yaml, split='test', project="d:/AI-Based Pothole Detection System/runs", name="pothole_eval", exist_ok=True)
    
    # Extract metrics
    precision = metrics.results_dict['metrics/precision(B)']
    recall = metrics.results_dict['metrics/recall(B)']
    map50 = metrics.results_dict['metrics/mAP50(B)']
    map50_95 = metrics.results_dict['metrics/mAP50-95(B)']

    print("\n" + "="*50)
    print(" EVALUATION METRICS EXPLAINED (For Viva)")
    print("="*50)
    
    explanation = f"""
1. Precision: {precision:.4f}
   - Explanation: Out of all the 'potholes' the model detected, this percentage were actually potholes. High precision means very few false alarms.

2. Recall: {recall:.4f}
   - Explanation: Out of all the real potholes in the test images, the model successfully found this percentage. High recall means it doesn't miss many potholes.

3. mAP@0.5 (mAP50): {map50:.4f}
   - Explanation: The Mean Average Precision calculated at an Intersection over Union (IoU) threshold of 50%. It means the model's bounding box overlapped with the real pothole by at least 50%.

4. mAP@0.5:0.95 (mAP50-95): {map50_95:.4f}
   - Explanation: The average mAP across multiple strict IoU thresholds (from 50% to 95%). This is a very strict metric; a high score here means the bounding boxes are drawn extremely accurately around the potholes.
"""
    print(explanation)

    # Save to report file
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    report_file = Path(report_dir) / "evaluation_report.txt"
    
    with open(report_file, 'w') as f:
        f.write("Pothole Detection - Test Evaluation Report\n")
        f.write("="*42 + "\n")
        f.write(explanation)

    print(f"[*] Evaluation report saved to: {report_file}")
    
    # YOLO automatically generates a confusion matrix and PR curve in the runs folder.
    # Let's copy them to our reports folder so the Streamlit app can easily find them later.
    runs_dir = Path("d:/AI-Based Pothole Detection System/runs/pothole_eval")
    
    plots_to_copy = {
        "confusion_matrix.png": "confusion_matrix.png",
        "PR_curve.png": "precision_recall_curve.png",
        "F1_curve.png": "f1_score_curve.png"
    }
    
    for src_name, dst_name in plots_to_copy.items():
        src_path = runs_dir / src_name
        dst_path = Path(report_dir) / dst_name
        if src_path.exists():
            shutil.copy(src_path, dst_path)
    
    print(f"[*] Visual plots (Confusion Matrix, PR Curve) copied to: {report_dir}")
    print("="*50)

if __name__ == "__main__":
    evaluate_model()
