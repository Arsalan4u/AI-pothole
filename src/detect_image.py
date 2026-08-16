import cv2
import argparse
from pathlib import Path
from ultralytics import YOLO
from severity import estimate_severity, calculate_road_condition

def detect_potholes_in_image(image_path, model_path="d:/AI-Based Pothole Detection System/models/best.pt", output_dir="d:/AI-Based Pothole Detection System/outputs/predictions"):
    """
    Runs YOLO inference on a single image, draws bounding boxes, and calculates severity.
    """
    img_path = Path(image_path)
    if not img_path.exists():
        print(f"[!] Error: Image not found at {image_path}")
        return

    model_file = Path(model_path)
    if not model_file.exists():
        print(f"[!] Error: Model not found at {model_path}")
        return

    # Load Model
    model = YOLO(model_path)
    
    # Read Image
    img = cv2.imread(str(img_path))
    if img is None:
        print("[!] Error: Could not read image.")
        return
        
    h, w, _ = img.shape
    image_area = h * w

    # Run Inference
    results = model(img)[0]
    
    detections = []
    severities = []
    
    # Process Results
    for box in results.boxes:
        # Bounding box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        
        # Calculate Severity
        box_area = (x2 - x1) * (y2 - y1)
        severity = estimate_severity(box_area, image_area, conf)
        severities.append(severity)
        
        # Determine Box Color based on Severity
        if severity == "HIGH":
            color = (0, 0, 255) # Red
        elif severity == "MEDIUM":
            color = (0, 165, 255) # Orange
        else:
            color = (0, 255, 0) # Green
            
        # Draw Bounding Box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        
        # Draw Label
        label = f"Pothole {conf:.2f} ({severity})"
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        detections.append({
            "confidence": conf,
            "severity": severity
        })

    # Calculate Overall Road Condition
    road_condition = calculate_road_condition(len(detections), severities)
    
    # Print Summary to Console
    print("\n" + "="*40)
    print(" DETECTION SUMMARY")
    print("="*40)
    print(f"Detected Potholes: {len(detections)}")
    
    for i, det in enumerate(detections, 1):
        print(f"Pothole {i}: Confidence: {det['confidence']:.2f}, Severity: {det['severity']}")
        
    print(f"\nAI Estimated Road Condition: {road_condition}")
    print("="*40)

    # Save Output Image
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_file = Path(output_dir) / f"detected_{img_path.name}"
    
    # Add Road condition text to image
    cv2.putText(img, f"Condition: {road_condition}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    cv2.putText(img, f"Condition: {road_condition}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
    
    cv2.imwrite(str(out_file), img)
    print(f"[*] Detected image saved to: {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect potholes in an image.")
    parser.add_argument("image", type=str, help="Path to the input image.")
    args = parser.parse_args()
    
    detect_potholes_in_image(args.image)
