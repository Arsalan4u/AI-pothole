import cv2
import argparse
import time
from pathlib import Path
from ultralytics import YOLO
from severity import estimate_severity, calculate_road_condition

def detect_potholes_in_video(video_path, model_path="d:/AI-Based Pothole Detection System/models/best.pt", output_dir="d:/AI-Based Pothole Detection System/outputs/videos", skip_frames=2):
    """
    Runs YOLO inference on a video, tracking potholes frame-by-frame.
    """
    vid_path = Path(video_path)
    if not vid_path.exists():
        print(f"[!] Error: Video not found at {video_path}")
        return

    # Load Model
    model = YOLO(model_path)
    
    # Open Video
    cap = cv2.VideoCapture(str(vid_path))
    if not cap.isOpened():
        print("[!] Error: Could not open video.")
        return
        
    # Get Video Properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    image_area = width * height
    
    # Prepare Video Writer
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_file = Path(output_dir) / f"detected_{vid_path.name}"
    
    # Use mp4v codec for better compatibility
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(out_file), fourcc, fps, (width, height))
    
    print(f"[*] Processing Video: {vid_path.name} (Resolution: {width}x{height}, FPS: {fps})")
    
    frame_count = 0
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Optimization: Skip frames to speed up processing
        if frame_count % skip_frames != 0:
            # We still write the unprocessed frame to keep the video smooth, 
            # but ideally, in a real app, you'd use a tracker (like ByteTrack).
            # For simplicity, we just run detection on every Nth frame.
            out.write(frame)
            continue
            
        # Run Inference
        results = model(frame, verbose=False)[0]
        
        detections = []
        severities = []
        
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            
            box_area = (x2 - x1) * (y2 - y1)
            severity = estimate_severity(box_area, image_area, conf)
            severities.append(severity)
            
            if severity == "HIGH":
                color = (0, 0, 255)
            elif severity == "MEDIUM":
                color = (0, 165, 255)
            else:
                color = (0, 255, 0)
                
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            label = f"Pothole {conf:.2f} ({severity})"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
        road_condition = calculate_road_condition(len(severities), severities)
        
        # Draw UI
        cv2.putText(frame, f"Potholes: {len(severities)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"Condition: {road_condition}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Calculate processing FPS
        elapsed = time.time() - start_time
        current_fps = frame_count / elapsed
        cv2.putText(frame, f"FPS: {current_fps:.1f}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        out.write(frame)
        
    cap.release()
    out.release()
    
    print("\n" + "="*40)
    print(f"[*] Video processing complete!")
    print(f"[*] Saved to: {out_file}")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect potholes in a video.")
    parser.add_argument("video", type=str, help="Path to the input video.")
    parser.add_argument("--skip", type=int, default=2, help="Process every Nth frame to speed up (default: 2).")
    args = parser.parse_args()
    
    detect_potholes_in_video(args.video, skip_frames=args.skip)
