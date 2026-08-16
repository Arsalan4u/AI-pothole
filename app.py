import streamlit as st
import cv2
import os
import time
import numpy as np
from PIL import Image
from pathlib import Path
from ultralytics import YOLO
import pandas as pd

# Important: Need to import from src
import sys
sys.path.append(os.path.abspath('src'))
from severity import estimate_severity, calculate_road_condition

# --- PAGE CONFIG ---
st.set_page_config(page_title="Pothole Detection AI", page_icon="🛣️", layout="wide")

# --- CUSTOM CSS FOR AESTHETICS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Sleek dark theme with animated gradient background */
    .stApp {
        background: radial-gradient(circle at 15% 50%, #1a1a2e, #16213e 50%, #0f3460);
        color: #e5e5e5;
    }
    
    .stApp::before {
        content: "";
        position: absolute;
        top: 0; left: 0; width: 100%; height: 100%;
        background: url('https://www.transparenttextures.com/patterns/stardust.png');
        opacity: 0.3;
        pointer-events: none;
    }

    /* Modern Glassmorphic Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(22, 33, 62, 0.65) !important;
        backdrop-filter: blur(15px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Glowing Title with fluid gradient */
    .main-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(120deg, #00F2FE 0%, #4FACFE 50%, #00F2FE 100%);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 40px;
        letter-spacing: -1px;
        animation: shine 3s linear infinite;
        text-shadow: 0px 4px 20px rgba(0, 242, 254, 0.3);
    }
    
    @keyframes shine {
        to {
            background-position: 200% center;
        }
    }

    /* Premium Glassmorphic Cards with Hover Micro-animations */
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-radius: 20px;
        padding: 25px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        text-align: center;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        height: 100%;
    }
    
    .metric-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 15px 40px rgba(0, 242, 254, 0.15);
        border: 1px solid rgba(0, 242, 254, 0.3);
        background: rgba(255, 255, 255, 0.05);
    }
    
    .metric-card h3, .metric-card h4 {
        margin-top: 0;
        color: #a0aec0;
        font-weight: 600;
        letter-spacing: 0.5px;
        font-size: 1.1rem;
        text-transform: uppercase;
    }
    
    .metric-value {
        font-size: 2.8rem;
        font-weight: 800;
        color: #00F2FE;
        text-shadow: 0 0 15px rgba(0, 242, 254, 0.4);
        margin-top: 10px;
    }

    /* Styling Buttons */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #00F2FE 0%, #4FACFE 100%);
        color: white;
        border: none;
        border-radius: 30px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 242, 254, 0.3);
    }
    
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 242, 254, 0.5);
    }

    /* File Uploader styling */
    [data-testid="stFileUploadDropzone"] {
        background: rgba(255, 255, 255, 0.02);
        border: 2px dashed rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploadDropzone"]:hover {
        border-color: #00F2FE;
        background: rgba(0, 242, 254, 0.05);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# --- GLOBAL VARIABLES ---
MODEL_PATH = "d:/AI-Based Pothole Detection System/models/best.pt"
OUTPUT_DIR = "d:/AI-Based Pothole Detection System/outputs/predictions"
REPORT_DIR = "d:/AI-Based Pothole Detection System/outputs/reports"

@st.cache_resource
def load_model():
    if Path(MODEL_PATH).exists():
        return YOLO(MODEL_PATH)
    return None

# Load model once
model = load_model()

# --- SIDEBAR NAV ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3253/3253258.png", width=100)
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to", ["🏠 Home", "📷 Image Detection", "🎥 Video Detection", "🔴 Live Detection", "📊 Model Performance", "ℹ️ About Project"])

st.sidebar.markdown("---")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.01, 1.0, 0.25, 0.05)
st.sidebar.markdown("---")
st.sidebar.markdown("**Project By:** 7th Semester CSE")
st.sidebar.markdown("**Tech:** Python, YOLO, OpenCV, Streamlit")

# --- HELPER FUNCTIONS ---
def draw_boxes(img, results):
    h, w, _ = img.shape
    image_area = h * w
    detections = []
    severities = []
    
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        
        box_area = (x2 - x1) * (y2 - y1)
        severity = estimate_severity(box_area, image_area, conf)
        severities.append(severity)
        
        if severity == "HIGH": color = (255, 0, 0) # BGR (Red in OpenCV, but Streamlit uses RGB, so we swap later)
        elif severity == "MEDIUM": color = (255, 165, 0)
        else: color = (0, 255, 0)
            
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        label = f"Pothole {conf:.2f} ({severity})"
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        detections.append(conf)
        
    return img, detections, severities

# --- PAGES ---

if menu == "🏠 Home":
    st.markdown('<div class="main-title">AI-Based Pothole Detection & Road Monitoring</div>', unsafe_allow_html=True)
    
    st.write("### Welcome to the Dashboard!")
    st.write("""
    This system uses **Deep Learning (YOLO)** to automatically detect potholes on roads, estimate their visual severity, and assess the overall road condition. 
    It is designed to help municipalities and drivers identify dangerous road segments quickly and efficiently.
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>⚡ Fast Inference</h3>
            <p>Real-time detection using optimized YOLO architecture.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 High Accuracy</h3>
            <p>Trained on a diverse dataset to generalize across lighting conditions.</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Actionable Reports</h3>
            <p>Calculates severity and generates automated road condition scores.</p>
        </div>
        """, unsafe_allow_html=True)

    if not model:
        st.error("⚠️ Model not found! Please train the model using `src/train.py` first.")

elif menu == "📷 Image Detection":
    st.markdown('<div class="main-title">Image Detection</div>', unsafe_allow_html=True)
    
    if not model:
        st.warning("Model not found. Please complete training.")
        st.stop()
        
    uploaded_file = st.file_uploader("Upload a Road Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        # Convert RGB to BGR for OpenCV processing
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        with st.spinner("Detecting Potholes..."):
            results = model(img_bgr, conf=conf_threshold)[0]
            detected_img, confidences, severities = draw_boxes(img_bgr.copy(), results)
            
            # Convert back to RGB for Streamlit
            detected_img_rgb = cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB)
            
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", width='stretch')
        with col2:
            st.image(detected_img_rgb, caption="Detected Image", width='stretch')
            
        # Analysis
        pothole_count = len(confidences)
        avg_conf = sum(confidences)/pothole_count if pothole_count > 0 else 0
        road_condition = calculate_road_condition(pothole_count, severities)
        
        st.markdown("### 📊 Detection Analysis")
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f'<div class="metric-card"><h4>Count</h4><div class="metric-value">{pothole_count}</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><h4>Avg Confidence</h4><div class="metric-value">{avg_conf:.2f}</div></div>', unsafe_allow_html=True)
        
        # Color code the condition
        cond_color = "#00F2FE"
        if road_condition == "CRITICAL": cond_color = "#ff4b4b"
        elif road_condition == "POOR": cond_color = "#ffa500"
        elif road_condition == "GOOD": cond_color = "#00ff00"
        
        c3.markdown(f'<div class="metric-card"><h4>Road Condition</h4><div class="metric-value" style="color:{cond_color}; text-shadow: 0 0 15px {cond_color}80;">{road_condition}</div></div>', unsafe_allow_html=True)
        
        high = severities.count("HIGH")
        med = severities.count("MEDIUM")
        low = severities.count("LOW")
        
        c4.markdown(f'<div class="metric-card"><h4>Severity</h4><p style="color:#ff4b4b;margin:5px 0;font-weight:600;">High: {high}</p><p style="color:#ffa500;margin:5px 0;font-weight:600;">Med: {med}</p><p style="color:#00ff00;margin:5px 0;font-weight:600;">Low: {low}</p></div>', unsafe_allow_html=True)
        
        # Save Report
        if st.button("Generate CSV Report"):
            report_data = {
                "Filename": [uploaded_file.name],
                "Pothole Count": [pothole_count],
                "Avg Confidence": [f"{avg_conf:.2f}"],
                "High Severity": [high],
                "Medium Severity": [med],
                "Low Severity": [low],
                "Road Condition": [road_condition]
            }
            df = pd.DataFrame(report_data)
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Report as CSV",
                data=csv,
                file_name=f"report_{uploaded_file.name}.csv",
                mime="text/csv",
            )

elif menu == "🎥 Video Detection":
    st.markdown('<div class="main-title">Video Detection</div>', unsafe_allow_html=True)
    
    if not model:
        st.warning("Model not found.")
        st.stop()
        
    uploaded_video = st.file_uploader("Upload a Road Video", type=["mp4", "avi", "mov"])
    
    if uploaded_video is not None:
        st.video(uploaded_video)
        
        if st.button("Process Video"):
            import tempfile
            with st.spinner("Processing video..."):
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(uploaded_video.read())
                tfile.close()
                
                cap = cv2.VideoCapture(tfile.name)
                
                st_frame = st.empty()
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    results = model(frame, conf=conf_threshold, verbose=False)[0]
                    frame, confidences, severities = draw_boxes(frame, results)
                    
                    # Convert BGR to RGB for Streamlit
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    st_frame.image(frame_rgb, channels="RGB")
                    
                cap.release()
                import os
                try:
                    os.unlink(tfile.name)
                except:
                    pass
                
            st.success("Video processing complete!")

elif menu == "🔴 Live Detection":
    st.markdown('<div class="main-title">Live Webcam Detection</div>', unsafe_allow_html=True)
    
    st.write("""
    Browser-based webcam access in Streamlit (like `st.camera_input`) only takes static photos. 
    To get a **true 30 FPS live video feed**, we use OpenCV directly.
    """)
    
    if st.button("Start Local Webcam"):
        st.info("Opening webcam... Press 'q' on the video window to close it.")
        
        # OpenCV Live Loop
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("Could not open webcam.")
        else:
            # We don't block the streamlit UI completely, but it will hang while OpenCV window is open.
            # This is expected for local desktop scripts.
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                results = model(frame, conf=conf_threshold, verbose=False)[0]
                frame, _, _ = draw_boxes(frame, results)
                
                cv2.imshow('Live Pothole Detection (Press Q to exit)', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            cap.release()
            cv2.destroyAllWindows()
            st.success("Webcam session ended.")

elif menu == "📊 Model Performance":
    st.markdown('<div class="main-title">Model Performance</div>', unsafe_allow_html=True)
    
    report_file = Path(REPORT_DIR) / "evaluation_report.txt"
    conf_matrix = Path(REPORT_DIR) / "confusion_matrix.png"
    pr_curve = Path(REPORT_DIR) / "precision_recall_curve.png"
    
    if report_file.exists():
        with open(report_file, 'r') as f:
            content = f.read()
        st.text_area("Evaluation Metrics", content, height=350)
    else:
        st.warning("Evaluation report not found. Run `python src/evaluate.py` first.")
        
    col1, col2 = st.columns(2)
    with col1:
        if conf_matrix.exists():
            st.image(str(conf_matrix), caption="Confusion Matrix")
    with col2:
        if pr_curve.exists():
            st.image(str(pr_curve), caption="Precision-Recall Curve")

elif menu == "ℹ️ About Project":
    st.markdown('<div class="main-title">About the Project</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Problem Statement
    Potholes cause severe damage to vehicles and lead to fatal road accidents. Manual inspection of roads is time-consuming and inefficient.
    
    ### Objective
    To build an automated, AI-driven system capable of detecting potholes from images, videos, and live camera feeds to evaluate road conditions dynamically.
    
    ### Methodology
    - **Dataset:** Annotated images of roads with potholes in YOLO format.
    - **Model:** Ultralytics YOLO (You Only Look Once) object detection architecture.
    - **Severity Estimation:** A proxy metric calculating the bounding box area relative to the image size.
    
    ### Future Scope
    - **GPS Integration:** Attaching latitude/longitude coordinates to every detection to build a map of potholes.
    - **Edge Deployment:** Running the model on a Raspberry Pi or Jetson Nano mounted on a dashboard camera.
    """)
