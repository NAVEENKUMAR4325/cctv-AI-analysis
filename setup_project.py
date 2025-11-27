import os

def create_file(path, content):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content.strip())
    print(f"✔ Created: {path}")

def main():
    # 1. Create Directory Structure
    dirs = ["modules", "input", "output"]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"✔ Directory: {d}/")

    # 2. Define File Contents
    
    requirements_txt = """
opencv-python>=4.8.0
ultralytics>=8.0.0
supervision>=0.18.0
ollama>=0.1.0
numpy
"""

    modules_init_py = ""

    vision_py = """
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv

class VisionSystem:
    def __init__(self, model_path='yolov8n.pt'):
        print(f"Loading Vision Model: {model_path}...")
        self.model = YOLO(model_path)
        
        # Initialize Tracker (ByteTrack)
        self.tracker = sv.ByteTrack(track_thresh=0.25, track_buffer=30, match_thresh=0.8, frame_rate=30)
        
        # Annotators
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()

    def detect_and_track(self, frame):
        # 1. Detect
        results = self.model(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)

        # 2. Track
        detections = self.tracker.update_with_detections(detections)

        # 3. Annotate
        labels = [
            f"#{tracker_id} {self.model.model.names[class_id]}"
            for class_id, tracker_id
            in zip(detections.class_id, detections.tracker_id)
        ]

        annotated_frame = self.box_annotator.annotate(
            scene=frame.copy(),
            detections=detections
        )
        annotated_frame = self.label_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
            labels=labels
        )

        return annotated_frame, detections
"""

    intelligence_py = """
import ollama
import cv2
import os
import time

class BrainSystem:
    def __init__(self, model_name='llava'):
        self.model_name = model_name
        self.last_analysis_time = 0
        self.cooldown = 5  # Seconds between AI checks

    def analyze_scene(self, frame, detections, current_time):
        # Heuristic: Only analyze if objects are present
        if len(detections) == 0:
            return None

        # Rate Limiting
        if (current_time - self.last_analysis_time) < self.cooldown:
            return None

        print("⚡ Triggering AI Analysis...")
        self.last_analysis_time = current_time

        # Save temp frame for Ollama
        temp_path = "temp_frame.jpg"
        cv2.imwrite(temp_path, frame)

        prompt = (
            "You are a CCTV security AI. Briefly describe the behavior of the people "
            "and vehicles in this image. Flag any suspicious activity like loitering, "
            "running, or fighting. Be concise."
        )

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': [temp_path]
                }]
            )
            
            description = response['message']['content']
            
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
            return description

        except Exception as e:
            print(f"Error during AI analysis: {e}")
            return None
"""

    main_py = """
import cv2
import time
import os
from modules.vision import VisionSystem
from modules.intelligence import BrainSystem

# CONFIGURATION
VIDEO_SOURCE = "input/sample.mp4" # Path to video or 0 for webcam
OUTPUT_PATH = "output/result.avi"
USE_LLAVA = True  # Set False if Ollama is not installed

def main():
    # Check for video file
    if not os.path.exists(VIDEO_SOURCE) and VIDEO_SOURCE != 0:
        print(f"WARNING: Video file '{VIDEO_SOURCE}' not found.")
        print("Please place a video in the 'input' folder and rename it 'sample.mp4'.")
        return

    # 1. Initialize Systems
    vision = VisionSystem()
    brain = BrainSystem() if USE_LLAVA else None

    # 2. Setup Video Capture
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"Error: Could not open video source {VIDEO_SOURCE}")
        return

    # 3. Setup Video Writer
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    print(f"Processing started... Press 'Q' to stop.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        current_time = time.time()
        
        # --- PHASE 1: SEE (Detection & Tracking) ---
        annotated_frame, detections = vision.detect_and_track(frame)

        # --- PHASE 2: THINK (LLM Analysis) ---
        ai_summary = None
        if brain:
            ai_summary = brain.analyze_scene(frame, detections, current_time)

        # --- PHASE 3: REPORT ---
        if ai_summary:
            print(f"\\n[AI REPORT @ {time.strftime('%H:%M:%S')}]:\\n{ai_summary}\\n" + "-"*50)
            cv2.putText(annotated_frame, "AI Analysis Active: See Console", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("CCTV AI System", annotated_frame)
        out.write(annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Processing complete. Video saved to", OUTPUT_PATH)

if __name__ == "__main__":
    main()
"""

    readme_md = """
# Local CCTV AI System

1. Install requirements: `pip install -r requirements.txt`
2. Install Ollama: https://ollama.com
3. Pull Model: `ollama pull llava`
4. Put video in `input/sample.mp4`
5. Run: `python main.py`
"""

    # 3. Write Files
    create_file("requirements.txt", requirements_txt)
    create_file("modules/__init__.py", modules_init_py)
    create_file("modules/vision.py", vision_py)
    create_file("modules/intelligence.py", intelligence_py)
    create_file("main.py", main_py)
    create_file("README.md", readme_md)

    print("\n✅ Project setup complete!")
    print("1. Add a video file to the 'input/' folder and name it 'sample.mp4'.")
    print("2. Run 'pip install -r requirements.txt'")
    print("3. Run 'python main.py'")

if __name__ == "__main__":
    main()