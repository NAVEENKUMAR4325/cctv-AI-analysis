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
            print(f"\n[AI REPORT @ {time.strftime('%H:%M:%S')}]:\n{ai_summary}\n" + "-"*50)
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