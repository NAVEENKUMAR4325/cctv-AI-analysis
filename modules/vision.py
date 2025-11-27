import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv

class VisionSystem:
    def __init__(self, model_path='yolov8n.pt'):
        print(f"Loading Vision Model: {model_path}...")
        self.model = YOLO(model_path)
        
        # Initialize Tracker (ByteTrack)
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=30,
            minimum_matching_threshold=0.8,
            frame_rate=30
        )
        
        # Annotators
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()

        # CLASS FILTER: Only track these IDs
        # 0=Person, 1=Bicycle, 2=Car, 3=Motorcycle, 5=Bus, 7=Truck
        # Potted Plant (58) is EXCLUDED.
        self.target_classes = [0, 1, 2, 3, 5, 7]

    def detect_and_track(self, frame):
        # 1. Detect
        # Added 'conf=0.45' to ignore weak detections (ghosts on tiles)
        results = self.model(frame, verbose=False, conf=0.45)[0]
        detections = sv.Detections.from_ultralytics(results)

        # 2. FILTER: Classes + Minimum Size
        # Filter A: Only track people and vehicles (remove plants)
        detections = detections[np.isin(detections.class_id, self.target_classes)]
        
        # Filter B: Remove tiny objects (noise/glitches often found on textured floors)
        # Any box smaller than 500 pixels is ignored
        detections = detections[detections.area > 500]

        # 3. Track (Only remaining valid objects)
        detections = self.tracker.update_with_detections(detections)

        # 4. Annotate
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