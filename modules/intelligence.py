from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import cv2
import torch

class BrainSystem:
    def __init__(self, model_name='ignored'):
        print("⚡ Loading BLIP Model (Optimized for CPU)...")
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        
        self.last_analysis_time = 0
        self.cooldown = 4
        self.last_desc = ""

    def analyze_scene(self, frame, detections, current_time):
        if len(detections) == 0: return None
        if (current_time - self.last_analysis_time) < self.cooldown: return None
            
        self.last_analysis_time = current_time
        print("⚡ Analyzing Frame (CPU Fast Mode)...")

        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            raw_image = Image.fromarray(rgb_frame)

            # --- FIX 1: ANCHOR PROMPTS ---
            # Check what YOLO found (Person vs Vehicle) and force the sentence start.
            # This stops the AI from hallucinating random locations.
            class_ids = detections.class_id
            if 0 in class_ids:  # 0 is Person
                start_text = "A person is"
            elif 2 in class_ids or 3 in class_ids: # Car or Motorcycle
                start_text = "A vehicle is"
            else:
                start_text = "A photo of"

            inputs = self.processor(raw_image, text=start_text, return_tensors="pt")
            
            # --- FIX 2: GENERATION CONSTRAINTS ---
            out = self.model.generate(
                **inputs, 
                max_new_tokens=20,          # Stop it before it invents a date
                repetition_penalty=1.5,     # Fixes "cci cci cci"
                min_length=5
            )
            description = self.processor.decode(out[0], skip_special_tokens=True)

            # Filter Duplicates
            if description.strip().lower() == self.last_desc.strip().lower():
                return None
            
            self.last_desc = description

            report = (
                f"OBSERVATION: {description.capitalize()}.\n"
                f"STATUS: Activity Detected."
            )
            return report

        except Exception as e:
            print(f"Error: {e}")
            return None