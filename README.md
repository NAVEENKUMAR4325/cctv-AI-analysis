
# 📹 CCTV Object & Activity Detection (CPU Optimized)

A real-time **Computer Vision system** for CCTV video analysis — capable of detecting people and vehicles, tracking their movement with unique IDs, and generating intelligent text descriptions of the scene.

This project is **CPU Optimized** — it avoids heavy GPU-based models and uses lightweight transformers (BLIP) so it can run smoothly on a normal laptop.

---

## 🔄 How the System Works (Full Pipeline Flow)

The system follows a simple but powerful 3-stage workflow:

---

### **1️⃣ The "Eyes" – Vision Module (YOLO + ByteTrack)**

This module reads each frame and extracts meaningful visual information.

#### ✔ Object Detection (YOLOv8)
- Scans every frame for objects.
- Confidence threshold increased to **0.45** for fewer false positives.
- Ignores "plants" and similar static objects.

#### ✔ Noise Filtering
- Removes micro-detections smaller than **500 pixels**.
- Prevents shadows, tiles, or noise from triggering events.

#### ✔ Object Tracking (ByteTrack)
- Every person/vehicle gets a **unique ID** (e.g., `ID 1`, `ID 2`).
- If an object disappears and reappears, the ID is preserved.

---

### **2️⃣ The "Brain" – Intelligence Module (BLIP Transformer)**

Runs **only when Vision finds a valid object** to reduce CPU usage.

#### ✔ Frame Captioning (BLIP)
- Generates a short description:
  - “A person is walking down the stairs”
  - “A man is entering the building”
  - etc.

#### ✔ Anti-Hallucination Layer
Modifications added to prevent BLIP hallucinations:
- Forced prompt anchoring (“A person is…”)
- Repetition penalty to stop loops like “cci cci cci…”
- Hard 20-word limit to prevent storytelling (“Mexico”, “2019”, etc.)

#### ✔ Change Detection
- If the new description is same as previous → **no new report printed**.

---

### **3️⃣ Output Layer**

#### 🎥 Video Output  
- Shows bounding boxes  
- Shows object IDs  
- Saved to: `output/result.avi`

#### 📝 Text Output  
Every time a new event happens, you see:


[AI REPORT @ 14:06:18]
OBSERVATION: A person is walking up the stairs.
OBJECTS DETECTED: 1 entities tracked.
STATUS: New Event Detected.

````

---

## 🚀 Installation & Usage

### **1. Install Dependencies**
```bash
pip install -r requirements.txt
````

### **2. Prepare Your CCTV Video**

1. Place your `.mp4` file inside the `input/` folder.
2. Rename it to:

```
sample.mp4
```

### **3. Run the System**

```bash
python main.py
```

> The first run may take 1–2 minutes while the BLIP model downloads (~900MB).
> All future runs will be instant.

---

## 📁 Project Structure

```
CCTV-AI/
│── main.py                     # Main execution file
│── requirements.txt
│
├── modules/
│   ├── vision.py               # YOLO + ByteTrack + filters
│   └── intelligence.py         # BLIP + anti-hallucination logic
│
├── input/
│   └── sample.mp4              # Your CCTV footage goes here
│
└── output/
    ├── result.avi              # Final processed video
    └── logs.txt                # Optional
```

---

## ⚙️ Configuration (Optional)

### **1. Make Detection Stricter**

Open `modules/vision.py`
Find:

```python
conf=0.45
```

Increase to:

```python
conf=0.60
```

### **2. Reduce AI Report Frequency**

Open `modules/intelligence.py`
Find:

```python
self.cooldown = 4
```

Set to a larger number (e.g., `8`).

### **3. Track More Object Types**

Open `modules/vision.py`
Find:

```python
self.target_classes = ["person", "car", "motorbike"]
```

Add/remove classes as needed.

---

## 🧠 Technologies Used

| Component       | Purpose                            |
| --------------- | ---------------------------------- |
| **YOLOv8**      | Object Detection (humans/vehicles) |
| **ByteTrack**   | Persistent ID tracking             |
| **BLIP (HF)**   | Lightweight image captioning (CPU) |
| **OpenCV**      | Video frame processing             |
| **Supervision** | Detection/tracking utilities       |

---

## ⭐ Key Features

* Runs on **any laptop** (no GPU required)
* Accurate object detection
* No hallucinations (custom prompt engineering)
* Real-time CCTV analysis
* Clean logs + annotated output video

---

## 📜 License

This project is open for educational and non-commercial use.

---

## 🔧 Author

Developed by **Naveen Kumar E**

```

