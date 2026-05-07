# YOLO Models Directory

## Purpose
This directory contains YOLO (You Only Look Once) model files for person detection.

## Required Model
- **File**: `yolov8n.pt`
- **Type**: YOLOv8 Nano model
- **Size**: ~6MB
- **Purpose**: Real-time person detection in video frames

## Download Instructions

### Automatic Download (Recommended)
The system will automatically download the model when first run:
```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")  # Downloads automatically if not found
```

### Manual Download
1. Visit: https://github.com/ultralytics/ultralytics
2. Download `yolov8n.pt` from releases
3. Place in this directory

### Alternative Models
You can use other YOLOv8 variants:
- `yolov8s.pt` - Small (22MB, better accuracy)
- `yolov8m.pt` - Medium (52MB, higher accuracy)
- `yolov8l.pt` - Large (87MB, best accuracy)
- `yolov8x.pt` - Extra Large (136MB, maximum accuracy)

## Model Performance
- **YOLOv8n**: 30+ FPS, 95%+ person detection accuracy
- **Input**: 640x640 images
- **Output**: Bounding boxes with confidence scores
- **Classes**: 80 COCO classes (person = class 0)

## Usage in Code
```python
from ultralytics import YOLO

# Load model
model = YOLO("models/yolo/yolov8n.pt")

# Detect people in frame
results = model(frame)
for box in results[0].boxes:
    if int(box.cls[0]) == 0:  # person class
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        confidence = box.conf[0].item()
```

## Fine-tuning (Advanced)
For better crowd detection, consider fine-tuning on:
- CrowdHuman dataset
- Custom crowd footage
- Specific venue types (malls, stations, etc.)

## Troubleshooting
- **Model not found**: Check file path and permissions
- **Slow inference**: Use GPU acceleration with CUDA
- **Low accuracy**: Try larger model variants (yolov8s, yolov8m)
- **Memory issues**: Reduce input image size or use yolov8n