#!/usr/bin/env python3
"""
Real-time Age and Gender Detection
Uses YOLO for person detection (works at distance) + DeepFace for demographics
"""

import cv2
import numpy as np
from deepface import DeepFace
from ultralytics import YOLO
import threading
import time

class RealtimeDetector:
    def __init__(self):
        self.running = False
        self.current_frame = None
        self.detections = []
        self.lock = threading.Lock()
        # Load YOLO model for person detection
        try:
            self.yolo_model = YOLO('yolov8n.pt')  # Nano model for speed
            print("✅ YOLO model loaded for person detection")
        except:
            self.yolo_model = None
            print("⚠️ YOLO not available, using DeepFace only")
        
    def analyze_frame(self, frame):
        """Analyze frame for people using YOLO + DeepFace"""
        try:
            detections = []

            # --- Fix dark frames: normalize brightness ---
            brightness = np.mean(frame)
            print(f"🌟 Frame brightness: {brightness:.1f}")
            if brightness < 60:
                # Auto-brighten the frame so YOLO can see
                frame = cv2.convertScaleAbs(frame, alpha=2.5, beta=40)
                print(f"🔆 Brightened frame (was {brightness:.1f})")

            # Use YOLO to detect people
            if self.yolo_model:
                results = self.yolo_model(frame, conf=0.10, iou=0.4, verbose=False)
                print(f"🔍 YOLO processing frame...")

                for result in results:
                    boxes = result.boxes
                    print(f"📦 YOLO found {len(boxes)} boxes total")
                    for box in boxes:
                        cls = int(box.cls[0])
                        conf_val = float(box.conf[0])
                        if cls != 0:  # Only persons
                            continue

                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)

                        if w < 20 or h < 30:
                            continue

                        print(f"✅ Person: conf={conf_val:.2f}, size={w}x{h} at ({x},{y})")

                        age = 25
                        gender = 'Female'
                        emotion = 'neutral'

                        try:
                            person_roi = frame[y:y+h, x:x+w]
                            if person_roi.size > 0 and w > 60 and h > 80:
                                face_result = DeepFace.analyze(
                                    person_roi,
                                    actions=['age', 'gender', 'emotion'],
                                    enforce_detection=False,
                                    detector_backend='opencv',
                                    silent=True
                                )
                                if isinstance(face_result, list):
                                    face_result = face_result[0]

                                age = int(face_result.get('age', 25))
                                gender_data = face_result.get('gender', {})
                                if isinstance(gender_data, dict):
                                    man_conf = gender_data.get('Man', 0)
                                    woman_conf = gender_data.get('Woman', 0)
                                    gender = 'Male' if man_conf > woman_conf and man_conf > 55 else 'Female'
                                else:
                                    gender_raw = face_result.get('dominant_gender', 'Woman')
                                    gender = 'Male' if gender_raw.lower() in ['man', 'male'] else 'Female'

                                emotion = face_result.get('dominant_emotion', 'neutral')
                        except Exception:
                            pass

                        detections.append({
                            'region': {'x': x, 'y': y, 'w': w, 'h': h},
                            'age': age,
                            'gender': gender,
                            'emotion': emotion,
                            'gender_confidence': {}
                        })

                print(f"YOLO detected {len(detections)} people")

            # Fallback: use DeepFace directly on full frame
            if len(detections) == 0:
                print("⚠️ YOLO found nothing, trying DeepFace on full frame...")
                try:
                    df_results = DeepFace.analyze(
                        frame,
                        actions=['age', 'gender', 'emotion'],
                        enforce_detection=False,
                        detector_backend='opencv',
                        silent=True
                    )
                    if not isinstance(df_results, list):
                        df_results = [df_results]

                    for r in df_results:
                        region = r.get('region', {})
                        rx = region.get('x', 0)
                        ry = region.get('y', 0)
                        rw = region.get('w', frame.shape[1])
                        rh = region.get('h', frame.shape[0])

                        gender_raw = r.get('dominant_gender', 'Woman')
                        gender = 'Male' if gender_raw.lower() in ['man', 'male'] else 'Female'

                        detections.append({
                            'region': {'x': rx, 'y': ry, 'w': rw, 'h': rh},
                            'age': int(r.get('age', 25)),
                            'gender': gender,
                            'emotion': r.get('dominant_emotion', 'neutral'),
                            'gender_confidence': r.get('gender', {})
                        })
                    print(f"✅ DeepFace fallback found {len(detections)} people")
                except Exception as e:
                    print(f"⚠️ DeepFace fallback failed: {e}")

            return detections

        except Exception as e:
            print(f"Detection error: {e}")
            import traceback
            traceback.print_exc()
            return []

    
    def draw_detections(self, frame, detections):
        """Draw bounding boxes and labels on frame - labels above head"""
        for idx, det in enumerate(detections):
            region = det['region']
            if not region:
                continue
                
            x, y, w, h = region['x'], region['y'], region['w'], region['h']
            age = det['age']
            gender = det['gender']
            emotion = det['emotion']
            
            # Determine color based on gender
            if gender.lower() == 'man' or gender.lower() == 'male':
                color = (255, 0, 0)  # Blue for male
                gender_label = 'Male'
            else:
                color = (255, 105, 180)  # Pink for female
                gender_label = 'Female'
            
            # Draw bounding box around person
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
            
            # Create label text - Gender and Age
            label_text = f"{gender_label}, {int(age)} years"
            
            # Calculate label size
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
            
            # Position label ABOVE the head (above the bounding box)
            label_x = x + (w - text_width) // 2  # Center horizontally
            label_y = y - 15  # Above the box
            
            # Draw background rectangle for label (above head)
            padding = 8
            cv2.rectangle(frame, 
                         (label_x - padding, label_y - text_height - padding),
                         (label_x + text_width + padding, label_y + baseline),
                         color, -1)
            
            # Draw white border around label
            cv2.rectangle(frame, 
                         (label_x - padding, label_y - text_height - padding),
                         (label_x + text_width + padding, label_y + baseline),
                         (255, 255, 255), 2)
            
            # Draw label text in white
            cv2.putText(frame, label_text, (label_x, label_y), 
                       font, font_scale, (255, 255, 255), thickness)
            
            # Optional: Draw emotion below the box (at feet level)
            emotion_text = f"{emotion}"
            cv2.putText(frame, emotion_text, (x + 5, y + h + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw summary at top of frame
        if detections:
            summary = f"Detected: {len(detections)} people"
            cv2.rectangle(frame, (10, 10), (350, 50), (0, 0, 0), -1)
            cv2.putText(frame, summary, (20, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return frame
    
    def get_statistics(self, detections):
        """Calculate statistics from detections"""
        if not detections:
            return {
                'total_people': 0,
                'male_count': 0,
                'female_count': 0,
                'children': 0,
                'adults': 0,
                'elderly': 0,
                'avg_age': 0,
                'panic_emotions': 0
            }
        
        male_count = sum(1 for d in detections if d['gender'].lower() in ['man', 'male'])
        female_count = len(detections) - male_count
        
        ages = [d['age'] for d in detections]
        children = sum(1 for age in ages if age < 18)
        elderly = sum(1 for age in ages if age > 60)
        adults = len(ages) - children - elderly
        
        panic_emotions = sum(1 for d in detections if d['emotion'] in ['fear', 'angry', 'sad'])
        
        return {
            'total_people': len(detections),
            'male_count': male_count,
            'female_count': female_count,
            'children': children,
            'adults': adults,
            'elderly': elderly,
            'avg_age': int(np.mean(ages)) if ages else 0,
            'panic_emotions': panic_emotions
        }

# Global detector instance
detector = RealtimeDetector()

def process_webcam_frame(frame_data):
    """Process a frame from webcam"""
    try:
        # Decode base64 frame
        import base64
        img_data = base64.b64decode(frame_data.split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        print(f"📷 Received frame: {frame.shape}, brightness={np.mean(frame):.1f}")
        
        # Analyze frame
        detections = detector.analyze_frame(frame)
        
        # Get statistics
        stats = detector.get_statistics(detections)
        
        # Format detections for frontend (with coordinates for drawing)
        formatted_detections = []
        for det in detections:
            region = det.get('region', {})
            if region:
                gender = det['gender']
                if gender.lower() in ['man', 'male']:
                    gender_label = 'Male'
                else:
                    gender_label = 'Female'
                
                formatted_detections.append({
                    'x': region.get('x', 0),
                    'y': region.get('y', 0),
                    'width': region.get('w', 0),
                    'height': region.get('h', 0),
                    'gender': gender_label,
                    'age': int(det['age']),
                    'emotion': det['emotion']
                })
        
        result = {
            'stats': stats,
            'detections': formatted_detections
        }
        
        print(f"✅ Returning {len(formatted_detections)} detections to frontend")
        print(f"📊 Stats: {stats['total_people']} people ({stats['male_count']}M, {stats['female_count']}F)")
        
        return result
        
    except Exception as e:
        print(f"Frame processing error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Test with webcam
    cap = cv2.VideoCapture(0)
    
    print("Starting real-time detection...")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Analyze every 5 frames for performance
        if cap.get(cv2.CAP_PROP_POS_FRAMES) % 5 == 0:
            detections = detector.analyze_frame(frame)
            frame = detector.draw_detections(frame, detections)
            stats = detector.get_statistics(detections)
            
            print(f"People: {stats['total_people']} | "
                  f"M:{stats['male_count']} F:{stats['female_count']} | "
                  f"Avg Age: {stats['avg_age']}")
        
        cv2.imshow('Real-time Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
