import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from PIL import Image

# CNN Architecture for Emotion Recognition (FER-2013 compatible)
class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling and dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # Convolutional layers with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout1(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout2(x)
        
        # Flatten for fully connected layers
        x = x.view(-1, 128 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return x

# Initialize model and face detector
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
emotion_model = EmotionCNN().to(device)

# Load pre-trained weights if available
model_path = "models/emotion/emotion_model.pth"
if os.path.exists(model_path):
    try:
        emotion_model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ Loaded emotion model from {model_path}")
        model_loaded = True
    except Exception as e:
        print(f"⚠️ Could not load emotion model: {e}")
        model_loaded = False
else:
    print("⚠️ Emotion model not found, using simulated values")
    model_loaded = False

emotion_model.eval()

# Face detection cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Emotion labels
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def preprocess_face(face_img):
    """Preprocess face image for emotion recognition"""
    if face_img is None or face_img.size == 0:
        return None
    
    # Resize to 48x48 (FER-2013 standard)
    face_resized = cv2.resize(face_img, (48, 48))
    
    # Normalize pixel values
    face_normalized = face_resized.astype('float32') / 255.0
    
    # Convert to tensor
    face_tensor = torch.from_numpy(face_normalized).unsqueeze(0).unsqueeze(0).to(device)
    
    return face_tensor

def detect_emotion_in_face(face_img):
    """Detect emotion in a single face"""
    if not model_loaded:
        # Return simulated emotion scores if model not available
        return np.random.dirichlet(np.ones(7))  # Random probability distribution
    
    face_tensor = preprocess_face(face_img)
    if face_tensor is None:
        return np.zeros(7)
    
    try:
        with torch.no_grad():
            outputs = emotion_model(face_tensor)
            probabilities = F.softmax(outputs, dim=1)
            return probabilities.cpu().numpy()[0]
    except Exception as e:
        print(f"Emotion detection error: {e}")
        return np.zeros(7)

def emotion_score(frame, person_boxes):
    """
    Advanced Emotion Analysis for Crowd Panic Detection
    
    Args:
        frame: Input video frame
        person_boxes: List of person bounding boxes from YOLO
    
    Returns:
        float: Panic emotion score (0.0 to 1.0)
    """
    if frame is None or len(person_boxes) == 0:
        return 0.1  # Default low emotion score
    
    try:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        panic_emotions = []
        face_count = 0
        
        # Analyze each detected person
        for box in person_boxes:
            x1, y1, x2, y2 = map(int, box)
            
            # Ensure coordinates are within frame bounds
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Extract person region
            person_roi = gray_frame[y1:y2, x1:x2]
            
            if person_roi.size == 0:
                continue
            
            # Detect faces within person region
            faces = face_cascade.detectMultiScale(
                person_roi, 
                scaleFactor=1.1, 
                minNeighbors=5,
                minSize=(20, 20)  # Minimum face size
            )
            
            # Analyze each detected face
            for (fx, fy, fw, fh) in faces:
                face_count += 1
                
                # Extract face region
                face_roi = person_roi[fy:fy+fh, fx:fx+fw]
                
                if face_roi.size == 0:
                    continue
                
                # Get emotion probabilities
                emotion_probs = detect_emotion_in_face(face_roi)
                
                # Calculate panic-related emotion score
                # Focus on Fear (index 2) and Anger (index 0)
                fear_prob = emotion_probs[2]
                anger_prob = emotion_probs[0]
                
                # Weight by face size (larger faces are more reliable)
                face_area = fw * fh
                face_weight = min(face_area / 1600, 1.0)  # Normalize by 40x40 face
                
                panic_emotion = (fear_prob + anger_prob) * face_weight
                panic_emotions.append(panic_emotion)
        
        # Calculate overall crowd emotion score
        if panic_emotions:
            # Use weighted average with confidence boosting for multiple faces
            avg_emotion = np.mean(panic_emotions)
            
            # Boost confidence if multiple faces show similar emotions
            if len(panic_emotions) > 1:
                emotion_std = np.std(panic_emotions)
                consistency_boost = max(0, (0.3 - emotion_std) / 0.3) * 0.2
                avg_emotion += consistency_boost
            
            # Boost score if many faces detected (crowd effect)
            crowd_boost = min(face_count / 10.0, 0.3) * 0.1
            avg_emotion += crowd_boost
            
            return min(avg_emotion, 1.0)
        else:
            # No faces detected, return low emotion score
            return 0.1
            
    except Exception as e:
        print(f"Emotion analysis error: {e}")
        return 0.1

def get_emotion_statistics():
    """Get emotion detection statistics"""
    return {
        "model_loaded": model_loaded,
        "device": str(device),
        "emotion_classes": EMOTION_LABELS,
        "face_detector": "Haar Cascade",
        "model_architecture": "Custom CNN (FER-2013 compatible)"
    }
