import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose_engine = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def calculate_pose_angles(landmarks):
    """Calculate key body angles for panic detection"""
    try:
        # Get key landmarks
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        
        # Calculate body tilt (falling detection)
        shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
        hip_center_y = (left_hip.y + right_hip.y) / 2
        body_vertical_ratio = abs(shoulder_center_y - hip_center_y)
        
        # Calculate leg angles (running/panic posture detection)
        def calculate_angle(p1, p2, p3):
            """Calculate angle between three points"""
            v1 = np.array([p1.x - p2.x, p1.y - p2.y])
            v2 = np.array([p3.x - p2.x, p3.y - p2.y])
            
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            return np.degrees(angle)
        
        # Knee angles (bent knees indicate running/panic)
        left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
        
        # Arm position (raised arms indicate panic/distress)
        left_arm_raised = left_shoulder.y > landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y
        right_arm_raised = right_shoulder.y > landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y
        
        return {
            'body_vertical_ratio': body_vertical_ratio,
            'left_knee_angle': left_knee_angle,
            'right_knee_angle': right_knee_angle,
            'arms_raised': left_arm_raised or right_arm_raised,
            'both_arms_raised': left_arm_raised and right_arm_raised
        }
        
    except Exception as e:
        print(f"Pose angle calculation error: {e}")
        return None

def detect_panic_postures(pose_data):
    """Detect panic-related body postures"""
    if pose_data is None:
        return 0.2
    
    panic_score = 0.0
    
    # 1. Falling detection (horizontal body)
    if pose_data['body_vertical_ratio'] < 0.15:  # Very horizontal
        panic_score += 0.8
    elif pose_data['body_vertical_ratio'] < 0.25:  # Leaning significantly
        panic_score += 0.4
    
    # 2. Running posture (bent knees)
    avg_knee_angle = (pose_data['left_knee_angle'] + pose_data['right_knee_angle']) / 2
    if avg_knee_angle < 140:  # Significantly bent knees
        panic_score += 0.3
    elif avg_knee_angle < 160:  # Moderately bent knees
        panic_score += 0.2
    
    # 3. Distress gestures (raised arms)
    if pose_data['both_arms_raised']:
        panic_score += 0.4
    elif pose_data['arms_raised']:
        panic_score += 0.2
    
    # 4. Asymmetric posture (imbalance, stumbling)
    knee_angle_diff = abs(pose_data['left_knee_angle'] - pose_data['right_knee_angle'])
    if knee_angle_diff > 30:  # Significant asymmetry
        panic_score += 0.3
    
    return min(panic_score, 1.0)

def pose_score(frame, person_boxes):
    """
    Advanced Pose Analysis for Crowd Panic Detection
    
    Args:
        frame: Input video frame
        person_boxes: List of person bounding boxes from YOLO
    
    Returns:
        tuple: (pose_score, pose_results) - panic pose score and MediaPipe results
    """
    if frame is None or len(person_boxes) == 0:
        return 0.2, None
    
    try:
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        pose_scores = []
        all_pose_results = []
        
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
            person_roi = rgb_frame[y1:y2, x1:x2]
            
            if person_roi.size == 0:
                continue
            
            # Process pose estimation
            results = pose_engine.process(person_roi)
            
            if results.pose_landmarks:
                all_pose_results.append(results)
                
                # Calculate pose angles and detect panic postures
                pose_data = calculate_pose_angles(results.pose_landmarks.landmark)
                person_pose_score = detect_panic_postures(pose_data)
                
                # Weight by person size (larger persons are more reliable)
                person_area = (x2 - x1) * (y2 - y1)
                person_weight = min(person_area / (w * h * 0.1), 1.0)  # Normalize by 10% of frame
                
                weighted_score = person_pose_score * person_weight
                pose_scores.append(weighted_score)
        
        # Calculate overall crowd pose score
        if pose_scores:
            # Use maximum pose score (most alarming person)
            max_pose_score = max(pose_scores)
            
            # Boost if multiple people show panic postures
            if len(pose_scores) > 1:
                high_scores = [s for s in pose_scores if s > 0.5]
                if len(high_scores) > 1:
                    crowd_panic_boost = min(len(high_scores) * 0.1, 0.3)
                    max_pose_score += crowd_panic_boost
            
            return min(max_pose_score, 1.0), all_pose_results[0] if all_pose_results else None
        else:
            # No pose detected, return default score
            return 0.2, None
            
    except Exception as e:
        print(f"Pose analysis error: {e}")
        return 0.2, None

def draw_pose_landmarks(frame, pose_results):
    """Draw pose landmarks on frame for visualization"""
    if pose_results and pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, 
            pose_results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
        )
    return frame

def get_pose_statistics():
    """Get pose detection statistics"""
    return {
        "pose_engine": "MediaPipe Pose",
        "model_complexity": 1,
        "detection_confidence": 0.5,
        "tracking_confidence": 0.5,
        "landmarks_count": 33,
        "panic_indicators": [
            "Falling (horizontal body)",
            "Running posture (bent knees)",
            "Distress gestures (raised arms)",
            "Asymmetric posture (imbalance)"
        ]
    }
