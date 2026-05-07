import cv2
import os
import numpy as np
from datetime import datetime
import threading
import time

class VideoEvidenceCapture:
    def __init__(self, buffer_size=150):  # 5 seconds at 30fps
        self.buffer_size = buffer_size
        self.frame_buffer = []
        self.recording = False
        self.evidence_dir = "data/event_clips"
        os.makedirs(self.evidence_dir, exist_ok=True)
    
    def add_frame(self, frame):
        """Add frame to circular buffer"""
        if len(self.frame_buffer) >= self.buffer_size:
            self.frame_buffer.pop(0)
        self.frame_buffer.append(frame.copy())
    
    def save_event_clip(self, camera_id="CAM-01", duration_after=2.0):
        """Save video clip when panic is detected"""
        if len(self.frame_buffer) < 30:  # Need at least 1 second of footage
            return None, None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = f"panic_event_{camera_id}_{timestamp}.mp4"
        screenshot_filename = f"panic_screenshot_{camera_id}_{timestamp}.jpg"
        
        video_path = os.path.join(self.evidence_dir, video_filename)
        screenshot_path = os.path.join(self.evidence_dir, screenshot_filename)
        
        try:
            # Get frame dimensions
            height, width = self.frame_buffer[0].shape[:2]
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))
            
            # Write buffered frames (pre-event)
            for frame in self.frame_buffer:
                out.write(frame)
            
            # Save screenshot of the panic moment
            cv2.imwrite(screenshot_path, self.frame_buffer[-1])
            
            out.release()
            
            print(f"📹 Video evidence saved: {video_filename}")
            print(f"📸 Screenshot saved: {screenshot_filename}")
            
            return video_path, screenshot_path
            
        except Exception as e:
            print(f"❌ Error saving evidence: {e}")
            return None, None

# Global evidence capture instance
evidence_capture = VideoEvidenceCapture()

def save_event(camera_id="CAM-01"):
    """Save panic event with video evidence"""
    video_path, screenshot_path = evidence_capture.save_event_clip(camera_id)
    
    if video_path and screenshot_path:
        print("📁 Panic event clip stored with video evidence")
        return {
            'video_path': video_path,
            'screenshot_path': screenshot_path,
            'status': 'success'
        }
    else:
        print("📁 Panic event logged (no video available)")
        return {
            'video_path': None,
            'screenshot_path': None,
            'status': 'no_video'
        }

def add_frame_to_buffer(frame):
    """Add frame to evidence buffer"""
    evidence_capture.add_frame(frame)
