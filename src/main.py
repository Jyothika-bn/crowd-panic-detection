import cv2
import sys
import os
import time
import threading
from datetime import datetime

# Add src to python path to allow imports
sys.path.append(os.path.join(os.path.dirname(__file__)))

from video_capture.capture import get_video
from preprocessing.preprocess import preprocess
from detection.person_detector import detect_people, draw_person_detections, get_crowd_density
from tracking.tracker import track_people
from motion_analysis.optical_flow import motion_score, draw_optical_flow, reset_motion_analysis
from pose_estimation.pose_estimator import pose_score, draw_pose_landmarks
from emotion_recognition.emotion_detector import emotion_score
from fusion.panic_score import compute_panic_detailed
from alerting.alert import alert, get_alert_statistics
from evidence.storage import save_event, add_frame_to_buffer
from utils.config import PANIC_THRESHOLD

class CrowdPanicDetectionSystem:
    def __init__(self):
        self.running = False
        self.frame_count = 0
        self.start_time = None
        self.stats = {
            'total_frames': 0,
            'total_people_detected': 0,
            'total_alerts': 0,
            'avg_processing_time': 0,
            'fps': 0
        }
    
    def process_frame(self, frame, camera_id="CAM-01", show_visualization=True):
        """
        Process a single frame through the complete AI pipeline
        
        Args:
            frame: Input video frame
            camera_id: Camera identifier
            show_visualization: Whether to draw visualizations
        
        Returns:
            dict: Processing results and panic analysis
        """
        if frame is None:
            return None
        
        frame_start_time = time.time()
        
        try:
            # 1. Preprocessing
            processed_frame = preprocess(frame.copy())
            
            # 2. Person Detection
            people_boxes = detect_people(processed_frame)
            crowd_info = get_crowd_density(people_boxes, processed_frame.shape)
            
            # 3. Tracking (currently pass-through, can be enhanced)
            tracks = track_people(people_boxes)
            
            # 4. Motion Analysis
            motion = motion_score(processed_frame, crowd_info['person_count'])
            
            # 5. Pose Estimation
            pose, pose_results = pose_score(processed_frame, people_boxes)
            
            # 6. Emotion Recognition
            emotion = emotion_score(processed_frame, people_boxes)
            
            # 7. Advanced Feature Fusion
            panic_analysis = compute_panic_detailed(motion, pose, emotion)
            panic = panic_analysis['panic_score']
            
            # 8. Visualization (if enabled)
            if show_visualization:
                # Draw person detections
                processed_frame = draw_person_detections(processed_frame, people_boxes)
                
                # Draw pose landmarks
                if pose_results:
                    processed_frame = draw_pose_landmarks(processed_frame, pose_results)
                
                # Add information overlay
                self._draw_info_overlay(processed_frame, panic_analysis, crowd_info, camera_id)
            
            # Add frame to evidence buffer (for video capture)
            add_frame_to_buffer(processed_frame)
            
            # 9. Alert Generation
            alert_triggered = False
            video_evidence = None
            if panic > PANIC_THRESHOLD:
                # Capture video evidence
                video_evidence = save_event(camera_id)
                
                alert_data = alert(
                    panic, motion, pose, emotion, camera_id, 
                    detailed_analysis=panic_analysis,
                    video_evidence=video_evidence
                )
                alert_triggered = True
            
            # Update statistics
            processing_time = time.time() - frame_start_time
            self._update_stats(processing_time, len(people_boxes), alert_triggered)
            
            return {
                'frame': processed_frame,
                'panic_analysis': panic_analysis,
                'crowd_info': crowd_info,
                'people_detected': len(people_boxes),
                'alert_triggered': alert_triggered,
                'processing_time': processing_time,
                'camera_id': camera_id
            }
            
        except Exception as e:
            print(f"Frame processing error: {e}")
            return None
    
    def _draw_info_overlay(self, frame, panic_analysis, crowd_info, camera_id):
        """Draw information overlay on frame"""
        h, w = frame.shape[:2]
        
        # Background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Panic score with color coding
        panic_score = panic_analysis['panic_score']
        if panic_score > PANIC_THRESHOLD:
            color = (0, 0, 255)  # Red
            status = "ALERT"
        elif panic_score > 0.5:
            color = (0, 165, 255)  # Orange
            status = "ELEVATED"
        else:
            color = (0, 255, 0)  # Green
            status = "NORMAL"
        
        # Draw text information
        y_offset = 30
        cv2.putText(frame, f"{camera_id} | Status: {status}", (15, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        y_offset += 25
        cv2.putText(frame, f"Panic Score: {panic_score:.3f}", (15, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        y_offset += 20
        cv2.putText(frame, f"Motion: {panic_analysis['feature_scores']['motion']:.2f} | "
                          f"Pose: {panic_analysis['feature_scores']['pose']:.2f} | "
                          f"Emotion: {panic_analysis['feature_scores']['emotion']:.2f}", 
                   (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_offset += 20
        cv2.putText(frame, f"People: {crowd_info['person_count']} | "
                          f"Crowd: {crowd_info['crowd_level']} | "
                          f"Confidence: {panic_analysis['confidence']:.2f}", 
                   (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_offset += 20
        cv2.putText(frame, f"Dominant: {panic_analysis['dominant_feature']} | "
                          f"Alert Level: {panic_analysis['alert_level']}", 
                   (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # FPS counter
        if self.stats['fps'] > 0:
            cv2.putText(frame, f"FPS: {self.stats['fps']:.1f}", (w - 100, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def _update_stats(self, processing_time, people_count, alert_triggered):
        """Update system statistics"""
        self.frame_count += 1
        self.stats['total_frames'] += 1
        self.stats['total_people_detected'] += people_count
        
        if alert_triggered:
            self.stats['total_alerts'] += 1
        
        # Update average processing time
        if self.stats['avg_processing_time'] == 0:
            self.stats['avg_processing_time'] = processing_time
        else:
            self.stats['avg_processing_time'] = (
                self.stats['avg_processing_time'] * 0.9 + processing_time * 0.1
            )
        
        # Calculate FPS
        if self.start_time and self.frame_count > 0:
            elapsed_time = time.time() - self.start_time
            self.stats['fps'] = self.frame_count / elapsed_time
    
    def run_video_analysis(self, video_source, camera_id="CAM-01"):
        """
        Run complete video analysis on a video source
        
        Args:
            video_source: Video file path or camera index
            camera_id: Camera identifier
        """
        print(f"🎬 Starting Crowd Panic Detection Analysis")
        print(f"📹 Camera: {camera_id} | Source: {video_source}")
        print("=" * 60)
        
        # Initialize video capture
        cap = get_video(video_source)
        if cap is None:
            print("❌ Failed to initialize video capture")
            return
        
        # Reset motion analysis state
        reset_motion_analysis()
        
        self.running = True
        self.start_time = time.time()
        self.frame_count = 0
        
        try:
            while self.running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("📹 End of video stream")
                    break
                
                # Process frame
                result = self.process_frame(frame, camera_id, show_visualization=True)
                
                if result:
                    # Display frame
                    cv2.imshow(f'Crowd Panic Detection - {camera_id}', result['frame'])
                    
                    # Print periodic updates
                    if self.frame_count % 30 == 0:  # Every 30 frames
                        self._print_status_update(result)
                
                # Check for exit key
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("🛑 User requested exit")
                    break
                elif key == ord('s'):
                    # Save current frame
                    cv2.imwrite(f"screenshot_{camera_id}_{int(time.time())}.jpg", result['frame'])
                    print("📸 Screenshot saved")
        
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
        
        finally:
            self.running = False
            cap.release()
            cv2.destroyAllWindows()
            self._print_final_statistics()
    
    def _print_status_update(self, result):
        """Print periodic status update"""
        panic_score = result['panic_analysis']['panic_score']
        people_count = result['people_detected']
        alert_level = result['panic_analysis']['alert_level']
        
        print(f"Frame {self.frame_count:4d} | "
              f"People: {people_count:2d} | "
              f"Panic: {panic_score:.3f} | "
              f"Level: {alert_level:8s} | "
              f"FPS: {self.stats['fps']:5.1f}")
    
    def _print_final_statistics(self):
        """Print final system statistics"""
        print("\n" + "=" * 60)
        print("📊 FINAL STATISTICS")
        print("=" * 60)
        print(f"Total Frames Processed: {self.stats['total_frames']}")
        print(f"Total People Detected: {self.stats['total_people_detected']}")
        print(f"Total Alerts Generated: {self.stats['total_alerts']}")
        print(f"Average Processing Time: {self.stats['avg_processing_time']:.3f}s")
        print(f"Average FPS: {self.stats['fps']:.1f}")
        
        if self.stats['total_frames'] > 0:
            alert_rate = (self.stats['total_alerts'] / self.stats['total_frames']) * 100
            people_per_frame = self.stats['total_people_detected'] / self.stats['total_frames']
            print(f"Alert Rate: {alert_rate:.2f}%")
            print(f"Average People per Frame: {people_per_frame:.1f}")
        
        # Get alert statistics from database
        alert_stats = get_alert_statistics(hours=1)
        if alert_stats.get('total_alerts', 0) > 0:
            print(f"\n📈 Recent Alert Statistics (Last Hour):")
            print(f"Total Alerts: {alert_stats['total_alerts']}")
            print(f"Alert Levels: {alert_stats['alert_counts']}")
            if alert_stats['averages']['response_time']:
                print(f"Average Response Time: {alert_stats['averages']['response_time']:.1f}s")

def main():
    """Main function to run the crowd panic detection system"""
    
    # Configuration
    video_sources = [
        {"id": "DEMO-CAM-01", "source": "data/input_videos/sample_crowd.mp4"},
        {"id": "WEBCAM", "source": 0}  # Webcam fallback
    ]
    
    print("🚨 CROWD PANIC DETECTION SYSTEM")
    print("=" * 50)
    print("🤖 AI Models: YOLOv8 + MediaPipe + CNN")
    print("📊 Features: Motion + Pose + Emotion Analysis")
    print("🎯 Real-time Processing with Advanced Fusion")
    print("=" * 50)
    
    # Initialize system
    system = CrowdPanicDetectionSystem()
    
    # Try each video source
    for source_config in video_sources:
        source_id = source_config["id"]
        source_path = source_config["source"]
        
        print(f"\n🎬 Attempting to use {source_id}: {source_path}")
        
        # Check if file exists (for file sources)
        if isinstance(source_path, str) and not os.path.exists(source_path):
            print(f"⚠️  Video file not found: {source_path}")
            continue
        
        try:
            # Run analysis
            system.run_video_analysis(source_path, source_id)
            break  # Exit after successful run
            
        except Exception as e:
            print(f"❌ Error with {source_id}: {e}")
            continue
    
    else:
        print("\n❌ No valid video sources found")
        print("💡 To use real video:")
        print("   1. Place MP4 files in data/input_videos/")
        print("   2. Connect a webcam")
        print("   3. Run simulation: python simulation_run.py")

if __name__ == "__main__":
    main()
