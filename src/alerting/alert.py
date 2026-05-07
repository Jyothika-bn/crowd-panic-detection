import requests
import json
import sqlite3
import os
from datetime import datetime
import threading
import time

# Database setup
DB_PATH = "data/alerts.db"

def init_alerts_database():
    """Initialize alerts database with enhanced schema"""
    os.makedirs("data", exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check if table exists and get current schema
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='alerts'")
    table_exists = cursor.fetchone()
    
    if not table_exists:
        # Create new table with full schema
        cursor.execute('''
            CREATE TABLE alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                panic_score REAL NOT NULL,
                motion_score REAL,
                pose_score REAL,
                emotion_score REAL,
                camera_id TEXT,
                alert_level TEXT DEFAULT 'HIGH',
                confidence REAL DEFAULT 0.8,
                dominant_feature TEXT DEFAULT 'motion',
                synergy_boost REAL DEFAULT 0.0,
                status TEXT DEFAULT 'active',
                response_time REAL,
                acknowledged BOOLEAN DEFAULT FALSE,
                acknowledged_by TEXT,
                acknowledged_at TEXT,
                notes TEXT
            )
        ''')
    else:
        # Add missing columns to existing table
        try:
            cursor.execute('ALTER TABLE alerts ADD COLUMN alert_level TEXT DEFAULT "HIGH"')
        except sqlite3.OperationalError:
            pass  # Column already exists
        
        try:
            cursor.execute('ALTER TABLE alerts ADD COLUMN confidence REAL DEFAULT 0.8')
        except sqlite3.OperationalError:
            pass
        
        try:
            cursor.execute('ALTER TABLE alerts ADD COLUMN dominant_feature TEXT DEFAULT "motion"')
        except sqlite3.OperationalError:
            pass
        
        try:
            cursor.execute('ALTER TABLE alerts ADD COLUMN synergy_boost REAL DEFAULT 0.0')
        except sqlite3.OperationalError:
            pass
        
        try:
            cursor.execute('ALTER TABLE alerts ADD COLUMN status TEXT DEFAULT "active"')
        except sqlite3.OperationalError:
            pass
        
        try:
            cursor.execute('ALTER TABLE alerts ADD COLUMN response_time REAL')
        except sqlite3.OperationalError:
            pass
        
        try:
            cursor.execute('ALTER TABLE alerts ADD COLUMN acknowledged BOOLEAN DEFAULT FALSE')
        except sqlite3.OperationalError:
            pass
        
        try:
            cursor.execute('ALTER TABLE alerts ADD COLUMN acknowledged_by TEXT')
        except sqlite3.OperationalError:
            pass
        
        try:
            cursor.execute('ALTER TABLE alerts ADD COLUMN acknowledged_at TEXT')
        except sqlite3.OperationalError:
            pass
        
        try:
            cursor.execute('ALTER TABLE alerts ADD COLUMN notes TEXT')
        except sqlite3.OperationalError:
            pass
    
    # Create indexes (only if they don't exist and columns exist)
    try:
        # Check if camera_id column exists before creating index
        cursor.execute("PRAGMA table_info(alerts)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if 'timestamp' in columns:
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON alerts(timestamp)')
        if 'camera_id' in columns:
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_camera_id ON alerts(camera_id)')
    except sqlite3.OperationalError as e:
        pass  # Indexes might already exist or columns might not exist
    
    conn.commit()
    conn.close()

# Initialize database on import
init_alerts_database()

class AlertManager:
    def __init__(self):
        self.alert_callbacks = []
        self.dashboard_url = "http://localhost:5000/api/alert"
        self.alert_queue = []
        self.processing_thread = None
        self.start_processing()
    
    def add_callback(self, callback):
        """Add callback function for alert notifications"""
        self.alert_callbacks.append(callback)
    
    def start_processing(self):
        """Start background thread for processing alerts"""
        if self.processing_thread is None or not self.processing_thread.is_alive():
            self.processing_thread = threading.Thread(target=self._process_alerts, daemon=True)
            self.processing_thread.start()
    
    def _process_alerts(self):
        """Background processing of alert queue"""
        while True:
            if self.alert_queue:
                alert_data = self.alert_queue.pop(0)
                self._send_to_dashboard(alert_data)
                self._execute_callbacks(alert_data)
            time.sleep(0.1)  # Small delay to prevent busy waiting
    
    def _send_to_dashboard(self, alert_data):
        """Send alert to web dashboard"""
        try:
            response = requests.post(
                self.dashboard_url,
                json=alert_data,
                timeout=1
            )
            if response.status_code == 200:
                print(f"📊 Alert sent to dashboard: {alert_data['alert_level']}")
        except requests.exceptions.RequestException:
            # Dashboard might not be running, continue silently
            pass
    
    def _execute_callbacks(self, alert_data):
        """Execute registered callback functions"""
        for callback in self.alert_callbacks:
            try:
                callback(alert_data)
            except Exception as e:
                print(f"Alert callback error: {e}")

# Global alert manager instance
alert_manager = AlertManager()

def alert(score, motion=0, pose=0, emotion=0, camera_id="CAM-01", detailed_analysis=None, video_evidence=None):
    """
    Enhanced alert function with comprehensive logging and notifications
    """
    
    # Determine alert level
    if score >= 0.8:
        alert_level = "CRITICAL"
        alert_icon = "🔴"
    elif score >= 0.65:
        alert_level = "HIGH"
        alert_icon = "🟠"
    elif score >= 0.5:
        alert_level = "MEDIUM"
        alert_icon = "🟡"
    else:
        alert_level = "LOW"
        alert_icon = "🟢"
    
    # Create comprehensive alert data
    alert_data = {
        'timestamp': datetime.now().isoformat(),
        'panic_score': float(score),
        'motion_score': float(motion),
        'pose_score': float(pose),
        'emotion_score': float(emotion),
        'camera_id': camera_id,
        'alert_level': alert_level,
        'confidence': detailed_analysis.get('confidence', 0.8) if detailed_analysis else 0.8,
        'dominant_feature': detailed_analysis.get('dominant_feature', 'motion') if detailed_analysis else 'motion',
        'synergy_boost': detailed_analysis.get('synergy_boost', 0.0) if detailed_analysis else 0.0,
        'feature_contributions': detailed_analysis.get('feature_contributions', {}) if detailed_analysis else {},
        'response_time': None,
        'acknowledged': False,
        'video_clip_path': video_evidence.get('video_path') if video_evidence else None,
        'screenshot_path': video_evidence.get('screenshot_path') if video_evidence else None
    }
    
    # Console output
    print(f"{alert_icon} PANIC DETECTED | Level: {alert_level} | Score: {score:.3f} | Camera: {camera_id}")
    print(f"   Motion: {motion:.3f} | Pose: {pose:.3f} | Emotion: {emotion:.3f}")
    
    if detailed_analysis:
        print(f"   Confidence: {alert_data['confidence']:.3f} | Dominant: {alert_data['dominant_feature']}")
    
    if video_evidence:
        print(f"   📹 Video Evidence: {video_evidence.get('video_path', 'N/A')}")
        print(f"   📸 Screenshot: {video_evidence.get('screenshot_path', 'N/A')}")
    
    # Store in database
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO alerts (
                timestamp, panic_score, motion_score, pose_score, emotion_score,
                camera_id, alert_level, confidence, dominant_feature, synergy_boost
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            alert_data['timestamp'],
            alert_data['panic_score'],
            alert_data['motion_score'],
            alert_data['pose_score'],
            alert_data['emotion_score'],
            alert_data['camera_id'],
            alert_data['alert_level'],
            alert_data['confidence'],
            alert_data['dominant_feature'],
            alert_data['synergy_boost']
        ))
        
        alert_data['alert_id'] = cursor.lastrowid
        conn.commit()
        conn.close()
        
        print(f"💾 Alert stored in database (ID: {alert_data['alert_id']})")
        
    except Exception as e:
        print(f"Database storage error: {e}")
    
    # Add to processing queue for dashboard and callbacks
    alert_manager.alert_queue.append(alert_data)
    
    return alert_data

def get_alert_statistics(hours=24):
    """Get alert statistics for the specified time period"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Calculate time threshold
        time_threshold = datetime.now().timestamp() - (hours * 3600)
        
        # Get total count
        cursor.execute('''
            SELECT COUNT(*) FROM alerts 
            WHERE datetime(timestamp) > datetime(?, 'unixepoch')
        ''', (time_threshold,))
        
        total_alerts = cursor.fetchone()[0]
        
        # Get average scores
        cursor.execute('''
            SELECT 
                AVG(panic_score) as avg_panic,
                AVG(motion_score) as avg_motion,
                AVG(pose_score) as avg_pose,
                AVG(emotion_score) as avg_emotion
            FROM alerts 
            WHERE datetime(timestamp) > datetime(?, 'unixepoch')
        ''', (time_threshold,))
        
        averages = cursor.fetchone()
        
        conn.close()
        
        return {
            'time_period_hours': hours,
            'total_alerts': total_alerts,
            'averages': {
                'panic_score': averages[0] or 0,
                'motion_score': averages[1] or 0,
                'pose_score': averages[2] or 0,
                'emotion_score': averages[3] or 0
            }
        }
        
    except Exception as e:
        print(f"Statistics error: {e}")
        return {'total_alerts': 0, 'averages': {'panic_score': 0}}

def get_alert_system_status():
    """Get alert system status and configuration"""
    return {
        'database_path': DB_PATH,
        'dashboard_url': alert_manager.dashboard_url,
        'processing_thread_active': alert_manager.processing_thread.is_alive() if alert_manager.processing_thread else False,
        'queue_size': len(alert_manager.alert_queue),
        'registered_callbacks': len(alert_manager.alert_callbacks),
        'alert_levels': ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'],
        'features': [
            'Real-time database storage',
            'Dashboard integration',
            'Response time tracking',
            'Alert acknowledgment',
            'Statistical analysis',
            'Callback system'
        ]
    }
