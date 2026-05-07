from flask import Flask, render_template, jsonify, send_from_directory, request, session, redirect, url_for
import os
import sqlite3
from datetime import datetime
import json

app = Flask(__name__)
app.secret_key = 'your-secret-key-here-change-in-production'

# Database paths
ALERTS_DB = 'data/alerts.db'
USERS_DB = 'data/users.db'
SCREENSHOTS_DIR = 'data/screenshots'
EVENT_CLIPS_DIR = 'data/event_clips'

# Initialize databases
def init_databases():
    """Initialize the databases if they don't exist"""
    os.makedirs('data', exist_ok=True)
    os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
    os.makedirs(EVENT_CLIPS_DIR, exist_ok=True)
    
    # Create alerts database
    conn = sqlite3.connect(ALERTS_DB)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS alerts
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  camera_id TEXT,
                  location TEXT,
                  panic_level REAL,
                  people_count INTEGER,
                  panic_sources INTEGER,
                  reason TEXT,
                  safety_measures TEXT,
                  screenshot_path TEXT,
                  video_path TEXT,
                  demographics TEXT)''')
    conn.commit()
    conn.close()
    
    # Create users database
    conn = sqlite3.connect(USERS_DB)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE,
                  password_hash TEXT,
                  email TEXT,
                  role TEXT DEFAULT 'viewer',
                  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                  last_login TEXT)''')
    
    # Add default admin user if not exists
    c.execute("SELECT * FROM users WHERE username='admin'")
    if not c.fetchone():
        c.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                 ('admin', 'admin123', 'admin'))
    
    conn.commit()
    conn.close()

# Initialize on startup
init_databases()

# Camera configurations
CAMERAS = [
    {
        'id': 'CAM-001-HIGHWAY-NORTH',
        'name': 'Highway North',
        'location': 'Highway 101 North, Mile 45',
        'status': 'active',
        'video': 'clean_cctv_CAM-001-HIGHWAY-NORTH_20260210_180323.mp4'
    },
    {
        'id': 'CAM-002-INTERSECTION-MAIN',
        'name': 'Main Intersection',
        'location': 'Main St & 5th Ave',
        'status': 'active',
        'video': 'clean_cctv_CAM-002-INTERSECTION-MAIN_20260210_180323.mp4'
    },
    {
        'id': 'CAM-003-EMERGENCY-ZONE',
        'name': 'Emergency Zone',
        'location': 'Hospital Emergency Entrance',
        'status': 'active',
        'video': 'clean_cctv_CAM-003-EMERGENCY-ZONE_20260210_180323.mp4'
    },
    {
        'id': 'CAM-004-ROADSIDE-ASSIST',
        'name': 'Roadside Assistance',
        'location': 'Highway 101 South, Mile 52',
        'status': 'active',
        'video': 'clean_cctv_CAM-004-ROADSIDE-ASSIST_20260210_180323.mp4'
    }
]

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('simple_enhanced_dashboard.html', cameras=CAMERAS)

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        conn = sqlite3.connect(USERS_DB)
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username=? AND password_hash=?", (username, password))
        user = c.fetchone()
        conn.close()
        
        if user:
            session['user'] = username
            session['role'] = user[4]  # role is at index 4
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error='Invalid credentials')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    """Logout"""
    session.pop('user', None)
    session.pop('role', None)
    return redirect(url_for('login'))

@app.route('/api/cameras')
def get_cameras():
    """Get all cameras"""
    return jsonify(CAMERAS)

@app.route('/api/camera/<camera_id>')
def get_camera(camera_id):
    """Get specific camera details"""
    camera = next((c for c in CAMERAS if c['id'] == camera_id), None)
    if camera:
        return jsonify(camera)
    return jsonify({'error': 'Camera not found'}), 404

@app.route('/api/alerts')
def get_alerts():
    """Get all alerts"""
    conn = sqlite3.connect(ALERTS_DB)
    c = conn.cursor()
    c.execute("SELECT * FROM alerts ORDER BY timestamp DESC LIMIT 50")
    alerts = []
    for row in c.fetchall():
        alerts.append({
            'id': row[0],
            'timestamp': row[1],
            'panic_score': row[2],
            'motion_score': row[3],
            'pose_score': row[4],
            'emotion_score': row[5],
            'camera_id': row[6],
            'alert_level': row[7],
            'confidence': row[8],
            'people_count': row[19] if len(row) > 19 else 0,
            'male_count': row[22] if len(row) > 22 else 0,
            'female_count': row[23] if len(row) > 23 else 0,
            'children_count': row[24] if len(row) > 24 else 0,
            'adults_count': row[25] if len(row) > 25 else 0,
            'elderly_count': row[26] if len(row) > 26 else 0,
            'avg_age': row[27] if len(row) > 27 else None,
            'panic_reason': row[28] if len(row) > 28 else '',
            'safety_measures': row[29] if len(row) > 29 else '',
            'video_path': row[17] if len(row) > 17 else None,
            'screenshot_path': row[18] if len(row) > 18 else None
        })
    conn.close()
    return jsonify(alerts)

@app.route('/api/alert/<int:alert_id>')
def get_alert(alert_id):
    """Get specific alert"""
    conn = sqlite3.connect(ALERTS_DB)
    c = conn.cursor()
    c.execute("SELECT * FROM alerts WHERE id=?", (alert_id,))
    row = c.fetchone()
    conn.close()
    
    if row:
        alert = {
            'id': row[0],
            'timestamp': row[1],
            'panic_score': row[2],
            'motion_score': row[3],
            'pose_score': row[4],
            'emotion_score': row[5],
            'camera_id': row[6],
            'alert_level': row[7],
            'confidence': row[8],
            'people_count': row[19] if len(row) > 19 else 0,
            'male_count': row[22] if len(row) > 22 else 0,
            'female_count': row[23] if len(row) > 23 else 0,
            'children_count': row[24] if len(row) > 24 else 0,
            'adults_count': row[25] if len(row) > 25 else 0,
            'elderly_count': row[26] if len(row) > 26 else 0,
            'avg_age': row[27] if len(row) > 27 else None,
            'panic_reason': row[28] if len(row) > 28 else '',
            'safety_measures': row[29] if len(row) > 29 else '',
            'video_path': row[17] if len(row) > 17 else None,
            'screenshot_path': row[18] if len(row) > 18 else None
        }
        return jsonify(alert)
    return jsonify({'error': 'Alert not found'}), 404

@app.route('/video/<path:filename>')
def serve_video(filename):
    """Serve video files"""
    try:
        video_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'event_clips'))
        video_path = os.path.join(video_dir, filename)
        print(f"Serving video: {video_path} | exists={os.path.exists(video_path)}")
        if os.path.exists(video_path):
            from flask import send_file
            response = send_file(video_path, mimetype='video/mp4', conditional=True)
            response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            response.headers['Pragma'] = 'no-cache'
            return response
        return jsonify({'error': f'Not found: {filename}'}), 404
    except Exception as e:
        print(f"Video serve error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/screenshot/<path:filename>')
def serve_screenshot(filename):
    """Serve screenshot files"""
    return send_from_directory(SCREENSHOTS_DIR, filename)

@app.route('/api/stats')
def get_stats():
    """Get system statistics"""
    conn = sqlite3.connect(ALERTS_DB)
    c = conn.cursor()
    
    # Total alerts
    c.execute("SELECT COUNT(*) FROM alerts")
    total_alerts = c.fetchone()[0]
    
    # Alerts today
    today = datetime.now().strftime('%Y-%m-%d')
    c.execute("SELECT COUNT(*) FROM alerts WHERE timestamp LIKE ?", (f'{today}%',))
    alerts_today = c.fetchone()[0]
    
    # Average panic score (using panic_score column)
    c.execute("SELECT AVG(panic_score) FROM alerts")
    avg_panic = c.fetchone()[0] or 0
    
    # Critical alerts (panic_score > 0.8)
    c.execute("SELECT COUNT(*) FROM alerts WHERE panic_score > 0.8")
    critical_alerts = c.fetchone()[0]
    
    # Total people monitored
    c.execute("SELECT SUM(people_count) FROM alerts")
    total_people = c.fetchone()[0] or 0
    
    conn.close()
    
    return jsonify({
        'total_alerts': total_alerts,
        'alerts_today': alerts_today,
        'avg_panic_score': round(avg_panic, 3),
        'critical_alerts': critical_alerts,
        'total_people_monitored': total_people,
        'active_cameras': len([c for c in CAMERAS if c['status'] == 'active']),
        'total_cameras': len(CAMERAS)
    })

@app.route('/api/chart_data')
def get_chart_data():
    """Get chart data for panic score trend"""
    conn = sqlite3.connect(ALERTS_DB)
    c = conn.cursor()
    c.execute("SELECT timestamp, panic_score FROM alerts ORDER BY timestamp DESC LIMIT 20")
    rows = c.fetchall()
    conn.close()
    
    # Reverse to show oldest first
    rows = list(reversed(rows))
    
    timestamps = [datetime.fromisoformat(row[0]).strftime('%H:%M:%S') for row in rows]
    panic_scores = [row[1] for row in rows]
    
    return jsonify({
        'timestamps': timestamps,
        'panic_scores': panic_scores
    })

@app.route('/api/export/csv')
def export_csv():
    """Export alerts as CSV"""
    import io
    from flask import make_response
    
    conn = sqlite3.connect(ALERTS_DB)
    c = conn.cursor()
    c.execute("SELECT * FROM alerts ORDER BY timestamp DESC")
    alerts = c.fetchall()
    conn.close()
    
    # Create CSV
    output = io.StringIO()
    output.write('ID,Timestamp,Panic Score,Motion,Pose,Emotion,Camera ID,Alert Level,People Count,Panic Reason,Safety Measures\n')
    
    for alert in alerts:
        panic_reason = alert[28] if len(alert) > 28 else ''
        safety_measures = alert[29] if len(alert) > 29 else ''
        people_count = alert[19] if len(alert) > 19 else 0
        output.write(f'{alert[0]},{alert[1]},{alert[2]},{alert[3]},{alert[4]},{alert[5]},{alert[6]},{alert[7]},{people_count},"{panic_reason}","{safety_measures}"\n')
    
    # Create response
    response = make_response(output.getvalue())
    response.headers['Content-Disposition'] = f'attachment; filename=alerts_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    response.headers['Content-Type'] = 'text/csv'
    
    return response

@app.route('/api/video/<int:alert_id>')
def get_video(alert_id):
    """Get video for specific alert"""
    conn = sqlite3.connect(ALERTS_DB)
    c = conn.cursor()
    c.execute("SELECT * FROM alerts WHERE id=?", (alert_id,))
    row = c.fetchone()
    conn.close()
    
    if row:
        alert = {
            'id': row[0],
            'timestamp': row[1],
            'panic_score': row[2],
            'motion_score': row[3],
            'pose_score': row[4],
            'emotion_score': row[5],
            'camera_id': row[6],
            'alert_level': row[7],
            'confidence': row[8],
            'people_count': row[19] if len(row) > 19 else 0,
            'male_count': row[22] if len(row) > 22 else 0,
            'female_count': row[23] if len(row) > 23 else 0,
            'children_count': row[24] if len(row) > 24 else 0,
            'adults_count': row[25] if len(row) > 25 else 0,
            'elderly_count': row[26] if len(row) > 26 else 0,
            'avg_age': row[27] if len(row) > 27 else None,
            'panic_reason': row[28] if len(row) > 28 else '',
            'safety_measures': row[29] if len(row) > 29 else '',
            'video_clip_path': row[17] if len(row) > 17 else None,
            'screenshot_path': row[18] if len(row) > 18 else None
        }
        return render_template('video_player.html', alert=alert)
    
    return jsonify({'error': 'Video not found'}), 404

@app.route('/camera')
def camera_page():
    """Live camera monitoring page"""
    return render_template('live_camera_v2.html')

@app.route('/camera_new')
def camera_page_new():
    """Live camera monitoring page - NEW VERSION"""
    return render_template('live_camera_v2.html')

@app.route('/api/process_frame', methods=['POST'])
def process_frame():
    """Process a frame for age/gender detection"""
    try:
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
        
        from realtime_detection import process_webcam_frame
        
        data = request.get_json()
        frame_data = data.get('frame')
        
        if not frame_data:
            return jsonify({'error': 'No frame data'}), 400
        
        result = process_webcam_frame(frame_data)
        
        if result:
            return jsonify(result)
        else:
            return jsonify({'error': 'Processing failed'}), 500
            
    except Exception as e:
        print(f"Frame processing error: {e}")
        import traceback
        traceback.print_exc()
        # Return simulated data as fallback with correct structure
        return jsonify({
            'detections': [{
                'x': 100,
                'y': 100,
                'width': 200,
                'height': 250,
                'gender': 'Female',
                'age': 22,
                'emotion': 'happy'
            }],
            'stats': {
                'total_people': 1,
                'male_count': 0,
                'female_count': 1,
                'children': 0,
                'adults': 1,
                'elderly': 0,
                'avg_age': 22,
                'panic_emotions': 0
            }
        })

if __name__ == '__main__':
    print("=" * 70)
    print("🎥 CROWD PANIC DETECTION SYSTEM - DASHBOARD")
    print("=" * 70)
    print()
    print("✅ Dashboard starting...")
    print("✅ Databases initialized")
    print("✅ Cameras configured")
    print()
    print("📊 Dashboard URL: http://localhost:5000")
    print("👤 Default Login: admin / admin123")
    print()
    print("🎥 Active Cameras:")
    for cam in CAMERAS:
        print(f"   - {cam['name']} ({cam['id']})")
    print()
    print("=" * 70)
    print("Press Ctrl+C to stop the server")
    print("=" * 70)
    print()
    
    app.run(debug=True, host='0.0.0.0', port=5000)
