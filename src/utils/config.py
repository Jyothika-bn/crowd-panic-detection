# Presentation Configuration
PANIC_THRESHOLD = 0.65

# Demo video sources (will fall back to simulation if files don't exist)
VIDEO_SOURCES = [
    {"id": "DEMO-CAM-01", "source": "data/input_videos/demo_crowd.mp4"},
    {"id": "DEMO-CAM-02", "source": 0}  # Webcam fallback
]

# Presentation settings
DEMO_MODE = True
AUTO_GENERATE_ALERTS = True
PRESENTATION_SPEED = 1.0  # Normal speed

# Dashboard settings for demo
DASHBOARD_REFRESH_RATE = 1000  # 1 second
MAX_CHART_POINTS = 20
ALERT_RETENTION_HOURS = 24
