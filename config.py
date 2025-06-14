"""
Configuration settings for the Intrusion Detection System.
Modify these settings to customize the system behavior.
"""

# Camera Settings
CAMERA_SOURCE = 0  # 0 for default webcam, or 'rtsp://user:pass@ip_address:port/path' for IP camera
CAMERA_WIDTH = 640  # Camera frame width
CAMERA_HEIGHT = 480  # Camera frame height
CAMERA_FPS = 30  # Camera frames per second

# Detection Settings
MOTION_DETECTION_ENABLED = True
OBJECT_DETECTION_ENABLED = True
FACE_DETECTION_ENABLED = False

# Motion Detection Settings
MOTION_THRESHOLD = 30  # Threshold for motion detection (0-255)
MIN_MOTION_AREA = 500  # Minimum area of motion to trigger detection
MOTION_DETECTION_INTERVAL = 1  # Frames to wait between motion detection

# Object Detection Settings
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for object detection
CLASSES_TO_DETECT = ['person']  # Classes to detect (person, car, etc.)
DETECTION_INTERVAL = 10  # Frames to wait between object detection runs

# Alert Settings
ALERT_ON_MOTION = True  # Alert when motion is detected
ALERT_ON_PERSON = True  # Alert when person is detected
ALERT_COOLDOWN = 30  # Seconds to wait between alerts

# Logging Settings
LOG_DETECTIONS = True  # Log detection events
SAVE_DETECTION_IMAGES = True  # Save images when detection occurs
DETECTION_IMAGES_PATH = 'detections/'  # Path to save detection images

# Display Settings
SHOW_VIDEO = True  # Display video feed
DRAW_DETECTION_BOXES = True  # Draw boxes around detected objects
DRAW_MOTION_AREAS = True  # Draw areas of detected motion

# Alert Methods
PLAY_SOUND_ALERT = True  # Play sound when alert is triggered
SEND_EMAIL_ALERT = False  # Send email when alert is triggered
EMAIL_SETTINGS = {
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'email_address': 'your_email@gmail.com',
    'email_password': 'your_app_password',
    'recipient_email': 'recipient_email@example.com'
}
