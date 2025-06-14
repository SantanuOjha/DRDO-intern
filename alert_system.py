"""
Alert system module for intrusion detection system.
Handles various alert methods when intrusions are detected.
"""

import os
import time
import datetime
import threading
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import cv2
import config

class AlertSystem:
    def __init__(self):
        """Initialize the alert system."""
        # Keep track of last alert time to prevent alert flooding
        self.last_alert_time = 0
        self.alert_cooldown = config.ALERT_COOLDOWN
        
        # Create detections directory if it doesn't exist
        if config.SAVE_DETECTION_IMAGES:
            os.makedirs(config.DETECTION_IMAGES_PATH, exist_ok=True)
        
        # Load alert sound if enabled
        self.sound_loaded = False
        if config.PLAY_SOUND_ALERT:
            try:
                # Using winsound for Windows systems
                import winsound
                self.winsound = winsound
                self.sound_loaded = True
            except ImportError:
                print("Warning: winsound module not found. Sound alerts will be disabled.")
                
    def trigger_alert(self, frame, alert_type, detection_info=None):
        """
        Trigger an alert based on detection.
        
        Args:
            frame: Current video frame
            alert_type: Type of alert ('motion' or 'person')
            detection_info: Additional information about the detection
            
        Returns:
            bool: Whether the alert was triggered (False if on cooldown)
        """
        # Check if we should alert based on settings
        if (alert_type == 'motion' and not config.ALERT_ON_MOTION) or \
           (alert_type == 'person' and not config.ALERT_ON_PERSON):
            return False
        
        # Check cooldown to prevent alert flooding
        current_time = time.time()
        if current_time - self.last_alert_time < self.alert_cooldown:
            return False
        
        # Update last alert time
        self.last_alert_time = current_time
        
        # Save detection image if enabled
        image_path = None
        if config.SAVE_DETECTION_IMAGES:
            image_path = self._save_detection_image(frame, alert_type)
        
        # Start alert methods in separate threads to avoid blocking
        if config.PLAY_SOUND_ALERT:
            threading.Thread(target=self._sound_alert).start()
            
        if config.SEND_EMAIL_ALERT:
            threading.Thread(target=self._email_alert, 
                            args=(image_path, alert_type, detection_info)).start()
        
        return True
    
    def _save_detection_image(self, frame, alert_type):
        """Save the frame when detection occurs."""
        # Create filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{alert_type}_{timestamp}.jpg"
        filepath = os.path.join(config.DETECTION_IMAGES_PATH, filename)
        
        # Save image
        cv2.imwrite(filepath, frame)
        
        return filepath
    
    def _sound_alert(self):
        """Play an alert sound."""
        if not self.sound_loaded:
            return
            
        try:
            # Play system beep sound
            self.winsound.Beep(1000, 500)  # Frequency: 1000Hz, Duration: 500ms
        except Exception as e:
            print(f"Error playing alert sound: {e}")
    
    def _email_alert(self, image_path, alert_type, detection_info):
        """Send email alert with detection information."""
        try:
            # Create email message
            msg = MIMEMultipart()
            msg['From'] = config.EMAIL_SETTINGS['email_address']
            msg['To'] = config.EMAIL_SETTINGS['recipient_email']
            msg['Subject'] = f"Security Alert: {alert_type.capitalize()} Detected"
            
            # Email body
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            body = f"Security alert triggered at {timestamp}\n\n"
            body += f"Alert type: {alert_type.capitalize()}\n"
            
            if detection_info:
                body += f"Detection details: {detection_info}\n"
                
            msg.attach(MIMEText(body, 'plain'))
            
            # Attach image if available
            if image_path and os.path.exists(image_path):
                with open(image_path, 'rb') as f:
                    img_data = f.read()
                    image = MIMEImage(img_data, name=os.path.basename(image_path))
                    msg.attach(image)
            
            # Connect to server and send email
            server = smtplib.SMTP(config.EMAIL_SETTINGS['smtp_server'], 
                                 config.EMAIL_SETTINGS['smtp_port'])
            server.starttls()
            server.login(config.EMAIL_SETTINGS['email_address'], 
                        config.EMAIL_SETTINGS['email_password'])
            server.send_message(msg)
            server.quit()
            
            print(f"Email alert sent to {config.EMAIL_SETTINGS['recipient_email']}")
            
        except Exception as e:
            print(f"Error sending email alert: {e}")
