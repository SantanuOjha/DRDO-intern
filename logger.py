"""
Logger module for intrusion detection system.
Handles logging of events and system status.
"""

import os
import datetime
import logging
import cv2
import config

class Logger:
    def __init__(self):
        """Initialize the logger for the intrusion detection system."""
        # Set up logging
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        date_format = '%Y-%m-%d %H:%M:%S'
        
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Set up file handler
        log_file = os.path.join('logs', f'intrusion_detection_{datetime.datetime.now().strftime("%Y%m%d")}.log')
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            datefmt=date_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()  # Also log to console
            ]
        )
        
        self.logger = logging.getLogger('intrusion_detection')
        self.logger.info("Intrusion Detection System Logger initialized")
        
    def log_system_start(self):
        """Log system startup."""
        self.logger.info("Intrusion Detection System started")
        
    def log_system_stop(self):
        """Log system shutdown."""
        self.logger.info("Intrusion Detection System stopped")
        
    def log_detection(self, detection_type, details=None):
        """
        Log a detection event.
        
        Args:
            detection_type: Type of detection (motion, person, etc.)
            details: Additional details about the detection
        """
        if not config.LOG_DETECTIONS:
            return
            
        if details:
            self.logger.warning(f"{detection_type.capitalize()} detected: {details}")
        else:
            self.logger.warning(f"{detection_type.capitalize()} detected")
    
    def log_alert(self, alert_type, details=None):
        """
        Log an alert event.
        
        Args:
            alert_type: Type of alert triggered
            details: Additional details about the alert
        """
        if details:
            self.logger.critical(f"Alert triggered ({alert_type}): {details}")
        else:
            self.logger.critical(f"Alert triggered: {alert_type}")
    
    def log_error(self, error_message):
        """
        Log an error.
        
        Args:
            error_message: The error message to log
        """
        self.logger.error(f"Error: {error_message}")
        
    def log_warning(self, warning_message):
        """
        Log a warning.
        
        Args:
            warning_message: The warning message to log
        """
        self.logger.warning(f"Warning: {warning_message}")
        
    def log_info(self, info_message):
        """
        Log an informational message.
        
        Args:
            info_message: The message to log
        """
        self.logger.info(info_message)
