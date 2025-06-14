"""
Motion detector module for intrusion detection system.
Implements algorithms for detecting and tracking motion in video frames.
"""

import cv2
import numpy as np
import config

class MotionDetector:
    def __init__(self):
        # Initialize background subtractor
        # Using MOG2 algorithm which is good for detecting moving objects
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,       # Number of frames used to build the background model
            varThreshold=16,   # Threshold to decide whether a pixel is foreground or background
            detectShadows=True # Whether to detect and mark shadows
        )
        
        # Keep track of previous frame for frame differencing
        self.prev_frame = None
        
        # Initialize counters
        self.frame_count = 0
        self.detection_interval = config.MOTION_DETECTION_INTERVAL
        
        # Initialize motion status
        self.motion_detected = False
        self.motion_areas = []
    
    def detect(self, frame):
        """
        Detect motion in the given frame.
        
        Args:
            frame: Current video frame
            
        Returns:
            motion_detected: Boolean indicating if motion was detected
            motion_areas: List of contours where motion was detected
            mask: Mask showing motion areas
        """
        # Increment frame counter
        self.frame_count += 1
        
        # Only process every n-th frame for performance
        if self.frame_count % self.detection_interval != 0:
            return self.motion_detected, self.motion_areas, None
        
        # Convert frame to grayscale for processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)  # Apply blur to reduce noise
        
        # If first frame, initialize and return
        if self.prev_frame is None:
            self.prev_frame = gray
            return False, [], None
        
        # Create mask using background subtraction
        mask = self.bg_subtractor.apply(gray)
        
        # Apply thresholding to get binary mask
        thresh = cv2.threshold(mask, config.MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
        
        # Dilate the threshold image to fill in holes
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # Find contours of moving objects
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area to reduce false positives
        significant_contours = []
        for contour in contours:
            if cv2.contourArea(contour) > config.MIN_MOTION_AREA:
                significant_contours.append(contour)
        
        # Update motion status
        self.motion_detected = len(significant_contours) > 0
        self.motion_areas = significant_contours
        
        # Save current frame as previous
        self.prev_frame = gray
        
        return self.motion_detected, self.motion_areas, thresh
    
    def draw_motion_areas(self, frame):
        """
        Draw rectangles around areas where motion was detected.
        
        Args:
            frame: Video frame to draw on
            
        Returns:
            frame: Frame with motion areas highlighted
        """
        if not config.DRAW_MOTION_AREAS:
            return frame
            
        for contour in self.motion_areas:
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Add text indicating motion detection
        if self.motion_detected:
            cv2.putText(frame, "Motion Detected", (10, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return frame
    
    def reset(self):
        """Reset the motion detector."""
        self.prev_frame = None
        self.frame_count = 0
        self.motion_detected = False
        self.motion_areas = []
