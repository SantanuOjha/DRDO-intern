"""
Main script for the Intrusion Detection System.
Integrates camera input, motion detection, object recognition, and alert systems.
"""

import cv2
import numpy as np
import time
import os
import argparse
import sys

# Import our modules
import config
from motion_detector import MotionDetector
from object_detector import ObjectDetector
from alert_system import AlertSystem
from logger import Logger
import utils

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Intrusion Detection System')
    
    parser.add_argument('--camera', type=int, default=config.CAMERA_SOURCE,
                       help='Camera source (default: from config.py)')
    
    parser.add_argument('--video', type=str, default=None,
                       help='Path to video file instead of camera')
    
    parser.add_argument('--width', type=int, default=config.CAMERA_WIDTH,
                       help='Camera width (default: from config.py)')
    
    parser.add_argument('--height', type=int, default=config.CAMERA_HEIGHT,
                       help='Camera height (default: from config.py)')
    
    parser.add_argument('--motion-threshold', type=int, default=config.MOTION_THRESHOLD,
                       help='Motion detection threshold (default: from config.py)')
    
    parser.add_argument('--no-display', action='store_true',
                       help='Run without displaying video feed')
    
    return parser.parse_args()

def setup_video_source(args):
    """Setup video source (camera or file)."""
    if args.video:
        # Use video file
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print(f"Error: Could not open video file {args.video}")
            sys.exit(1)
    else:
        # Use camera
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print(f"Error: Could not open camera {args.camera}")
            sys.exit(1)
        
        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    return cap

def main():
    """Main function for the Intrusion Detection System."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Initialize the logger
    logger = Logger()
    logger.log_system_start()
    
    try:
        # Setup video source
        cap = setup_video_source(args)
        
        # Initialize detectors and alert system
        motion_detector = MotionDetector()
        object_detector = ObjectDetector()
        alert_system = AlertSystem()
        
        # Create directories if they don't exist
        os.makedirs(config.DETECTION_IMAGES_PATH, exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        
        # Variables for FPS calculation
        start_time = time.time()
        frame_count = 0
        fps = 0
        
        # Main processing loop
        while True:
            # Read frame from video source
            ret, frame = cap.read()
            
            if not ret:
                # End of video or camera disconnected
                logger.log_warning("No frame received, ending processing")
                break
            
            # Increment frame counter
            frame_count += 1
            
            # Calculate FPS every 10 frames
            if frame_count % 10 == 0:
                fps = utils.calculate_fps(start_time, frame_count)
                # Reset counter and timer periodically to get current FPS
                if frame_count > 100:
                    start_time = time.time()
                    frame_count = 0
            
            # Create a copy of the frame for display
            display_frame = frame.copy() if config.SHOW_VIDEO else None
            
            # Process frame with motion detector
            if config.MOTION_DETECTION_ENABLED:
                motion_detected, motion_areas, motion_mask = motion_detector.detect(frame)
                
                # Draw motion detection results
                if config.SHOW_VIDEO and config.DRAW_MOTION_AREAS:
                    display_frame = motion_detector.draw_motion_areas(display_frame)
                
                # Log motion detection
                if motion_detected:
                    logger.log_detection("motion", f"Detected in {len(motion_areas)} areas")
                    
                    # Trigger alert if motion detected and alerts are enabled
                    if config.ALERT_ON_MOTION:
                        alert_triggered = alert_system.trigger_alert(
                            frame, 'motion', f"Motion in {len(motion_areas)} areas")
                        
                        if alert_triggered:
                            logger.log_alert("motion", "Alert triggered due to motion detection")
            
            # Process frame with object detector if motion was detected
            if config.OBJECT_DETECTION_ENABLED and (not config.MOTION_DETECTION_ENABLED or motion_detected):
                detections = object_detector.detect(frame)
                
                # Draw object detection results
                if config.SHOW_VIDEO and config.DRAW_DETECTION_BOXES:
                    display_frame = object_detector.draw_detections(display_frame)
                
                # Process detections
                for detection in detections:
                    class_name = detection['class']
                    confidence = detection['confidence']
                    
                    # Log object detection
                    logger.log_detection(class_name, f"Confidence: {confidence:.2f}")
                    
                    # Trigger alert if person detected and alerts are enabled
                    if class_name == 'person' and config.ALERT_ON_PERSON:
                        alert_triggered = alert_system.trigger_alert(
                            frame, 'person', f"Person detected with confidence {confidence:.2f}")
                        
                        if alert_triggered:
                            logger.log_alert("person", f"Alert triggered due to person detection")
            
            # Display the frame if required
            if config.SHOW_VIDEO and not args.no_display:
                # Add timestamp and FPS counter
                display_frame = utils.add_timestamp(display_frame)
                display_frame = utils.add_fps_display(display_frame, fps)
                
                # Show the frame
                cv2.imshow('Intrusion Detection System', display_frame)
                
                # Break the loop when 'q' key is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.log_info("User requested exit")
                    break
    
    except KeyboardInterrupt:
        logger.log_info("Keyboard interrupt received, shutting down")
    
    except Exception as e:
        logger.log_error(f"Unexpected error: {str(e)}")
    
    finally:
        # Clean up
        if 'cap' in locals() and cap is not None:
            cap.release()
        
        if config.SHOW_VIDEO and not args.no_display:
            cv2.destroyAllWindows()
        
        logger.log_system_stop()
        print("Intrusion Detection System stopped.")

if __name__ == "__main__":
    main()
