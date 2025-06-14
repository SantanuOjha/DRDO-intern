"""
Utility functions for the intrusion detection system.
"""

import cv2
import numpy as np
import time
import datetime

def resize_frame(frame, width=None, height=None):
    """
    Resize a frame while maintaining aspect ratio.
    
    Args:
        frame: The frame to resize
        width: Target width (if None, will be calculated from height)
        height: Target height (if None, will be calculated from width)
        
    Returns:
        Resized frame
    """
    if width is None and height is None:
        return frame
        
    h, w = frame.shape[:2]
    
    if width is None:
        # Calculate width to maintain aspect ratio
        aspect_ratio = w / h
        width = int(height * aspect_ratio)
    elif height is None:
        # Calculate height to maintain aspect ratio
        aspect_ratio = h / w
        height = int(width * aspect_ratio)
    
    # Resize the frame
    return cv2.resize(frame, (width, height))

def add_timestamp(frame):
    """
    Add a timestamp to a frame.
    
    Args:
        frame: The frame to add the timestamp to
        
    Returns:
        Frame with timestamp
    """
    # Get current time
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Add timestamp to frame
    cv2.putText(frame, timestamp, (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame

def calculate_fps(start_time, frame_count):
    """
    Calculate frames per second.
    
    Args:
        start_time: Start time in seconds
        frame_count: Number of frames processed
        
    Returns:
        FPS as float
    """
    elapsed_time = time.time() - start_time
    return frame_count / elapsed_time if elapsed_time > 0 else 0

def add_fps_display(frame, fps):
    """
    Add FPS counter to a frame.
    
    Args:
        frame: The frame to add the FPS counter to
        fps: Current FPS value
        
    Returns:
        Frame with FPS counter
    """
    # Add FPS text to frame
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return frame

def draw_grid(frame, grid_size=50):
    """
    Draw a grid on the frame for debugging or reference.
    
    Args:
        frame: The frame to draw the grid on
        grid_size: Size of grid cells in pixels
        
    Returns:
        Frame with grid
    """
    h, w = frame.shape[:2]
    
    # Draw vertical lines
    for x in range(0, w, grid_size):
        cv2.line(frame, (x, 0), (x, h), (50, 50, 50), 1)
    
    # Draw horizontal lines
    for y in range(0, h, grid_size):
        cv2.line(frame, (0, y), (w, y), (50, 50, 50), 1)
    
    return frame

def get_roi_mask(frame, points):
    """
    Create a mask for a region of interest (ROI).
    
    Args:
        frame: The reference frame for shape
        points: List of points defining the ROI polygon
        
    Returns:
        Binary mask for the ROI
    """
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    
    # Convert points to numpy array
    roi_corners = np.array([points], dtype=np.int32)
    
    # Fill the ROI polygon
    cv2.fillPoly(mask, roi_corners, 255)
    
    return mask

def apply_text_with_background(frame, text, position, font_scale=0.5, thickness=1, 
                              text_color=(255, 255, 255), bg_color=(0, 0, 0)):
    """
    Add text with background to a frame.
    
    Args:
        frame: The frame to add text to
        text: The text to add
        position: Position (x, y) of the text
        font_scale: Font scale
        thickness: Font thickness
        text_color: Text color (B, G, R)
        bg_color: Background color (B, G, R)
        
    Returns:
        Frame with text and background
    """
    # Get text size
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Calculate background rectangle
    padding = 5
    bg_rect = (
        position[0] - padding, 
        position[1] - text_height - padding,
        text_width + (padding * 2), 
        text_height + (padding * 2)
    )
    
    # Draw background rectangle
    cv2.rectangle(
        frame, 
        (bg_rect[0], bg_rect[1]), 
        (bg_rect[0] + bg_rect[2], bg_rect[1] + bg_rect[3]), 
        bg_color, 
        -1
    )
    
    # Draw text
    cv2.putText(
        frame, 
        text, 
        position, 
        font, 
        font_scale, 
        text_color, 
        thickness
    )
    
    return frame
