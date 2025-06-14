"""
Object detector module for intrusion detection system.
Uses OpenCV's built-in capabilities for object detection as a lightweight alternative to TensorFlow.
"""

import cv2
import numpy as np
import config
import os
import time

# Try to import TensorFlow, but continue even if it's not available
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. Using fallback detection methods.")

class ObjectDetector:
    def __init__(self):
        """Initialize the object detector."""
        # Initialize parameters
        self.model = None
        self.classes = []
        self.frame_count = 0
        self.detection_interval = config.DETECTION_INTERVAL
        self.confidence_threshold = config.CONFIDENCE_THRESHOLD
        self.classes_to_detect = config.CLASSES_TO_DETECT
        
        # Initialize detection results
        self.detections = []
        
        # Flag to check if model is loaded
        self.model_loaded = False
        
        # Use OpenCV's HOG detector as a fallback
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
    def load_model(self):
        """Load the pre-trained object detection model."""
        # If TensorFlow is not available, we'll use the HOG detector only
        if not TENSORFLOW_AVAILABLE:
            print("Using OpenCV's HOG detector for person detection")
            return False
            
        try:
            # Create a directory for models if it doesn't exist
            os.makedirs('models', exist_ok=True)
            
            # Path to the model - in real implementation, ensure this model exists
            model_path = 'models/ssd_mobilenet_v2'
            
            # Check if model exists, otherwise we would download it
            if not os.path.exists(model_path):
                print("Model not found. Using OpenCV's HOG detector as fallback.")
                return False
            
            # Load the model
            self.model = tf.saved_model.load(model_path)
            
            # Load classes
            with open('models/coco_classes.txt', 'r') as f:
                self.classes = [line.strip() for line in f.readlines()]
            
            self.model_loaded = True
            return True
            
        except Exception as e:
            print(f"Error loading TensorFlow model: {e}")
            print("Using OpenCV's HOG detector as fallback.")
            return False
    
    def detect(self, frame):
        """
        Detect objects in the given frame.
        
        Args:
            frame: Current video frame
            
        Returns:
            detections: List of detection results (class, confidence, bounding box)
        """
        # Increment frame counter
        self.frame_count += 1
        
        # Only process every n-th frame for performance
        if self.frame_count % self.detection_interval != 0:
            return self.detections
            
        # Clear previous detections
        self.detections = []
        
        # Try TensorFlow model if available
        if TENSORFLOW_AVAILABLE and not self.model_loaded:
            self.load_model()
            
        if self.model_loaded:
            # Use TensorFlow model for detection
            try:
                # Prepare image for the model
                input_tensor = self._preprocess_image(frame)
                
                # Run inference
                output_dict = self.model(input_tensor)
                
                # Process results
                self._process_detections(output_dict, frame.shape)
                
                return self.detections
                
            except Exception as e:
                print(f"Error during TensorFlow detection: {e}")
                # Fall back to HOG detection if TensorFlow fails
                self._detect_with_hog(frame)
        else:
            # Use HOG detector as fallback
            self._detect_with_hog(frame)
            
        return self.detections
    
    def _detect_with_hog(self, frame):
        """Use OpenCV's HOG detector to detect people."""
        # Convert to grayscale for better detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect people in the image
        boxes, weights = self.hog.detectMultiScale(
            frame, 
            winStride=(8, 8),
            padding=(4, 4),
            scale=1.05
        )
        
        # Add detections to the list
        for i, (x, y, w, h) in enumerate(boxes):
            confidence = float(weights[i])
            if confidence >= self.confidence_threshold:
                self.detections.append({
                    'class': 'person',
                    'confidence': confidence,
                    'box': (x, y, x + w, y + h)
                })
    
    def _preprocess_image(self, frame):
        """Preprocess the image for the model."""
        # Resize to model's expected input dimensions
        input_size = (300, 300)  # Common size for SSD MobileNet
        resized = cv2.resize(frame, input_size)
        
        # Convert to RGB (OpenCV uses BGR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize pixel values
        normalized = rgb / 255.0
        
        # Add batch dimension
        input_tensor = np.expand_dims(normalized, axis=0)
        
        return input_tensor
    
    def _process_detections(self, output_dict, original_shape):
        """Process detection results from the model."""
        # Extract detection results
        boxes = output_dict['detection_boxes'][0].numpy()
        scores = output_dict['detection_scores'][0].numpy()
        classes = output_dict['detection_classes'][0].numpy().astype(np.int32)
        
        height, width, _ = original_shape
        
        # Filter detections by confidence and class
        for i in range(len(scores)):
            if scores[i] >= self.confidence_threshold:
                # Convert class index to class name
                class_name = self.classes[classes[i] - 1]  # COCO classes are 1-indexed
                
                # Check if class is in the list of classes to detect
                if class_name in self.classes_to_detect:
                    # Convert normalized box coordinates to pixel values
                    ymin, xmin, ymax, xmax = boxes[i]
                    left = int(xmin * width)
                    top = int(ymin * height)
                    right = int(xmax * width)
                    bottom = int(ymax * height)
                    
                    # Add detection to the list
                    self.detections.append({
                        'class': class_name,
                        'confidence': float(scores[i]),
                        'box': (left, top, right, bottom)
                    })
    
    def _simulate_detections(self, frame):
        """Simulate object detections for testing without a model."""
        # Clear previous detections
        self.detections = []
        
        # Only simulate a detection periodically and if motion has been detected
        # This is for demonstration only - in production, use a real model
        height, width, _ = frame.shape
        
        # Simulate a person detection in a random location of the frame
        # This is just for testing - a real implementation would use the model
        if np.random.random() < 0.3:  # 30% chance of detection
            # Create a detection in the bottom half of the frame
            left = int(np.random.uniform(0, width * 0.7))
            top = int(np.random.uniform(height * 0.5, height * 0.8))
            right = left + int(np.random.uniform(width * 0.1, width * 0.3))
            bottom = top + int(np.random.uniform(height * 0.1, height * 0.2))
            
            # Add simulated detection
            self.detections.append({
                'class': 'person',
                'confidence': np.random.uniform(0.5, 0.9),
                'box': (left, top, right, bottom)
            })
    
    def draw_detections(self, frame):
        """
        Draw bounding boxes around detected objects.
        
        Args:
            frame: Video frame to draw on
            
        Returns:
            frame: Frame with detection boxes
        """
        if not config.DRAW_DETECTION_BOXES:
            return frame
            
        for detection in self.detections:
            # Extract detection information
            label = f"{detection['class']}: {detection['confidence']:.2f}"
            left, top, right, bottom = detection['box']
            
            # Draw bounding box
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
            
            # Draw label background
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (left, top - 20), (left + label_size[0], top), (255, 0, 0), -1)
            
            # Draw label text
            cv2.putText(frame, label, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return frame
