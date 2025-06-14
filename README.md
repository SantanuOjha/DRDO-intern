# Intrusion Detection System

A Python-based intrusion detection system using computer vision to analyze camera feeds and detect potential intruders.

## Features

- Real-time video processing from camera feed
- Motion detection and tracking
- Human detection using pre-trained models
- Alert system for intrusion events
- Event logging and screenshot capture
- User-configurable sensitivity and detection zones

## Requirements

- Python 3.8+
- OpenCV
- NumPy
- TensorFlow or PyTorch (for object detection)
- Pillow (for image processing)

## Installation

1. Clone this repository:

```
git clone <repository-url>
```

2. Install required packages:

```
pip install -r requirements.txt
```

## Usage

1. Configure the settings in `config.py` to match your camera setup and detection preferences.

2. Run the main script:

```
python intrusion_detection.py
```

3. The system will start processing the camera feed and display the video with detection overlays.

4. When an intrusion is detected, the system will trigger alerts according to your configuration.

## Configuration

Edit `config.py` to customize:

- Camera source (webcam, IP camera, video file)
- Detection sensitivity
- Alert methods (sound, email, SMS)
- Logging preferences
- Detection zones (areas to monitor)

## Project Structure

- `intrusion_detection.py`: Main script for running the system
- `motion_detector.py`: Handles motion detection algorithms
- `object_detector.py`: Implements object detection using ML models
- `alert_system.py`: Manages different types of alerts
- `config.py`: Configuration settings
- `logger.py`: Logging functions
- `utils.py`: Utility functions

## License

[MIT License](LICENSE)
