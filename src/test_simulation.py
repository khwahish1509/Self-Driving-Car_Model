import socketio
import eventlet
import numpy as np
from flask import Flask
import tensorflow as tf
from tensorflow.keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import cv2
import argparse
from src.preprocess import preprocess

# Enable loading Lambda layers (safe for our own models)
tf.keras.config.enable_unsafe_deserialization()

# Initialize Flask and SocketIO
sio = socketio.Server()
app = Flask(__name__)

# Global variables
model = None
speed_limit = 30  # Maximum speed


def preprocess_image(image):
    """
    Preprocess image for model prediction
    """
    # Convert PIL Image to numpy array
    img = np.array(image)
    # Convert RGB to BGR (OpenCV format)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # Apply preprocessing pipeline
    img = preprocess(img)
    return img


@sio.on('telemetry')
def telemetry(sid, data):
    """
    Receives telemetry data from simulator and sends back steering angle
    """
    if data:
        # Current steering angle from simulator
        # steering_angle = float(data["steering_angle"])
        # Current throttle
        # throttle = float(data["throttle"])
        # Current speed
        speed = float(data["speed"])
        
        # Current image from center camera
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        
        try:
            # Preprocess image
            image_array = preprocess_image(image)
            
            # Add batch dimension: (66, 200, 3) -> (1, 66, 200, 3)
            image_array = np.expand_dims(image_array, axis=0)
            
            # Predict steering angle
            steering_angle = float(model.predict(image_array, verbose=0)[0][0])
            
            # Compute throttle based on speed and steering
            # Slow down on sharp turns
            global speed_limit
            if abs(steering_angle) > 0.2:
                throttle = 0.1
            elif speed > speed_limit:
                throttle = -0.1
            else:
                throttle = 0.3
            
            # Print telemetry
            print(f'Steering: {steering_angle:.4f} | Throttle: {throttle:.2f} | Speed: {speed:.2f}')
            
            # Send control commands back to simulator
            send_control(steering_angle, throttle)
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            send_control(0, 0)
    else:
        # Manual mode
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    """
    Called when simulator connects
    """
    print("Connected to simulator")
    send_control(0, 0)


def send_control(steering_angle, throttle):
    """
    Send control commands to simulator
    """
    sio.emit(
        "steer",
        data={
            'steering_angle': str(steering_angle),
            'throttle': str(throttle)
        },
        skip_sid=True
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        '--model',
        type=str,
        default='logs/models/model_best.h5',
        help='Path to model h5 file.'
    )
    parser.add_argument(
        '--speed',
        type=int,
        default=30,
        help='Maximum speed limit'
    )
    args = parser.parse_args()
    
    # Load the trained model with compile=False to avoid Keras compatibility issues
    print(f"Loading model from: {args.model}")
    try:
        # First try: Load without compiling (avoids MSE deserialization error)
        model = load_model(args.model, compile=False)
        print("✓ Model loaded successfully!")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("\nTrying alternative method...")
        try:
            # Second try: Load architecture and weights separately
            from src.model import nvidia_model
            model = nvidia_model()
            model.load_weights(args.model)
            print("✓ Model weights loaded successfully!")
        except Exception as e2:
            print(f"✗ Failed to load model: {e2}")
            print("\nPlease ensure:")
            print("  1. Model file exists at:", args.model)
            print("  2. You've run training: python -m src.train ...")
            exit(1)
    
    speed_limit = args.speed
    
    # Wrap Flask app with socketio's middleware
    app = socketio.Middleware(sio, app)
    
    # Deploy server
    print("\n" + "="*50)
    print("🚗 Server is running on port 4567")
    print("Now launch the simulator in AUTONOMOUS MODE")
    print("="*50 + "\n")
    
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)