import os
print("Setting Up ...")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import base64
import cv2
import numpy as np
import socketio
import eventlet
from io import BytesIO
from PIL import Image
from flask import Flask
from tensorflow.keras.models import load_model
import time

sio = socketio.Server(async_mode='eventlet')
app = Flask(__name__)

maxSpeed = 10.0

def preProcessing(img):
    img = img[60:135, :, :] 
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV) 
    img = cv2.GaussianBlur(img, (3, 3), 0) 
    img = cv2.resize(img, (200, 66))
    img = img / 255 
    return img
@sio.on('connect')
def connect(sid, environ):
    print("Connected to simulator!")
    sendControl(0.0, 0.0)

@sio.on('telemetry')
def telemetry(sid, data):
    if not data:
        print("telemetry data dont received.")
        return

    try:
        print(f"Telemetry data gotit: {data}")
        speed = float(data['speed'])
        print(f"Speed is: {speed}")

        image = Image.open(BytesIO(base64.b64decode(data['image'])))
        image = np.asarray(image)
        image = preProcessing(image)
        image = np.array([image])

        steering = float(model.predict(image))
        throttle = 1.0 - speed / maxSpeed

        print(f"Steering: {steering}, Throttle: {throttle}")
        sendControl(steering, throttle)
    except Exception as e:
        print(f"Error in processing telemetry: {e}")

def sendControl(steering, throttle):
    print(f"Sending control: steering={steering}, throttle={throttle}")
    sio.emit('steer',
             data={
                 'steering_angle': str(steering),
                 'throttle': str(throttle)
             },
             skip_sid=True)

if __name__ == "__main__":
    try:
        model = load_model('model_final.h5', compile=False)
        print("Model loaded.")

        # Start the server
        app = socketio.WSGIApp(sio, app)
        eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
    except Exception as e:
        print(f"Error starting server: {e}")