import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# Load and preprocess data

def load_data(log_path):
    image_paths = []
    steering_angles = []

    with open(log_path) as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            path = row[0].strip()     
            angle = float(row[3])    

            image_paths.append(path)
            steering_angles.append(angle)

    return image_paths, steering_angles

# Data augmentation functions
def random_flip(image, angle):
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        angle = -angle
    return image, angle


def random_brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    factor = 0.5 + np.random.uniform()
    hsv[:, :, 2] = hsv[:, :, 2] * factor
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# Random shifting
def random_shift(image, angle, range_x=80, range_y=25):
    tx = range_x * (np.random.rand() - 0.5)
    ty = range_y * (np.random.rand() - 0.5)
    angle += tx * 0.002   
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, M, (width, height))

    return image, angle
def crop_image(image):
    return image[60:-25, :, :]


def convert_to_yuv(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2YUV)


def resize_image(image):
    return cv2.resize(image, (200, 66))


def gaussian_blur(image):
    return cv2.GaussianBlur(image, (3, 3), 0)


def normalize(image):
    return image.astype(np.float32) / 127.5 - 1.0

# Full preprocessing pipeline
def preprocess_image(image):
    image = crop_image(image)
    image = convert_to_yuv(image)
    image = resize_image(image)
    image = gaussian_blur(image)
    image = normalize(image)
    return image
def batch_generator(image_paths, steering_angles, batch_size, training=True):

    while True:
        for i in range(0, len(image_paths), batch_size):

            batch_img = []
            batch_angle = []

            for img_path, angle in zip(image_paths[i:i+batch_size],
                                       steering_angles[i:i+batch_size]):

                image = cv2.imread(img_path)
                if training:
                    image, angle = random_flip(image, angle)
                    image = random_brightness(image)
                    image, angle = random_shift(image, angle)

                image = preprocess_image(image)

                batch_img.append(image)
                batch_angle.append(angle)

            yield (np.array(batch_img), np.array(batch_angle))
# Define the NVIDIA model architecture
def nvidia_model():
    model = Sequential([
        Conv2D(24, (5,5), strides=(2,2), activation="relu", input_shape=(66,200,3)),
        Conv2D(36, (5,5), strides=(2,2), activation="relu"),
        Conv2D(48, (5,5), strides=(2,2), activation="relu"),
        Conv2D(64, (3,3), activation="relu"),
        Conv2D(64, (3,3), activation="relu"),

        Flatten(),
        Dense(100, activation="relu"),
        Dropout(0.5),
        Dense(50, activation="relu"),
        Dense(10, activation="relu"),
        Dense(1)
    ])

    model.compile(optimizer=Adam(1e-4), loss="mse")
    return model
# Main training script
if __name__ == "__main__":

    log_path = "driving_log.csv"

    print("Loading data...")
    image_paths, angles = load_data(log_path)

    print("Total samples:", len(image_paths))

    # Split dataset
    X_train, X_valid, y_train, y_valid = train_test_split(
        image_paths, angles, test_size=0.2, random_state=42
    )

    print("Training samples:", len(X_train))
    print("Validation samples:", len(X_valid))

    batch_size = 64

    # Generators
    train_gen = batch_generator(X_train, y_train, batch_size, training=True)
    valid_gen = batch_generator(X_valid, y_valid, batch_size, training=False)

    model = nvidia_model()
    print(model.summary())

    # Save best model
    checkpoint = ModelCheckpoint(
        "model_best.h5",
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )

    # Train model
    history = model.fit(
        train_gen,
        steps_per_epoch=len(X_train) // batch_size,
        validation_data=valid_gen,
        validation_steps=len(X_valid) // batch_size,
        epochs=8,
        callbacks=[checkpoint]
    )

    # Save final model
    model.save("model_final.h5")
    print("Training complete! Model saved as model_final.h5")
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.legend()
    plt.grid()
    plt.savefig("training_graph.png")
    plt.show()