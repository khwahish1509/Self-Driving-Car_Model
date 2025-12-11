from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout


def nvidia_model(input_shape=(66, 200, 3)):
    """
    NVIDIA CNN architecture for self-driving car
    Input: 66x200x3 YUV image (normalized to [-0.5, 0.5])
    Output: Steering angle (continuous value)
    """
    model = Sequential()
    
    # Convolutional layers
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='elu', input_shape=input_shape))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    
    # Flatten
    model.add(Flatten())
    
    # Fully connected layers
    model.add(Dense(100, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='elu'))
    
    # Output layer (steering angle)
    model.add(Dense(1))
    
    return model