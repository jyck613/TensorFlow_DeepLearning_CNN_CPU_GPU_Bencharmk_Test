import tensorflow as tf
from tensorflow.keras.datasets import mnist 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
import numpy as np
import time

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape((60000,28,28,1)).astype('float32') / 255
x_test = x_test.reshape((10000,28,28,1)).astype('float32') / 255
y_train = tf.keras.utils.to_categorical(y_train,10)
y_test  = tf.keras.utils.to_categorical(y_test,10) 

# Define the model
def create_model():
    model = Sequential([
        Conv2D(32,(3,3), activation='relu', input_shape=(28,28,1)),
        MaxPooling2D((2,2)),
        Conv2D(64,(3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to train the model and measure time
def train_model_on_device(device_name):
    with tf.device(device_name):
        model = create_model()
        start_time = time.time()
        model.fit(x_train, y_train, epochs=5, batch_size=64, verbose=0)  # Train for five epoch
        end_time = time.time()
    return end_time - start_time

# Train on CPU
cpu_time = train_model_on_device('/CPU:0')
print(f"Time taken on CPU: {cpu_time:.2f} seconds")

# Check if GPU is available and train on GPU
if tf.config.list_physical_devices('GPU'):
    gpu_time = train_model_on_device('/GPU:0')
    print(f"Time taken on GPU: {gpu_time:.2f} seconds")
else:
    print("GPU is not available on this system.")
