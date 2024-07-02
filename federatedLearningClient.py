import sys
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist

def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(784,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    return model

client_id = int(sys.argv[1])
(x_train, y_train), _ = mnist.load_data()
x_train = x_train.reshape((x_train.shape[0], -1)) / 255.0

# Simulating distribution of data
num_clients = 5
data_size = len(x_train) // num_clients
start = client_id * data_size
end = start + data_size
x_train, y_train = x_train[start:end], y_train[start:end]

model = build_model()
model.load_weights('global_weights.h5')  # Assuming weights are manually updated by server operator
model.fit(x_train, y_train, epochs=1, verbose=1)
model.save_weights(f'client_weights_{client_id}.npy')
