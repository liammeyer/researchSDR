import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

#Distributing
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Normalize the data
x_train, x_test = x_train / 255.0, x_test / 255.0

#Flatten the images for a simple neural network input
x_train = x_train.reshape((x_train.shape[0], -1))
x_test = x_test.reshape((x_test.shape[0], -1))



#Dividing between number clients 
def distribute_data(x, y, num_clients):
    size = len(x) // num_clients
    return [(x[i*size:(i+1)*size], y[i*size:(i+1)*size]) for i in range(num_clients)]

num_clients = 5
clients = distribute_data(x_train, y_train, num_clients)


def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer (input_shape = (784,)),
        tf.keras.layers.Dense (128, activation = 'relu'),
        tf.keras.layers.Dropout (0.2),
        tf.keras.layers.Dense (10)
    ])
    return model


global_model = build_model()
global_model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

for x in range (10):
    client_models = train_on_clients(clients, global_model)
    new_weights = federated_average(client_models)
    global_model.set_weights(new_weights)

# Evaluate the model
global_model.evaluate(x_test, y_test)

#Creates a local model
def train_on_clients(clients, model):
    client_models = [] 
    for X, y in clients:
        local_model = build_model()
        local_model.compile(optimizer='adam', 
                            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
                            metrics=['accuracy'])
        local_model.set_weights(model.get_weights())  # Initialize with global model weights
        local_model.fit(X, y, epochs=1, verbose=0)  # Train locally on client's data
        client_models.append(local_model)
    return client_models

def federated_average(models):
    new_weights = []
    # Average weights from all client models
    for layer_tuple in zip(*[model.get_weights() for model in models]):
        new_weights.append(np.mean(layer_tuple, axis=0))
    return new_weights
