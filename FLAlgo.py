import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import matplotlib
from matplotlib import pyplot as plt

#Load the data in
emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()

len(emnist_train.client_ids)
emnist_train.element_type_structure

example_dataset = emnist_train.create_tf_dataset_for_client(
    emnist_train.client_ids[0])

example_element = next(iter(example_dataset))

example_element['label'].numpy()

plt.imshow(example_element['pixels'].numpy(), cmap='gray', aspect='equal')
plt.grid(False)
_ = plt.show()
