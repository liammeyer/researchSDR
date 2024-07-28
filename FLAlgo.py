import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import matplotlib
from matplotlib import pyplot as plt
import collections
from collections.abc import Callable


#Load the dataset in as training and testing
emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()

len(emnist_train.client_ids) #checking number of clients - 3383

#describes the data format (structure) of the training data (same as testing)
#Label (tensor) and pixels (28x28 grid) it reads
emnist_train.element_type_structure 

#Tensorflow dataset for specific client based on individual ID
#example_dataset holds dataset for first client (used to train model or inspect data)
example_dataset = emnist_train.create_tf_dataset_for_client(emnist_train.client_ids[0])


#Each label is a number 0-9 that identifies the related 28x28 grid which is a drawn number they are learning off of
example_dataset = emnist_train.create_tf_dataset_for_client(emnist_train.client_ids[0])
example_element = next(iter(example_dataset))
example_element['label'].numpy()



plt.imshow(example_element['pixels'].numpy(), cmap='gray', aspect='equal')
plt.grid(False)
_ = plt.show()


#Example MNIST digits for one client (only first 40)
#Looking at client1's first 40 numbers
figure = plt.figure(figsize=(20, 4))
j = 0
for example in example_dataset.take(40): #first 40 clients
  plt.subplot(4, 10, j+1)
  plt.imshow(example['pixels'].numpy(), cmap='gray', aspect='equal')
  plt.axis('off')
  j += 1
plt.show()


#Number of examples per layer 6 clients
#How many numbers (0-9) exist for each of the first 6 clients
f = plt.figure(figsize=(12, 7))
f.suptitle('Label Counts for a Sample of Clients')
for i in range(6): #first 6 clients only
  client_dataset = emnist_train.create_tf_dataset_for_client(emnist_train.client_ids[i])
  plot_data = collections.defaultdict(list)
  for example in client_dataset:
    #Append counts individually per label to make plots
    label = example['label'].numpy()
    plot_data[label].append(label)
  plt.subplot(2, 3, i+1)
  plt.title('Client {}'.format(i))
  for j in range(10):
    plt.hist(
        plot_data[j],
        density=False,
        bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) #making a histogram where each example goes into a category so we can see distribution
plt.show()




#Each client is slightly different, each client nudges model in individual direction locally
for i in range(5):
  client_dataset = emnist_train.create_tf_dataset_for_client(emnist_train.client_ids[i]) #getting client specific dataset
  plot_data = collections.defaultdict(list)
  for example in client_dataset:
    plot_data[example['label'].numpy()].append(example['pixels'].numpy())
  f = plt.figure(i, figsize=(12, 5))
  f.suptitle("Client #{}'s Mean Image Per Label".format(i))
  for j in range(10):
    mean_img = np.mean(plot_data[j], 0)
    plt.subplot(2, 5, j+1)
    plt.imshow(mean_img.reshape((28, 28)))
    plt.axis('off')
plt.show()


NUM_CLIENTS = 10
NUM_EPOCHS = 5 #num times repeated
BATCH_SIZE = 20 #how many are processed together - 
SHUFFLE_BUFFER = 100 #randomness
PREFETCH_BUFFER = 10 #preloaded batches


#transforms data into format for training ML models - Neural Networks here using Keras and TensorFlow
def preprocess(dataset):

  def batch_format_fn(element): #flattening of the pixels
    """Flatten a batch `pixels` and return the features as an `OrderedDict`."""
    return collections.OrderedDict(
        x=tf.reshape(element['pixels'], [-1, 784]),
        y=tf.reshape(element['label'], [-1, 1]))


#goes through the flattening num_epochs times
#shuffles data based on buffer and seed ensures consistent reproducibility
#determines data points fed into model at single time during training
#applies function to each element (for flattening and reshaping)
#fetches next batch to decrease latency and improve throughput
  return dataset.repeat(NUM_EPOCHS).shuffle(SHUFFLE_BUFFER, seed=1).batch(BATCH_SIZE).map(batch_format_fn).prefetch(PREFETCH_BUFFER) 

preprocessed_example_dataset = preprocess(example_dataset) #verifying function worked

sample_batch = tf.nest.map_structure(lambda x: x.numpy(), next(iter(preprocessed_example_dataset)))


#Print x and y ordered dict
#X represents an image (784 digits ling) and y represents that label (0-9)
#Both are arrays
print (sample_batch)



def make_federated_data(client_data, client_ids):
  return [
      preprocess(client_data.create_tf_dataset_for_client(x))
      for x in client_ids
  ]

sample_clients = emnist_train.client_ids[0:NUM_CLIENTS] #choosing number of clients, sample is same each time

federated_train_data = make_federated_data(emnist_train, sample_clients)

print(f'Number of client datasets: {len(federated_train_data)}')
print(f'First dataset: {federated_train_data[0]}')

'''
def create_keras_model():
  return tf.keras.models.Sequential([
      tf.keras.layers.InputLayer(input_shape=(784,)),
      tf.keras.layers.Dense(10, kernel_initializer='zeros'),
      tf.keras.layers.Softmax(),
  ])

def model_fn():
  # We _must_ create a new model here, and _not_ capture it from an external
  # scope. TFF will call this within different graph contexts.
  keras_model = create_keras_model()
  return tff.learning.models.from_keras_model(
      keras_model,
      input_spec=preprocessed_example_dataset.element_spec,
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

training_process = tff.learning.algorithms.build_weighted_fed_avg(
    model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0))

print(training_process.initialize.type_signature.formatted_representation())
train_state = training_process.initialize()

result = training_process.next(train_state, federated_train_data)
train_state = result.state
train_metrics = result.metrics
print('round  1, metrics={}'.format(train_metrics))

NUM_ROUNDS = 11
for round_num in range(2, NUM_ROUNDS):
  result = training_process.next(train_state, federated_train_data)
  train_state = result.state
  train_metrics = result.metrics
  print('round {:2d}, metrics={}'.format(round_num, train_metrics))
  
'''
