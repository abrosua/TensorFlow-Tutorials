# Packages
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
from mnist import MNIST

# 1. Define NN Configuration
# conv 1
filter_size1 = 5
filter_num1 = 16
# conv 2
filter_size2 = 5
filter_num2 = 36
# FC layer
fc_size = 128

# 2. Import data set
# *Each data sets are already labelled as training, test or validation
data = MNIST(data_dir="data/MNIST/")

# 3. Store important data information
img_size = data.img_size
img_size_flat = data.img_size_flat
img_shape = data.img_shape
num_channels = data.num_channels
num_classes = data.num_classes

# COMPUTATION GRAPH
# 4. Define variables (weight and biases) as function
def weight(w_shape):
    return tf.Variable(tf.random.truncated_normal(w_shape, stddev=0.05))

def bias(b_length):
    return tf.Variable(tf.constant(0.05, shape=[b_length]))

# 5. Define NN architecture
# Construct conv. layer
def convolution_layer(input, num_channel,
                      filter_size, num_filter,
                      use_pooling=True):

    shape = [filter_size, filter_size, num_channel, num_filter]
    weights = weight(shape)
    biases = bias(num_filter)

    # Convolution layer
    layer = tf.nn.conv2d(input=input, filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    # Add bias
    layer += biases

    # Option for pooling usage
    if use_pooling:
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME')

    # Activation function with ReLU
    layer = tf.nn.relu(layer)

    return layer, weights

# Flatten conv. result
def flatten_layer(input):

    layer_shape = input.get_shape()

    num_features = layer_shape[1:4].num_elements()

    layer_flat = tf.reshape(input, [-1, num_features])

    return layer_flat, num_features

# Construct FC layer
def fc_layer(input, num_inputs,
             num_outputs,
             use_relu=True):

    shape = [num_inputs, num_inputs]
    weights = weight(shape)
    biases = bias(num_outputs)

    layer = tf.linalg.matmul(weights, input) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer

# 6. Define placeholders
