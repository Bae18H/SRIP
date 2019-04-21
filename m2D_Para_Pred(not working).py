# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 2017

@author: Hyun Jun Bae
"""

# Importing libraries
import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as tcl
import os
import pdb

# Limiting TensorFlow to one GPU device
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Parameters to control the efficiency of the network
learning_rate = 1e-4
epochs = 200
batch_size = 100

# Input data and output data
x = tf.placeholder(tf.float32, shape=[None, 235200]) #420 x 560
x_shaped = tf.reshape(x, [-1, 420, 560, 1])
y = tf.placeholder(tf.float32, [None, 2])

# I'm not 100% certain what these do
ymax = tf.placeholder(tf.float32, shape=[None,1])
pattern = tf.placeholder(tf.float32, shape=[None,1])

# Function for creating convolutional layers
def create_new_conv_layer(input_data, num_input_channels, num_filters, 
                          filter_shape, name, bias=False):
    conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels,
                       num_filters]
    weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.02), name=name+'_W')
    out_layer = tf.nn.conv2d(input_data, weights, [1,2,2,1], padding='SAME')
    if bias:
        b = tf.Variable(tf.truncated_normal([num_filters],stddev=0.02), name=name+'_b')
        out_layer += b
    out_layer = tf.nn.relu(tcl.batch_norm(out_layer))
    
    return out_layer

# Creating the convolutional layers
layer1 = create_new_conv_layer(x_shaped, 1, 8, [5,5], name='layer1')
layer2 = create_new_conv_layer(layer1, 8, 16, [5,5], name='layer2')
layer3 = create_new_conv_layer(layer2, 16, 32, [5,5], name='layer3')

# Creating the fully connected layer
flattened = tcl.flatten(layer3)
fc = tf.concat([layer3, ymax, pattern], axis=1)
logits = tcl.fully_connected(inputs=fc, num_outputs=2, activation_fn=None)
ppred = tf.sigmoid(logits)

# Optimization process
loss = tf.reduce_mean(tf.square(ppred - y))

varset = tf.trainable_variables()

optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5, beta2=0.999)
grads = optimizer.compute_gradients(loss, varset)
clip_grads = [(tf.clip_by_norm(grad, 5), var) for grad, var in grads if grad is not None]
optimizer = optimizer.apply_gradients(clip_grads)

# Importing image data
result = tf.image.decode_png("E2F vs kMYC Distribution.png")
with tf.Session() as sess:
    print(sess.run(result))