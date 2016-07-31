# Copyright 2016-present Tony Peng

"""
VGG network model
"""
import numpy as np
import scipy.io
import tensorflow as tf


def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')


def max_pool(x, ksize):
    return tf.nn.max_pool(x, ksize=[1, ksize, ksize, 1], strides=[1, ksize, ksize, 1], padding='SAME')


def avg_pool(x, ksize):
    return tf.nn.avg_pool(x, ksize=[1, ksize, ksize, 1], strides=[1, ksize, ksize, 1], padding='SAME')


def net(file_name, x, pooling_function='MAX'):
    mat_dict = scipy.io.loadmat(file_name)
    img_mean = mat_dict['meta'][0][0][1][0][0][0][0][0]
    layers = mat_dict['layers'][0]
    vgg = x
    content_activations = {}
    relu_num = 1
    pool_num = 1
    for layer_data in layers:
        layer = layer_data[0][0]
        layer_type = layer[1][0]
        if layer_type == 'conv':
            weights, biases, *rest = layer[2][0]
            # permute `weights` elements for input to TensorFlow
            weights = np.transpose(weights, (1, 0, 2, 3))
            W_conv = tf.constant(weights)
            # convert `biases` shape from [n,1] to [n]
            biases = biases.reshape(-1)
            b_conv = tf.constant(biases)
            vgg = conv2d(vgg, W_conv, 1) + b_conv
        elif layer_type == 'relu':
            vgg = tf.nn.relu(vgg)
            content_activations["relu"+str(pool_num)+"_"+str(relu_num)] = vgg
            relu_num += 1
        elif layer_type == 'pool':
            if pooling_function == 'AVG':
                vgg = avg_pool(vgg, 2)
            else:
                vgg = max_pool(vgg, 2)
            pool_num += 1
            relu_num = 1
    return vgg, content_activations, img_mean
