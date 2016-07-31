# Copyright 2016-present Tony Peng

import numpy as np
import os
import scipy.misc
import tensorflow as tf


def preprocess_image(image, img_mean):
    return (image - img_mean).astype(np.float32)


def postprocess_image(image, img_mean):
    return img_mean + image.reshape(image.shape[1:])


def initialize_image(shape, img_mean, path=None):
    if path == None:
        x = tf.random_normal(shape) * 0.001
    else:
        x = np.array([read_image(path)])
        x = preprocess_image(x, img_mean)
    return tf.Variable(x)


def read_image(path, type=np.float):
    return scipy.misc.imread(path).astype(type)


def write_image(img, path):
    img = np.clip(img, 0, 255).astype(np.uint8)
    scipy.misc.imsave(path, img)


def stylized_path(content_path, initial_path, style_path, suffix=None):
    content_path, content_filename = os.path.split(initial_path or content_path)
    style_path, style_filename = os.path.split(style_path)
    # remove extensions
    content_filename_parts = os.path.splitext(content_filename)
    content_filename = content_filename_parts[0]
    content_ext = content_filename_parts[1]
    style_filename = os.path.splitext(style_filename)[0]
    return os.path.join(content_path, content_filename + '+' + style_filename
        + ("_" + suffix if suffix != None else "") + content_ext)
