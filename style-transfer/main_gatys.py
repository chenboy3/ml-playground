# Copyright 2016-present Tony Peng

"""
Implementation of the paper "A Neural Algorithm of Artistic Style" by Leon A. Gatys, Alexander S. Ecker, and
Matthias Bethge.
"""

import numpy as np
import tensorflow as tf
import utils
import vgg

# Neural style parameters
CONTENT_PATH = 'KillianCourt.jpg'
STYLE_PATH = 'ASundayOnLaGrandeJatte.jpg'
INITIAL_IMAGE_PATH = None
CONTENT_LAYER = 'relu4_2'
STYLE_LAYERS = {'relu1_1': 0.2, 'relu2_1': 0.2, 'relu3_1': 0.2, 'relu4_1': 0.2, 'relu5_1': 0.2}
DEVICE = '/cpu:0'
CONTENT_WEIGHT = 0.005
ITERATIONS = 1000

# Network parameters
VGG_19_PATH = 'models/imagenet-vgg-verydeep-19.mat'
LEARNING_RATE = 10

# Load images
content_image = utils.read_image(CONTENT_PATH)
style_image = utils.read_image(STYLE_PATH)
print(style_image.shape)

g = tf.Graph()
with g.device(DEVICE), g.as_default(), tf.Session() as sess:
    # 1. Compute content representation
    print("1. Computing content representation...")
    content_shape = (1,) + content_image.shape  # add batch size dimension
    x = tf.placeholder(tf.float32, content_shape)
    net, activations, img_mean = vgg.net(VGG_19_PATH, x)

    # Pre-process image
    content_image_pp = utils.preprocess_image(content_image, img_mean)

    content_representation = activations[CONTENT_LAYER].eval(feed_dict={x: np.array([content_image_pp])})

    # 2. Compute style Gram matrices
    print("2. Computing style Gram matrices...")
    style_shape = (1,) + style_image.shape  # add batch size dimension
    x = tf.placeholder(tf.float32, style_shape)
    net, activations, _ = vgg.net(VGG_19_PATH, x)

    # Pre-process image
    style_image_pp = utils.preprocess_image(style_image, img_mean)

    style_layer_shapes = {}
    gram_matrices = {}
    for style_layer, _ in STYLE_LAYERS.items():
        style_features = activations[style_layer].eval(feed_dict={x: np.array([style_image_pp])})
        style_layer_shapes[style_layer] = style_features.shape
        style_features = style_features.reshape((-1, style_features.shape[3]))
        gram_matrices[style_layer] = tf.matmul(style_features.T, style_features)

    # 3. Set up loss
    print("3. Setting up loss...")
    styled_image = utils.initialize_image(content_shape, img_mean, path=INITIAL_IMAGE_PATH)
    net, activations, _ = vgg.net(VGG_19_PATH, styled_image)

    # Content less
    loss_content = tf.nn.l2_loss(activations[CONTENT_LAYER] - content_representation) / 2.0

    # Style loss
    loss_style = 0
    for layer, w_l in STYLE_LAYERS.items():
        features = activations[layer]
        _, height, width, features_num = style_layer_shapes[layer]
        N_l = features_num
        M_l = width * height
        features = tf.reshape(features, (-1, features_num))
        gram_matrix = tf.matmul(tf.transpose(features), features)
        loss_style += w_l / (4 * N_l**2 * M_l **2) * tf.nn.l2_loss(gram_matrix - gram_matrices[layer])

    loss = CONTENT_WEIGHT * loss_content + (1 - CONTENT_WEIGHT) * loss_style

    # 4. Optimize
    print("4. Optimizing...")
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

    sess.run(tf.initialize_all_variables())
    best_loss = float('inf')
    best_styled_image = None
    for n in range(ITERATIONS):
        sess.run(optimizer)
        curr_loss_content, curr_loss_style = sess.run([loss_content, loss_style])
        if curr_loss_content + curr_loss_style < best_loss:
            best_loss = curr_loss_content + curr_loss_style
            best_styled_image = styled_image.eval()

        if n % 10 == 0:
            current_styled_image = utils.postprocess_image(styled_image.eval(), img_mean)
            output_path = utils.stylized_path(CONTENT_PATH, INITIAL_IMAGE_PATH, STYLE_PATH, 'step'+str(n))
            utils.write_image(current_styled_image, output_path)

        print("Iteration " + str(n + 1) + ": L_content=" + str(curr_loss_content) + "; L_style=" + str(curr_loss_style)
              + "; Loss=" + str(curr_loss_content + curr_loss_style))

best_styled_image = utils.postprocess_image(best_styled_image, img_mean)
output_path = utils.stylized_path(CONTENT_PATH, INITIAL_IMAGE_PATH, STYLE_PATH)
utils.write_image(best_styled_image, output_path)

