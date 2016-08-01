import math
import tensorflow as tf
from random import shuffle
import utils

# Initialization parameters
STD_DEV = 0.1
BIAS_BIAS = 0.1

# Network parameters
INPUT_DIMENSION = 28
INPUT_SIZE = 784  # 28x28
OUTPUT_SIZE = 10  # 0-9
CONV_POOL_LAYERS = [(5, 32, 1, 2), (5, 64, 1, 2)]  # filter_size, features, conv_stride, pool_ksize
FULLY_CONNECTED_NEURONS = 1024
P_KEEP = 0.5

# Learning hyperparameters
EPOCHS = 50
ETA = 0.0025
LAMBDA = 0.0001
MINI_BATCH_SIZE = 32


def initialize_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=STD_DEV))


def initialize_biases(shape):
    return tf.Variable(tf.constant(BIAS_BIAS, shape=shape))


def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')


def max_pool(x, ksize):
    return tf.nn.max_pool(x, ksize=[1, ksize, ksize, 1], strides=[1, ksize, ksize, 1], padding='SAME')


def cnn(x, keep_prob):
    x = tf.reshape(x, [-1, INPUT_DIMENSION, INPUT_DIMENSION, 1])

    # L2 regularization to prevent overfitting
    l2_loss = 0

    # Create conv/pool layers
    out_shape = (INPUT_DIMENSION, INPUT_DIMENSION)
    prev_features = 1
    for layer in CONV_POOL_LAYERS:
        filter_size, features, conv_stride, pool_ksize = layer
        W_conv = initialize_weights([filter_size, filter_size, prev_features, features])
        b_conv = initialize_biases([features])
        prev_features = features
        l2_loss += tf.nn.l2_loss(W_conv)

        conv_layer = tf.nn.relu(conv2d(x, W_conv, conv_stride) + b_conv)
        out_shape = (
            calc_conv_pool_same_output(out_shape[0], conv_stride),
            calc_conv_pool_same_output(out_shape[1], conv_stride),
        )
        pool_layer = max_pool(conv_layer, pool_ksize)
        out_shape = (
            calc_conv_pool_same_output(out_shape[0], pool_ksize),
            calc_conv_pool_same_output(out_shape[1], pool_ksize),
        )
        x = pool_layer

    # Create fully-connected layer
    W_fc = initialize_weights([prev_features * out_shape[0] * out_shape[1], FULLY_CONNECTED_NEURONS])
    b_fc = initialize_biases([FULLY_CONNECTED_NEURONS])
    l2_loss += tf.nn.l2_loss(W_fc)

    conv_pool_flat = tf.reshape(x, [-1, prev_features * out_shape[0] * out_shape[1]])
    fc = tf.nn.relu(tf.matmul(conv_pool_flat, W_fc) + b_fc)

    # Drop-out regularization to prevent overfitting
    dropout = tf.nn.dropout(fc, keep_prob)

    W_out = initialize_weights([FULLY_CONNECTED_NEURONS, OUTPUT_SIZE])
    b_out = initialize_biases([OUTPUT_SIZE])
    l2_loss += tf.nn.l2_loss(W_out)

    out = tf.matmul(dropout, W_out) + b_out
    return out, l2_loss


def calc_conv_pool_same_output(inp, stride):
    return int(math.ceil(float(inp) / float(stride)))


def format_for_tf(data_x, data_y):
    data = [(feature.reshape(INPUT_SIZE), utils.vectorize_y(label).reshape(OUTPUT_SIZE)) for feature, label in
            zip(data_x, data_y)]
    return map(list, zip(*data))


# Declare network parameter placeholderes
x = tf.placeholder(tf.float32, [None, INPUT_SIZE])
y = tf.placeholder(tf.float32, [None, OUTPUT_SIZE])
keep_prob = tf.placeholder(tf.float32)

# Create model
net, l2_loss = cnn(x, keep_prob)

# Set up cost/optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(net, y) + LAMBDA * l2_loss)
optimizer = tf.train.AdamOptimizer(learning_rate=ETA).minimize(cost)

# Create function to measure accuracy
eval_output = tf.equal(tf.argmax(net, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(eval_output, tf.float32))

init = tf.initialize_all_variables()

# Get labeled data and re-format for TF
train_x, train_y, valid_x, valid_y, test_x, test_y = utils.load_mnist_data()
train_data = [(feature.reshape(INPUT_SIZE), utils.vectorize_y(label).reshape(OUTPUT_SIZE)) for feature, label in
              zip(train_x, train_y)]
valid_x, valid_y = format_for_tf(valid_x, valid_y)
test_x, test_y = format_for_tf(test_x, test_y)

sess = tf.InteractiveSession()
sess.run(init)

for n in range(EPOCHS):
    shuffle(train_data)

    # mini-batch gradient descent
    for s in range(0, len(train_data), MINI_BATCH_SIZE):
        mini_batch = train_data[s:s + MINI_BATCH_SIZE]
        batch_x, batch_y = map(list, zip(*mini_batch))
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: P_KEEP})

    curr_cost, curr_accuracy = sess.run([cost, accuracy], feed_dict={x: valid_x, y: valid_y, keep_prob: 1.0})
    print("Epoch " + str(n + 1) + ": " + str(curr_accuracy) + "; cost=" + str(curr_cost))

final_cost, final_accuracy = sess.run([cost, accuracy], feed_dict={x: test_x, y: test_y, keep_prob: 1.0})
print("----------------------------------------")
print("Test accuracy: " + str(final_accuracy) + "; cost=" + str(final_cost))
