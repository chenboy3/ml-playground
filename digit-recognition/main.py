from random import shuffle
import numpy as np
import utils
import nn

HIDDEN_NEURONS = 100
def validation_test(output, label):
    return np.argmax(output) == np.argmax(label)
def deriv_objective(activation, y):
    return (activation - y)

train_x, train_y, valid_x, valid_y, test_x, test_y = utils.load_mnist_data()
train_data = [(feature.reshape((784,1)), utils.vectorize_y(label)) for feature, label in zip(train_x, train_y)]
valid_data = [(feature.reshape((784, 1)), utils.vectorize_y(label)) for feature, label in zip(valid_x, valid_y)]
test_data = [(feature.reshape(784,1), utils.vectorize_y(label)) for feature, label in zip(test_x, test_y)]

net = nn.Network([28*28, HIDDEN_NEURONS, 10])
net.train(train_data, deriv_objective, 3, 20, mini_batch_size=32, validation_data=valid_data, validation_test=validation_test)

test_correct = net.evaluate(test_data, validation_test)
print("----------------------------------------")
print("Test accuracy: "+str(test_correct)+"/"+str(len(test_data))+";"+str(test_correct/len(test_data)*100))
