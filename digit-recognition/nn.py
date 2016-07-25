import numpy as np
from random import shuffle


class SigmoidActivation:
    @classmethod
    def fn(cls, z):
        return 1.0 / (1.0 + np.exp(-z))

    @classmethod
    def deriv(cls, z):
        s = cls.fn(z)
        return s * (1 - s)


class QuadraticCost:
    @classmethod
    def fn(cls, activation, y):
        return 0.5 * np.linalg.norm(activation - y) ** 2

    @classmethod
    def delta(cls, activation, y):
        return activation - y


class CrossEntropyCost:
    @classmethod
    def fn(cls, activation, y):
        return np.sum(np.nan_to_num(-y * np.log(activation) - (1 - y) * np.log(1 - activation)))

    @classmethod
    def delta(cls, activation, y):
        return (activation - y) / (activation * (1 - activation))


class Network:
    def __init__(self, layer_sizes, weights=None, biases=None, activation=SigmoidActivation):
        self.layer_sizes = layer_sizes
        self.velocities = [np.zeros((o, i)) for o, i in zip(layer_sizes[1:], layer_sizes[:-1])]
        self.weights = weights if weights is not None else [np.random.randn(o, i) / np.sqrt(i) for o, i in
                                                            zip(layer_sizes[1:], layer_sizes[:-1])]
        self.biases = biases if biases is not None else [np.random.randn(n, 1) for n in layer_sizes[1:]]
        self.activation = activation

    def train(self, train_data, epochs, eta, regularization_lambda, momentum=0, mini_batch_size=1, cost=QuadraticCost,
              validation_data=None, validation_test=None):
        if validation_data is not None:
            print("eta=" + str(eta) + "; lambda=" + str(regularization_lambda) + "; momentum=" +
                  str(momentum) + "; epochs=" + str(epochs) + "; mini_batch_size=" + str(mini_batch_size))

        for n in range(epochs):
            shuffle(train_data)
            # mini-batch SGD
            for s in range(0, len(train_data), mini_batch_size):
                mini_batch = train_data[s:s + mini_batch_size]
                self.gd(mini_batch, len(train_data), cost, eta, regularization_lambda, momentum)
            if validation_data is not None:
                correct = self.evaluate(validation_data, validation_test)
                print("Epoch " + str(n + 1) + ": " + str(correct) + "/" + str(len(validation_data)) + ";" + str(
                    correct / len(validation_data) * 100))

    def predict(self, x):
        inp = x
        for w_l, b_l in zip(self.weights, self.biases):
            inp = self.activation.fn(np.dot(w_l, inp) + b_l)
        return inp

    def gd(self, data, n, cost, eta, regularization_lambda, momentum):
        partials_w = [np.zeros(w.shape) for w in self.weights]
        partials_b = [np.zeros(b.shape) for b in self.biases]

        for x, y in data:
            partials_w_p, partials_b_p = self.calc_partials(x, y, cost)
            partials_w = [p_w + p_w_p for p_w, p_w_p in zip(partials_w, partials_w_p)]
            partials_b = [p_b + p_b_p for p_b, p_b_p in zip(partials_b, partials_b_p)]

        self.velocities = [momentum * V - eta * (partial_w / len(data) + regularization_lambda * W / n)
                           for V, W, partial_w in zip(self.velocities, self.weights, partials_w)]
        self.weights = [W + V for W, V in zip(self.weights, self.velocities)]
        self.biases = [B - eta * partial_b / len(data) for B, partial_b in zip(self.biases, partials_b)]

    def calc_partials(self, x, y, cost):
        partials_w = [np.zeros(w.shape) for w in self.weights]
        partials_b = [np.zeros(b.shape) for b in self.biases]

        # calculate output z and activations
        inp = x
        activations = [x]
        zs = [x]
        for w_l, b_l in zip(self.weights, self.biases):
            z = np.dot(w_l, inp) + b_l
            inp = self.activation.fn(z)
            zs.append(z)
            activations.append(inp)

        # calculate partials for last layer, L
        # first, find delta_L (dJ/dz_L_j)
        delta = cost.delta(activations[-1], y) * self.activation.deriv(zs[-1])
        partials_w[-1] = np.dot(delta, activations[-2].T)
        partials_b[-1] = delta

        # backpropagate through
        for o in range(2, len(self.layer_sizes)):
            # calculate delta_l
            delta = np.dot(self.weights[-o + 1].T, delta) * self.activation.deriv(zs[-o])
            partials_w[-o] = np.dot(delta, activations[-o - 1].T)
            partials_b[-o] = delta

        return partials_w, partials_b

    def evaluate(self, data, test):
        correct = 0
        for (feature, label) in data:
            output = self.predict(feature)
            if test(output, label):
                correct += 1
        return correct
