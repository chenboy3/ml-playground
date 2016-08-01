import pickle
import gzip
import numpy as np


def read_pickle_data(file_name):
    f = gzip.open(file_name, 'rb')
    data = pickle.load(f, encoding='latin1')
    f.close()
    return data


def load_mnist_data():
    trainSet, validSet, testSet = read_pickle_data('mnist_6036.pkl.gz')
    trainX, trainY = trainSet
    validX, validY = validSet
    testX, testY = testSet
    return (trainX, trainY, validX, validY, testX, testY)


def vectorize_y(y):
    v = np.zeros((10, 1))
    v[y] = 1
    return v
