# -*- coding: utf-8 -*-
import numpy as np
import scipy.io as sio

def generate_data(P, N):
    data = np.zeros((P, N))
    for i in range(P):
        data[i] = np.random.randn(N)

    labels = np.random.random(P) * 2 - 1

    return data, labels

def const_eta(eta):
    def eta_func(t):
        return eta
    return eta_func

def train(P, N, t_max, eta_func=const_eta(0.05)):
    data, labels, w_1, w_2 = setup(P, N)
    for t in range(t_max):
        idx = np.random.random_integers(P) - 1
        eta = eta_func(t)
        w_1, w_2 = train_round(idx, data, labels, w_1, w_2, eta)

def regression_train(t_max, P=100, Q=100, eta_func=const_eta(0.05)):
    train, test, train_labels, test_labels, w_1, w_2 = regression_setup(P, Q)
    for t in range(t_max):
        idx = np.random.random_integers(P) - 1
        eta = eta_func(t)
        w_1, w_2 = train_round(idx, train, train_labels, w_1, w_2, eta, False)
        if t % P == 0:
            err = regression_error(train, train_labels, w_1, w_2)
            err_test = regression_error(test, test_labels, w_1, w_2)
            print(t, err, err_test)

def setup(P, N):
    data, labels = generate_data(P, N)
    print(labels.shape)

    w_1 = np.random.randn(N)
    w_2 = np.random.randn(N)
    w_1 = w_1 / np.linalg.norm(w_1, ord=2)
    w_2 = w_2 / np.linalg.norm(w_2, ord=2)
    return data, labels, w_1, w_2

def regression_setup(P=100, Q=100):
    # Load mat file and find the value of parameters
    mat = sio.loadmat('data3.mat')
    data = np.transpose(mat['xi'])
    labels = np.reshape(mat['tau'], -1)
    N = data.shape[1]
    P = data.shape[0] if P > data.shape[0] else P
    Q = data.shape[0] - P if Q > (data.shape[0] - P) else Q

    # Create test and dataset
    train_set = data[:P,:]
    train_labels = labels[:P]
    test_set = data[P:(Q+P),:]
    test_labels = labels[P:Q+P]

    # Generate initial w_1 and w_2
    w_1 = np.random.randn(N)
    w_2 = np.random.randn(N)
    w_1 = w_1 / np.linalg.norm(w_1, ord=2)
    w_2 = w_2 / np.linalg.norm(w_2, ord=2)

    return train_set, test_set, train_labels, test_labels, w_1, w_2

def train_round(idx, data, labels, w_1, w_2, eta, verbose=True):
    out = output(data[idx,:], w_1, w_2)
    err = 0.5 * (out - labels[idx]) ** 2
    nabla_1 = calc_nabla(w_1, data[idx,:], labels[idx])
    nabla_2 = calc_nabla(w_2, data[idx,:], labels[idx])

    w_1 = w_1 - eta * nabla_1 * err
    w_2 = w_2 - eta * nabla_2 * err
    if verbose:
        print(idx, labels[idx], out, err)

    return w_1, w_2

def output(xi, w_1, w_2):
    return np.tanh(w_1.dot(xi)) + np.tanh(w_2.dot(xi))

def calc_nabla(w, xi, tau):
    return (np.tanh(w.dot(xi)) - tau) * (1 - np.tanh(w.dot(xi))**2) * xi

def regression_error(data, labels, w_1, w_2):
    Err = 0
    for mu in range(labels.shape[0]):
        err = (output(data[mu], w_1, w_2) - labels[mu]) ** 2
        Err += err
    Err /= 2 * labels.shape[0]
    return Err

if __name__ == '__main__':
    regression_train(100000, 2000, 100)
    #train(10, 5, 10)
