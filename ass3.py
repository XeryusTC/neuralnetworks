# -*- coding: utf-8 -*-
import numpy as np

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

def setup(P, N):
    data, labels = generate_data(P, N)

    w_1 = np.random.randn(N)
    w_2 = np.random.randn(N)
    w_1 = w_1 / np.linalg.norm(w_1, ord=2)
    w_2 = w_2 / np.linalg.norm(w_2, ord=2)
    return data, labels, w_1, w_2

def train_round(idx, data, labels, w_1, w_2, eta):
    out = output(data[idx,:], w_1, w_2)
    err = 0.5 * (out - labels[idx]) ** 2
    nabla_1 = calc_nabla(w_1, data[idx,:], labels[idx])
    nabla_2 = calc_nabla(w_2, data[idx,:], labels[idx])

    w_1 = w_1 - eta * nabla_1 * err
    w_2 = w_2 - eta * nabla_2 * err
    print(idx, labels[idx], out, err)

    return w_1, w_2

def output(xi, w_1, w_2):
    return np.tanh(w_1.dot(xi)) + np.tanh(w_2.dot(xi))

def calc_nabla(w, xi, tau):
    return (np.tanh(w.dot(xi)) - tau) * (1 - np.tanh(w.dot(xi))**2) * xi

if __name__ == '__main__':
    train(10, 5, 10000)
