# -*- coding: utf-8 -*-
import numpy as np

def generate_data(P, N):
    data = np.zeros((P, N))
    for i in range(P):
        data[i] = np.random.randn(N)

    labels = np.random.choice((-1, 1), P)

    return data, labels

def train_rosenblat(data, labels, n_max):
    N = data.shape[1]
    w = np.zeros(N)
    for n in range(n_max):
        print('Starting training epoch', n)
        success = True
        for t in range(labels.size):
            E = w.dot(data[t]) * labels[t]
            if E <= 0:
                success = False
                w = w + (1/N) * data[t] * labels[t]
        # Stop training if all E > 0
        if success:
            break
    return w

if __name__ == '__main__':
    data, labels = generate_data(10, 20)

    train_rosenblat(data, labels, 5)
