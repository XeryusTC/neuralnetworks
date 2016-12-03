# -*- coding: utf-8 -*-
import numpy as np

def generate_data(P, N):
    data = np.zeros((P, N))
    for i in range(P):
        data[i] = np.random.randn(N)

    labels = np.random.choice((-1, 1), P)

    return data, labels

if __name__ == '__main__':
    print(generate_data(10, 20))
