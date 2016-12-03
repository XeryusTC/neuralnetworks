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
        success = True
        for t in range(labels.size):
            E = w.dot(data[t]) * labels[t]
            if E <= 0:
                success = False
                w = w + (1/N) * data[t] * labels[t]
        # Stop training if all E > 0
        if success:
            return True
    return False

def experiment(N, alphas, n_D, n_max):
    with open('rosenblat_results.csv', 'w') as f:
        f.write('alpha,n,result\n')
        for a in alphas:
            P = int(N * a)
            print('Running experiment for P={}'.format(P))
            for n in range(n_D):
                data, labels = generate_data(P, N)
                res = train_rosenblat(data, labels, n_max)
                f.write('{},{},{}\n'.format(a, n, res))

if __name__ == '__main__':
    alphas = np.linspace(0.75, 3.0, 10)
    experiment(20, alphas, 50, 100)
