# -*- coding: utf-8 -*-
import numpy as np

class Teacher():
    def __init__(self, N):
        self.wstar = np.random.randn(N)
        self.wstar /= np.linalg.norm(self.wstar) / np.sqrt(N)

    def classify(self, xi):
        return np.sign(self.wstar.dot(xi))

def generate_data(P, N):
    R = Teacher(N)
    data = np.zeros((P, N))
    labels = np.zeros(P)
    for i in range(P):
        data[i] = np.random.randn(N)
        labels[i] = R.classify(data[i])

    return data, labels, R

def train_minover(data, labels, R, n_max):
    P = data.shape[0]
    N = data.shape[1]
    w = np.zeros(N)
    stability = 0
    last_kappa = 0
    for t in range(n_max):
        # Find the lowest kappa^nu(t)
        kappa_min = 999999
        mu = 0
        for nu in range(P):
            kappa = (w.dot(data[nu]) * labels[nu])
            if kappa < kappa_min:
                kappa_min = kappa
                mu = nu
        # Perform the Hebbian update
        w = w + (1/N) * data[mu] * labels[mu]
        w /= np.linalg.norm(w)
        if abs(kappa_min - last_kappa) < 0.1:
            stability += 1
            if stability == P:
                break # Stop training
        else:
            stability = 0
        last_kappa = kappa_min
    error = (1/np.pi) * np.arccos(w.dot(R.wstar) / (np.linalg.norm(w) * \
            np.linalg.norm(R.wstar)))
    print(error)
    return t, error

def experiment(N, alphas, n_D, n_max):
    with open('minover_results.csv', 'w') as f:
        f.write('alpha,n,t,err\n')
        for a in alphas:
            P = int(N * a)
            print('Running experiment for P={}'.format(P))
            for n in range(n_D):
                data, labels, R = generate_data(P, N)
                t, err = train_minover(data, labels, R, n_max)
                f.write('{},{},{},{}\n'.format(a, n, t, err))
            f.flush()

if __name__ == '__main__':
    #alphas = np.linspace(0.5, 5.0, 20)
    alphas = np.arange(0.05, 9.1, 0.1)
    print(alphas)
    experiment(20, alphas, 20, 25000)
