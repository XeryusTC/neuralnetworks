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

def regression_train(t_max, P=100, Q=100, eta_func=const_eta(0.05),
    filename='results.csv'):
    train, test, train_labels, test_labels, w_1, w_2 = regression_setup(P, Q)
    with open(filename, 'a') as f:
        for t in range(t_max):
            idx = np.random.random_integers(P) - 1
            eta = eta_func(t)
            w_1, w_2 = train_round(idx, train, train_labels, w_1, w_2, eta,
                False)
            if t % 50 == 0:
                err = regression_error(train, train_labels, w_1, w_2)
                err_test = regression_error(test, test_labels, w_1, w_2)
                #print(t, err, err_test)
                f.write('{},{},{},{},{},{}\n'.format(P, Q, t, eta, err,
                    err_test))
    return w_1, w_2

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
    xi = np.transpose(mat['xi'])
    tau = np.reshape(mat['tau'], -1)
    N = xi.shape[1]
    P = xi.shape[0] if P > xi.shape[0] else P
    Q = xi.shape[0] - P if Q > (xi.shape[0] - P) else Q
    indexes = np.random.permutation(np.arange(N))

    # Create test and dataset
    train_set = np.zeros((P, N))
    train_labels = np.zeros(P)
    test_set = np.zeros((Q, N))
    test_labels = np.zeros(Q)
    for i in range(P):
        train_set[i] = xi[i]
        train_labels[i] = tau[i]
    for i in range(Q):
        test_set[i] = xi[i+P]
        test_labels[i] = tau[i+P]

    # Generate initial w_1 and w_2
    w_1 = np.random.randn(N)
    w_2 = np.random.randn(N)
    w_1 = w_1 / np.linalg.norm(w_1, ord=2)
    w_2 = w_2 / np.linalg.norm(w_2, ord=2)

    return train_set, test_set, train_labels, test_labels, w_1, w_2

def train_round(idx, data, labels, w_1, w_2, eta, verbose=True):
    out = output(data[idx,:], w_1, w_2)
    err = 0.5 * (out - labels[idx]) ** 2
    nabla_1 = calc_nabla(w_1, data[idx,:], labels[idx], out)
    nabla_2 = calc_nabla(w_2, data[idx,:], labels[idx], out)

    w_1 = w_1 - eta * nabla_1 * err
    w_2 = w_2 - eta * nabla_2 * err
    if verbose:
        print(idx, labels[idx], out, err)

    return w_1, w_2

def output(xi, w_1, w_2):
    return np.tanh(w_1.dot(xi)) + np.tanh(w_2.dot(xi))

def calc_nabla(w, xi, tau, sigma):
    return (sigma - tau) * (1 - np.tanh(w.dot(xi))**2) * xi

def regression_error(data, labels, w_1, w_2):
    Err = 0
    for mu in range(labels.shape[0]):
        err = (output(data[mu], w_1, w_2) - labels[mu]) ** 2
        Err += err
    Err /= 2 * labels.shape[0]
    return Err

if __name__ == '__main__':
    print("Getting w_1, w_2")
    w_1, w_2 = regression_train(5000, 1000, 200, filename='weights.csv')
    with open('weights.csv', 'w') as f:
        for i in range(w_1.shape[0]):
            if i > 0:
                f.write(',')
            f.write('{}'.format(w_1[i]))
        f.write('\n')
        for i in range(w_2.shape[0]):
            if i > 0:
                f.write(',')
            f.write('{}'.format(w_2[i]))
        f.write('\n')

    with open('gradient_descent_results.csv', 'w') as f:
        f.write('P,Q,t,eta,E,E_test\n')
    for i in range(25):
        print("error", 2000, 2000, 200)
        regression_train(2000, 2000, 200,
            filename='gradient_descent_results.csv')

    with open('eta_sweep.csv', 'w') as f:
        f.write('P,Q,t,eta,E,E_test\n')
    for i in range(25):
        for eta in [0.1, 0.05, 0.01, 0.005, 0.001]:
            print("eta", 5000, 2000, 200, eta)
            regression_train(5000, 2000, 200, const_eta(eta), 'eta_sweep.csv')

    with open('train_sweep.csv', 'w') as f:
        f.write('P,Q,t,eta,E,E_test\n')
    for i in range(25):
        for P in [100, 200, 400, 500, 1000, 1500, 2000, 4000]:
            print("train", 2000, P, 200)
            regression_train(2000, P, 200, filename='train_sweep.csv')

    #train(10, 5, 10)
