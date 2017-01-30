# -*- coding: utf-8 -*-
import sys

cache = {}

def C(P, N):
    if P == 1 or N == 1:
        return 2
    try:
        return cache[(P, N)]
    except KeyError:
        res = C(P-1, N) + C(P-1, N-1)
        cache[(P, N)] = res
        return res

if __name__ == '__main__':
    sys.setrecursionlimit(2000)
    N = 20
    C(10*N, N)
    with open('epsilon.csv', 'w') as f:
        f.write('N,P,alpha,epsilon\n')
        for i in range(10*N):
            P = i + 1
            alpha = P / N
            f.write('{},{},{},{}\n'.format(N, P, alpha, C(P, N - 1) / (2 * C(P, N))))
            f.flush()
