import numpy as np  
from numba import jit, njit


@njit(cache=True)
def infnumbanjit(A):
    M, N = A.shape
    maxval = 0
    for i in range(M):
        infnorm = 0.0
        for j in range(N):
            infnorm = np.abs(A[i, j])
            if infnorm > maxval:
                maxval = infnorm
    return infnorm


@jit(cache=True)
def infnumbajit(A):
    M, N = A.shape
    maxval = 0
    for i in range(M):
        infnorm = 0.0
        for j in range(N):
            infnorm = np.abs(A[i, j])
            if infnorm > maxval:
                maxval = infnorm
    return infnorm


@jit(cache=True)
def infnumbaarr(A):
    M, N = A.shape
    sumarr = np.zeros(M)
    for i in range(M):
        for j in range(N):
            sumarr[i] += np.abs(A[i,j])
    return max(sumarr)


def infcomp(A):
    return max([sum(i) for i in A])

def infpure(A):
    M, N = A.shape
    maxval = 0
    for i in range(M):
        infnorm = 0.0
        for j in range(N):
            infnorm = abs(A[i, j])
            if infnorm > maxval:
                maxval = infnorm
    return infnorm


if __name__ == "__name__":
    print("Testing Testing")
