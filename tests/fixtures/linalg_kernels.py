import numpy as np


def solve_system(A, b):
    return np.linalg.solve(A, b)

def cholesky_decomp(A):
    return np.linalg.cholesky(A)
