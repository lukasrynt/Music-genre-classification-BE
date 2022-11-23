import numpy as np

from scipy.spatial.distance import euclidean


def dtw(first, second, window: int = 10, distance=euclidean):
    n = len(first)
    m = len(second)
    window = max(window, abs(n - m))
    matrix = np.full((n + 1, m + 1), np.inf)
    matrix[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(max(1, i - window), min(m, i + window) + 1):
            matrix[i, j] = 0

    for i in range(1, n + 1):
        for j in range(max(1, i - window), min(m, i + window) + 1):
            curr_i = i - 1
            curr_j = j - 1
            cost = distance(first[curr_i], second[curr_j])
            matrix[i, j] = cost + min(matrix[curr_i, j],
                                      matrix[i, curr_j],
                                      matrix[curr_i, curr_j])
    return matrix[n, m]
