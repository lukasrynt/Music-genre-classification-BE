import numpy as np

from scipy.spatial.distance import euclidean


def dtw(first, second, window: int = 10, distance=euclidean):
    n = len(first)
    m = len(second)
    window = max(window, abs(n - m))
    matrix = np.full((n + 1, m + 1), np.inf)

    for i in range(n + 1):
        for j in range(max(1, i - window), min(m, i + window) + 1):
            matrix[i, j] = 0

    for i in range(n + 1):
        for j in range(max(1, i - window), min(m, i + window) + 1):
            curr_i = i - 1
            curr_j = j - 1
            cost = distance(first[curr_i], second[curr_j])
            matrix[i, j] = cost + min(matrix[curr_i - 1, curr_j],  # insertion
                                      matrix[curr_i, curr_j - 1],  # deletion
                                      matrix[curr_i - 1, curr_j - 1])  # match
    return matrix[n, m]
