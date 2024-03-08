import itertools
import numpy as np


def generate_permutation_matrix_group(N):
    # Generate all permutations of size N
    permutations = list(itertools.permutations(range(N)))
    permutation_matrices = []

    for perm in permutations:
        matrix = np.zeros((N, N), dtype=int)
        # Fill the matrix with ones based on the permutation
        for i, j in enumerate(perm):
            matrix[i, j] = 1
        permutation_matrices.append(matrix)
    return permutation_matrices
