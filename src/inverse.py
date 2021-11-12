from typing import Union
import numpy as np
from src.det import det, build_cofactor_matrix


# with extra space
# def transpose(matrix, size):
#     transpose_matrix = matrix.copy()
#
#     for i in range(size):
#         for j in range(size):
#             transpose_matrix[i][j] = matrix[j][i]
#
#     return transpose_matrix

# optimization of both space and time
def transpose(matrix, size):
    for i in range(size):
        for j in range(i, size):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    return matrix


def adjoint(matrix, size):
    adj = [[0 for _ in range(size)] for _ in range(size)]
    for i in range(size):
        for j in range(size):
            cofactor_matrix = build_cofactor_matrix(matrix, i, j, size)
            adj[i][j] = det(cofactor_matrix, size - 1) * pow(-1, i + j)

    return transpose(adj, size)


def inverse(matrix: Union[np.array, list], size):
    matrix_det = det(matrix, size)
    matrix_adj = adjoint(matrix, size)

    for i in range(size):
        for j in range(size):
            matrix_adj[i][j] = matrix_adj[i][j] / matrix_det

    return matrix_adj
