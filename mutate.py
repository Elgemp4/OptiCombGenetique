import random
import numpy as np
import scipy
from scipy.optimize import lsq_linear

from solution import Solution


def loop(min_value: int, max_value: int, value: int) -> int:
    return ((value - min_value) % (max_value - min_value)) + min_value


def stochastic_hill_climbing(solution: Solution, lower_w: int, higher_w: int, lower_h: int, higher_h: int,
                             X: np.ndarray, iterations=3500):
    W = solution.get_W()
    H = solution.get_H()

    M, R = W.shape
    R, N = H.shape

    current_best_score = solution.score

    for _ in range(iterations):

        if random.random() < 0.5:
            i = random.randint(0, M - 1)
            r = random.randint(0, R - 1)

            old_val = W[i, r]

            delta = random.choice([-1, 1])
            new_val = old_val + delta

            if not (lower_w <= new_val <= higher_w):
                continue

            solution.change_w_at(i, r, new_val)

            if solution.score < current_best_score:
                current_best_score = solution.score
            else:
                solution.change_w_at(i, r, old_val)

        else:
            r = random.randint(0, R - 1)
            j = random.randint(0, N - 1)

            old_val = H[r, j]

            delta = random.choice([-1, 1])
            new_val = old_val + delta

            if not (lower_h <= new_val <= higher_h):
                continue

            solution.change_h_at(r, j, new_val)

            if solution.score < current_best_score:
                current_best_score = solution.score
            else:
                solution.change_h_at(r, j, old_val)

    return solution

def nnls_mutation(solution: Solution, lower_w: int, higher_w: int, lower_h: int, higher_h: int, X: np.ndarray):
    """
    Do a mutation by row or column (depending on wheter we modify W or H)
    :param solution:
    :param lower_w:
    :param higher_w:
    :param lower_h:
    :param higher_h:
    :param X:
    :return:
    """
    for i in range(5):
        L = solution.residu
        L_abs = np.abs(L)
        W = solution.get_W()
        H = solution.get_H()
        M, R = W.shape
        R, N = H.shape

        if random.random() < 0.5:
            error_per_row = np.sum(L_abs, axis=1)

            if np.sum(error_per_row) == 0:
                row_index = np.random.randint(0, M)
                probabilities = error_per_row / np.sum(error_per_row)
                row_index = np.random.choice(M, p=probabilities)

            new_W_row_float = np.linalg.lstsq(H.T, X[row_index, :], rcond=None)[0]

            new_W_row_int = np.round(new_W_row_float).astype(int)
            new_W_row_clamped = np.clip(new_W_row_int, lower_w, higher_w)
            solution.change_w_row_at(row_index, new_W_row_clamped)


        else:
            error_per_col = np.sum(L_abs, axis=0)
            if np.sum(error_per_col) == 0:
                col_index = np.random.randint(0, N)
            else:
                probabilities = error_per_col / np.sum(error_per_col)
                col_index = np.random.choice(N, p=probabilities)

            new_H_col_float = np.linalg.lstsq(W, X[:, col_index], rcond=None)[0]

            new_H_col_int = np.round(new_H_col_float).astype(int)
            new_H_col_clamped = np.clip(new_H_col_int, lower_h, higher_h)
            solution.change_h_col_at(col_index, new_H_col_clamped)

        return solution
