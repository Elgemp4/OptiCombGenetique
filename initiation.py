import numpy as np
from scipy.linalg import lu, qr

from solution import Solution
from sklearn.decomposition import NMF, TruncatedSVD, PCA, FastICA

import numpy as np
import scipy.linalg
from solution import Solution  # Assurez-vous que l'import correspond à votre projet

import numpy as np
import scipy.linalg
import random
from solution import Solution


def generate_smart_solution(X: np.ndarray, M: int, N:int, r: int, lower_w: int, higher_w: int, lower_h: int,
                            higher_h: int) -> Solution:

    """
    Generate one of the initial solution randomly, or thanks to SVD.
    :param X:
    :param M:
    :param N:
    :param r:
    :param lower_w:
    :param higher_w:
    :param lower_h:
    :param higher_h:
    :return:
    """
    strategy_roll = random.random()

    if strategy_roll < 0.05:
        try:
            U, S, Vt = scipy.linalg.svd(X, full_matrices=False)

            sqrt_S = np.sqrt(S[:r])

            W_float = U[:, :r] * sqrt_S
            H_float = sqrt_S[:, np.newaxis] * Vt[:r, :]

            W_curr = np.clip(np.round(W_float), lower_w, higher_w).astype(int)
            H_curr = np.clip(np.round(H_float), lower_h, higher_h).astype(int)

            for _ in range(20):
                W_res = scipy.linalg.lstsq(H_curr.T, X.T)
                W_curr = W_res[0].T

                W_curr = np.clip(np.round(W_curr), lower_w, higher_w)

                H_res = scipy.linalg.lstsq(W_curr, X)
                H_curr = H_res[0]

                H_curr = np.clip(np.round(H_curr), lower_h, higher_h)

            sol = Solution(W_curr.astype(int), H_curr.astype(int))
            sol.compute_score(X)
            return sol
        except Exception:
            pass

    if strategy_roll < 0.70:
        W_curr = np.random.randint(lower_w, higher_w + 1, size=(M, r)).astype(float)
        H_curr = np.random.randint(lower_h, higher_h + 1, size=(r, N)).astype(float)

        for _ in range(20):
            W_res = scipy.linalg.lstsq(H_curr.T, X.T)
            W_curr = W_res[0].T

            W_curr = np.clip(np.round(W_curr), lower_w, higher_w)

            H_res = scipy.linalg.lstsq(W_curr, X)
            H_curr = H_res[0]

            H_curr = np.clip(np.round(H_curr), lower_h, higher_h)

        sol = Solution(W_curr.astype(int), H_curr.astype(int))
        sol.compute_score(X)
        return sol

    W_rand = np.random.randint(lower_w, higher_w + 1, size=(M, r))
    H_rand = np.random.randint(lower_h, higher_h + 1, size=(r, N))


    sol = Solution(W_rand, H_rand)
    sol.compute_score(X)
    return sol