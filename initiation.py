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

    # Decision roll
    strategy_roll = random.random()

    # ====================================================
    # STRATEGY 1: SVD (Singular Value Decomposition) ~ 5%
    # ====================================================
    if strategy_roll < 0.15:
        try:
            # SVD separates X into U * S * Vt
            U, S, Vt = scipy.linalg.svd(X, full_matrices=False)

            # We keep only the top r components
            # We distribute the square root of Sigma (S) to balance values between W and H
            sqrt_S = np.sqrt(S[:r])

            W_float = U[:, :r] * sqrt_S
            H_float = sqrt_S[:, np.newaxis] * Vt[:r, :]

            # SVD gives the best reconstruction, but violates integer/bounds constraints
            # We force it back into the box
            W_curr = np.clip(np.round(W_float), lower_w, higher_w).astype(int)
            H_curr = np.clip(np.round(H_float), lower_h, higher_h).astype(int)

            for _ in range(20):
                # Fix H, Solve W:  H.T * W.T = X.T
                # lstsq solves Ax = B. It's fast and robust (handles negatives).
                W_res = scipy.linalg.lstsq(H_curr.T, X.T)
                W_curr = W_res[0].T  # Transpose back

                # Clamp immediately
                W_curr = np.clip(np.round(W_curr), lower_w, higher_w)

                # Fix W, Solve H:  W * H = X
                H_res = scipy.linalg.lstsq(W_curr, X)
                H_curr = H_res[0]

                # Clamp immediately
                H_curr = np.clip(np.round(H_curr), lower_h, higher_h)

            sol = Solution(W_curr.astype(int), H_curr.astype(int))
            sol.compute_score(X)
            return sol
        except Exception:
            # If SVD fails (rare convergence issue), fallback to Pure Random
            pass

            # ====================================================
    # STRATEGY 2: ALS Hill Climbing (Smart Random) ~ 65%
    # ====================================================
    if strategy_roll < 0.70:
        # 1. Random Start
        W_curr = np.random.randint(lower_w, higher_w + 1, size=(M, r)).astype(float)
        H_curr = np.random.randint(lower_h, higher_h + 1, size=(r, N)).astype(float)

        # 2. Fast Optimization Loop (5 iterations)
        # Uses Least Squares to quickly fit the random matrices to X
        for _ in range(20):
            # Fix H, Solve W:  H.T * W.T = X.T
            # lstsq solves Ax = B. It's fast and robust (handles negatives).
            W_res = scipy.linalg.lstsq(H_curr.T, X.T)
            W_curr = W_res[0].T  # Transpose back

            # Clamp immediately
            W_curr = np.clip(np.round(W_curr), lower_w, higher_w)

            # Fix W, Solve H:  W * H = X
            H_res = scipy.linalg.lstsq(W_curr, X)
            H_curr = H_res[0]

            # Clamp immediately
            H_curr = np.clip(np.round(H_curr), lower_h, higher_h)

        sol = Solution(W_curr.astype(int), H_curr.astype(int))
        sol.compute_score(X)
        return sol
    # ====================================================
    # STRATEGY 3: Pure Random (Chaos) ~ 30%
    # ====================================================
    # Standard random generation within bounds
    W_rand = np.random.randint(lower_w, higher_w + 1, size=(M, r))
    H_rand = np.random.randint(lower_h, higher_h + 1, size=(r, N))


    sol = Solution(W_rand, H_rand)
    sol.compute_score(X)
    return sol