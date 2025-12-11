from utils import fobj
import numpy as np


class Solution:
    def __init__(self, W: np.ndarray, H: np.ndarray):
        self.residu = None
        self.W = W
        self.H = H
        self.score = None

    def get_score(self) -> float:
        return self.score

    def compute_score(self, X):
        """
        Compute the score of the solution
        :param X:
        :return:
        """
        self.score, self.residu = fobj(X, self.W, self.H)
        self.score = round(self.score)

    def get_W(self) -> np.ndarray:
        return self.W

    def get_H(self) -> np.ndarray:
        return self.H

    def change_w_at(self, i: int, r: int, new_value: float):
        """
        Change the row for W and incrementaly recompute the score
        :param i:
        :param r:
        :param new_value:
        :return:
        """
        if self.score is None or self.residu is None:
            raise ValueError("Score and residu must be initialized with compute_score(X) before calling change_w_at.")

        delta = new_value - self.W[i, r]

        if delta == 0.0:
            return

        E_i = self.residu[i, :]
        H_r = self.H[r, :]
        Terme1 = np.sum(E_i * H_r)

        Terme2 = np.sum(H_r ** 2)

        new_score = self.score - 2 * delta * Terme1 + delta ** 2 * Terme2

        self.residu[i, :] -= delta * H_r

        self.W[i, r] = new_value
        self.score = round(new_score)

    def change_h_at(self, r: int, j: int, new_value: float):
        """
        Change the column for H and incrementaly recompute the score
        :param r:
        :param j:
        :param new_value:
        :return:
        """
        if self.score is None or self.residu is None:
            raise ValueError("Score and residu must be initialized with compute_score(X) before calling change_h_at.")

        delta = new_value - self.H[r, j]

        if delta == 0.0:
            return

        E_j = self.residu[:, j]
        W_r = self.W[:, r]

        first_term = np.sum(E_j * W_r)

        second_term = np.sum(W_r ** 2)

        new_score = self.score - 2 * delta * first_term + delta ** 2 * second_term

        self.residu[:, j] -= delta * W_r

        self.H[r, j] = new_value
        self.score = round(new_score)

    def clone(self):
        new = Solution(self.W.copy(), self.H.copy())
        new.score = self.score
        new.residu = self.residu.copy()
        return new

    def change_w_row_at(self, i: int, new_row: np.ndarray):
        """
        Change a single value at a random position in H and incrementaly recompute the score
        :param i:
        :param new_row:
        :return:
        """
        old_row = self.W[i, :]
        self.W[i, :] = new_row

        delta_W_row = old_row - new_row

        update_term = delta_W_row @ self.H

        self.residu[i, :] += update_term

        self.compute_score_from_residu()

    def change_h_col_at(self, j: int, new_col: np.ndarray):
        """
        Change a single value at a random position in H and incrementaly recompute the score
        :param j:
        :param new_col:
        :return:
        """
        old_col = self.H[:, j]
        self.H[:, j] = new_col


        delta_H_col = old_col - new_col

        update_term = self.W @ delta_H_col

        self.residu[:, j] += update_term

        self.compute_score_from_residu()

    def compute_score_from_residu(self):
        """
        Recompute the score after a change

        :return:
        """
        self.score = round(np.linalg.norm(self.residu, ord='fro') ** 2)

    def __eq__(self, other):
        return np.array_equal(self.W, other.W) and np.array_equal(self.H, other.H)

    def __hash__(self):
        return hash((self.W.tobytes(), self.H.tobytes()))