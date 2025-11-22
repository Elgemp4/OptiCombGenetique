from utils import fobj
import numpy as np


class Solution:
    def __init__(self, W: np.ndarray, H: np.ndarray):
        self.residu = None  # Matrice d'erreur E = X - W@H
        self.W = W
        self.H = H
        self.score = None  # L = ||E||_F^2

    def get_score(self) -> float:
        return self.score

    def compute_score(self, X):
        """Calcule le score initial et le résidu (à appeler au moins une fois)."""
        # Assurez-vous que fobj renvoie bien (score, residu)
        self.score, self.residu = fobj(X, self.W, self.H)

    def get_W(self) -> np.ndarray:
        return self.W

    def get_H(self) -> np.ndarray:
        return self.H

    ## ----------------------------------------------------
    ## MÉTHODES AVEC CALCUL INCRÉMENTAL
    ## ----------------------------------------------------

    def change_w_at(self, i: int, r: int, new_value: float):
        """
        Modifie W[i, r] et met à jour incrémentalement le score et le résidu.
        i : ligne de W (correspond à la ligne de X)
        r : colonne de W (correspond au rang)
        """
        if self.score is None or self.residu is None:
            # Sécurité : recalcule si les valeurs ne sont pas initialisées
            raise ValueError("Score and residu must be initialized with compute_score(X) before calling change_w_at.")

        # 1. Calculer le changement (delta)
        delta = new_value - self.W[i, r]

        # S'il n'y a pas de changement, ne rien faire.
        if delta == 0.0:
            return

        # --- TERMES DE LA FORMULE L' = L - 2*delta*Terme1 + delta^2*Terme2 ---

        # E_i est la ligne i du résidu (E[i, :])
        E_i = self.residu[i, :]
        # H_r est la ligne r de H (H[r, :])
        H_r = self.H[r, :]

        # Terme 1: Somme des (E_ik * H_rk)
        Terme1 = np.sum(E_i * H_r)

        # Terme 2: Somme des (H_rk)^2
        Terme2 = np.sum(H_r ** 2)

        # 2. Mise à jour du score L' = L - 2*delta*Terme1 + delta^2*Terme2
        new_score = self.score - 2 * delta * Terme1 + delta ** 2 * Terme2

        # 3. Mise à jour du résidu (E)
        # La ligne i du résidu change de -delta * H[r, :]
        self.residu[i, :] -= delta * H_r

        # 4. Appliquer le changement et mettre à jour le score
        self.W[i, r] = new_value
        self.score = new_score

    def change_h_at(self, r: int, j: int, new_value: float):
        """
        Modifie H[r, j] et met à jour incrémentalement le score et le résidu.
        r : ligne de H (correspond au rang)
        j : colonne de H (correspond à la colonne de X)
        """
        if self.score is None or self.residu is None:
            raise ValueError("Score and residu must be initialized with compute_score(X) before calling change_h_at.")

        # 1. Calculer le changement (delta)
        delta = new_value - self.H[r, j]

        if delta == 0.0:
            return

        # --- TERMES DE LA FORMULE L' = L - 2*delta*Terme1 + delta^2*Terme2 ---

        # E_j est la colonne j du résidu (E[:, j])
        E_j = self.residu[:, j]
        # W_r est la colonne r de W (W[:, r])
        W_r = self.W[:, r]

        # Terme 1: Somme des (E_ij * W_ir)
        # Note : On utilise la transposée de la formule pour le changement de H
        Terme1 = np.sum(E_j * W_r)

        # Terme 2: Somme des (W_ir)^2
        Terme2 = np.sum(W_r ** 2)

        # 2. Mise à jour du score L' = L - 2*delta*Terme1 + delta^2*Terme2
        new_score = self.score - 2 * delta * Terme1 + delta ** 2 * Terme2

        # 3. Mise à jour du résidu (E)
        # La colonne j du résidu change de -delta * W[:, r]
        self.residu[:, j] -= delta * W_r

        # 4. Appliquer le changement et mettre à jour le score
        self.H[r, j] = new_value
        self.score = new_score

    def clone(self):
        new = Solution(self.W.copy(), self.H.copy())
        new.score = self.score
        new.residu = self.residu.copy()
        return new

    def __eq__(self, other):
        return np.array_equal(self.W, other.W) and np.array_equal(self.H, other.H)

    def __hash__(self):
        return hash((self.W.tobytes(), self.H.tobytes()))