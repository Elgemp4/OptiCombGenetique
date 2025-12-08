import random

import numpy as np

import solution
from solution import Solution
import random
import numpy as np
from solution import Solution


def loop(min_value: int, max_value: int, value: int) -> int:
    return ((value - min_value) % (max_value - min_value)) + min_value


def stochastic_hill_climbing(solution: Solution, lower_w: int, higher_w: int, lower_h: int, higher_h: int,
                             X: np.ndarray, iterations=500):
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

def gradient_mutation(solution: Solution, lower_w: int, higher_w: int, lower_h: int, higher_h: int, X: np.ndarray):
    """
    Effectue une mutation par bloc (ligne de W ou colonne de H) en appliquant
    un pas d'optimisation vectoriel.
    """
    for i in range(10000):
        E = solution.residu
        E_abs = np.abs(E)
        W = solution.get_W()
        H = solution.get_H()
        M, R = W.shape
        R, N = H.shape

        # 1. Sélection Biaisée du Bloc à Muter (W ou H)
        if random.random() < 0.5:
            # --- MUTATION SUR W (Ligne complète) ---

            # a. Sélection Biaisée de la LIGNE 'i' (la plus grande erreur)
            error_per_row = np.sum(E_abs, axis=1)
            if np.sum(error_per_row) == 0:
                row_index = np.random.randint(0, M)
            else:
                probabilities = error_per_row / np.sum(error_per_row)
                row_index = np.random.choice(M, p=probabilities)

            # b. Calcul du Pas Optimal VECTORIEL pour la ligne W[i, :]

            # Le pas optimal de descente de gradient pour une ligne i est:
            # W_i_opt = W_i_old + (E_i @ H^T) @ (H @ H^T)^-1
            # Pour simplifier et accélérer, nous allons utiliser une approche inspirée de la NMF.

            # Gradient (Pente) : -2 * E[i, :] @ H.T
            # Terme_Gradient est le vecteur (1 x R) de toutes les dérivées partielles pour la ligne i
            Terme_Gradient_Vec = E[row_index, :] @ H.T

            # Matrice de Courbure (Hessienne) : 2 * H @ H.T
            Terme_Courbure_Mat = H @ H.T

            # c. Calcul du Mouvement (Résolution du système linéaire)
            # La valeur optimale est obtenue en résolvant le système (Terme_Courbure_Mat @ Delta_W_i) = Terme_Gradient_Vec

            # Sécurité : vérifier que la matrice de courbure est bien conditionnée
            try:
                # np.linalg.solve trouve le déplacement optimal pour la ligne W[i, :]
                Delta_W_Vec = np.linalg.solve(Terme_Courbure_Mat, Terme_Gradient_Vec)

                # Application du Pas
                new_W_row_float = W[row_index, :] + Delta_W_Vec
            except np.linalg.LinAlgError:
                # En cas de problème de matrice singulière, utiliser la méthode précédente
                print("Avertissement: Matrice de courbure singulière. Utilisation de la mutation aléatoire.")
                return solution  # On peut choisir de ne rien faire ou d'utiliser la mutation aléatoire simple ici

            # d. Clamping et Mise à Jour
            new_W_row_int = np.round(new_W_row_float).astype(int)

            # Appliquer le clamping à tous les éléments de la ligne
            new_W_row_clamped = np.clip(new_W_row_int, lower_w, higher_w)

            # Mise à jour de la ligne entière (cette mise à jour doit être implémentée dans Solution)
            solution.change_w_row_at(row_index, new_W_row_clamped)


        else:
            # --- MUTATION SUR H (Colonne complète) ---

            # a. Sélection Biaisée de la COLONNE 'j'
            error_per_col = np.sum(E_abs, axis=0)
            if np.sum(error_per_col) == 0:
                col_index = np.random.randint(0, N)
            else:
                probabilities = error_per_col / np.sum(error_per_col)
                col_index = np.random.choice(N, p=probabilities)

            # b. Calcul du Pas Optimal VECTORIEL pour la colonne H[:, j]

            # Terme_Gradient est le vecteur (R x 1)
            Terme_Gradient_Vec = W.T @ E[:, col_index]

            # Matrice de Courbure : W.T @ W
            Terme_Courbure_Mat = W.T @ W

            try:
                Delta_H_Vec = np.linalg.solve(Terme_Courbure_Mat, Terme_Gradient_Vec)
                new_H_col_float = H[:, col_index] + Delta_H_Vec
            except np.linalg.LinAlgError:
                print("Avertissement: Matrice de courbure singulière. Utilisation de la mutation aléatoire.")
                return solution

            # c. Clamping et Mise à Jour
            new_H_col_int = np.round(new_H_col_float).astype(int)
            new_H_col_clamped = np.clip(new_H_col_int, lower_h, higher_h)

            # Mise à jour de la colonne entière (doit être implémentée dans Solution)
            solution.change_h_col_at(col_index, new_H_col_clamped)

        return solution
