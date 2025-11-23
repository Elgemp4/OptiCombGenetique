import random

import numpy as np

import solution
from solution import Solution


def loop(min_value: int, max_value: int, value: int) -> int:
    return ((value - min_value) % (max_value - min_value)) + min_value

def mutate_local_search_addition(solution: Solution, lower_w: int, higher_w: int, lower_h: int, higher_h: int, X: np.ndarray) -> Solution:
    best_solution = solution
    while True:
        (best,has_changed) = get_best_neighbour(best_solution, lower_w, higher_w, lower_h, higher_h, X)
        best_solution = best
        if not has_changed:
            break

    return best_solution


import random
import numpy as np
from solution import Solution


def stochastic_hill_climbing(solution: Solution, lower_w: int, higher_w: int, lower_h: int, higher_h: int,
                             X: np.ndarray, iterations=1000):
    """
    Applies Stochastic Hill Climbing using the incremental update methods of the Solution class.
    Strategy: "Tentative Move" -> If score improves, keep it. If not, revert immediately.
    """

    # Pre-fetch dimensions to avoid calling len() inside the loop
    # We access the arrays directly for reading speed
    W = solution.get_W()
    H = solution.get_H()

    M, R = W.shape
    R, N = H.shape

    # We track the current best score locally to compare quickly
    current_best_score = solution.score

    for _ in range(iterations):

        # 50% chance to modify W, 50% chance to modify H
        if random.random() < 0.5:
            # --- MODIFY W ---
            i = random.randint(0, M - 1)
            r = random.randint(0, R - 1)

            old_val = W[i, r]

            # Try +1 or -1
            delta = random.choice([-1, 1])
            new_val = old_val + delta

            # Check bounds (Clamping)
            if not (lower_w <= new_val <= higher_w):
                continue

            # 1. APPLY (Incremental Update)
            solution.change_w_at(i, r, new_val)

            # 2. CHECK & DECIDE
            if solution.score < current_best_score:
                # Improvement: Update local tracker and continue
                current_best_score = solution.score
            else:
                # Degradation or Stagnation: REVERT immediately
                # Reverting uses the same incremental logic, so it's mathematically consistent
                solution.change_w_at(i, r, old_val)

        else:
            # --- MODIFY H ---
            r = random.randint(0, R - 1)
            j = random.randint(0, N - 1)

            old_val = H[r, j]

            delta = random.choice([-1, 1])
            new_val = old_val + delta

            if not (lower_h <= new_val <= higher_h):
                continue

            # 1. APPLY
            solution.change_h_at(r, j, new_val)

            # 2. CHECK & DECIDE
            if solution.score < current_best_score:
                current_best_score = solution.score
            else:
                solution.change_h_at(r, j, old_val)

    return solution

def get_best_neighbour(solution: Solution, lower_w: int, higher_w: int, lower_h: int, higher_h: int, X: np.ndarray) -> tuple[Solution, bool]:
    height_w = len(solution.get_W())
    width_w = len(solution.get_W()[0])
    height_h = len(solution.get_H())
    width_h = len(solution.get_H()[0])
    best = solution
    has_changed = False

    for y in range(height_w):
        for x in range(width_w):

            new = solution.clone()
            new.change_w_at(y,x,loop(lower_w, higher_w, solution.get_W()[y, x] + random.randint(lower_w, higher_w)))

            if best is None or best.score > new.score:
                best = new
                has_changed = True

    for y in range(height_h):
        for x in range(width_h):
            new = solution.clone()
            new.change_h_at(y,x,loop(lower_h, higher_h, solution.get_H()[y, x] + random.randint(lower_h, higher_h)))

            if best is None or best.score > new.score:
                best = new
                has_changed = True

    return best, has_changed

def sparse_mutation(solution: Solution, lower_w: int, higher_w: int, lower_h: int, higher_h: int, X: np.ndarray):
    if random.random() < 0.35:
        return mutate(solution,lower_w,higher_w,lower_h, higher_h, X)
    else:
        return block_mutation(solution, lower_w, higher_w, lower_h, higher_h, X)


def mutate(solution: Solution, lower_w: int, higher_w: int, lower_h: int, higher_h: int, X: np.ndarray):
    if random.randint(0,100) < 75:
        height = len(solution.get_W())
        width = len(solution.get_W()[0])
        solution.change_w_at(random.randint(0,height-1),random.randint(0,width-1), loop(lower_w, higher_w, random.randint(lower_w, higher_w)))

        height = len(solution.get_H())
        width = len(solution.get_H()[0])
        solution.change_h_at(random.randint(0,height-1), random.randint(0, width-1), loop(lower_h, higher_h, random.randint(lower_h, higher_h)))

    return solution

def gradient_mutation(solution: Solution, lower_w: int, higher_w: int, lower_h: int, higher_h: int, X: np.ndarray):
    for i in range(100):
        E = solution.residu
        E_abs = np.abs(E)
        W = solution.get_W()
        H = solution.get_H()
        M,R = W.shape
        R,N = H.shape

        error_per_row = np.sum(E_abs, axis=1)
        if np.sum(error_per_row) == 0:
            index = np.random.randint(0, M)
        else:
            # Sélection biaisée (Roulette Wheel Selection)
            proba = error_per_row / np.sum(error_per_row)
            index = np.random.choice(M, p=proba)
        r_index = np.random.randint(R)
        E_i = E[index, :]  # Ligne i du résidu
        H_r = H[r_index, :]  # Ligne r de H

        # Le terme du gradient est : -2 * np.sum(E_i * H_r)
        Terme1 = np.sum(E_i * H_r)

        # Le dénominateur du pas optimal est : 2 * np.sum(H_r^2)
        Terme2 = np.sum(H_r ** 2)
        if Terme2 == 0:
            new_value = W[index, r_index]  # Ne change rien
        else:
            new_value_float = W[index, r_index] + Terme1 / Terme2

            # Application de l'arrondi et du clamping
            new_value = int(np.clip(np.round(new_value_float), lower_w, higher_w))

            # d. Appliquer le changement et mettre à jour le score (méthode incrémentale)
        solution.change_w_at(index, r_index, new_value)

        E = solution.residu
        E_abs = np.abs(E)
        W = solution.get_W()
        H = solution.get_H()
        M, R = W.shape
        R, N = H.shape


        error_per_col = np.sum(E_abs, axis=0)

        if np.sum(error_per_col) == 0:
            col_index = np.random.randint(0, N)
        else:
            probabilities = error_per_col / np.sum(error_per_col)
            col_index = np.random.choice(N, p=probabilities)

        # b. Le rang r à muter est choisi aléatoirement
        rank_index = np.random.randint(0, R)

        # c. Calculer le pas optimal (mini-optimisation locale)

        # Dérivée = -2 * (W_r^T @ E_j)
        W_r = W[:, rank_index]  # Colonne r de W
        E_j = E[:, col_index]  # Colonne j du résidu

        Terme1 = np.sum(W_r * E_j)
        Terme2 = np.sum(W_r ** 2)

        if Terme2 == 0:
            new_value = H[rank_index, col_index]
        else:
            new_value_float = H[rank_index, col_index] + Terme1 / Terme2

            # Application de l'arrondi et du clamping
            new_value = int(np.clip(np.round(new_value_float), lower_h, higher_h))

        # d. Appliquer le changement et mettre à jour le score
        solution.change_h_at(rank_index, col_index, new_value)
    return solution


def block_mutation(solution: Solution, lower_w: int, higher_w: int, lower_h: int, higher_h: int, X: np.ndarray):
    """
    Effectue une mutation par bloc (ligne de W ou colonne de H) en appliquant
    un pas d'optimisation vectoriel.
    """
    for i in range(100):
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
