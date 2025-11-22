import random

import numpy as np

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
    if random.random() <0.1:
        return mutate(solution,lower_w,higher_w,lower_h, higher_h, X)
    else:
        return gradient_mutation(solution, lower_w, higher_w, lower_h, higher_h, X)


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