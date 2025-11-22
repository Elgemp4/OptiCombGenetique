import numpy as np

from solution import Solution
from sklearn.decomposition import NMF, TruncatedSVD

def factorize_and_quantize(X, rank, lower_w, higher_w, lower_h, higher_h, method='nmf'):
    """
    Factorise X et quantifie les facteurs W et H aux entiers.

    Args:
        X (np.ndarray): Matrice d'entrée (M x N).
        rank (int): Le rang R souhaité.
        lower_w, higher_w (int): Bornes pour W.
        lower_h, higher_h (int): Bornes pour H.
        method (str): 'nmf' (par défaut, nécessite X >= 0) ou 'svd'.

    Returns:
        tuple: (W_entier, H_entier)
    """
    m, n = X.shape

    if method == 'nmf':
        # NMF est préférée si X >= 0 car elle produit naturellement des facteurs non-négatifs.
        if np.any(X < 0):
            print("Attention : NMF est utilisée mais X contient des valeurs négatives. Utilisation de SVD à la place.")
            return factorize_and_quantize(X, rank, lower_w, higher_w, lower_h, higher_h, method='svd')

        model = NMF(n_components=rank, init='random', max_iter=50000)
        W_float = model.fit_transform(X)
        H_float = model.components_

    elif method == 'svd':
        # SVD Troncquée pour les matrices générales (contient des valeurs négatives si X en a).
        # Création des facteurs W et H pour X ≈ W @ H
        svd = TruncatedSVD(n_components=rank)
        W_float = svd.fit_transform(X)
        Sigma = np.diag(svd.singular_values_)
        H_float = svd.components_

        # Pour le format W @ H, on peut intégrer la matrice Sigma à l'un des facteurs.
        W_float = W_float @ np.sqrt(Sigma)
        H_float = np.sqrt(Sigma) @ H_float

    else:
        raise ValueError("Méthode doit être 'nmf' ou 'svd'.")

    # 1. Arrondi aux entiers (Quantification)
    W_entier = np.round(W_float).astype(int)
    H_entier = np.round(H_float).astype(int)

    # 2. Clamping pour respecter les bornes de la métaheuristique
    # C'est crucial pour que ces solutions soient dans l'espace de recherche valide.
    W_clamped = np.clip(W_entier, lower_w, higher_w)
    H_clamped = np.clip(H_entier, lower_h, higher_h)

    return W_clamped, H_clamped

def initiate_randomly(X, m, n, rank, lower_w, higher_w, lower_h, higher_h, count):
    population = []

    # --- 🎯 NOUVELLE PARTIE : Initialisation Biaisée (Factorisation) ---

    # 1. Déterminer la méthode de factorisation
    if np.any(X < 0):
        # Si X contient des négatifs, on utilise la SVD
        method = 'svd'
    else:
        # Si X est non-négative, la NMF est préférable
        method = 'nmf'

    print(f"Génération d'une solution par {method.upper()}...")

    try:
        W_fact, H_fact = factorize_and_quantize(
            X, rank, lower_w, higher_w, lower_h, higher_h, method=method
        )

        # 2. Créer l'objet Solution et l'ajouter
        sol_fact = Solution(W_fact, H_fact)
        sol_fact.compute_score(X)
        population.append(sol_fact)

        # S'assurer que le nombre total d'individus reste 'count'
        remaining_count = count - 1

    except Exception as e:
        # En cas d'erreur de factorisation (ex: rang trop grand pour SVD)
        print(f"Erreur lors de la factorisation : {e}. On continue avec une population aléatoire complète.")
        remaining_count = count

    # --- PARTIE EXISTANTE : Remplissage avec des solutions Aléatoires ---

    for i in range(remaining_count):
        # Utilisation de np.random.randint pour les valeurs entières dans les bornes
        w = np.random.randint(lower_w, higher_w + 1, (m, rank))
        h = np.random.randint(lower_h, higher_h + 1, (rank, n))
        sol = Solution(w, h)
        sol.compute_score(X)
        population.append(sol)

    return population


def initiate_algo(X, m, n, rank, lower_w, higher_w, lower_h, higher_h, count):
    population = []

    # --- Solutions basées sur NMF/SVD ---
    method = 'nmf' if np.all(X >= 0) else 'svd'

    for k in range(5):
        # Utiliser un seed différent ou laisser la fonction NMF/SVD
        # (si init='random') générer un départ aléatoire différent.
        try:
            W_fact, H_fact = factorize_and_quantize(
                X, rank, lower_w, higher_w, lower_h, higher_h, method=method
            )
            sol_fact = Solution(W_fact, H_fact)
            sol_fact.compute_score(X)

            # S'assurer de ne pas ajouter de doublons (peu probable avec NMF)
            if sol_fact not in population:
                population.append(sol_fact)

        except Exception as e:
            print(f"Erreur lors de la factorisation ({k + 1}/{num_factorized}) : {e}")


    remaining_count = count

    # --- PARTIE EXISTANTE : Remplissage avec des solutions Aléatoires ---

    for i in range(remaining_count):
        # Utilisation de np.random.randint pour les valeurs entières dans les bornes
        w = np.random.randint(lower_w, higher_w + 1, (m, rank))
        h = np.random.randint(lower_h, higher_h + 1, (rank, n))
        sol = Solution(w, h)
        sol.compute_score(X)
        population.append(sol)

    return population