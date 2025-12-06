from email.policy import default

import click

from crossover import two_point_crossover, one_point_crossover, uniform_crossover
from genetic import genetic
from initiation import initiate_randomly, initiate_algo
from mutate import mutate_local_search_addition, mutate, gradient_mutation, sparse_mutation, stochastic_hill_climbing, \
    block_mutation
from parser import read_file
from select_population import roulette_selection, select_replacement
from utils import solutionIsFeasible


@click.command()
@click.argument("file")
def enter_point(file):
    best = genetic(file=file,
                   select_reproduction=roulette_selection,
                   select_replacement=select_replacement,
                   duration=10,
                   crossover=uniform_crossover,
                   initiate_population=initiate_algo,
                   mutate_search=block_mutation,
                   mutate_intensify=stochastic_hill_climbing,
                   reproduce_count=110,
                   select_count=200,
                   initial_count=300)


    X, m, n, rank, lower_w, upper_w, lower_h, upper_h = read_file(file)

    print(best.get_W())
    print(best.get_H())
    print(best.score)
    best.compute_score(X)
    print(best.score)

    print(solutionIsFeasible(best.get_W(), best.get_H(), rank, lower_w, upper_w, lower_h, upper_h))

    import numpy as np
    import matplotlib.pyplot as plt

    # 1. Reconstruire la matrice finale (Produit Matriciel)
    # L'opérateur '@' fait la multiplication matricielle (dot product)
    x_recontructed = best.get_W() @ best.get_H()

    # Si vous n'utilisez pas Python 3.5+, utilisez : np.dot(W_best, H_best)

    # 2. Afficher le résultat
    plt.figure(figsize=(12, 6))

    # --- Image A : La matrice Originale (Cible) ---
    plt.subplot(1, 3, 1)
    plt.title("Originale (Cible)")
    # 'cmap' définit les couleurs (ex: 'gray', 'viridis', 'plasma')
    plt.imshow(X, cmap='viridis', aspect='auto')
    plt.savefig("original.png")
    plt.colorbar()

    # --- Image B : La matrice Reconstruite (Résultat) ---
    plt.subplot(1, 3, 2)
    plt.title("Reconstruction (W x H)")
    plt.imshow(x_recontructed, cmap='viridis', aspect='auto')
    plt.savefig("reconstructed.png")
    plt.colorbar()

    # --- Image C : La différence (L'erreur visible) ---
    # C'est très utile pour voir ce que l'algo a raté
    plt.subplot(1, 3, 3)
    plt.title("Différence (Erreur)")
    plt.imshow(np.abs(X - x_recontructed), cmap='magma', aspect='auto')
    plt.savefig("error.png")
    plt.colorbar()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    enter_point()


