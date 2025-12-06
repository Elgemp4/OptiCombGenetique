import concurrent.futures

import click
import matplotlib.pyplot as plt
import numpy as np

from crossover import two_point_crossover, one_point_crossover, uniform_crossover
from initiation import initiate_randomly, initiate_algo
from mutate import mutate_local_search_addition, mutate, gradient_mutation, sparse_mutation, stochastic_hill_climbing
from parser import read_file
from select_population import roulette_selection, select_replacement
from utils import solutionIsFeasible

from genetic import Genetic

def island_wrapper(island):
    """
    This function configures one 'Island'.
    It sets up the specific arguments for that island.
    """
    print(f"Island {island.island_id} starting...")

    # You can even give different islands different parameters!
    # e.g., Island 0 searches fast, Island 1 searches deep.
    result = island.run()
    print("Island finished successfully")
    return result

def plot_score_evolution(score_history, time_history, y_label='Score (Erreur L)'):
    """
    Crée le graphique de l'évolution du score en fonction du temps.
    """

    # Créer la figure et l'axe
    fig, ax = plt.subplots(figsize=(10, 6))

    # Tracer les données (Score vs. Temps)
    ax.plot(time_history, score_history, marker='o', linestyle='-', color='b', markersize=3)

    # Ajouter des labels et un titre
    ax.set_title("Évolution du Meilleur Score au fil du Temps", fontsize=14)
    ax.set_xlabel("Temps écoulé (secondes)", fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)

    # Ajouter une grille pour faciliter la lecture
    ax.grid(True, linestyle='--', alpha=0.7)

    # Mettre l'axe Y à l'échelle logarithmique si le score change énormément
    # (Utile si vous commencez à 300 millions et finissez à 1 million)
    # ax.set_yscale('log')
    filename = "evolution_score.png"
    plt.savefig(filename)

    print(f"\nGraphique enregistré sous : {filename}")
    # Afficher le graphique
    plt.show()

@click.command()
@click.argument("file")
def enter_point(file):
    num_islands = 4

    islands = []

    for i in range(num_islands):
        # 1. Create the instance
        island = Genetic(
            file=file,
            duration=10000,  # 10 generation
            select_reproduction=roulette_selection,
            crossover=uniform_crossover,
            mutate_search=sparse_mutation,
            mutate_intensify=stochastic_hill_climbing,
            select_replacement=select_replacement,
            initiate_population=initiate_algo,
            initial_count=30,
            reproduce_count=2,
            select_count=30,
            island_id=i
        )

        islands.append(island)

    results = []
    future_to_island = {}

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for island in islands:
            # We submit the OBJECT instance itself
            future = executor.submit(island_wrapper, island)
            future_to_island[future] = island.island_id


        for future in concurrent.futures.as_completed(future_to_island):
            try:
                res = future.result()
                results.append(res)
                print(f"Island {res['island_id']} finished with score: {res['solution'].score}")
            except Exception as exc:
                print(f"Island failed with exception: {exc}")

    X, m, n, rank, lower_w, upper_w, lower_h, upper_h = read_file(file)

    best_of_all = min(results, key=lambda x: x['solution'].score)
    best = best_of_all["solution"]
    print(best.score)

    print(solutionIsFeasible(best.get_W(), best.get_H(), rank, lower_w, upper_w, lower_h, upper_h))

    plot_score_evolution(best_of_all["score_historic"], best_of_all["time_historic"])

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


