import random
import time
from typing import Callable

import matplotlib.pyplot as plt
from parser import read_file
from solution import Solution


def genetic(file, duration,
            select_reproduction: Callable[[list[Solution], int],list[Solution]],
            crossover: Callable[[Solution, Solution, int, int, int],tuple[Solution, Solution]],
            mutate,
            select_replacement,
            initiate_population,
            initial_count,
            reproduce_count,
            select_count):
    X, m, n, rank, lower_w, upper_w, lower_h, upper_h = read_file(file)
    thousand_iteration = 0
    iteration = 0

    score_historic = []
    time_historic = []
    population = initiate_population(X, m, n, rank, lower_w, upper_w, lower_h, upper_h, initial_count)

    limit_sec = duration * 60
    start_time = time.perf_counter()

    best_solution = None
    try:
        while True:
            reproducing_population = select_reproduction(population, reproduce_count)

            shuffled_population = reproducing_population[:]

            random.shuffle(shuffled_population)

            for i in range(len(shuffled_population) -1):
                (child1, child2) = crossover(shuffled_population[i], shuffled_population[i+1], m, n, rank)
                child1.compute_score(X)
                child2.compute_score(X)
                child1 = mutate(child1, lower_w, upper_w, lower_h, upper_h, X)
                child2 = mutate(child2, lower_w, upper_w, lower_h, upper_h, X)
                population.append(child1)
                population.append(child2)
            population = select_replacement(population, select_count)

            for pop in population:
                if best_solution is None or pop.score < best_solution.score:
                    score_historic.append(pop.score)
                    time_historic.append(time.perf_counter() - start_time)
                    best_solution = pop
                    print(f"New best solution with cost : {0}", best_solution.score)
                    if(best_solution.score == 0):
                        break;

            iteration = iteration + 1
            if iteration % 1000 == 0:
                thousand_iteration = thousand_iteration + 1
                iteration = 0
                print("hundred iterations")
                if time.perf_counter() - start_time > limit_sec:
                    print("stop")
                    break

        plot_score_evolution(score_historic,time_historic)
        return best_solution
    except (KeyboardInterrupt):

        plot_score_evolution(score_historic,time_historic)
        return best_solution




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

# --- Appel de la fonction de tracé après la boucle ---
# plot_score_evolution(score_history, time_history)