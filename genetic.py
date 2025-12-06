import random
import time
from concurrent.futures.process import ProcessPoolExecutor
from typing import Callable

import matplotlib.pyplot as plt
from parser import read_file
from solution import Solution


def worker_create_one(args):
    """
    Creates ONE individual and computes its score.
    """
    (seed_val, init, m, n, rank, lower_w, upper_w, lower_h, upper_h, X) = args

    random.seed(seed_val)

    sol = init(X, m, n, rank, lower_w, upper_w, lower_h, upper_h)

    return sol


def worker_process_offspring(args):
    """
    Mutates ONE child and computes its score.
    """
    (child, mutation_func, lower_w, upper_w, lower_h, upper_h, X) = args
    child.compute_score(X)
    # 1. Mutate
    # (No need to re-seed here as we are transforming existing data)
    child = mutation_func(child, lower_w, upper_w, lower_h, upper_h, X)

    # 2. Score
    child.compute_score(X)

    return child

def genetic(file, duration,
            select_reproduction: Callable[[list[Solution], int],list[Solution]],
            crossover: Callable[[Solution, Solution, int, int, int],tuple[Solution, Solution]],
            mutate_search,
            mutate_intensify,
            select_replacement,
            initiate_population,
            initial_count,
            reproduce_count,
            select_count):
    best_solution = None
    try:
        with ProcessPoolExecutor() as executor:
            X, m, n, rank, lower_w, upper_w, lower_h, upper_h = read_file(file)

            score_historic = []
            time_historic = []

            limit_sec = duration * 60
            start_time = time.perf_counter()

            method = ""
            last_score = 0

            init_tasks = []
            for i in range(initial_count):
                # We pass 'time + i' to ensure every worker gets a unique random seed
                seed_val = time.time() + i
                init_tasks.append((seed_val, initiate_population, m, n, rank, lower_w, upper_w, lower_h, upper_h, X))

            population = list(executor.map(worker_create_one, init_tasks))

            while True:
                reproducing_population = select_reproduction(population, reproduce_count)

                shuffled_population = reproducing_population[:]

                random.shuffle(shuffled_population)
                evolution_tasks = []

                for i in range(len(shuffled_population) -1):
                    (child1, child2) = crossover(shuffled_population[i], shuffled_population[i+1], m, n, rank)
                    if random.random() < .5:
                        method= "search"
                        mutate = mutate_search
                    else:
                        method = "intensify"
                        mutate = mutate_intensify

                    evolution_tasks.append((child1, mutate, lower_w, upper_w, lower_h, upper_h, X))
                    evolution_tasks.append((child2, mutate, lower_w, upper_w, lower_h, upper_h, X))


                new_childs = list(executor.map(worker_process_offspring, evolution_tasks))
                population.extend(new_childs)
                population = select_replacement(population, select_count)

                for pop in population:
                    if best_solution is None or pop.score < best_solution.score:
                        score_historic.append(pop.score)
                        time_historic.append(time.perf_counter() - start_time)
                        best_solution = pop
                        print(f"New best solution with cost : {best_solution.score}, ({last_score - best_solution.score}) - {method}")
                        last_score = best_solution.score
                        if(best_solution.score == 0):
                            break;

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