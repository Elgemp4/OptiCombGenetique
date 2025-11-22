import random
import numpy as np

def roulette_selection(population, count ):
    scores = np.array(list(map(lambda x: x.score, population)))

    max_score = np.max(scores)

    weights = max_score - scores + 1

    return random.choices(population, weights=weights, k=count)


def select_replacement(combined_population, max_size):
    """
    Selects the best individuals, removing duplicates.
    """
    # 1. Sort the entire population by score (e.g., ascending for minimization)
    sorted_population = sorted(combined_population, key=lambda x: x.score)

    final_population = []
    seen_hashes = set()

    for individual in sorted_population:
        individual_hash = hash(individual)

        # Check for uniqueness and size limit
        if individual_hash not in seen_hashes and len(final_population) < max_size:
            final_population.append(individual)
            seen_hashes.add(individual_hash)

    return final_population

# Your main loop:
# population = select_replacement(population, 50)