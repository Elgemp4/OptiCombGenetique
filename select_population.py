import random
import numpy as np

def roulette_selection(population, count, factor=1.1):
    """
    Roulette selection, the lower the factor argument, the more less idea solutions will be chosed
    :param population:
    :param count:
    :param factor:
    :return:
    """
    scores = np.array(list(map(lambda x: x.score, population)))

    max_score = np.max(scores)

    weights = max_score - scores + 1

    weights = np.power(weights, factor)

    return random.choices(population, weights=weights, k=count)


def select_replacement(combined_population, max_size):
    """
    Selects the best individuals, removing duplicates.
    :param combined_population:
    :param max_size:
    :return:
    """
    sorted_population = sorted(combined_population, key=lambda x: x.score)

    final_population = []
    seen_hashes = set()

    for individual in sorted_population:
        individual_hash = hash(individual)

        if individual_hash not in seen_hashes and len(final_population) < max_size:
            final_population.append(individual)
            seen_hashes.add(individual_hash)

    return final_population
