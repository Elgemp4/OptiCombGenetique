import random
import time
from typing import Callable


from parser import read_file
from solution import Solution

class Genetic:

    def __init__(self, file, duration,
                select_reproduction: Callable[[list[Solution], int],list[Solution]],
                crossover: Callable[[Solution, Solution, int, int, int],tuple[Solution, Solution]],
                mutate_search,
                mutate_intensify,
                select_replacement,
                initiate_population,
                initial_count,
                reproduce_count,
                select_count,
                island_id):

        self.initialized = False
        self.file = file
        self.duration = duration
        self.select_reproduction = select_reproduction
        self.crossover = crossover
        self.mutate_search = mutate_search
        self.mutate_intensify = mutate_intensify
        self.select_replacement = select_replacement
        self.initiate_population = initiate_population
        self.initial_count = initial_count
        self.reproduce_count = reproduce_count
        self.select_count = select_count
        self.island_id = island_id
        self.init = False

    def initialize(self):
        random.seed(time.time() + self.island_id)
        self.X, self.m, self.n, self.rank, self.lower_w, self.upper_w, self.lower_h, self.upper_h = read_file(self.file)

        self.score_historic = []
        self.time_historic = []
        self.population = self.initiate_population(self.X, self.m, self.n, self.rank, self.lower_w, self.upper_w, self.lower_h, self.upper_h, self.initial_count)
        self.init = True

    def run(self):

        if(not self.init):
            self.initialize()


        method=""
        last_score=0

        generation = 0
        best_solution = None
        try:
            while True:
                reproducing_population = self.select_reproduction(self.population, self.reproduce_count)

                shuffled_population = reproducing_population[:]

                random.shuffle(shuffled_population)

                for i in range(len(shuffled_population) -1):
                    (child1, child2) = self.crossover(shuffled_population[i], shuffled_population[i+1], self.m, self.n, self.rank)
                    child1.compute_score(self.X)
                    child2.compute_score(self.X)
                    if random.random() < 0.5:
                        method= "search"
                        mutate = self.mutate_search
                    else:
                        method = "intensify"
                        mutate = self.mutate_intensify
                    child1 = mutate(child1, self.lower_w, self.upper_w, self.lower_h, self.upper_h, self.X)
                    child2 = mutate(child2, self.lower_w, self.upper_w, self.lower_h, self.upper_h, self.X)
                    self.population.append(child1)
                    self.population.append(child2)
                population = self.select_replacement(self.population, self.select_count)

                for pop in population:
                    if best_solution is None or pop.score < best_solution.score:
                        best_solution = pop
                        print(f"New best solution with cost : {best_solution.score}, ({last_score - best_solution.score}) - {method}")
                        last_score = best_solution.score
                        if(best_solution.score == 0):
                            break;

                generation = generation + 1
                if generation > self.duration:
                    print("stop")
                    break

            return {
                "solution": best_solution,
                "island_id": self.island_id,
            }
        except (KeyboardInterrupt):
            return {
                "solution": best_solution,
                "island_id": self.island_id,
            }






# --- Appel de la fonction de tracé après la boucle ---
# plot_score_evolution(score_history, time_history)