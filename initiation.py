import numpy as np

from solution import Solution




def initiate_randomly(X, m,n,rank,lower_w, higher_w, lower_h, higher_h, count):

    population = []

    for i in range(count):
        w = np.random.randint(lower_w, higher_w+1, (m,rank))
        h = np.random.randint(lower_h, higher_h+1, (rank,n))
        sol = Solution(w,h)
        sol.compute_score(X)
        population.append(sol)

    return population