import random
import numpy
import numpy as np

from solution import Solution


def one_point_crossover(parent1, parent2):
    return n_crossover(parent1,parent2,1)

def two_point_crossover(parent1, parent2):
    return n_crossover(parent1, parent2, 2)


def n_crossover(parent1 : Solution, parent2 : Solution, nb_section: int):
    (child1_W, child2_W) = n_crossover_an_array(parent1.get_W().T, parent2.get_W().T, nb_section)
    (child1_H, child2_H) = n_crossover_an_array(parent1.get_H(), parent2.get_H(), nb_section)

    child1 = Solution(child1_W.T, child1_H)
    child2 = Solution(child2_W.T, child2_H)

    return (child1,child2)

def n_crossover_an_array(arr1,arr2, nb_sections):
    size = len(arr1[0])
    rank = len(arr1)
    cuts = random.sample(range(0,size), nb_sections)
    new_arr1, new_arr2 = np.empty((rank,0))
    flag = False
    previous = 0
    for i in cuts:
        if flag:
            cut1 = arr1[:, previous:i].reshape(rank, 0)
            cut2 = arr2[:, previous:i].reshape(rank, 0)
        else:
            cut2 = arr1[:, previous:i].reshape(rank, 0)
            cut1 = arr2[:, previous:i].reshape(rank, 0)

        new_arr1 = np.hstack((new_arr1, cut1))
        new_arr2 = np.hstack((new_arr2, cut2))

    return (new_arr1, new_arr2)