import random
from pyexpat.errors import XML_ERROR_NO_MEMORY

import numpy as np

from solution import Solution


def uniform_crossover(parent1, parent2, m,n,rank):
    parent1_w = parent1.get_W().T
    parent2_w = parent2.get_W().T
    parent1_h = parent1.get_H()  # (rank, n)
    parent2_h = parent2.get_H()  # (rank, n)

    mask_w = np.random.rand(m) > 0.5

    child_w1 = np.empty((rank, m))
    child_w2 = np.empty((rank, m))

    child_w1[:, mask_w] = parent1_w[:, mask_w]
    child_w2[:, mask_w] = parent2_w[:, mask_w]

    child_w1[:, ~mask_w] = parent2_w[:, ~mask_w]
    child_w2[:, ~mask_w] = parent1_w[:, ~mask_w]

    mask_h = np.random.rand(n) > 0.5

    child_h1 = np.empty((rank, n))
    child_h2 = np.empty((rank, n))

    child_h1[:, mask_h] = parent1_h[:, mask_h]
    child_h2[:, mask_h] = parent2_h[:, mask_h]

    child_h1[:, ~mask_h] = parent2_h[:, ~mask_h]
    child_h2[:, ~mask_h] = parent1_h[:, ~mask_h]

    child1 = Solution(child_w1.T, child_h1)
    child2 = Solution(child_w2.T, child_h2)

    return child1, child2

def one_point_crossover(parent1, parent2,m,n,rank):
    return n_crossover(parent1,parent2,1,m,n,rank)

def two_point_crossover(parent1, parent2,m,n,rank):
    return n_crossover(parent1, parent2, 2,m,n,rank)


def n_crossover(parent1 : Solution, parent2 : Solution, nb_section: int, m,n,rank):
    (child1_w, child2_w) = n_crossover_an_array(parent1.get_W().T, parent2.get_W().T, nb_section,m,rank)
    (child1_h, child2_h) = n_crossover_an_array(parent1.get_H(), parent2.get_H(), nb_section,n,rank)

    child1 = Solution(child1_w.T, child1_h)
    child2 = Solution(child2_w.T, child2_h)

    return child1, child2

def n_crossover_an_array(arr1,arr2, nb_sections, size, rank):
    cuts = random.sample(range(0,size), nb_sections)

    cuts.extend([0, size])
    cuts.sort()
    new_arr1 = np.empty((rank, 0))
    new_arr2 = np.empty((rank, 0))
    flag = False
    previous = 0
    for i in cuts:
        if flag:
            cut1 = arr1[:, previous:i].reshape(rank, i-previous)
            cut2 = arr2[:, previous:i].reshape(rank, i-previous)
        else:
            cut2 = arr2[:, previous:i].reshape(rank, i-previous)
            cut1 = arr1[:, previous:i].reshape(rank, i-previous)

        previous = i
        flag = not flag
        new_arr1 = np.hstack((new_arr1, cut1))
        new_arr2 = np.hstack((new_arr2, cut2))

    return new_arr1, new_arr2