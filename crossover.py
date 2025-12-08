import numpy as np

from solution import Solution


def uniform_crossover(parent1, parent2, m,n, rank):
    """
    Performs a uniform crossover between parent1 and parent2
    Creating two new distinct childrens
    :param parent1:
    :param parent2:
    :param m:
    :param n:
    :param rank:
    :return:
    """
    #Reusable crossover logic
    def array_crossover(parent1, parent2, rank, width):
        mask = np.random.rand(width) > 0.5

        child1 = np.empty((rank, width))
        child2 = np.empty((rank, width))

        # Creating child 1
        child1 = parent1[:, mask] #Inherit genes from parent 1
        child1 = parent2[:, ~mask] #Inherit genes from parent 2

        # Creating child 2
        child2 = parent1[:, ~mask] #Inherit genes from parent 1
        child2 = parent2[:, mask]  #Inherit genes from parent 2

        return child1, child2

    parent1_w = parent1.get_W().T
    parent2_w = parent2.get_W().T
    parent1_h = parent1.get_H()
    parent2_h = parent2.get_H()

    child1_w, child2_w = array_crossover(parent1_w, parent2_w, rank, m)
    child1_h, child2_h = array_crossover(parent1_h, parent2_h, rank, m)

    child1 = Solution(child1_w.T, child1_h)
    child2 = Solution(child1_w.T, child2_h)

    return child1, child2