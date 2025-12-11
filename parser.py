import numpy as np

def read_file(filename="donnee.txt"):
    """
    Read the provided file
    :param filename:
    :return:
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    params = [int(x) for x in lines[0].split()]
    m, n, r, LW, UW, LH, UH = params

    X = []
    for i in range(1, 1 + m):
        ligne = [int(x) for x in lines[i].split()]
        X.append(ligne)

    return np.array(X), m, n, r, LW, UW, LH, UH


def write_output(filepath, solution):
    """
    Write the solution to the provided file
    :param filepath:
    :param solution:
    :return:
    """
    with open(filepath, 'w') as f:
        f.write(f"{solution.score}\n")
        np.savetxt(f, solution.get_W(), fmt='%d')
        np.savetxt(f, solution.get_H(), fmt='%d')