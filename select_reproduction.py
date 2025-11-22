import random


def selection_reproduction_roulette(population, percentage = 0.4):
    count = round(len(population) * percentage)
    return random.choices(population, weights=list(map(lambda x : x.score, population)), k=count)



