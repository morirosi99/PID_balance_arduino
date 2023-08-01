from algorithm import Algorithm

import numpy as np

populationSize = 5
fitness = np.zeros(populationSize)

algorithm = Algorithm(fitness, populationSize=populationSize)

population = algorithm.getPopulation()


def hypothesis_tester(i):
    return population[i]


def fitness_reader(x, y, z, i):
    fitness[i] = abs(x) + abs(y) + abs(z)


def evolution_process():
    global population

    index = min(range(len(fitness)), key=lambda i: fitness[i])

    individual = population[index]

    algo = Algorithm(fitness, population=population, populationSize=populationSize)
    population = algo.evolution()

    Kp = individual[0]
    Ki = individual[1]
    Kd = individual[2]

    return [Kp, Ki, Kd, fitness[index]]
