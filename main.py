import math
from typing import List

import matplotlib.pyplot as plt
from numpy import random

from Evolution.evolution import Evolution


def ackley_quality_function(parameters: List[float]):
    n = len(parameters)

    sum_of_squares = sum([x ** 2 for x in parameters])
    sqrt_of_mean_of_squares = math.sqrt(sum_of_squares / n)
    sum_of_cosines = sum([math.cos(2 * math.pi * x) for x in parameters])

    return (-20 * math.exp(-0.2 * sqrt_of_mean_of_squares)) - math.exp(sum_of_cosines / n) + 20 + math.e


random.seed(42)

ackley_parameter_bounds = [(-32, 32)] * 10
number_of_experiment_repetitions = 25

small_population_size = 20
big_population_size = 100

big_population_last_iterations_populations = []
small_population_last_iterations_populations = []

evolution = Evolution(100, 0.25, ackley_quality_function, 10, ackley_parameter_bounds, 1,
                      minimize=True,
                      type_of_selection=Evolution.TOURNAMENT_SELECTION,
                      type_of_crossing=Evolution.NO_CROSSING,
                      type_of_succession=Evolution.GENERATION_SUCCESSION)

evolution.evolve(Evolution.MAX_QUALITY_FUNCTION_CALLS, max_quality_function_calls=10000)
evolution.evolve(Evolution.MAX_QUALITY_FUNCTION_CALLS, max_quality_function_calls=10000)

evolution.print_best(5)
plt.plot(evolution.mean_scores)
plt.plot(evolution.best_scores)

plt.ylabel('Mean And Max Values')
plt.show()
