import random
from typing import Callable, List, Tuple

import matplotlib.pyplot as plt
from numpy import random as np_random


class Evolution:
    # Evolution stop method
    MAX_ITERATIONS = 'iterations'
    MAX_QUALITY_FUNCTION_CALLS = 'quality calls'

    # Selection types
    TOURNAMENT_SELECTION = 'Tournament selection'

    # Crossing types
    NO_CROSSING = 'No crossing'

    # Succession types
    GENERATION_SUCCESSION = 'Generation succession'

    def __init__(self, population_size: int, sigma: float,
                 quality_function: Callable, num_of_parameters: int,
                 parameter_bounds: List[Tuple[float, float]],
                 crossing_probability: float,
                 minimize: bool = True,
                 *,
                 type_of_selection: str,
                 type_of_crossing: str,
                 type_of_succession: str,
                 ):

        # Different methods are stored in dictionaries to avoid creating
        # convoluted if-else statements for algorithm parametrization

        # TODO add crossing_probability checking
        # TODO add sigma checking

        selection_methods = {self.TOURNAMENT_SELECTION: self.tournament_selection}
        crossing_methods = {self.NO_CROSSING: self.no_crossing}
        succession_methods = {self.GENERATION_SUCCESSION: self.generation_succession}

        self.population_size = population_size
        self.sigma = sigma
        self.quality_function = quality_function
        self.num_of_parameters = num_of_parameters
        self.parameter_bounds = parameter_bounds
        self.crossing_probability = crossing_probability

        self.select: Callable = selection_methods[type_of_selection]
        self.cross: Callable = crossing_methods[type_of_crossing]
        self.apply_succession: Callable = succession_methods[type_of_succession]

        self.quality_function_calls = 0

        self.population = []
        self.population_scores = []

        # Parameter used to determine if we want to minimize or maximize function
        self.minimize = minimize

        self.best_scores = []
        self.mean_scores = []

    def evolve(self, stop_parameter: str, *, max_iterations: int = None, max_quality_function_calls: int = None):
        if stop_parameter == self.MAX_ITERATIONS:
            if max_iterations is None or not isinstance(max_iterations, int) or max_iterations < 0:
                raise ValueError('max_iterations must be an integer bigger than 0')

        elif stop_parameter == self.MAX_QUALITY_FUNCTION_CALLS:
            if max_quality_function_calls is None or not isinstance(max_quality_function_calls, int) \
                    or max_quality_function_calls < 0:
                raise ValueError('max_quality_function_calls must be an integer bigger than 0')

        t = 0

        self.population = self.generate_first_generation()
        self.population_scores = self.score(self.population)

        while not self._is_done(stop_parameter, t, max_iterations, max_quality_function_calls):
            temporal_population = self.select(self.population, self.population_scores)
            temporal_population = self.mutate(temporal_population)
            temporal_population = self.cross(temporal_population, self.crossing_probability)

            temporal_population_scores = self.score(temporal_population)

            self.population, self.population_scores = self.apply_succession(
                self.population, temporal_population,
                self.population_scores, temporal_population_scores
            )

            self.mean_scores.append(sum(self.population_scores) / self.population_size)
            self.best_scores.append(max(self.population_scores))

            t += 1

    def _is_done(self, stop_parameter: str, iterations: int, max_iterations: int, max_quality_function_calls: int):
        is_done = False

        if stop_parameter == self.MAX_ITERATIONS:
            if iterations >= max_iterations:
                is_done = True
        else:
            if self.quality_function_calls >= max_quality_function_calls:
                is_done = True

        return is_done

    def generate_first_generation(self):
        population = []

        for _ in range(self.population_size):
            candidate = []

            for i in range(self.num_of_parameters):
                candidate.append(random.uniform(
                    self.parameter_bounds[i][0],
                    self.parameter_bounds[i][1]
                ))

            population.append(candidate)

        return population

    def score(self, population: List[List[float]]) -> List[float]:
        scores = []
        for candidate in population:
            scores.append(self.quality_function(candidate))
            self.quality_function_calls += 1
        return scores

    def tournament_selection(self, population: List[List[float]], scores: List[float]) -> List[List[float]]:
        selected_candidates = []

        for i in range(self.population_size):
            contenders_indexes = random.choices(range(self.population_size), k=2)

            first_score = scores[contenders_indexes[0]]
            second_score = scores[contenders_indexes[1]]

            if self.minimize is True:
                if first_score < second_score:
                    better_one = population[contenders_indexes[0]]
                else:
                    better_one = population[contenders_indexes[1]]
            else:
                if first_score < second_score:
                    better_one = population[contenders_indexes[1]]
                else:
                    better_one = population[contenders_indexes[0]]

            selected_candidates.append(better_one)

        return selected_candidates

    def no_crossing(self, population: List[List[float]], crossing_probability: float) -> List[List[float]]:
        return population

    def generation_succession(self, population: List[List[float]], temporal_population: List[List[float]],
                              population_scores: List[float], termporal_population_scores: List[float]) \
            -> Tuple[List[List[float]], List[float]]:
        return temporal_population, termporal_population_scores

    def mutate(self, population: List[List[float]]) -> List[float]:
        mutated_population = []
        for candidate in population:
            modification_arr = np_random.normal(loc=0, scale=self.sigma, size=self.num_of_parameters)

            lower_than_zero = len([x for x in modification_arr if x < 0])

            mutated_candidate = [value + modifier for value, modifier in zip(candidate, modification_arr)]
            mutated_population.append(mutated_candidate)

        mutated_population = self.assert_parameters_are_bound(mutated_population)
        return mutated_population

    def assert_parameters_are_bound(self, mutated_values: List[List[float]]):

        for index, parameters in enumerate(mutated_values):
            for parameter_number, value in enumerate(parameters):
                lower_bound = self.parameter_bounds[parameter_number][0]
                high_bound = self.parameter_bounds[parameter_number][1]
                if value < lower_bound:
                    mutated_values[index][parameter_number] = lower_bound
                elif value > high_bound:
                    mutated_values[index][parameter_number] = high_bound

        return mutated_values

    def print_best(self, k: int):
        extended_population = list(zip(self.population, self.population_scores))

        extended_population.sort(key=lambda a: a[1], reverse=not self.minimize)
        for i in range(k):
            print('Score: {:.3f}, candidate {}'.format(extended_population[i][1], extended_population[i][0]))


def quality_function(parameters: List[float]):
    return sum([x ** 2 for x in parameters])


if __name__ == '__main__':
    parameter_bounds = [(-10, 10), (-10, 10), (-10, 10), (-10, 10), (-10, 10),
                        (-10, 10), (-10, 10), (-10, 10), (-10, 10), (-10, 10)]

    evolution = Evolution(20, 0.5, quality_function, 10, parameter_bounds, 1,
                          minimize=True,
                          type_of_selection=Evolution.TOURNAMENT_SELECTION,
                          type_of_crossing=Evolution.NO_CROSSING,
                          type_of_succession=Evolution.GENERATION_SUCCESSION)
    evolution.evolve(Evolution.MAX_QUALITY_FUNCTION_CALLS, max_quality_function_calls=10000)
    evolution.print_best(5)

    plt.plot(evolution.mean_scores)
    plt.plot(evolution.best_scores)
    plt.ylabel('Mean And Max Values')
    plt.show()
