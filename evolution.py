from __future__ import annotations  # Python 3.7 or higher required

import math
from abc import ABC, abstractmethod
from typing import Callable, List, Tuple

import matplotlib.pyplot as plt
from numpy import random


class SelectionStrategy(ABC):
    def __init__(self, evolution: Evolution):
        self.evolution = evolution

    @abstractmethod
    def select(self, population: List[List[float]], scores: List[float]) -> List[List[float]]:
        pass


class TournamentSelectionStrategy(SelectionStrategy):
    def select(self, population: List[List[float]], scores: List[float]) -> List[List[float]]:
        """
        This method applies tournament selection to a given population and returns selected candidates.

        :param population: Population on which selection should be performed
        :type population: List[List[float]
        :param scores: List of scores for whole population
        :type scores: List[float]
        :return: Selected candidates
        :rtype: List[List[float]]
        """
        selected_candidates = []

        for i in range(self.evolution.population_size):
            contenders_indexes = random.choice(range(self.evolution.population_size), size=2, replace=True)

            first_score = scores[contenders_indexes[0]]
            second_score = scores[contenders_indexes[1]]

            if self.evolution.minimize is True:
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


class CrossingStrategy(ABC):
    def __init__(self, evolution: Evolution):
        self.evolution = evolution

    @abstractmethod
    def cross(self, population: List[List[float]], crossing_probability: float) -> List[List[float]]:
        pass


class NoCrossingStrategy(CrossingStrategy):
    def cross(self, population: List[List[float]], crossing_probability: float) -> List[List[float]]:
        """
        Just returns given population.
        :param population:
        :type population:
        :param crossing_probability: Probability of crossing a single candidate with any randomly selected candidate.
        This parameter is not used in method, but is required for crossing methods interface uniformity.
        :type crossing_probability: float
        :return: Returns given population, because no crossing is performed.
        :rtype: List[List[float]]
        """
        return population


class SuccessionStrategy(ABC):
    def __init__(self, evolution: Evolution):
        self.evolution = evolution

    @abstractmethod
    def apply_succession(self, population: List[List[float]], temporal_population: List[List[float]],
                         population_scores: List[float], temporal_population_scores: List[float]) \
            -> Tuple[List[List[float]], List[float]]:
        pass


class GenerationSuccessionStrategy(SuccessionStrategy):
    def apply_succession(self, population: List[List[float]], temporal_population: List[List[float]],
                         population_scores: List[float], temporal_population_scores: List[float]) \
            -> Tuple[List[List[float]], List[float]]:
        """
        Applies generation succession by returning just mutated and crossed candidates (temporal population)
        and their scores.
        :param population: Population from last iteration of evolution algorithm
        :type population: List[List[float]]
        :param temporal_population: Population created by mutation and crossing processes in current iteration of
        evolution algorithm
        :type temporal_population: List[List[float]]
        :param population_scores:
        :type population_scores:
        :param temporal_population_scores:
        :type temporal_population_scores:
        :return: Population and scores
        :rtype: Tuple[List[List[float]], List[float]]
        """
        return temporal_population, temporal_population_scores


class Evolution:
    # Those constants are used to create one clear way of parametrizing objects of this class.

    # Evolution stop method
    MAX_ITERATIONS = 'iterations'
    MAX_QUALITY_FUNCTION_CALLS = 'quality calls'

    # Selection methods
    TOURNAMENT_SELECTION = 'Tournament selection'

    # Crossing methods
    NO_CROSSING = 'No crossing'

    # Succession methods
    GENERATION_SUCCESSION = 'Generation succession'

    def __init__(self, population_size: int, sigma: float,
                 quality_function: Callable, dimensionality: int,
                 parameter_bounds: List[Tuple[float, float]],
                 crossing_probability: float,
                 minimize: bool = True,
                 *,
                 type_of_selection: str,
                 type_of_crossing: str,
                 type_of_succession: str,
                 ):
        """
        :param population_size: Defines population size to be used in evolution process
        :type population_size: int
        :param sigma: Defines standard deviation of distribution used in mutation process
        :type sigma: float
        :param quality_function: Function used to define quality of single candidate. This function should return float
        :type quality_function: Callable
        :param dimensionality: Defines number of dimensions of each candidate
        :type dimensionality: int
        :param parameter_bounds: Defines low and high bounds of each of the parameters. Bounds should be floats
        :type parameter_bounds: List[Tuple[float, float]]
        :param crossing_probability: Probability of crossing single candidate with any other candidate
        :type crossing_probability: float
        :param minimize: Parameter defining if algorithm should minimize of maximize value of quality function.
                        If True is given then algorithm will minimize output of quality function.
        :type minimize: bool
        :param type_of_selection: Parameter deciding what algorithm of selection is used in algorithm.
                                  Inside class there are constants defined in form <NAME OF SELECTION TYPE>_SELECTION,
                                  that should be passed as value of this parameter.
        :type type_of_selection: str
        :param type_of_crossing: Parameter deciding what algorithm of crossing is used in algorithm.
                                Inside class there are constants defined in form <NAME OF CROSSING TYPE>_CROSSING,
                                that should be passed as value of this parameter.
        :type type_of_crossing: str
        :param type_of_succession: Parameter deciding what algorithm of succession is used in algorithm.
                                  Inside class there are constants defined in form <NAME OF SUCCESSION TYPE>_SUCCESSION,
                                  that should be passed as value of this parameter.
        :type type_of_succession: str
        """
        if crossing_probability < 0 or crossing_probability > 1:
            raise ValueError('Probability of crossing must be in range [0, 1]')

        self.population_size = population_size
        self.sigma = sigma
        self.quality_function = quality_function
        self.num_of_parameters = dimensionality
        self.parameter_bounds = parameter_bounds
        self.crossing_probability = crossing_probability
        self.minimize = minimize

        self.select_strategy: SelectionStrategy = self.selection_factory(type_of_selection)
        self.crossing_strategy: CrossingStrategy = self.crossing_factory(type_of_crossing)
        self.succession_strategy: SuccessionStrategy = self.succession_factory(type_of_succession)

        self.quality_function_calls = 0

        self.population = []
        self.population_scores = []

        self.best_scores = []
        self.mean_scores = []

    def selection_factory(self, selection_type: str) -> SelectionStrategy:
        """
        Factory method for selection strategy objects.

        :param selection_type: Parameter deciding what algorithm of selection is used in algorithm.
                               Inside class there are constants defined in form <NAME OF SELECTION TYPE>_SELECTION,
                               that should be passed as value of this parameter.
        :type selection_type: str
        :return: Selection strategy object
        :rtype: SelectionStrategy
        """
        selection_methods = {self.TOURNAMENT_SELECTION: TournamentSelectionStrategy}

        selection = selection_methods.get(selection_type)
        if selection is None:
            raise ValueError('Selection method not found')

        selection_object = selection(self)
        return selection_object

    def crossing_factory(self, crossing_type: str) -> CrossingStrategy:
        """
        Factory method for crossing strategy objects.

        :param crossing_type: Parameter deciding what algorithm of crossing is used in algorithm.
                              Inside class there are constants defined in form <NAME OF CROSSING TYPE>_CROSSING,
                              that should be passed as value of this parameter.
        :type crossing_type: str
        :return: Crossing strategy object
        :rtype: CrossingStrategy
        """
        crossing_methods = {self.NO_CROSSING: NoCrossingStrategy}

        crossing_method = crossing_methods.get(crossing_type)
        if crossing_method is None:
            raise ValueError('Crossing method not found')

        crossing_object = crossing_method(self)
        return crossing_object

    def succession_factory(self, succession_type: str) -> SuccessionStrategy:
        """

        :param succession_type: Parameter deciding what algorithm of succession is used in algorithm.
                                Inside class there are constants defined in form <NAME OF SUCCESSION TYPE>_SUCCESSION,
                                that should be passed as value of this parameter.
        :type succession_type: str
        :return: Succession strategy object
        :rtype: SuccessionStrategy
        """
        succession_methods = {self.GENERATION_SUCCESSION: GenerationSuccessionStrategy}

        succession_method = succession_methods.get(succession_type)
        if succession_method is None:
            raise ValueError('Succession method not found')

        succession_object = succession_method(self)
        return succession_object

    def evolve(self, stop_parameter: str, *, max_iterations: int = None, max_quality_function_calls: int = None):
        """
        Main method of this class, it performs evolution algorithm.

        :param stop_parameter: This parameter defines when does evolution process end. There are two possible values of
        this parameter defined in static constants - MAX_ITERATIONS and MAX_QUALITY_FUNCTION_CALLS.
        :type stop_parameter: str
        :param max_iterations: Defines maximum number of iterations in evolution algorithm. If stop_parameter is set to
        MAX_ITERATIONS this parameter must be passed.
        :type max_iterations: int
        :param max_quality_function_calls: Defines maximum number of calls of quality function in evolution algorithm.
        If stop_parameter is set to MAX_QUALITY_FUNCTION_CALLS this parameter must be passed.
        :type max_quality_function_calls: int
        """
        if stop_parameter not in [self.MAX_ITERATIONS, self.MAX_QUALITY_FUNCTION_CALLS]:
            raise ValueError('Stop parameter not found')

        if stop_parameter == self.MAX_ITERATIONS:
            if max_iterations is None or not isinstance(max_iterations, int) or max_iterations < 0:
                raise ValueError('Max_iterations must be an integer bigger than 0')

        elif stop_parameter == self.MAX_QUALITY_FUNCTION_CALLS:
            if max_quality_function_calls is None or not isinstance(max_quality_function_calls, int) \
                    or max_quality_function_calls < 0:
                raise ValueError('Max_quality_function_calls must be an integer bigger than 0')

        self.best_scores = []
        self.mean_scores = []
        self.quality_function_calls = 0

        t = 0

        self.population = self.generate_first_generation()
        self.population_scores = self.score(self.population)

        while not self._is_done(stop_parameter, t, max_iterations, max_quality_function_calls):

            temporal_population = self.select_strategy.select(self.population, self.population_scores)
            temporal_population = self.mutate(temporal_population)
            temporal_population = self.crossing_strategy.cross(temporal_population, self.crossing_probability)

            temporal_population_scores = self.score(temporal_population)

            self.population, self.population_scores = self.succession_strategy.apply_succession(
                self.population, temporal_population,
                self.population_scores, temporal_population_scores
            )

            self.mean_scores.append(sum(self.population_scores) / self.population_size)
            if self.minimize:
                self.best_scores.append(min(self.population_scores))
            else:
                self.best_scores.append(max(self.population_scores))

            t += 1

    def _is_done(self, stop_parameter: str, iterations: int, max_iterations: int, max_quality_function_calls: int):
        """
        Helper function for deciding if evolution process should terminate.

        :param stop_parameter: This parameter defines when does evolution process end. There are two possible values of
        this parameter defined in static constants - MAX_ITERATIONS and MAX_QUALITY_FUNCTION_CALLS.
        :type stop_parameter: str
        :param iterations: Current number of iterations of evolution process
        :type iterations:  int
        :param max_iterations: Maximum number of iterations of evolution process
        :type max_iterations: int
        :param max_quality_function_calls: Maximum number of
        :type max_quality_function_calls:
        :return:
        :rtype:
        """
        is_done = False

        if stop_parameter == self.MAX_ITERATIONS:
            if iterations >= max_iterations:
                is_done = True
        else:
            if self.quality_function_calls >= max_quality_function_calls:
                is_done = True

        return is_done

    def generate_first_generation(self) -> List[List[float]]:
        """"
        This method is used to generate first population for evolution process. Each parameter of candidates is
        generated using uniform distribution on a interval given for that parameter in parameter bounds.

        :rtype: List[List[float]]
        """
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
        """
        Creates list of scores for given population by applying quality function for each candidate and updates number
        of quality function calls.

        :param population: Population on which scoring process should be performed
        :type population: List[List[float]]
        :return: Scores for each corresponding candidate in population
        :rtype: List[float]
        """
        scores = [self.quality_function(candidate) for candidate in population]
        self.quality_function_calls += self.population_size

        return scores

    def mutate(self, population: List[List[float]]) -> List[List[float]]:
        """
        Applies gaussian mutation to all parameters for all candidates in population.

        :param population: Population from last iteration of evolution algorithm
        :type population: List[List[float]]
        :return: Mutated population
        :rtype: List[List[float]]
        """
        mutated_population = []
        for candidate in population:
            modification_arr = random.normal(loc=0, scale=self.sigma, size=self.num_of_parameters)
            mutated_candidate = [value + modifier for value, modifier in zip(candidate, modification_arr)]

            mutated_population.append(mutated_candidate)

        mutated_population = self._assert_parameters_are_bound(mutated_population)
        return mutated_population

    def _assert_parameters_are_bound(self, mutated_population: List[List[float]]):
        """
        This method checks all parameters for all candidates in population. If parameter is out of bound it is set to
        maximum or minimum value for that parameter depending if it was above upper bound or below lower bound.

        :param mutated_population: Mutated population
        :type mutated_population: List[List[float]]
        :return: Mutated population with parameters values in specified bounds
        :rtype: List[List[float]]
        """
        for index, parameters in enumerate(mutated_population):
            for parameter_number, value in enumerate(parameters):
                lower_bound = self.parameter_bounds[parameter_number][0]
                high_bound = self.parameter_bounds[parameter_number][1]
                if value < lower_bound:
                    mutated_population[index][parameter_number] = lower_bound
                elif value > high_bound:
                    mutated_population[index][parameter_number] = high_bound

        return mutated_population

    def get_best_candidates(self, n: int) -> List[List[float]]:
        """
        Returns n best candidates after evolution process.

        :param n: Number of best candidates to return
        :type n: int
        :return: List of best candidates in descending order. At position 0 is the best candidate
        :rtype: List[List[float]]
        """
        if len(self.population) == 0:
            raise AssertionError('Population is empty')

        sorted_population = sorted(self.population, key=lambda a: self.quality_function(a), reverse=not self.minimize)

        return sorted_population[:n]

    def print_best(self, n: int):
        """
        Prints n best candidates and their scores to the console

        :param n: Number of best candidates to print
        :type n: int
        """
        n_best_candidates = self.get_best_candidates(n)

        for candidate in n_best_candidates:
            print('Score: {:.3f}, candidate {}'.format(self.quality_function(candidate), candidate))


def ackley_quality_function(parameters: List[float]):
    n = len(parameters)

    sum_of_squares = sum([x ** 2 for x in parameters])
    sqrt_of_mean_of_squares = math.sqrt(sum_of_squares / n)
    sum_of_cosines = sum([math.cos(2 * math.pi * x) for x in parameters])

    return (-20 * math.exp(-0.2 * sqrt_of_mean_of_squares)) - math.exp(sum_of_cosines / n) + 20 + math.e


if __name__ == '__main__':
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
