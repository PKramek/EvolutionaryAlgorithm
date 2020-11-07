from __future__ import annotations  # Python 3.7 or higher required

from typing import Callable, List, Tuple

import matplotlib.pyplot as plt
from numpy import random

from Evolution.benchmark import Benchmark
from Evolution.crossing import CrossingStrategy, NoCrossingStrategy
from Evolution.selection import SelectionStrategy, TournamentSelectionStrategy
from Evolution.succession import SuccessionStrategy, GenerationSuccessionStrategy


class Evolution:
    # Those constants are used to create one clear way of parametrizing objects of this class.

    # If new methods of selection, crossing or succession are needed, user should create according classes by extending
    # correct abstract base class, create corresponding static constant in this class and update factory method to
    # include new class.

    # Evolution stop method
    MAX_ITERATIONS = 'iterations'
    MAX_OBJECTIVE_FUNCTION_CALLS = 'objective calls'

    # Selection methods
    TOURNAMENT_SELECTION = 'Tournament selection'

    # Crossing methods
    NO_CROSSING = 'No crossing'

    # Succession methods
    GENERATION_SUCCESSION = 'Generation succession'

    def __init__(self, population_size: int, sigma: float,
                 objective_function: Callable, dimensionality: int,
                 parameter_bounds: List[Tuple[float, float]],
                 crossing_probability: float,
                 minimize: bool = True,
                 *,
                 type_of_selection: str,
                 type_of_crossing: str,
                 type_of_succession: str
                 ):
        """
        :param population_size: Defines population size to be used in evolution process
        :type population_size: int
        :param sigma: Defines standard deviation of distribution used in mutation process
        :type sigma: float
        :param objective_function: Function used to define quality of single candidate. This function should
        return float
        :type objective_function: Callable
        :param dimensionality: Defines number of dimensions of each candidate
        :type dimensionality: int
        :param parameter_bounds: Defines low and high bounds of each of the parameters. Bounds should be floats
        :type parameter_bounds: List[Tuple[float, float]]
        :param crossing_probability: Probability of crossing single candidate with any other candidate
        :type crossing_probability: float
        :param minimize: Parameter defining if algorithm should minimize of maximize value of objective function.
                        If True is given then algorithm will minimize output of objective function.
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
        self.objective_function = objective_function
        self.num_of_parameters = dimensionality
        self.parameter_bounds = parameter_bounds
        self.crossing_probability = crossing_probability
        self.minimize = minimize

        self.select_strategy: SelectionStrategy = self.selection_factory(type_of_selection)
        self.crossing_strategy: CrossingStrategy = self.crossing_factory(type_of_crossing)
        self.succession_strategy: SuccessionStrategy = self.succession_factory(type_of_succession)

        self.object_function_calls = 0

        self.population = []
        self.population_scores = []

        self.best_scores = []
        self.mean_scores = []

    def evolve(self, stop_parameter: str, *, benchmark: Benchmark,
               max_iterations: int = None, max_objective_function_calls: int = None,
               ):
        """
        Main method of this class, it performs evolution algorithm.


        :param stop_parameter: This parameter defines when does evolution process end. There are two possible values of
        this parameter defined in static constants - MAX_ITERATIONS and MAX_OBJECTIVE_FUNCTION_CALLS.
        :type stop_parameter: str
        :param max_iterations: Defines maximum number of iterations in evolution algorithm. If stop_parameter is set to
        MAX_ITERATIONS this parameter must be passed.
        :type max_iterations: int
        :param max_objective_function_calls: Defines maximum number of calls of objective function in
        evolution algorithm. If stop_parameter is set to MAX_OBJECTIVE_FUNCTION_CALLS this parameter must be passed.
        :type max_objective_function_calls: int
        :param benchmark: Benchmark object class, which collect_data method will be called every iteration of evolution
        algorithm
        :type benchmark: Benchmark
        """
        if stop_parameter not in [self.MAX_ITERATIONS, self.MAX_OBJECTIVE_FUNCTION_CALLS]:
            raise ValueError('Stop parameter not found')

        if stop_parameter == self.MAX_ITERATIONS:
            if max_iterations is None or not isinstance(max_iterations, int) or max_iterations < 0:
                raise ValueError('Max_iterations must be an integer bigger than 0')

        elif stop_parameter == self.MAX_OBJECTIVE_FUNCTION_CALLS:
            if max_objective_function_calls is None or not isinstance(max_objective_function_calls, int) \
                    or max_objective_function_calls < 0:
                raise ValueError('Max_objective_function_calls must be an integer bigger than 0')

        assert isinstance(benchmark, Benchmark)

        self.best_scores = []
        self.mean_scores = []
        self.object_function_calls = 0

        t = 0

        self.population = self.generate_first_generation()
        self.population_scores = self.score(self.population)

        while not self._is_done(stop_parameter, t, max_iterations, max_objective_function_calls):
            benchmark.collect_data(self)

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
        benchmark.collect_data(self)

    def _is_done(self, stop_parameter: str, iterations: int, max_iterations: int,
                 max_objective_function_calls: int) -> bool:
        """
        Helper function for deciding if evolution process should terminate.

        :param stop_parameter: This parameter defines when does evolution process end. There are two possible values of
        this parameter defined in static constants - MAX_ITERATIONS and MAX_OBJECTIVE_FUNCTION_CALLS.
        :type stop_parameter: str
        :param iterations: Current number of iterations of evolution process
        :type iterations:  int
        :param max_iterations: Maximum number of iterations of evolution process
        :type max_iterations: int
        :param max_objective_function_calls: Maximum number of objective function calls
        :type max_objective_function_calls: int
        :return: Information if evolution process should be terminated
        :rtype: bool
        """
        is_done = False

        if stop_parameter == self.MAX_ITERATIONS:
            if iterations >= max_iterations:
                is_done = True
        else:
            if self.object_function_calls >= max_objective_function_calls:
                is_done = True

        return is_done

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
        Factory method for succession strategy objects.

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
        Creates list of scores for given population by applying objective function for each candidate and updates number
        of objective function calls.

        :param population: Population on which scoring process should be performed
        :type population: List[List[float]]
        :return: Scores for each corresponding candidate in population
        :rtype: List[float]
        """
        scores = [self.objective_function(candidate) for candidate in population]
        self.object_function_calls += self.population_size

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

        sorted_population = sorted(self.population, key=lambda a: self.objective_function(a), reverse=not self.minimize)

        return sorted_population[:n]

    def print_best(self, n: int):
        """
        Prints n best candidates and their scores to the console

        :param n: Number of best candidates to print
        :type n: int
        """
        n_best_candidates = self.get_best_candidates(n)

        for candidate in n_best_candidates:
            print('Score: {:.2f}, candidate {}'.format(self.objective_function(candidate), candidate))

    def plot_evolution(self):
        """
        Creates a plot of best and mean values in evolution process.

        :return: None
        :rtype: None
        """
        plt.xlabel('Iterations of evolution algorithm')
        plt.ylabel('Best and Mean values of objective function')
        plt.plot(self.best_scores, label='Best score')
        plt.plot(self.mean_scores, label='Mean score')
        plt.legend()
        plt.show()
