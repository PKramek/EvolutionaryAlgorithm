import random
from typing import Callable, List

from numpy import random as np_random


class Evolution:
    # Selection types
    TOURNAMENT_SELECTION = 'Tournament selection'

    # Crossing types
    NO_CROSSING = 'No crossing'

    # Succesion types
    GENERATION_SUCCESSION = 'Generation succesion'

    def __init__(self, population_size: int, sigma: float, quality_function: Callable, num_of_parameters: int,
                 type_of_selection: str, type_of_crossing: str, type_of_succession: str, crossing_probability: float):
        selection_methods = {self.TOURNAMENT_SELECTION: self.tournament_selection}
        crossing_methods = {self.NO_CROSSING: self.no_crossing}
        succession_methods = {self.GENERATION_SUCCESSION: self.generation_succesion}

        self.population = []

        self.population_size = population_size
        self.sigma = sigma
        self.quality_function = quality_function
        self.num_of_parameters = num_of_parameters
        self.crossing_probability = crossing_probability

        self.selection = selection_methods[type_of_selection]
        self.crossing = crossing_methods[type_of_crossing]
        self.succession = succession_methods[type_of_succession]

    def evolve(self):


    def tournament_selection(self, population: List[List[float]]) -> List[List[float]]:
        selected_candidates = []

        for i in range(self.population_size):
            contenders = random.choices(population, k=2)

            first_quality = self.quality_function(contenders[0])
            second_quality = self.quality_function(contenders[1])

            better_one = contenders[0]
            if first_quality < second_quality:
                better_one = contenders[1]

            selected_candidates.append(better_one)

        return selected_candidates

    def no_crossing(self, population: List[List[float]], crossing_probability: float) -> List[List[float]]:
        return population

    def generation_succesion(self, population: List[List[float]], temporal_population: List[List[float]]):
        return temporal_population

    def mutate(self):
        modification_arr = np_random.normal(loc=0, scale=self.sigma, size=self.population_size)

        # TODO add checking allowed intervals
        mutated_values = [value + modifier for value, modifier in zip(self.population, modification_arr)]

        return mutated_values
