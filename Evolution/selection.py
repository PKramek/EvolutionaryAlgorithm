from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING

from numpy import random

if TYPE_CHECKING:
    from Evolution.evolution import Evolution


class SelectionStrategy(ABC):
    def __init__(self, evolution: 'Evolution'):
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
