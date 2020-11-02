from abc import ABC, abstractmethod
from typing import List, Tuple

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Evolution.evolution import Evolution


class SuccessionStrategy(ABC):
    def __init__(self, evolution: 'Evolution'):
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
