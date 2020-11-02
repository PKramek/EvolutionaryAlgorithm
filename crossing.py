from abc import ABC, abstractmethod
from typing import List

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evolution import Evolution


class CrossingStrategy(ABC):
    def __init__(self, evolution: 'Evolution'):
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
