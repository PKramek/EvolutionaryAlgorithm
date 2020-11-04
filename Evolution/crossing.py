from abc import ABC, abstractmethod
from typing import List

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Evolution.evolution import Evolution


class CrossingStrategy(ABC):
    """
    Abstract base class for crossing strategy objects.
    """

    def __init__(self, evolution: 'Evolution'):
        """
        :param evolution: Evolution object to which Strategy object is passed.
        :type evolution: Evolution
        """
        self.evolution = evolution

    @abstractmethod
    def cross(self, population: List[List[float]], crossing_probability: float) -> List[List[float]]:
        """
        Abstract method for crossing.

        :param population: Population on which crossing is to be performed
        :type population: List[List[float]]
        :param crossing_probability: Probability of crossing a single candidate with any randomly selected candidate.
        This parameter is not used in method, but is required for crossing methods interface uniformity.
        :type crossing_probability: float
        :return: Returns given population, because no crossing is performed.
        :rtype: List[List[float]]
        """
        pass


class NoCrossingStrategy(CrossingStrategy):
    """
    Class implementing no crossing approach.
    """

    def cross(self, population: List[List[float]], crossing_probability: float) -> List[List[float]]:
        """
        Just returns given population.

        :param population: Population on which crossing is to be performed
        :type population: List[List[float]]
        :param crossing_probability: Probability of crossing a single candidate with any randomly selected candidate.
        This parameter is not used in method, but is required for crossing methods interface uniformity.
        :type crossing_probability: float
        :return: Returns given population, because no crossing is performed.
        :rtype: List[List[float]]
        """
        return population
