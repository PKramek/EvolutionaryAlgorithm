from abc import abstractmethod
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from numpy import median, mean, std

if TYPE_CHECKING:
    from Evolution.evolution import Evolution


class Benchmark:
    """
    This abstract base class for benchmarks. Members of this class should be passed as parameter evolution function.
    Every iteration of evolution process collect_data is called, which allows objects of this class to collect necessary
    data from Evolution object
    """

    @abstractmethod
    def collect_data(self, evolution: 'Evolution'):
        """
        This method is called every iteration of evolution process.

        :param evolution: Object of class Evolution, from which data will be collected
        :type evolution: Evolution
        :return: None
        :rtype: None
        """
        pass

    @abstractmethod
    def get_results(self) -> dict:
        """
        Method that calculates necessary results and returns them in a dictionary

        :return: Dictionary containing calculated results
        :rtype: dict
        """
        pass


class NoBenchmark(Benchmark):
    """
    Class representing situation when user does not want to perform any data collection in evolution process.
    """

    def collect_data(self, evolution: 'Evolution'):
        pass

    def get_results(self) -> dict:
        pass


class MyBenchmark(Benchmark):
    """
    This benchmark collects information about scores of whole population after 100, 1000, 10000 and 100000 quality
    function calls. When get_results is called it calculates best, worst, mean and median scores as well as standard
    deviation of all scores
    """

    def __init__(self, is_algorithm_minimizing_function: bool):
        """
        :param is_algorithm_minimizing_function:
        :type is_algorithm_minimizing_function: bool
        """
        super().__init__()
        self.scores_100 = []
        self.scores_1000 = []
        self.scores_10000 = []
        self.scores_100000 = []

        self.is_algorithm_minimizing_function = is_algorithm_minimizing_function

    def collect_data(self, evolution: 'Evolution'):
        if evolution.quality_function_calls == 100:
            self.scores_100.extend(evolution.population_scores)
        elif evolution.quality_function_calls == 1000:
            self.scores_1000.extend(evolution.population_scores)
        elif evolution.quality_function_calls == 10000:
            self.scores_10000.extend(evolution.population_scores)
        elif evolution.quality_function_calls == 100000:
            self.scores_100000.extend(evolution.population_scores)

    def get_results(self) -> dict:
        results = {}
        data_dict = {'100': self.scores_100,
                     '1000': self.scores_1000,
                     '10000': self.scores_10000,
                     '100000': self.scores_100000,
                     }

        for key, dataset in data_dict.items():
            if len(dataset) != 0:
                results[key] = {
                    'median': round(median(dataset), 2),
                    'mean': round(mean(dataset), 2),
                    'std': round(std(dataset), 2)
                }
                if self.is_algorithm_minimizing_function is True:
                    results[key]['best'] = round(min(dataset), 2)
                    results[key]['worst'] = round(max(dataset), 2)
                else:
                    results[key]['best'] = round(max(dataset), 2)
                    results[key]['worst'] = round(min(dataset), 2)

        return results

    def create_and_save_boxplot(self, name_of_the_file: str):
        array_of_vectors = [self.scores_100, self.scores_1000, self.scores_10000, self.scores_100000]
        plt.boxplot(array_of_vectors)
        plt.ylabel('Quality function value')
        plt.xlabel('Quality function calls')
        plt.xticks([1, 2, 3, 4], ['100', '1000', '10000', '100000'])
        plt.savefig(name_of_the_file)
        plt.show()
