import json
from pprint import pprint
from time import time
from typing import List

from numpy import random

from Evolution.benchmark import MyBenchmark
from Evolution.evolution import Evolution


def sphere_quality_function(parameters: List[float]):
    return sum([x ** 2 for x in parameters])


if __name__ == '__main__':
    random.seed(42)

    start_time = time()

    number_of_experiment_repetitions = 25

    is_algorithm_minimizing_function = True
    sigma = 0.125
    small_population_size = 10
    big_population_size = 100

    # Variables describing optimized function
    sphere_dimensionality = 10
    sphere_parameter_bounds = [(-20, 20)] * sphere_dimensionality
    sphere_optimal_quality_function_value = 0

    # Objects used for collection and processing of data from evolution process
    benchmark_small_population = MyBenchmark(is_algorithm_minimizing_function, sphere_optimal_quality_function_value)
    benchmark_big_population = MyBenchmark(is_algorithm_minimizing_function, sphere_optimal_quality_function_value)

    results_dict = {'10k': {},
                    '100k': {}}
    results_filename = 'results.json'
    results_directory_name = 'results/'

    evolution_small_population = Evolution(small_population_size, sigma,
                                           sphere_quality_function,
                                           sphere_dimensionality,
                                           sphere_parameter_bounds, 1,
                                           minimize=is_algorithm_minimizing_function,
                                           type_of_selection=Evolution.TOURNAMENT_SELECTION,
                                           type_of_crossing=Evolution.NO_CROSSING,
                                           type_of_succession=Evolution.GENERATION_SUCCESSION,
                                           )

    evolution_big_population = Evolution(big_population_size, sigma,
                                         sphere_quality_function,
                                         sphere_dimensionality,
                                         sphere_parameter_bounds, 1,
                                         minimize=is_algorithm_minimizing_function,
                                         type_of_selection=Evolution.TOURNAMENT_SELECTION,
                                         type_of_crossing=Evolution.NO_CROSSING,
                                         type_of_succession=Evolution.GENERATION_SUCCESSION,
                                         )

    # Dictionary used to shorten and improve readability of output files and displayed results
    shorthands = {
        10000: '10k',
        100000: '100k'
    }

    for max_quality_fun_calls in [10000, 100000]:

        for i in range(number_of_experiment_repetitions):
            evolution_small_population.evolve(Evolution.MAX_QUALITY_FUNCTION_CALLS,
                                              max_quality_function_calls=max_quality_fun_calls,
                                              benchmark=benchmark_small_population)

        for i in range(number_of_experiment_repetitions):
            evolution_big_population.evolve(Evolution.MAX_QUALITY_FUNCTION_CALLS,
                                            max_quality_function_calls=max_quality_fun_calls,
                                            benchmark=benchmark_big_population)

        print('Small population results - {} quality function calls'.format(shorthands[max_quality_fun_calls]))
        small_population_results = benchmark_small_population.get_results()
        pprint(small_population_results)
        benchmark_small_population.create_and_save_boxplot(
            results_directory_name + 'small_population_' + shorthands[max_quality_fun_calls] + '_calls.png')

        print('Big population results - {} quality function calls'.format(shorthands[max_quality_fun_calls]))
        big_population_results = benchmark_big_population.get_results()
        pprint(big_population_results)
        benchmark_big_population.create_and_save_boxplot(
            results_directory_name + 'big_population_' + shorthands[max_quality_fun_calls] + '_calls.png')

        results_dict[shorthands[max_quality_fun_calls]]['small population'] = small_population_results
        results_dict[shorthands[max_quality_fun_calls]]['big population'] = big_population_results

    end_time = time()

    with open(results_directory_name + results_filename, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, ensure_ascii=False, indent=4)

    print('Time of execution: {}s'.format(int(end_time - start_time)))
