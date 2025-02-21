from __future__ import annotations
import logging
import math
import numbers
from abc import ABC, abstractmethod
import numpy
import scipy
from tracts.demography.param_types import ParamType


class BaseFounderEvent(ABC):

    def __init__(self, found_time, source_population, remainder_population):
        self.found_time = found_time
        self.source_population = source_population
        self.remainder_population = remainder_population

    @abstractmethod
    def execute(self, parametrized_demography: BaseParametrizedDemography, params):
        pass


class FounderEvent(BaseFounderEvent):

    def __init__(self, found_time, source_population, remainder_population):
        super().__init__(found_time=found_time, source_population=source_population,
                         remainder_population=remainder_population)

    def execute(self, parametrized_demography: BaseParametrizedDemography, params):
        true_start_time = parametrized_demography.get_param_value(self.found_time, params)
        start_time = math.ceil(true_start_time)
        migration_matrix = numpy.zeros((start_time + 1, len(parametrized_demography.population_indices)))

        remaining_rate = 1

        # Fraction of migrants that get repeated in the next generation,
        # to ensure continuous behaviour for fractional start times.
        repeated_migrant_fraction = start_time - true_start_time
        # TODO: We have different rates for both sexes
        for source_population, rate_param in self.source_population.items():
            rate = parametrized_demography.get_param_value(rate_param, params)
            migration_matrix[start_time, parametrized_demography.population_indices[source_population]] = rate
            migration_matrix[start_time - 1, parametrized_demography.population_indices[source_population]] = (
                    rate * repeated_migrant_fraction)
            remaining_rate -= rate

        if remaining_rate < 0:
            logging.warning('Founding migration rates add up to more than 1')

        migration_matrix[
            start_time, parametrized_demography.population_indices[self.remainder_population]] = remaining_rate
        migration_matrix[start_time - 1, parametrized_demography.population_indices[
            self.remainder_population]] = remaining_rate * repeated_migrant_fraction

        return migration_matrix

class BaseMigrationEvent(ABC):

    def __init__(self, rate_parameter, source_population):
        self.rate_parameter = rate_parameter
        self.source_population = source_population

    @abstractmethod
    def execute(self, parametrized_demography: BaseParametrizedDemography, migration_matrix: numpy.ndarray, params):
        pass

class BaseParametrizedDemography(ABC):
    logger = logging.getLogger(__name__)

    def __init__(self, name: str = "", min_time=2, max_time=numpy.inf):
        self.name = name
        self.min_time = min_time
        self.max_time = max_time
        self.events: list[BaseMigrationEvent] = []
        self.constraints = []
        self.params_fixed_by_ancestry = []
        self.params_not_fixed_by_ancestry = []
        self.free_params = {}
        self.dependent_params = {}
        self.constant_params = {}
        self.population_indices = {}
        self.reduced_constraints = []
        self.finalized = False
        self.known_ancestry_proportions = None
        self.has_been_fixed = False
        self.founder_event: FounderEvent = None

    @staticmethod
    def proportions_from_matrix(migration_matrix):
        current_ancestry_proportions = migration_matrix[-1, :]
        for row in migration_matrix[-2::-1, :]:
            current_ancestry_proportions = current_ancestry_proportions * (1 - row.sum()) + row
            if not numpy.isclose(current_ancestry_proportions.sum(), 1):
                raise ValueError('Current ancestry proportions do not sum to 1.')
        return current_ancestry_proportions

    def add_founder_event(self, source_populations: dict[str, str], remainder_population: str, found_time: str) -> None:
        """
        Adds a founder event. A parametrized demography must have exactly one founder event.
        source_populations is a dict where each key is a population
        and each value is the name of the parameter defining the migration ratio of each population
        remainder_population is the source of the remaining migrants, such that the total migration ratio adds up to 1.
        found_time is the name of the parameter defining the time of migration.
        """

        if self.founder_event:
            raise ValueError('Population cannot have more than one founder event.')

        for population, rate_param in source_populations.items():
            self.add_population(population)
            self.add_parameter(rate_param, param_type=ParamType.RATE)

        self.add_population(remainder_population)

        self.add_parameter(found_time, param_type=ParamType.TIME)
        self.founder_event = FounderEvent(
            found_time=found_time,
            source_population=source_populations,
            remainder_population=remainder_population,
        )

    def get_founding_time(self):
        if not self.founder_event:
            raise ValueError('Missing a founder event.')
        return self.founder_event.found_time

    def finalize(self):
        self.finalized = True
        for index, param_name in enumerate(self.free_params):
            self.free_params[param_name]['index'] = index
        for index, population_name in enumerate(self.population_indices):
            self.population_indices[population_name] = index

    def add_parameter(self, param_name: str, param_type: ParamType=None, bounds=None):
        """
        Adds the given parameter name to the parameters of the model
        """
        self.finalized = False
        if param_name not in self.dependent_params:
            if bounds is None:
                if param_type == ParamType.TIME:
                    bounds = (self.min_time, self.max_time)
                else:
                    bounds = param_type.bounds
            self.free_params[param_name] = {'type': param_type, 'bounds': bounds}

    def add_population(self, population_name: str):
        """
        Adds the given population name to the populations of the model
        """
        if not self.has_been_fixed:
            # population_indices will be given values when the model is finalized
            self.finalized = False
            self.population_indices[population_name] = None
        else:
            if population_name not in self.population_indices:
                raise ValueError('Cannot add populations to a model after fixing ancestry proportions.')
            self.population_indices[population_name] = None

    def execute_migration_events(self, migration_matrix, params):
        for event in self.events:
            event.execute(self, migration_matrix, params)

    def get_index(self, time_param_name: str, population_name: str, params: list[float]):
        """
        Returns the matrix index as a tuple from the position and time. Reduces repetitive code
        """

        return self.get_param_value(time_param_name, params), self.population_indices[population_name]

    def is_time_param(self):
        if not self.has_been_fixed:
            return [param['type'] == ParamType.TIME for param in self.free_params.values()]
        time_param_list = []
        for param_name, param in self.free_params.items():
            if param_name not in self.params_fixed_by_ancestry:
                time_param_list.append(param['type'] == ParamType.TIME)
        return time_param_list

    def compute_dependent_params(self, params):
        if not self.has_been_fixed:
            raise Exception("The demography has not been fixed yet.")
        self.logger.info(f'Params before fixed-ancestry solving: {params}')
        if len(params) == len(self.free_params):
            full_params = params
            migration_matrix = self.get_migration_matrices(full_params, has_been_fixed=False)[0]
            # TODO: We should have known_ancestry_proportions_auto and known_ancestry_proportions_x
            #  self.proportions_from_matrix should be generalized as well
            #  We have 2 matrices and 4 known_ancestry_proportions
            #  fix_parameters_from_ancestry_proportions should be two fields in the driver file for
            #  the sex-specific case
            if numpy.allclose(self.proportions_from_matrix(migration_matrix)[:-1],
                              self.known_ancestry_proportions):
                return full_params
        else:
            full_params = params.copy()

        def param_objective_func(parametrized_demography: BaseParametrizedDemography, params_to_solve):
            nonlocal full_params
            params_to_solve[numpy.isnan(params_to_solve)] = 0
            full_params = parametrized_demography.insert_params(full_params, params_to_solve)
            # self.logger.info(f'Full params: {full_params}')
            parametrized_demography_migration_matrix = parametrized_demography.get_migration_matrices(
                full_params,
                has_been_fixed=False)[0]
            found_props = parametrized_demography.proportions_from_matrix(parametrized_demography_migration_matrix)[:-1]
            fixed_props = parametrized_demography.known_ancestry_proportions
            diff = found_props - fixed_props
            return diff

        solved_params = scipy.optimize.fsolve(lambda params_to_solve: param_objective_func(self, params_to_solve),
                                              numpy.ones(len(self.params_fixed_by_ancestry)) * .2)
        full_params = self.insert_params(full_params, solved_params)
        self.logger.info(f'Params after solving with ancestry proportions: {full_params}')
        return full_params

    def get_param_value(self, param_name: str, params: list[float]):
        """
        Gets the correct value from the name of the parameter and the list of passed params.
        If param_name is a number instead, uses the number directly
        """
        if isinstance(param_name, numbers.Number):
            return param_name
        if param_name in self.free_params:
            return params[self.free_params[param_name]['index']]
        if param_name in self.constant_params:
            return self.constant_params[param_name]['value']
        if param_name in self.dependent_params:
            return self.dependent_params[param_name](self, params)
        raise KeyError(f'Parameter "{param_name}" could not be found')

    def get_violation_score(self, params: list[float]):
        """
        Takes in a list of params equal to the length of free_params
        and returns a negative violation score if the resulting matrix would be or is invalid.
        """
        if self.has_been_fixed:
            if len(params) != len(self.free_params):
                full_params = self.insert_params(params.copy(), [0 for _ in self.params_fixed_by_ancestry])
            else:
                full_params = params
            violation_score = min(self.check_bounds(full_params), self.check_constraints(full_params))
            if violation_score < 0:
                return violation_score
            params = self.compute_dependent_params(params)
        self.logger.info(f'Running bounds check.')
        violation_score = min(self.check_bounds(params), self.check_constraints(params))
        if violation_score < 0:
            return violation_score
        for migration_matrix in self.get_migration_matrices(params):
            totmig = migration_matrix.sum(1).max()
            if 1 - totmig < violation_score:
                violation_score = 1 - totmig
        return violation_score

    def check_constraints(self, params: list[float]):
        """
        Constraints take the form of a dict {'param_subset':Tuple[String], 'expression': lambda (param_subset)}
        The violation score is the largest negative value from all the constraints
        """
        violation_score = 0
        if not self.has_been_fixed:
            for constraint in self.constraints:
                violation = constraint['expression'](
                    [self.get_param_value(param_name, params) for param_name in constraint['param_subset']])
                if violation < violation_score:
                    violation_score = violation
                    logging.warning(f'{constraint["message"]} Out of bounds by: {-violation}.')
        else:
            if len(params) != len(self.free_params):
                full_params = self.insert_params(params.copy(), [0 for _ in self.params_fixed_by_ancestry])
            else:
                full_params = params
            for constraint in self.constraints:
                violation = constraint['expression'](
                    [self.get_param_value(param_name, full_params) for param_name in constraint['param_subset']])
                if violation < violation_score:
                    logging.warning(f'{constraint["message"]} Out of bounds by: {-violation}.')
                    violation_score = violation
        return violation_score

    def insert_params(self, params, params_to_solve):
        if not self.params_fixed_by_ancestry:
            raise Exception("The insert_params method must be called only on fixed demographies")
        # self.logger.info(f'Params: {params}, params')
        if len(params_to_solve) != len(self.params_fixed_by_ancestry):
            raise ValueError('Incorrect number of parameters to be solved')
        if len(params) + len(params_to_solve) == len(self.free_params):
            iter_params = iter(params)
            iter_params_to_solve = iter(params_to_solve)
            params = [next(iter_params_to_solve) if (param_name in self.params_fixed_by_ancestry) else next(iter_params)
                      for param_name in self.free_params]
            return params
        if len(params) == len(self.free_params):
            for param_name, value in zip(self.params_fixed_by_ancestry, params_to_solve):
                params[self.free_params[param_name]['index']] = value
            return params
        raise ValueError('Parameters fixed by ancestry proportions could not be resolved with the given parameters.')

    def check_bounds(self, params: list[float]):
        """
        Checks the bounds on parameters.
        Bounds should be absolute restrictions on possible parameter values,
        whereas Constraints should be restrictions on parameter values relative to each other.
        """
        violation_score = 0
        if not self.has_been_fixed:
            for param_name, param_info in self.free_params.items():
                violation = self.get_param_value(param_name, params) - param_info['bounds'][0]
                if violation < violation_score:
                    logging.warning(
                        f'Lower bound for parameter {param_name} is {param_info["bounds"][0]}. '
                        f'Out of bounds by: {-violation}.')
                    violation_score = violation
                violation = param_info['bounds'][1] - self.get_param_value(param_name, params)
                if violation < violation_score:
                    logging.warning(
                        f'Upper bound for parameter {param_name} is {param_info["bounds"][1]}. '
                        f'Out of bounds by: {-violation}.')
                    violation_score = violation
        else:
            if len(params) != len(self.free_params):
                full_params = self.insert_params(params.copy(), [0 for _ in self.params_fixed_by_ancestry])
            else:
                full_params = params
            # print(full_params, self.free_params)
            for param_name, param_info in self.free_params.items():
                if param_name in self.params_fixed_by_ancestry:
                    continue
                violation = self.get_param_value(param_name, full_params) - param_info['bounds'][0]
                if violation < violation_score:
                    logging.warning(
                        f'Lower bound for parameter {param_name} is {param_info["bounds"][0]}. '
                        f'Current value is {self.get_param_value(param_name, full_params)}.')
                    violation_score = violation
                violation = param_info['bounds'][1] - self.get_param_value(param_name, full_params)
                if violation < violation_score:
                    logging.warning(
                        f'Upper bound for parameter {param_name} is {param_info["bounds"][1]}. '
                        f'Current value is {self.get_param_value(param_name, full_params)}.')
                    violation_score = violation
        return violation_score

    @staticmethod
    def parse_proportions(ancestor_names: list[str], proportions: list[str]) -> tuple[dict[str:str], str]:
        """
        Parses the ancestry proportions used in a founding event into a dict of parametrized source populations
        and a remainder population.
        May later be folded into the add_founder_event() method.
        TODO: add support for int arguments in proportions
        """
        remainder_population = None
        remainder_proportion_string = None
        source_populations = {}
        population_and_proportion = zip(ancestor_names, proportions)
        for population, proportion in population_and_proportion:
            if isinstance(proportion, str) and proportion.startswith('1-'):
                if remainder_population is not None:
                    raise ValueError(
                        'More than one population detected whose proportion parameter begins with "1-".\n'
                        'This syntax is reserved for the population whose proportion is fixed'
                        ' by the proportions of the other populations '
                        'such that the sum of all proportions is 1.\n'
                        'Only one proportion should be an expression beginning with "1-"')
                remainder_population = population
                remainder_proportion_string = proportion
            else:
                if '-' in proportion:
                    raise ValueError('Parameter names cannot contain "-" when used in founding events.')
                source_populations.update({population: proportion})

        # Check that a remainder population was found
        assert remainder_population, ('The given proportions are not guaranteed to sum to 1.\n'
                                      'When using parametrized founding proportions, a population must be specified '
                                      'whose proportion takes the form "1-[the other proportions]".\n'
                                      'For example, in a three-population founder event, '
                                      'if two of the proportions are "a" and "b", the other must be "1-a-b"')

        # Check if the "1-" expression correctly contains all the other parameters.
        if not all(p1 == p2 for p1, p2 in
                   zip(source_populations.values(), remainder_proportion_string.split('-')[1:])):
            raise ValueError(
                'The given proportions are not guaranteed to sum to 1.\n'
                'When using parametrized founding proportions, a population must be specified '
                'whose proportion takes the form "1-[the other proportions]".\n'
                'For example, in a three-population founder event, if two of the proportions are "a" and "b",'
                ' the other must be "1-a-b"')

        return source_populations, remainder_population

    @abstractmethod
    def fix_ancestry_proportions(self, params_to_fix, proportions):
        pass

    @abstractmethod
    def get_migration_matrices(self, params: list[float], has_been_fixed: bool = None) -> list[numpy.ndarray]:
        pass

    @abstractmethod
    def add_pulse_migration(self, source_population, rate_param, time_param):
        pass

    @abstractmethod
    def add_continuous_migration(self, source_population, rate_param, start_param, end_param):
        pass
