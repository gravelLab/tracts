from __future__ import annotations

import math
from pathlib import Path

import numpy
import ruamel.yaml

from tracts.demography.base_parametrized_demography import BaseParametrizedDemography, BaseMigrationEvent, FounderEvent
from tracts.demography.param_types import ParamType

"""
TODO: 
Add continuous migrations
Add fixed parameters and parameter equations
Add penalization function for non-resolvable constraints
Add function to solve constraints shape
Auto-create constraint function tyo give known ancestry fractions
Complete function to run the optimizer
"""

sex_ration_yaml_key = "male_to_female_ratio"
population_ancestor_name_yaml_key = "name"

class PulseEvent(BaseMigrationEvent):

    def __init__(self, rate_parameter, time_parameter, source_population):
        super().__init__(rate_parameter=rate_parameter, source_population=source_population)
        self.time_parameter = time_parameter

    def execute(self, parametrized_demography: BaseParametrizedDemography, migration_matrix: numpy.ndarray, params):
        t = parametrized_demography.get_param_value(self.time_parameter, params)
        a = parametrized_demography.get_param_value(self.rate_parameter, params)
        t2 = math.floor(t)
        # print(f'Pulse average time: {t}. Pulse start time: {t2}.
        # Founding time: {self.get_param_value(self.founding_time_param, params)}')
        r2 = a * (t2 + 1 - t)
        migration_matrix[t2, parametrized_demography.population_indices[self.source_population]] += r2
        migration_matrix[t2 + 1, parametrized_demography.population_indices[self.source_population]] += (
                a * (t - t2) / (1 - r2))


class ContinuousEvent(BaseMigrationEvent):

    def __init__(self, start_parameter, end_parameter, rate_parameter, source_population):
        super().__init__(rate_parameter=rate_parameter, source_population=source_population)
        self.start_parameter = start_parameter
        self.end_parameter = end_parameter

    def execute(self, parametrized_demography: ParametrizedDemography, migration_matrix: numpy.ndarray, params):
        start_time = parametrized_demography.get_param_value(self.start_parameter, params)
        end_time = parametrized_demography.get_param_value(self.end_parameter, params) if self.end_parameter else 2
        t1 = math.floor(start_time)
        t2 = math.ceil(end_time)
        a = parametrized_demography.get_param_value(self.rate_parameter, params)

        migration_matrix[t1 - 1, parametrized_demography.population_indices[self.source_population]] += (
                a * (end_time - t1))

        for t in range(t1, t2):
            migration_matrix[t, parametrized_demography.population_indices[self.source_population]] += a

        migration_matrix[t1, parametrized_demography.population_indices[self.source_population]] += a * (
                start_time - t1)
        migration_matrix[t2, parametrized_demography.population_indices[self.source_population]] += a * (t2 - end_time)


class ParametrizedDemography(BaseParametrizedDemography):
    """
    A class representing a demographic history for a population, with parametrized migrations from other populations.
    TODO: add support for int (constant) parameters
    """

    def __init__(self, name: str = "", min_time=2, max_time=numpy.inf):
        super().__init__(name=name, min_time=min_time, max_time=max_time)

    def get_migration_matrices(self, params: list[float], has_been_fixed: bool = None) -> list[numpy.ndarray]:
        """
        Takes in a list of params equal to the length of free_params
        and returns a p*g migration matrix where p is the number of incoming populations and g is
        the number of generations
        If one of the parameters (time or migration) is incorrect, returns an empty matrix
        """
        if has_been_fixed is None:
            has_been_fixed = self.has_been_fixed
        if has_been_fixed:
            self.logger.info(f'Generating migration matrix.')
            params = self.compute_dependent_params(params)
        if self.finalized is not True:
            self.finalize()

        if len(params) != len(self.free_params):
            raise ValueError(
                f'Number of supplied parameters ({len(params)}) does not match the number of'
                f' model parameters ({len(self.free_params)}).')

        if not self.founder_event:
            raise ValueError('Population is missing a founder event.')

        migration_matrix = self.founder_event.execute(self, params)
        self.execute_migration_events(migration_matrix=migration_matrix, params=params)

        return [migration_matrix]

    def fix_ancestry_proportions(self, params_to_fix, proportions):
        """
        Tells the model to calculate certain rate parameters based on the known
        ancestry proportions of the sample population
        Proportions are calculated in driver.py
        """
        for param_name in params_to_fix:
            if param_name not in self.free_params:
                if param_name in self.dependent_params:
                    raise KeyError(f'{param_name} is already specified by another equation.')
                raise KeyError(f'{param_name} is not a parameter of this model.')
            if self.free_params[param_name]['type'] != 'rate':
                raise ValueError(f'{param_name} is not a rate parameter.')
        if len(proportions) != len(self.population_indices):
            raise ValueError(f'Number of given ancestry proportions is not equal to the number of population indices.')
        if len(params_to_fix) != len(self.population_indices) - 1:
            raise ValueError(f'Number of parameters to fix is not equal to the number of population indices - 1.')
        self.has_been_fixed = True
        self.params_fixed_by_ancestry = {param_name: '' for param_name in self.free_params if
                                         param_name in params_to_fix}
        self.known_ancestry_proportions = proportions[:-1]
        self.reduced_constraints = [constraint for constraint in self.constraints if any(
            param_name in self.params_fixed_by_ancestry for param_name in constraint['param_subset'])]

    def add_pulse_migration(self, source_population, rate_param, time_param):
        """
        Adds a pulse migration from source population A, parametrized by time and rate
        """
        self.add_population(source_population)
        self.add_parameter(rate_param, param_type=ParamType.RATE)
        self.add_parameter(time_param, param_type=ParamType.TIME)
        founding_time = self.get_founding_time()
        self.constraints.append({
            'param_subset': (founding_time, time_param),
            'expression': lambda param_subset: param_subset[0] - param_subset[1] - 1,
            'message': 'Pulses cannot occur before or during the founding of the population.'
        })
        pulse_migration_event = PulseEvent(
            rate_parameter=rate_param,
            time_parameter=time_param,
            source_population=source_population
        )
        self.events.append(pulse_migration_event)

    def add_continuous_migration(self, source_population, rate_param, start_param, end_param):
        """
        Adds a continuous migration from source population A, parametrized by start_time, end_time, and magnitude
        """
        self.add_population(source_population)
        self.add_parameter(rate_param, param_type=ParamType.RATE)
        self.add_parameter(start_param, param_type=ParamType.TIME)
        founding_time = self.get_founding_time()
        self.constraints.append({
            'param_subset': (founding_time, start_param),
            'expression': lambda param_subset: param_subset[0] - 1 - param_subset[1],
            'message': 'Migrations cannot start before or during the founding of the population.'
        })

        if end_param:
            self.add_parameter(end_param, param_type=ParamType.TIME)
            self.constraints.append({
                'param_subset': (start_param, end_param),
                'expression': lambda param_subset: param_subset[0] - param_subset[1],
                'message': 'Migrations start time cannot be more recent than end time.'
            })

        continuous_migration_event = ContinuousEvent(
            rate_parameter=rate_param,
            start_parameter=start_param,
            end_parameter=end_param,
            source_population=source_population
        )
        self.events.append(continuous_migration_event)

    def get_migration_matrix(self, params: list[float], has_been_fixed: bool = None):
        return self.get_migration_matrices(params, has_been_fixed)[0]

    @staticmethod
    def load_from_YAML(filepath: str | Path) -> ParametrizedDemography:
        """
        Creates an instance of ParametrizedDemography from a YAML file
        """
        demography = ParametrizedDemography('')
        with open(filepath) as file, ruamel.yaml.YAML(typ="safe") as yaml:
            demes_data = yaml.load(file)
            assert isinstance(demes_data, dict), ".yaml file was invalid."
            demography.name = demes_data['model_name'] if 'model_name' in demes_data else 'Unnamed Model'
            for population in demes_data['demes']:
                if 'ancestors' in population:
                    parametrized_population = population['name']
                    source_populations, remainder_population = ParametrizedDemography.parse_proportions(
                        population['ancestors'], population['proportions'])
                    demography.add_founder_event(source_populations, remainder_population, population['start_time'])
            if 'pulses' in demes_data:
                for pulse in demes_data['pulses']:
                    if pulse['dest'] == parametrized_population:
                        for source, proportion in zip(pulse['sources'], pulse['proportions']):
                            demography.add_pulse_migration(source, proportion, pulse['time'])
            if 'migrations' in demes_data:
                for migration in demes_data['migrations']:
                    if 'dest' in migration and migration['dest'] == parametrized_population:
                        demography.add_continuous_migration(migration['source'], migration['rate'],
                                                            migration['start_time'], migration['end_time'])
            demography.finalize()
        return demography


class SexBiasedParametrizedDemography(BaseParametrizedDemography):

    def __init__(self, name: str = "", min_time=2, max_time=numpy.inf):
        super().__init__(name=name, min_time=min_time, max_time=max_time)

    @staticmethod
    def load_from_YAML(filename: str, name: str = '') -> SexBiasedParametrizedDemography:
        """
        Creates an instance of ParametrizedDemography from a YAML file
        """
        demography = SexBiasedParametrizedDemography(name=name)
        with open(filename) as file, ruamel.yaml.YAML(typ="safe") as yaml:
            demes_data = yaml.load(file)
            assert isinstance(demes_data, dict), ".yaml file was invalid."
            demography.name = demes_data['model_name'] if 'model_name' in demes_data else 'Unnamed Model'
            for population in demes_data['demes']:
                if 'ancestors' in population:
                    parametrized_population = population['name']
                    ancestors_info = population['ancestors']
                    population_proportions = population['proportions']
                    # Extract male_to_female_ratio for each ancestor
                    source_populations = []
                    ancestor_to_male_female_ratios = {}
                    for ancestor in ancestors_info:
                        ancestor_name = ancestor['name']
                        source_populations.append(ancestor_name)
                        sex_ratio_param_name = ancestor.get(sex_ration_yaml_key)
                        ancestor_to_male_female_ratios[ancestor_name] = sex_ratio_param_name
                        demography.add_parameter(param_name=sex_ratio_param_name, bounds=[-1, 1])
                    # Parse proportions
                    source_populations, remainder_population = SexBiasedParametrizedDemography.parse_proportions(
                        source_populations, population_proportions
                    )
                    start_time = population['start_time']
                    print(f"Population: {parametrized_population}, "
                          f"Male-to-Female Ratios: {ancestor_to_male_female_ratios}")
                    demography.add_founder_event(source_populations, remainder_population, start_time)
            if 'pulses' in demes_data:
                for pulse in demes_data['pulses']:
                    if pulse['dest'] == parametrized_population:
                        for source, proportion in zip(pulse['sources'], pulse['proportions']):
                            pulse_time = pulse['time']
                            demography.add_pulse_migration(source, proportion, pulse_time)
                    if sex_ration_yaml_key in pulse:
                        pulse_sex_ratio_param_name = pulse[sex_ration_yaml_key]
                        demography.add_parameter(param_name=pulse_sex_ratio_param_name, bounds=[-1, 1])
            if 'migrations' in demes_data:
                for migration in demes_data['migrations']:
                    if 'dest' in migration and migration['dest'] == parametrized_population:
                        demography.add_continuous_migration(migration['source'], migration['rate'],
                                                            migration['start_time'], migration['end_time'])
                    if sex_ration_yaml_key in migration:
                        migration_sex_ratio_param_name = migration[sex_ration_yaml_key]
                        demography.add_parameter(param_name=migration_sex_ratio_param_name, bounds=[-1, 1])
        demography.finalize()
        return demography

    def fix_ancestry_proportions(self, params_to_fix, proportions):
        pass

    def get_migration_matrices(self, params: list[float], has_been_fixed: bool = None) -> list[numpy.ndarray]:
        pass

    def add_pulse_migration(self, source_population, rate_param, time_param):
        pass

    def add_continuous_migration(self, source_population, rate_param, start_param, end_param):
        pass
