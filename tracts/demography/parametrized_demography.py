from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import ruamel.yaml

from tracts.demography.base_parametrized_demography import BaseParametrizedDemography, BaseMigrationEvent, FounderEvent
from tracts.demography.parameter import ParamType


#TODO: 
#Add constant parameters and parameter equations
#Add penalization function for non-resolvable constraints
#Add function to solve constraints shape


class PulseEvent(BaseMigrationEvent):

    def __init__(self, rate_parameter, time_parameter, source_population):
        super().__init__(rate_parameter=rate_parameter, source_population=source_population)
        self.time_parameter = time_parameter

    def execute(self, parametrized_demography: BaseParametrizedDemography, migration_matrix: np.ndarray, params):
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

    def execute(self, parametrized_demography: ParametrizedDemography, migration_matrix: np.ndarray, params):
        start_time = parametrized_demography.get_param_value(self.start_parameter, params)
        end_time = parametrized_demography.get_param_value(self.end_parameter, params) if self.end_parameter else 2
        t2 = math.floor(start_time)
        t1 = math.ceil(end_time)
        a = parametrized_demography.get_param_value(self.rate_parameter, params)

        migration_matrix[t1 - 1, parametrized_demography.population_indices[self.source_population]] += (
                a * (t1 - end_time))
        
        for t in range(t1, t2+1):
            migration_matrix[t, parametrized_demography.population_indices[self.source_population]] += a

        migration_matrix[t2+1, parametrized_demography.population_indices[self.source_population]] += a * (start_time - t2)


class ParametrizedDemography(BaseParametrizedDemography):
    """
    A class representing a demographic history for multiple populations of interest, with parametrized migrations from other populations.
    """
    #TODO: add support for int (constant) parameters.

    def __init__(self, name: str = "", min_time=2, max_time=np.inf):
        super().__init__(name=name, min_time=min_time, max_time=max_time)
        
    def execute_migration_events(self, migration_matrices, params):
        for population, events in self.events.items():
            for event in events:
                event.execute(self, migration_matrices[population], params)

    def get_migration_matrices(self, params: list[float]) -> dict[str, np.ndarray]:
        """
        Takes in a list of params equal to the length of model_base_params
        and returns a *pg* migration matrix for each population of interest
        where *p* is the number of incoming populations and *g* is the number of generations.
        If one of the parameters (time or migration) is incorrect, returns an empty matrix.
        """
        #This is old code that handled parameter fixing within this function. Now, parameter fixing is handled externally.
        #if solve_using_known_proportions is None: # If unspecified, determine from class state
        #    solve_using_known_proportions = self.parameter_handler.has_been_fixed
        #if insert_fixed_parameters and len(self.parameter_handler.user_params_fixed_by_value)>0: #insert fixed parameters. This happens before we fix parameters by ancestry
        #    
        #    params = self.parameter_handler.insert_fixed_params( model_base_params=self.model_base_params, 
        #                                                                fixed_params=self.fixed_parameter_values, params_to_optimize=params)
        #
        #if solve_using_known_proportions:
        #    self.logger.info(f'Generating migration matrix.')
        #    params = self.parameter_handler.compute_params_fixed_by_ancestry(params)
        if self.finalized is not True:
            self.finalize()

        if len(params) != len(self.model_base_params):
            raise ValueError(
                f'Number of supplied parameters ({len(params)}) does not match the number of'
                f' model parameters ({len(self.model_base_params)}).')

        if not self.founder_events:
            raise ValueError('Demography contains no founder events.')
        
        migration_matrices = {population_of_interest: founder_event.execute(self, params) 
                              for population_of_interest, founder_event in self.founder_events.items()}

        self.execute_migration_events(migration_matrices=migration_matrices, params=params)

        return migration_matrices
    
    def get_founding_time(self, population):
        if population not in self.founder_events:
            raise ValueError(f'Population {population} is missing a founder event.')
        return self.founder_events[population].found_time
    
    def add_pulse_migration(self, dest_population, source_population, rate_param, time_param):
        """
        Adds a pulse migration from source population A, parametrized by time and rate.
        """
        self.add_population(source_population)
        self.add_parameter(rate_param, param_type=ParamType.RATE)
        self.add_parameter(time_param, param_type=ParamType.TIME)
        founding_time = self.get_founding_time(dest_population)
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
        self.events[dest_population].append(pulse_migration_event)

    def add_continuous_migration(self, dest_population, source_population, rate_param, start_param, end_param):
        """
        Adds a continuous migration from source population A, parametrized by start_time, end_time, and magnitude.
        """
        self.add_population(source_population)
        self.add_parameter(rate_param, param_type=ParamType.RATE)
        self.add_parameter(start_param, param_type=ParamType.TIME)
        founding_time = self.get_founding_time(dest_population)
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
        self.events[dest_population].append(continuous_migration_event)

    def add_founder_event(self, dest_population: str, source_populations: dict[str, str], 
                          remainder_population: str, found_time: str, end_time: str = None) -> None:
        """
        Adds a founder event. A parametrized demography must have exactly one founder event.
        source_populations is a dict where each key is a population
        and each value is the name of the parameter defining the migration ratio of each population
        remainder_population is the source of the remaining migrants, such that the total migration ratio adds up to 1.
        found_time is the name of the parameter defining the time of migration.
        """
        if dest_population in self.founder_events:
            raise ValueError(f'Population {dest_population} cannot have more than one founder event.')

        for population, rate_param in source_populations.items():
            self.add_population(population)
            self.add_parameter(rate_param, param_type=ParamType.RATE)

        

        self.add_parameter(found_time, param_type=ParamType.TIME)
        if end_time is None:
            self.add_population(remainder_population)
        else:
            self.add_parameter(end_time, param_type=ParamType.TIME)
        
        self.founder_events[dest_population] = FounderEvent(
            found_time=found_time,
            source_populations=source_populations,
            remainder_population=remainder_population,
            end_time = end_time
            )
        
        self.events[dest_population]=[]

    def get_random_parameters():
        return None

    @staticmethod
    def load_from_YAML(source: str | Path) -> ParametrizedDemography:
        """
        Creates an instance of ParametrizedDemography from a YAML file.
        """
        yaml = ruamel.yaml.YAML(typ="safe")
        if isinstance(source, (str, bytes, os.PathLike)):
            with open(source, 'r') as file:
                demes_data=yaml.load(file)
        else:
            # Assume it's a file-like object
            demes_data=yaml.load(source)
        
        assert isinstance(demes_data, dict), ".yaml file was invalid."        
        demography = ParametrizedDemography('Unnamed Model')

    
        if 'model_name' in demes_data:
            demography.name = demes_data['model_name']

        for population in demes_data['demes']:
            if 'ancestors' in population:
                demography.parametrized_populations.append(population['name'])
                if 'end_time' in population.keys(): # Continuous founding
                    source_populations = {pop:label for pop,label in zip(population['ancestors'], population['proportions'])}               
                    demography.add_founder_event(population['name'], source_populations, None, population['start_time'], population['end_time'])
                else:    
                    source_populations, remainder_population = ParametrizedDemography.parse_proportions(
                        population['ancestors'], population['proportions'])
                    demography.add_founder_event(population['name'], source_populations, remainder_population, population['start_time'])
        if 'pulses' in demes_data:
            for pulse in demes_data['pulses']:
                if 'dest' in pulse and pulse['dest'] in demography.parametrized_populations:
                    for source, proportion in zip(pulse['sources'], pulse['proportions']):
                        demography.add_pulse_migration(pulse['dest'], source, proportion, pulse['time'])
        if 'migrations' in demes_data:
            for migration in demes_data['migrations']:
                if 'dest' in migration and migration['dest'] in demography.parametrized_populations:
                    demography.add_continuous_migration(migration['dest'], migration['source'], migration['rate'],
                                                        migration['start_time'], migration['end_time'])
        demography.finalize()
        return demography
