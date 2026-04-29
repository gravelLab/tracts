from __future__ import annotations
import math
import os
from pathlib import Path
import numpy as np
import ruamel.yaml
from tracts.demography.base_parametrized_demography import BaseParametrizedDemography, BaseMigrationEvent, FounderEvent
from tracts.demography.parameter import ParamType
import logging
logger = logging.getLogger(__name__)

#TODO: Add constant parameters and parameter equations.
#TODO: Add penalization function for non-resolvable constraints.
#TODO: Add function to solve constraints shape.
class PulseEvent(BaseMigrationEvent):
    """
    A class representing a pulse migration event, parametrized by time and rate.

    Attributes
    ----------
    time_parameter : str
        The name of the parameter defining the time of the pulse migration.
    rate_parameter : str
        The name of the parameter defining the rate of the pulse migration.
    source_population : str
        The name of the source population for the pulse migration.   
    """

    def __init__(self, rate_parameter: str, time_parameter: str, source_population: str):
        """
        Initializes a PulseEvent object.

        Parameters
        ----------
        rate_parameter : str
            The name of the parameter defining the rate of the pulse migration.
        time_parameter : str
            The name of the parameter defining the time of the pulse migration.
        source_population : str
            The name of the source population for the pulse migration.
        """
        super().__init__(rate_parameter=rate_parameter,
                        source_population=source_population)
        self.time_parameter = time_parameter

    def execute(self, parametrized_demography: BaseParametrizedDemography, migration_matrix: np.ndarray, params: list[float]):
        """
        Adds the pulse migration event to the migration matrix.

        Parameters
        ----------
        parametrized_demography : BaseParametrizedDemography
            The parametrized demographic model.
        migration_matrix : np.ndarray
            The migration matrix to update.
        params : list[float]
            The list of parameter values.
        """
        t = parametrized_demography.get_param_value(param_name=self.time_parameter,
                                                    params=params)
        a = parametrized_demography.get_param_value(param_name=self.rate_parameter,
                                                   params=params)
        t2 = math.floor(t)
        r2 = a * (t2 + 1 - t)
        migration_matrix[t2, parametrized_demography.population_indices[self.source_population]] += r2
        migration_matrix[t2 + 1, parametrized_demography.population_indices[self.source_population]] += a * (t - t2) / (1 - r2)

class ContinuousEvent(BaseMigrationEvent):
    """
    A class representing a continuous migration event, parametrized by start time, end time, and rate.

    Attributes
    ----------
    start_parameter : str
        The name of the parameter defining the start time of the continuous migration.
    end_parameter : str
        The name of the parameter defining the end time of the continuous migration.
    rate_parameter : str
        The name of the parameter defining the rate of the continuous migration.
    source_population : str
        The name of the source population for the continuous migration.
    """

    def __init__(self, start_parameter: str, end_parameter: str, rate_parameter: str, source_population: str):
        """
        Initializes a ContinuousEvent object.

        Parameters
        ----------
        start_parameter : str
            The name of the parameter defining the start time of the continuous migration.
        end_parameter : str
            The name of the parameter defining the end time of the continuous migration.
        rate_parameter : str
            The name of the parameter defining the rate of the continuous migration.
        source_population : str
            The name of the source population for the continuous migration.
        """
        super().__init__(rate_parameter=rate_parameter,
                        source_population=source_population)
        self.start_parameter = start_parameter
        self.end_parameter = end_parameter

    def execute(self, parametrized_demography: ParametrizedDemography, migration_matrix: np.ndarray, params: list[float]):
        """
        Adds the continuous migration event to the migration matrix.

        Parameters
        ----------
        parametrized_demography : ParametrizedDemography
            The parametrized demographic model.
        migration_matrix : np.ndarray
            The migration matrix to update.
        params : list[float]
            The list of parameter values.
        """
        start_time = parametrized_demography.get_param_value(param_name=self.start_parameter,
                                                            params=params)
        end_time = parametrized_demography.get_param_value(param_name=self.end_parameter,
                                                            params=params) if self.end_parameter else 1
        t2 = math.floor(start_time)
        t1 = math.ceil(end_time)
        a = parametrized_demography.get_param_value(param_name=self.rate_parameter,
                                                    params=params)

        migration_matrix[t1 - 1, parametrized_demography.population_indices[self.source_population]] += a * (t1 - end_time)
        for t in range(t1, t2+1):
            migration_matrix[t, parametrized_demography.population_indices[self.source_population]] += a
        migration_matrix[t2+1, parametrized_demography.population_indices[self.source_population]] += a * (start_time - t2)


class ParametrizedDemography(BaseParametrizedDemography):
    """
    A class representing a demographic history for multiple populations of interest, with parametrized migrations from other populations.

    Attributes
    ----------
    name : str
        The name of the demographic model.
    min_time : float
        The minimum time for the demographic model, in generations. Default is 1 generation ago.
    max_time : float
        The maximum time for the demographic model, in generations. Default is infinity.
    logger: logging.Logger
        The logger.
    """
    #TODO: Add support for int (constant) parameters.

    def __init__(self, name: str | None = None, min_time: float = 1, max_time: float = np.inf):
        """
        Initializes a ParametrizedDemography object.

        Parameters
        ----------
        name : str | None, optional
            The name of the demographic model. Default is None, which will be set to an empty string.
        min_time : float, optional
            The minimum time for the demographic model, in generations. Default is 1 generation ago.
        max_time : float, optional
            The maximum time for the demographic model, in generations. Default is infinity.
        """
        name = name if name is not None else ""
        super().__init__(name=name,
                        min_time=min_time,
                        max_time=max_time)
        self.logger = logger
        
    def execute_migration_events(self, migration_matrices: dict[str, np.ndarray], params: list[float]):
        """
        Adds the migration events to the migration matrices.

        Parameters
        ----------
        migration_matrices : dict[str, np.ndarray]
            A dictionary mapping population names to their corresponding migration matrices.
        params : list[float]
            The list of parameter values.
        """
        for population, events in self.events.items():
            for event in events:
                event.execute(self, migration_matrices[population], params)

    def get_migration_matrices(self, params: list[float]) -> dict[str, np.ndarray]:
        """
        Takes in a list of params equal to the length of :py:attr:`~tracts.demography.base_parametrized_demography.BaseParametrizedDemography.model_base_params` and returns a  :math:`P \\times G` migration matrix for each population of interest
        where *P* is the number of incoming populations and *G* is the number of generations. If one of the parameters (time or migration) is incorrect, returns an empty matrix.

        Parameters
        ----------
        params : list[float]
            A list of parameter values, where the order of the values corresponds to the order of the parameters in :py:attr:`~tracts.demography.base_parametrized_demography.BaseParametrizedDemography.model_base_params`.
        
        Returns
        -------
        dict[str, np.ndarray]
            A dictionary mapping population names to their corresponding migration matrices.
        """
        if self.finalized is not True:
            self.finalize()

        if len(params) != len(self.model_base_params):
            raise ValueError(
                f'Number of supplied parameters ({len(params)}) does not match the number of model parameters ({len(self.model_base_params)}).')

        if not self.founder_events:
            raise ValueError('Demography contains no founder events.')
        
        migration_matrices = {population_of_interest: founder_event.execute(self, params) 
                              for population_of_interest, founder_event in self.founder_events.items()}

        self.execute_migration_events(migration_matrices=migration_matrices,
                                    params=params)

        return migration_matrices
    
    def get_founding_time(self, population: str) -> str:
        """
        Gets the name of the parameter defining the founding time of a population.

        Parameters
        ----------
        population : str
            The name of the population for which to get the founding time.
        
        Returns
        -------
        str
            The name of the parameter defining the founding time of the population.
        """
        if population not in self.founder_events:
            raise ValueError(f'Population {population} is missing a founder event.')
        return self.founder_events[population].found_time
    
    def add_pulse_migration(self, dest_population: str, source_population: str, rate_param: str, time_param: str):
        """
        Adds a pulse migration from a source population, parametrized by time and rate.

        Parameters
        ----------
        dest_population : str
            The name of the destination population for the pulse migration.
        source_population : str
            The name of the source population for the pulse migration.
        rate_param : str
            The name of the parameter defining the rate of the pulse migration.
        time_param : str
            The name of the parameter defining the time of the pulse migration.
        """
        self.add_population(population_name=source_population)
        self.add_parameter(param_name=rate_param,
                        param_type=ParamType.RATE)
        self.add_parameter(param_name=time_param,
                           param_type=ParamType.TIME)
        founding_time = self.get_founding_time(population=dest_population)
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

    def add_continuous_migration(self, dest_population: str, source_population: str, rate_param: str, start_param: str, end_param: str = None):
        """
        Adds a continuous migration from a source population, parametrized by start_time, end_time, and magnitude.
        """
        self.add_population(population_name=source_population)
        self.add_parameter(param_name=rate_param,
                           param_type=ParamType.RATE)
        self.add_parameter(param_name=start_param,
                           param_type=ParamType.TIME)
        founding_time = self.get_founding_time(population=dest_population)
        self.constraints.append({
            'param_subset': (founding_time, start_param),
            'expression': lambda param_subset: param_subset[0] - 1 - param_subset[1],
            'message': 'Migrations cannot start before or during the founding of the population.'
        })

        if end_param:
            self.add_parameter(param_name=end_param,
                               param_type=ParamType.TIME)
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

        Parameters
        ----------
        dest_population : str
            The name of the destination population for the founder event.
        source_populations : dict[str, str]
            A dictionary where each key is a source population and each value is the name of the parameter defining the migration ratio of each population.
        remainder_population : str
            The name of the source population for the remaining migrants, such that the total migration ratio adds up to 1.
        found_time : str
            The name of the parameter defining the time of the founder event.
        end_time : str, optional
            The name of the parameter defining the end time of the founder event, for continuous founding. If None, the founder event is a pulse founding. Default is None.
        """
        if dest_population in self.founder_events:
            raise ValueError(f'Population {dest_population} cannot have more than one founder event.')

        for population, rate_param in source_populations.items():
            self.add_population(population_name=population)
            self.add_parameter(param_name=rate_param,
                               param_type=ParamType.RATE)     

        self.add_parameter(param_name=found_time,
                        param_type=ParamType.TIME)
        
        if end_time is None:
            self.add_population(population_name=remainder_population)
        else:
            self.add_parameter(param_name=end_time,
                               param_type=ParamType.TIME)
        
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
        Creates an instance of :class:`~tracts.demography.parametrized_demography.ParametrizedDemography` from a YAML file.

        Parameters
        ----------
        source : str | Path
            The path to the YAML file containing the demographic model. See online documentations for details on how to specify demographic models in YAML files.
        
        Returns
        -------
        ParametrizedDemography
            An instance of :class:`~tracts.demography.parametrized_demography.ParametrizedDemography` representing the demographic model specified in the YAML file.
        """
        yaml = ruamel.yaml.YAML(typ="safe")
        if isinstance(source, (str, bytes, os.PathLike)):
            with open(source, 'r') as file:
                demes_data=yaml.load(file)
        else:
            demes_data=yaml.load(source) # Assume it's a file-like object.
        
        assert isinstance(demes_data, dict), "The provided .yaml file is invalid."        
        demography = ParametrizedDemography('Unnamed Model')

        if 'model_name' in demes_data:
            demography.name = demes_data['model_name']

        for population in demes_data['demes']:
            if 'ancestors' in population:
                demography.parametrized_populations.append(population['name'])
                
                if 'end_time' in population.keys(): # Continuous founding
                    source_populations = {pop:label for pop,label in zip(population['ancestors'], population['proportions'])}               
                    demography.add_founder_event(dest_population=population['name'],
                                                source_populations=source_populations,
                                                remainder_population=None,
                                                found_time=population['start_time'],
                                                end_time=population['end_time'])
                else:    
                    source_populations, remainder_population = ParametrizedDemography.parse_proportions(ancestor_names=population['ancestors'],
                                                                                                        proportions=population['proportions'])
                    demography.add_founder_event(dest_population=population['name'],
                                                source_populations=source_populations,
                                                remainder_population=remainder_population,
                                                found_time=population['start_time'],
                                                end_time=None)
        if 'pulses' in demes_data:
            for pulse in demes_data['pulses']:
                if 'dest' in pulse and pulse['dest'] in demography.parametrized_populations:
                    for source, proportion in zip(pulse['sources'], pulse['proportions']):
                        demography.add_pulse_migration(dest_population=pulse['dest'],
                                                        source_population=source,
                                                        rate_param=proportion,
                                                        time_param=pulse['time'])
        if 'migrations' in demes_data:
            for migration in demes_data['migrations']:
                if 'dest' in migration and migration['dest'] in demography.parametrized_populations:
                    demography.add_continuous_migration(dest_population=migration['dest'],
                                                        source_population=migration['source'],
                                                        rate_param=migration['rate'],
                                                        start_param=migration['start_time'],
                                                        end_param=migration['end_time'])
        demography.finalize()
        return demography
