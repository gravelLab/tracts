from __future__ import annotations
import os
from pathlib import Path
from typing import Callable
import numpy
import ruamel.yaml
from tracts.demography.base_parametrized_demography import BaseParametrizedDemography
from tracts.demography.parametrized_demography import ParametrizedDemography
from tracts.demography.parameter import ParamType
from enum import Enum
import logging
logger = logging.getLogger(__name__)

class SexType(Enum):
    """
    A class representing the sex-specific parameters in a demographic model.

    Attributes
    ----------
    suffix : str
        The suffix to be added to the parameter name to indicate the sex-specific parameter.
    expression : Callable[[str, str], Callable[[BaseParametrizedDemography, list[float]], float]]
        A function that takes in the overall rate parameter and the sex bias parameter and returns a function to compute the corresponding rate for that sex in a demography.
    MALE : tuple[str, Callable[[str, str], Callable[[BaseParametrizedDemography, list[float]], float]]]
        A tuple representing the male-specific parameters, where the first element is the suffix for male parameters and the second element is the function to compute the male-specific rate.
    FEMALE : tuple[str, Callable[[str, str], Callable[[BaseParametrizedDemography, list[float]], float]]]
        A tuple representing the female-specific parameters, where the first element is the suffix for female parameters and the second element is the function to compute the female-specific rate.
    """
    @staticmethod
    def male_female_sex_type_function(multiplier: float) -> Callable[[str,str], Callable[[BaseParametrizedDemography,list[float]], float]]:
        r"""
        Returns a function :math:`f`(``[multiplier]``) to compute the sex-specific migration rates. Given a rate parameter and the accompanying sex bias, 
        :math:`f`(``[multiplier])(rate, sex_bias)`` will output a function to compute the corresponding rate in a demography.

        Parameters
        ----------
        multiplier : float
            A float parameter that indicates the sex for which the function will compute the migration rate. It should be -1 for male-specific parameters and +1 for female-specific parameters. 

        Returns
        -------
        Callable[[str, str], Callable[[BaseParametrizedDemography, list[float]], float]]
            A function that takes in the overall rate parameter and the sex bias parameter and returns a function to compute the corresponding rate for that sex in a demography.
        """
        return (lambda rate_param, sex_bias_param:
                    (lambda demography, params: 
                        demography.get_param_value(param_name=rate_param,
                                                params=params)+
                        multiplier*demography.get_param_value(param_name=sex_bias_param,
                                                            params=params)*
                        (1/2-numpy.abs(demography.get_param_value(param_name=rate_param,
                                                                params=params)-1/2))
                    )
                )

    MALE=(
        "_male",
        male_female_sex_type_function(-1)
    )

    FEMALE=(
        "_female",
        male_female_sex_type_function(1)
    )
    
    def __init__(self, suffix: str, expression: Callable[[str,str], Callable[[BaseParametrizedDemography,list[float]], float]]):
        """
        Initializes a SexType object.

        Parameters
        ----------
        suffix : str
            The suffix to be added to the parameter name to indicate the sex-specific parameter.
        expression : Callable[[str, str], Callable[[BaseParametrizedDemography, list[float]], float]]
            A function that takes in the overall rate parameter and the sex bias parameter and returns a function to compute the corresponding rate for that sex in a demography.
        """
        self.suffix=suffix
        self.expression=expression

sex_types=[SexType.MALE, SexType.FEMALE]

class ParametrizedDemographySexBiased(ParametrizedDemography):
    """
    A class representing a demographic history with varying rates of male and female migration. The classc onstructs a separate migration matrix for male and female individuals,
    where each entry represents the proportion admixed individuals of that sex that is replaced during that migration.
    The matrices are implemented as two subpopulations whose migrations have independent rate parameters but linked time parameters.

    Attributes
    ----------
    name : str
        The name of the demographic model.
    min_time : float
        The minimum time for any demographic event in the model.
    max_time : float
        The maximum time for any demographic event in the model.
    allosome_label : str | None
        The label for the allosome chromosome in the model, if applicable.
    logger: logging.Logger
        The logger.
    """

    def __init__(self, name: str | None = None, min_time: float = 1, max_time: float = numpy.inf, allosome_label: str | None = None):

        name = name if name is not None else ""
        super().__init__(name=name,
                        min_time=min_time,
                        max_time=max_time)
        self.allosome_label=allosome_label
        self.logger = logger

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
            The name of the parameter defining the migration rate for the pulse migration.
        time_param : str
            The name of the parameter defining the time of the pulse migration.
        """
        self.add_parameter(param_name=rate_param,
                        param_type=ParamType.RATE)
        sex_bias_param=f'{rate_param}_sex_bias'
        self.add_parameter(param_name=sex_bias_param,
                            param_type=ParamType.SEX_BIAS)
        for sex_type in sex_types: # Sex-specific rates are computed from an overall rate (rate_param) and a corresponding sex bias.  
            self.add_dependent_parameter(param_name=f"{rate_param}{sex_type.suffix}",
                                        expression=sex_type.expression(rate_param, sex_bias_param),
                                        param_type=ParamType.RATE)
            super().add_pulse_migration(dest_population=f"{dest_population}{sex_type.suffix}",
                                        source_population=source_population,
                                        rate_param=f"{rate_param}{sex_type.suffix}",
                                        time_param=time_param)

    def add_continuous_migration(self, dest_population: str, source_population: str, rate_param: str, start_param: str, end_param: str):
        """
        Adds a continuous migration from a source population, parametrized by start_time, end_time, and magnitude.

        Parameters
        ----------
        dest_population : str
            The name of the destination population for the continuous migration.
        source_population : str
            The name of the source population for the continuous migration.
        rate_param : str
            The name of the parameter defining the migration rate for the continuous migration.
        start_param : str
            The name of the parameter defining the start time of the continuous migration.
        end_param : str
            The name of the parameter defining the end time of the continuous migration.
        """
        self.add_parameter(param_name=rate_param,
                           param_type=ParamType.RATE)
        sex_bias_param=f'{rate_param}_sex_bias'
        self.add_parameter(param_name=sex_bias_param,
                           param_type=ParamType.SEX_BIAS)
        for sex_type in sex_types:
            self.add_dependent_parameter(param_name=f"{rate_param}{sex_type.suffix}",
                                        expression=sex_type.expression(rate_param, sex_bias_param),
                                        param_type=ParamType.RATE)
            super().add_continuous_migration(dest_population=f"{dest_population}{sex_type.suffix}",
                                            source_population=source_population,
                                            rate_param=f"{rate_param}{sex_type.suffix}",
                                            start_param=start_param, end_param=end_param)

    def add_founder_event(self, dest_population: str, source_populations: dict[str, str], remainder_population: str, found_time: str, end_time: str | None = None):
        """
        Adds a founder event. A parametrized demography must have exactly one founder event.
        
        Parameters
        ----------
        dest_population : str
            The name of the destination population for the founder event.
        source_populations : dict[str, str]
            A dictionary where the keys are the names of the source populations for the founder event and the values are the names of the parameters defining the proportions contributed by each source population.
        remainder_population : str
            The name of the population that contributes the remaining proportion for the founder event, if applicable. If None, the proportions defined by the parameters in source_populations should sum to 1.
        found_time : str
            The name of the parameter defining the time of the founder event.
        end_time : str | None
            The name of the parameter defining the end time of the founder event, if applicable. If None, the founder event is assumed to be instantaneous.
        """
        self.parametrized_populations.append(dest_population)

        for population, rate_param in source_populations.items():
            self.add_parameter(param_name=rate_param,
                               param_type=ParamType.RATE)
            sex_bias_param=f'{rate_param}_sex_bias'
            self.add_parameter(param_name=sex_bias_param,
                               param_type=ParamType.SEX_BIAS)
            for sex_type in sex_types:
                self.add_dependent_parameter(param_name=f"{rate_param}{sex_type.suffix}",
                                            expression=sex_type.expression(rate_param, sex_bias_param),
                                            param_type=ParamType.RATE)

        for sex_type in sex_types:
            super().add_founder_event(dest_population=f"{dest_population}{sex_type.suffix}",
                                    source_populations={population: f"{rate_param}{sex_type.suffix}" for population, rate_param in source_populations.items()},
                                    remainder_population=remainder_population,
                                    found_time=found_time,
                                    end_time=end_time)
        
    def proportions_from_matrices(self, migration_matrices: dict[str, numpy.ndarray]):
        """
        Computes the ancestry proportions for each population and chromosome type from the migration matrices. 

        Parameters
        ----------
        migration_matrices : dict[str, numpy.ndarray]
            A dictionary where the keys are the names of the migration matrices for each population and sex (formatted as '{population}{sex_type.suffix}') and the values are the corresponding migration matrices as numpy arrays.
        
        Returns
        -------
        proportions : dict[str, numpy.ndarray]
            A dictionary where the keys are the names of the populations and chromosome types and the values are the corresponding ancestry proportions as numpy arrays.
        """
        proportions={}
        for population in self.parametrized_populations:
            male_matrix=migration_matrices[f'{population}{SexType.MALE.suffix}']
            female_matrix=migration_matrices[f'{population}{SexType.FEMALE.suffix}']
            proportions[f'{population}_autosomal']=self.proportions_from_matrix(migration_matrix=(male_matrix+female_matrix)/2)
            current_male_proportions=male_matrix[-1,:]
            current_female_proportions=female_matrix[-1,:]
            for male_row, female_row in zip(male_matrix[-2::-1, :], female_matrix[-2::-1, :]):
                (current_male_proportions, current_female_proportions) = (
                    current_female_proportions * (1 - female_row.sum()) + female_row,
                    1/2*(current_male_proportions * (1 - male_row.sum()) + male_row+current_female_proportions * (1 - female_row.sum()) + female_row)
                )
            proportions[f'{population}_{self.allosome_label}']=(current_male_proportions+2*current_female_proportions)/3
        return proportions
    
    def proportions_from_matrices_return_keys(self):
        """
        Returns the keys for the ancestry proportions computed from the migration matrices, which are formatted as '{population}_{chromosome_type}' where chromosome_type is either 'autosomal' or the allosome label specified for the demography.

        Returns
        -------
        set[str]
             The set of keys for the ancestry proportions computed from the migration matrices.
        """
        if not self.allosome_label:
            self.logger.warning("The allosome label for this demography has not been specified. Defaulting to 'X'.")
            self.allosome_label="X"
        return set([f'{population}_{label}' for label in ['autosomal', self.allosome_label] for population in self.parametrized_populations])

    @staticmethod
    def load_from_YAML(source: str | Path) -> ParametrizedDemographySexBiased:
        """
        Creates an instance of :class:`~tracts.demography.parametrized_demography_sex_biased.ParametrizedDemographySexBiased` from a YAML file.

        Parameters
        ----------
        source : str | Path
            The file path to the YAML file containing the demographic model specification. See online documentation for the expected format of the YAML file.
        
        Returns
        -------
        ParametrizedDemographySexBiased
            An instance of :class:`~tracts.demography.parametrized_demography_sex_biased.ParametrizedDemographySexBiased` representing the demographic model specified in the YAML file.
        """
        yaml = ruamel.yaml.YAML(typ="safe")
        if isinstance(source, (str, bytes, os.PathLike)):
            with open(source, 'r') as file:
                demes_data=yaml.load(file)
        else:
            demes_data=yaml.load(source) # Assume it's a file-like object
        
        assert isinstance(demes_data, dict), ".yaml file was invalid." 
        demography = ParametrizedDemographySexBiased(name='Unnamed Model')
        if 'model_name' in demes_data:
            demography.name = demes_data['model_name']

        for population in demes_data['demes']:
            if 'ancestors' in population:
                demography.parametrized_populations.append(population['name'])
                if 'end_time' in population.keys(): # Continuous founder event
                    source_populations = {pop:param for pop,param in zip(population['ancestors'], population['proportions'])}
                    remainder_population = None
                    end_time =  population['end_time']
                else:
                    source_populations, remainder_population = ParametrizedDemographySexBiased.parse_proportions(ancestor_names=population['ancestors'],
                                                                                                                proportions=population['proportions'])
                    end_time = None
                
                demography.add_founder_event(dest_population=population['name'],
                                            source_populations=source_populations,
                                            remainder_population=remainder_population,
                                            found_time=population['start_time'],
                                            end_time=end_time)
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
