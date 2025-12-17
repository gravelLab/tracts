from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Callable

import numpy
import ruamel.yaml
import logging

from tracts.demography.base_parametrized_demography import BaseParametrizedDemography, FixedProportionsHandler
from tracts.demography.parametrized_demography import ParametrizedDemography
from tracts.demography.parameter import ParamType
from enum import Enum


class SexType(Enum):
    @staticmethod
    def male_female_sex_type_function(multiplier: float) -> Callable[[str,str], Callable[[BaseParametrizedDemography,list[float]], float]]:
        return (lambda rate_param, sex_bias_param:
                    (lambda demography, params: 
                        demography.get_param_value(rate_param, params)+
                        multiplier*demography.get_param_value(sex_bias_param, params)*
                        (1/2-numpy.abs(demography.get_param_value(rate_param, params)-1/2))
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
        self.suffix=suffix
        self.expression=expression

sex_types=[SexType.MALE, SexType.FEMALE]

class ParametrizedDemographySexBiased(ParametrizedDemography):
    """
    A class representing a demographic history with varying rates of male and female migration.
    Constructs a separate migration matrix for male and female individuals.
    Each entry represent the proportion of that sex that is replaced during that migration.
    The matrices are implemented as two subpopulations whose migrations have independent rate parameters but linked time parameters.
    """

    def __init__(self, name: str = "", min_time=2, max_time=numpy.inf, allosome_label=None):
        super().__init__(name=name, min_time=min_time, max_time=max_time)
        self.allosome_label=allosome_label

    def add_pulse_migration(self, dest_population, source_population, rate_param, time_param):
        """
        Adds a pulse migration from source population A, parametrized by time and rate.
        """
        self.add_parameter(rate_param, ParamType.RATE)
        sex_bias_param=f'{rate_param}_sex_bias'
        self.add_parameter(sex_bias_param, ParamType.SEX_BIAS)
        for sex_type in sex_types:
            self.add_dependent_parameter( f"{rate_param}{sex_type.suffix}", sex_type.expression(rate_param, sex_bias_param), ParamType.RATE)
            super().add_pulse_migration(f"{dest_population}{sex_type.suffix}", source_population, f"{rate_param}{sex_type.suffix}", time_param)

    def add_continuous_migration(self, dest_population, source_population, rate_param, start_param, end_param):
        """
        Adds a continuous migration from source population A, parametrized by start_time, end_time, and magnitude.
        """
        self.add_parameter(rate_param, ParamType.RATE)
        sex_bias_param=f'{rate_param}_sex_bias'
        self.add_parameter(sex_bias_param, ParamType.SEX_BIAS)
        for sex_type in sex_types:
            self.add_dependent_parameter( f"{rate_param}{sex_type.suffix}", sex_type.expression(rate_param, sex_bias_param), ParamType.RATE)
            super().add_continuous_migration( f"{dest_population}{sex_type.suffix}", source_population,  f"{rate_param}{sex_type.suffix}", start_param, end_param)

    def add_founder_event(self, dest_population, source_populations: dict[str, str], remainder_population: str, found_time: str) -> None:
        """
        Adds a founder event. A parametrized demography must have exactly one founder event.
        source_populations is a dict where each key is a population
        and each value is the name of the parameter defining the migration ratio of each population
        remainder_population is the source of the remaining migrants, such that the total migration ratio adds up to 1.
        found_time is the name of the parameter defining the time of migration.
        """
        self.parametrized_populations.append(dest_population)

        for population, rate_param in source_populations.items():
            self.add_parameter(rate_param, ParamType.RATE)
            sex_bias_param=f'{rate_param}_sex_bias'
            self.add_parameter(sex_bias_param, ParamType.SEX_BIAS)
            for sex_type in sex_types:
                self.add_dependent_parameter( f"{rate_param}{sex_type.suffix}", sex_type.expression(rate_param, sex_bias_param), ParamType.RATE)

        for sex_type in sex_types:
            super().add_founder_event(f"{dest_population}{sex_type.suffix}", {population: f"{rate_param}{sex_type.suffix}" 
                                                                              for population, rate_param in source_populations.items()},
                                                                              remainder_population, found_time)
        
    def proportions_from_matrices(self, migration_matrices: dict[str, numpy.ndarray]):
        proportions={}
        for population in self.parametrized_populations:
            male_matrix=migration_matrices[f'{population}{SexType.MALE.suffix}']
            female_matrix=migration_matrices[f'{population}{SexType.FEMALE.suffix}']
            proportions[f'{population}_autosomal']=self.proportions_from_matrix((male_matrix+female_matrix)/2)
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
        if not self.allosome_label:
            self.logger.warning("The allosome label for this demography has not been specified. Defaulting to 'X'")
            self.allosome_label="X"
        return set([f'{population}_{label}' for label in ['autosomal', self.allosome_label] for population in self.parametrized_populations])

    @staticmethod
    def load_from_YAML(source: str | Path) -> ParametrizedDemographySexBiased:
        """
        Creates an instance of ParametrizedDemographySexBiased from a YAML file.
        """
        yaml = ruamel.yaml.YAML(typ="safe")
        if isinstance(source, (str, bytes, os.PathLike)):
            with open(source, 'r') as file:
                demes_data=yaml.load(file)
        else:
            # Assume it's a file-like object
            demes_data=yaml.load(source)
        
        assert isinstance(demes_data, dict), ".yaml file was invalid." 
        demography = ParametrizedDemographySexBiased('Unnamed Model')
        if 'model_name' in demes_data:
            demography.name = demes_data['model_name']

        for population in demes_data['demes']:
            if 'ancestors' in population:
                demography.parametrized_populations.append(population['name'])
                source_populations, remainder_population = ParametrizedDemographySexBiased.parse_proportions(
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
