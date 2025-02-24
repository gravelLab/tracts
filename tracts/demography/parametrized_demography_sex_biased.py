from __future__ import annotations

import math
from pathlib import Path

import numpy
import ruamel.yaml

from tracts.demography.base_parametrized_demography import BaseParametrizedDemography, BaseMigrationEvent, FounderEvent
from tracts.demography.parametrized_demography import PulseEvent, ContinuousEvent
from tracts.demography.parametrized_demography_multipop import ParametrizedDemographyMultiPop
from tracts.demography.param_types import ParamType

def male(param):
    return f'{param}_male'
    
def female(param):
    return f'{param}_female'

variants=[male, female]

class ParametrizedDemographySexBiased(ParametrizedDemographyMultiPop):
    """
    A class representing a demographic history with varying rates of male and female migration.
    Constructs a separate migration matrix for male and female individuals.
    Each entry represent the proportion of that sex that is replaced during that migration.
    The matrices are implemented as two subpopulations whose migrations have independent rate parameters but linked time parameters.
    """

    def __init__(self, name: str = "", min_time=2, max_time=numpy.inf):
        super().__init__(name=name, min_time=min_time, max_time=max_time)

    def add_pulse_migration(self, dest_population, source_population, rate_param, time_param):
        """
        Adds a pulse migration from source population A, parametrized by time and rate
        """
        for variant in variants:
            super().add_pulse_migration(variant(dest_population), source_population, variant(rate_param), time_param)


    def add_continuous_migration(self, dest_population, source_population, rate_param, start_param, end_param):
        """
        Adds a continuous migration from source population A, parametrized by start_time, end_time, and magnitude
        """
        for variant in variants:
            super().add_continuous_migration(variant(dest_population), source_population, variant(rate_param), start_param, end_param)

    def add_founder_event(self, dest_population, source_populations: dict[str, str], remainder_population: str, found_time: str) -> None:
        """
        Adds a founder event. A parametrized demography must have exactly one founder event.
        source_populations is a dict where each key is a population
        and each value is the name of the parameter defining the migration ratio of each population
        remainder_population is the source of the remaining migrants, such that the total migration ratio adds up to 1.
        found_time is the name of the parameter defining the time of migration.
        """
        for variant in variants:
            super().add_founder_event(variant(dest_population), {population: variant(rate) for population, rate in source_populations.items()},remainder_population, found_time)

    @staticmethod
    def load_from_YAML(filepath: str | Path) -> ParametrizedDemographySexBiased:
        """
        Creates an instance of ParametrizedDemographySexBiased from a YAML file
        """
        demography = ParametrizedDemographySexBiased('Unnamed Model')
        parametrized_populations=set()
        with open(filepath) as file, ruamel.yaml.YAML(typ="safe") as yaml:
            demes_data = yaml.load(file)
            assert isinstance(demes_data, dict), ".yaml file was invalid."
            if 'model_name' in demes_data:
                demography.name = demes_data['model_name']

            for population in demes_data['demes']:
                if 'ancestors' in population:
                    parametrized_populations.add(population['name'])
                    source_populations, remainder_population = ParametrizedDemographySexBiased.parse_proportions(
                        population['ancestors'], population['proportions'])
                    demography.add_founder_event(population['name'], source_populations, remainder_population, population['start_time'])
            if 'pulses' in demes_data:
                for pulse in demes_data['pulses']:
                    print(pulse)
                    if 'dest' in pulse and pulse['dest'] in parametrized_populations:
                        for source, proportion in zip(pulse['sources'], pulse['proportions']):
                            demography.add_pulse_migration(pulse['dest'], source, proportion, pulse['time'])
            if 'migrations' in demes_data:
                for migration in demes_data['migrations']:
                    if 'dest' in migration and migration['dest'] in parametrized_populations:
                        demography.add_continuous_migration(migration['dest'], migration['source'], migration['rate'],
                                                            migration['start_time'], migration['end_time'])
            demography.finalize()
        return demography

