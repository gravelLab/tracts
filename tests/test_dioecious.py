import os
import sys

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir+'\\..')
from tracts.demography.parametrized_demography_sex_biased import ParametrizedDemographySexBiased

model = ParametrizedDemographySexBiased.load_from_YAML('tests/pp_px.yaml')
print('Founder events:', model.founder_events)
print('Events:', model.events)
print('Free parameters:' ,model.free_params)
print('Dependent parameters:', model.dependent_params)

#print(model.get_migration_matrices([5, 0.5, 0.3, 3, 0.5, 0.7]))