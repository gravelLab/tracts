import os
import sys

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir+'\\..')
from tracts.demography.parametrized_demography_multipop import ParametrizedDemographyMultiPop

model = ParametrizedDemographyMultiPop.load_from_YAML('tests/pp_px_multipop.yaml')
print(model.founder_events)
print(model.events)
print(len(model.free_params))

for population,matrix in model.get_migration_matrices([0.8,5,0.3,7,0,3,0,3]).items():
    print(population)
    print(matrix)
