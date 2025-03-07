import os
import sys

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir+'\\..')
from tracts.demography.parametrized_demography_multipop import ParametrizedDemographyMultiPop

model = ParametrizedDemographyMultiPop.load_from_YAML('tests/pp_px.yaml')
print(model.founder_events)
print(model.events)
print(len(model.free_params))
