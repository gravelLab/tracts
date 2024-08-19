import sys
from pathlib import Path
sys.path.insert(1, str(Path(__file__).parent.parent))

from tracts.core import *
from tracts.parametrized_demography import ParametrizedDemography, FixedAncestryDemography
from tracts import legacy_models
from tracts import driver
from tracts import logs
from tracts import legacy
'''
from core import *
from parametrized_demography import ParametrizedDemography, FixedAncestryDemography
import legacy_models
import driver
import logs
'''


#from tracts.driver import run_tracts

run_tracts = driver.run_tracts
show_INFO = logs.show_INFO