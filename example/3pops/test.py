import sys
from pathlib import Path
import numpy
sys.path.append(str(Path(__file__).parent.parent.parent))
#path to tracts, may have to adjust if the file is moved
#print(sys.path)
import tracts
import logging


logging.basicConfig()
tracts.ParametrizedDemography.logger.setLevel(logging.INFO)

tracts.run_tracts('taino.yaml')
