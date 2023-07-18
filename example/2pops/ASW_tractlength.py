import sys
from pathlib import Path
import numpy
sys.path.append(str(Path(__file__).parent.parent.parent))
#path to tracts, may have to adjust if the file is moved
import tracts
import logging

logging.basicConfig()
tracts.driver.logger.setLevel(logging.INFO)
tracts.run_tracts('ASW_tractlength_one_pulse.yaml')
tracts.run_tracts('ASW_tractlength_two_pulse.yaml')
