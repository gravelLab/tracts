import sys
from pathlib import Path
import numpy
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent.parent))
#path to tracts, may have to adjust if the file is moved
#print(sys.path)
import tracts

tracts.run_tracts('ASW_tractlength_no_pulse.yaml')
tracts.run_tracts('ASW_tractlength_one_pulse.yaml')
