import sys
from pathlib import Path

sys.path.append('.')

from tracts.driver import run_tracts

script_path = Path(sys.argv[0]).resolve()
script_directory = script_path.parent

script_path = Path(sys.argv[0]).resolve()
script_directory = script_path.parent

script_path = "/home/jgonzale/Documents/PhaseType/tracts/example/3pops_sexbiased"

driver_filename = "ASW_tractlength_one_pulse.yaml"

run_tracts(driver_filename = driver_filename, script_dir = script_path)


