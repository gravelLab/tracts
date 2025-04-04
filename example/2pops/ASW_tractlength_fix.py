import sys
from pathlib import Path

sys.path.append('.')

from tracts.driver import run_tracts

script_path = Path(sys.argv[0]).resolve()
script_directory = script_path.parent

one_pulse_driver_filename = "ASW_fix_one_pulse.yaml"
two_pulse_driver_filename = "ASW_fix_two_pulse.yaml"

run_tracts(driver_filename=one_pulse_driver_filename, script_dir=script_directory)
run_tracts(driver_filename=two_pulse_driver_filename, script_dir=script_directory)
