from tracts.driver import run_tracts

from pathlib import Path
import sys

script_path = Path(sys.argv[0]).resolve()
script_directory = script_path.parent

driver_filename = "sex_biased_example.yaml"

run_tracts(driver_filename=driver_filename, script_dir=script_directory)
