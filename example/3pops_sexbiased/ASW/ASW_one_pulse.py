"""
ASW inference - One pulse model
===============================

This example implements inference for the ASW population under a one pulse model of admixture, using the tracts package.
Inference is performed using autosomal and X chromosome data, allowing for the specification of sex-biased admixture. 

"""

import sys
from pathlib import Path

sys.path.append('.')

from tracts.driver import run_tracts

script_path = Path(sys.argv[0]).resolve()

driver_filename = "ASW_one_pulse.yaml"

run_tracts(driver_filename = driver_filename, script_dir = script_path)


