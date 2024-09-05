import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
# path to tracts, may have to adjust if the file is moved
# print(sys.path)
import tracts

# tracts.show_INFO(tracts.driver)
# tracts.show_INFO(tracts.ParametrizedDemography)

tracts.run_tracts('taino.yaml')
