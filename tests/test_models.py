# import sys
# from pathlib import Path
# sys.path.append(str(Path(__file__).parent.parent))
from tracts import ParametrizedDemography
# from tracts.legacy_models.models_2pop import pp
import tracts
import numpy


def test_founding_2pop():
    model = ParametrizedDemography()
    model.add_founder_event({'A': 'm1_A'}, 'B', 't0')
    model.finalize()
    m = model.get_migration_matrix([0.4, 4.5])
    m2 = tracts.legacy_models.models_2pop.pp([0.4, 0.045])
    # TODO: The m2 array has 7 elements while m has 6 which causes this test to fail
    assert numpy.allclose(m, m2)


def test_pulse():
    return


def test_migration():
    return
