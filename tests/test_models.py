import time
import numpy as np
import sys
from tracts import ParametrizedDemography
import tracts
import pytest
import random


def test_founding_2pop(params):
    model = ParametrizedDemography()
    model.add_founder_event({'A': 'm1_A'}, 'B', 't0')
    model.finalize()
    m = model.get_migration_matrix([0.4, 4.5])
    print(m)
    m2 = tracts.legacy_models.models_2pop.pp([0.4,0.045])
    print(m2)
    return

def test_pulse():
    return

def test_migration():
    return
