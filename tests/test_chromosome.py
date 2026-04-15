import matplotlib.pyplot as plt
import numpy
import pytest
import scipy


from tracts import chromosome
from tracts.tract import Tract

"""
Tests for component methods of tracts core
"""



def test_smooth_unknown_leading():
    tracts = [
        Tract(0, 1, "UNKNOWN"),
        Tract(1, 2, "A"),
        Tract(2, 3, "B"),
    ]

    copy = chromosome.Chrom(tracts=tracts)
    copy.unknown_labels = ["UNKNOWN"]
    copy.smooth_unknown()

    assert len(copy.tracts) == 2
    assert copy.tracts[0].label == "A"
    assert copy.tracts[0].start == 1
    assert copy.tracts[0].end == 2


def test_smooth_unknown_trailing():
    tracts = [
        Tract(0, 1, "A"),
        Tract(1, 2, "B"),
        Tract(2, 3, "UNKNOWN"),
    ]

    copy = chromosome.Chrom(tracts=tracts)
    copy.unknown_labels = ["UNKNOWN"]
    copy.smooth_unknown()

    assert len(copy.tracts) == 2
    assert copy.tracts[-1].label == "B"
    assert copy.tracts[-1].end == 2


def test_smooth_unknown_multiple_internal():
    tracts = [
        Tract(0, 1, "A"),
        Tract(1, 2, "UNKNOWN"),
        Tract(2, 3, "UNKNOWN"),
        Tract(3, 4, "B"),
    ]

    copy = chromosome.Chrom(tracts=tracts)
    copy.unknown_labels = ["UNKNOWN"]
    copy.smooth_unknown()

    assert len(copy.tracts) == 2
    # midpoint between 1 and 3 is (3+1)/2 = 2
    assert copy.tracts[0].end == 2
    assert copy.tracts[1].start == 2


def test_smooth_unknown_all_unknown():
    tracts = [
        Tract(0, 1, "UNKNOWN"),
        Tract(1, 2, "UNKNOWN"),
        Tract(2, 3, "UNKNOWN"),
    ]

    copy = chromosome.Chrom(tracts=tracts)
    copy.unknown_labels = ["UNKNOWN"]
    copy.smooth_unknown()

    assert copy.tracts == []


def test_smooth_unknown_no_unknowns():
    tracts = [
        Tract(0, 1, "A"),
        Tract(1, 2, "B"),
    ]

    copy = chromosome.Chrom(tracts=tracts)
    copy.unknown_labels = ["UNKNOWN"]
    copy.smooth_unknown()

    assert len(copy.tracts) == 2
    assert copy.tracts[0].end == 1
    assert copy.tracts[1].start == 1


def test_smooth_unknown_three_segments():
    # A - UNKNOWN - C  → midpoint = 1.5
    tracts = [
        Tract(0, 1, "A"),
        Tract(1, 2, "UNKNOWN"),
        Tract(2, 3, "C"),
    ]

    copy = chromosome.Chrom(tracts=tracts)
    copy.unknown_labels = ["UNKNOWN"]
    copy.smooth_unknown()

    assert len(copy.tracts) == 2
    assert copy.tracts[0].label == "A"
    assert copy.tracts[1].label == "C"
    assert copy.tracts[0].end == 1.5
    assert copy.tracts[1].start == 1.5


def test_smooth_unknown_adjacent_knowns():
    # No UNKNOWN, no changes
    tracts = [
        Tract(0, 1, "A"),
        Tract(1, 3, "B"),
    ]

    copy = chromosome.Chrom(tracts=tracts)
    copy.unknown_labels = ["UNKNOWN"]
    copy.smooth_unknown()

    assert len(copy.tracts) == 2
    assert copy.tracts[0].end == 1
    assert copy.tracts[1].start == 1




