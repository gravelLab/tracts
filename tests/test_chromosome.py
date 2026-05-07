from tracts import chromosome
from tracts.tract import Tract

"""
This test suite checks the functionality of the `smooth_unknown` method in the `Chrom` class, which is responsible for smoothing out "UNKNOWN" labels in a chromosome's tracts.
The tests cover various scenarios, including leading and trailing unknowns, multiple internal unknowns, cases where all tracts are unknown, and cases with no unknowns. 
Each test verifies that the resulting tracts after smoothing match the expected outcomes based on the input tracts and the defined unknown labels.
"""

def test_smooth_unknown_leading():
    """
    Checks that leading "UNKNOWN" tracts are smoothed correctly by assigning them the label of the first known tract and adjusting the start position accordingly.
    """
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
    """
    Checks that trailing "UNKNOWN" tracts are smoothed correctly by assigning them the label of the last known tract and adjusting the end position accordingly.
    """
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
    """
    Checks that multiple internal "UNKNOWN" tracts are smoothed correctly by assigning them the label of the nearest known tract and adjusting the positions accordingly.
    """
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
    # Midpoint between 1 and 3 is (3+1)/2 = 2
    assert copy.tracts[0].end == 2
    assert copy.tracts[1].start == 2


def test_smooth_unknown_all_unknown():
    """
    Checks that if all tracts are labeled as "UNKNOWN", the smoothing process results in an empty list of tracts, as there are no known labels to assign.
    """
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
    """
    Checks that if there are no "UNKNOWN" tracts, the smoothing process has no effect.
    """
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
    """
    Checks that if there is a single "UNKNOWN" tract between two known tracts, the smoothing process correctly assigns the "UNKNOWN" tract to the nearest known label and adjusts the positions accordingly.
    In this case, the "UNKNOWN" tract is between "A" and "C", so it should be split at the midpoint and assigned to "A" and "C" respectively.
    """
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
    """
    Checks that if there are adjacent known tracts with no "UNKNOWN" tracts in between, the smoothing process does not alter the tracts.
    """
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




