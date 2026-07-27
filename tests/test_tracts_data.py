"""
Tests for tracts/tracts_data.py: TractsData.
"""

import numpy as np
from unittest.mock import MagicMock

from tracts.tracts_data import TractsData
from tracts.demography.parametrized_demography_sex_biased import SexType

N_BINS = 5
P_DICT = {"A": 0, "B": 1}


def _make_population():
    pop = MagicMock()
    bins = np.linspace(0, 1, N_BINS + 1)
    autosome_counts = {"A": [1, 2, 3, 4, 5], "B": [6, 7, 8, 9, 10]}
    female_counts = {"A": [1, 1, 1, 1, 1], "B": [2, 2, 2, 2, 2]}
    male_counts = {"A": [3, 3, 3, 3, 3], "B": [4, 4, 4, 4, 4]}

    pop.get_global_tractlengths.return_value = (bins, autosome_counts)
    pop.get_global_allosome_tractlengths.return_value = (
        bins,
        {SexType.FEMALE: female_counts, SexType.MALE: male_counts},
    )
    pop.num_males = 7
    pop.num_females = 9
    pop.allosome_lengths = {"X": 1.5}
    return pop


class TestTractsDataHasAllosomeData:

    def test_true_when_allosome_bins_present(self):
        data = TractsData(population=MagicMock(), autosome_bins=np.array([0, 1]), autosome_data_mapped=[[0]],
                           allosome_bins=np.array([0, 1]))
        assert data.has_allosome_data is True

    def test_false_when_allosome_bins_absent(self):
        data = TractsData(population=MagicMock(), autosome_bins=np.array([0, 1]), autosome_data_mapped=[[0]])
        assert data.has_allosome_data is False


class TestTractsDataFromPopulation:

    def test_autosome_data_mapped_indexed_by_p_dict(self):
        pop = _make_population()
        data = TractsData.from_population(pop, p_dict=P_DICT, npts=N_BINS, include_allosomes=False)

        assert data.autosome_data_mapped[P_DICT["A"]] == [1, 2, 3, 4, 5]
        assert data.autosome_data_mapped[P_DICT["B"]] == [6, 7, 8, 9, 10]

    def test_excludes_allosome_fields_when_not_requested(self):
        pop = _make_population()
        data = TractsData.from_population(pop, p_dict=P_DICT, npts=N_BINS, include_allosomes=False)

        assert data.has_allosome_data is False
        assert data.allosome_length is None
        assert data.female_data_mapped is None
        assert data.male_data_mapped is None
        assert data.num_females is None
        assert data.num_males is None
        pop.get_global_allosome_tractlengths.assert_not_called()

    def test_includes_allosome_fields_when_requested(self):
        pop = _make_population()
        data = TractsData.from_population(pop, p_dict=P_DICT, npts=N_BINS, include_allosomes=True)

        assert data.has_allosome_data is True
        assert data.allosome_length == 1.5
        assert data.female_data_mapped[P_DICT["A"]] == [1, 1, 1, 1, 1]
        assert data.female_data_mapped[P_DICT["B"]] == [2, 2, 2, 2, 2]
        assert data.male_data_mapped[P_DICT["A"]] == [3, 3, 3, 3, 3]
        assert data.male_data_mapped[P_DICT["B"]] == [4, 4, 4, 4, 4]
        assert data.num_females == 9
        assert data.num_males == 7

    def test_forwards_npts_and_exclude_tracts_below_cm(self):
        pop = _make_population()
        TractsData.from_population(pop, p_dict=P_DICT, npts=42, exclude_tracts_below_cM=3.5, include_allosomes=False)

        pop.get_global_tractlengths.assert_called_once_with(npts=42, exclude_tracts_below_cM=3.5)

    def test_population_reference_preserved(self):
        pop = _make_population()
        data = TractsData.from_population(pop, p_dict=P_DICT, npts=N_BINS, include_allosomes=False)
        assert data.population is pop

    def test_p_dict_not_mutated(self):
        pop = _make_population()
        p_dict = dict(P_DICT)
        TractsData.from_population(pop, p_dict=p_dict, npts=N_BINS, include_allosomes=False)
        assert p_dict == P_DICT
