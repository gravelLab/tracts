"""
Bundles the population and mapped tract-length histogram data used to evaluate
a demographic model's likelihood, to avoid threading autosome/allosome bins,
mapped data arrays, and sample counts as separate parameters through
:mod:`tracts.core`'s optimization functions.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from tracts.population import Population
from tracts.demography.parametrized_demography_sex_biased import SexType

@dataclass
class TractsData:
    """
    Observed tract-length histogram data for a population, mapped to a demographic
    model's population ordering, for autosomes and (optionally) allosomes.

    Attributes
    ----------
    population: Population
        The population providing chromosome lengths and sample sizes.
    autosome_bins: np.ndarray
        Bin edges for the autosomal tract-length histogram.
    autosome_data_mapped: list of list
        Observed autosomal tract counts per population, indexed to match the
        demographic model's population ordering.
    allosome_bins: np.ndarray | None
        Bin edges for the allosomal tract-length histogram. None if allosomal
        data is not available.
    allosome_length: float | None
        Length of the X chromosome in Morgans. None if allosomal data is not
        available.
    female_data_mapped: list of list | None
        Observed X-chromosome tract counts for females per population, indexed
        to match the demographic model's population ordering. None if allosomal
        data is not available.
    male_data_mapped: list of list | None
        Observed X-chromosome tract counts for males per population, indexed to
        match the demographic model's population ordering. None if allosomal
        data is not available.
    num_females: int | None
        Number of female samples. None if allosomal data is not available.
    num_males: int | None
        Number of male samples. None if allosomal data is not available.
    """
    population: Population
    autosome_bins: np.ndarray
    autosome_data_mapped: list
    allosome_bins: np.ndarray | None = None
    allosome_length: float | None = None
    female_data_mapped: list | None = None
    male_data_mapped: list | None = None
    num_females: int | None = None
    num_males: int | None = None

    @property
    def has_allosome_data(self) -> bool:
        """
        Whether allosomal tract-length data is available (``allosome_bins`` is not None).
        """
        return self.allosome_bins is not None

    @classmethod
    def from_population(
        cls,
        population: Population,
        p_dict: dict,
        npts: int,
        exclude_tracts_below_cM: float = 0,
        include_allosomes: bool = True,
    ) -> "TractsData":
        """
        Extracts and maps tract-length histograms from a population, for autosomes
        and, if requested, allosomes.

        Parameters
        ----------
        population: Population
            The population to extract tract-length histograms from.
        p_dict: dict
            A dictionary mapping population labels to their corresponding indices
            in the demographic model.
        npts: int
            Number of bins for the tract-length histograms.
        exclude_tracts_below_cM: float, default 0
            Minimum tract length in centimorgans to include in the histograms.
        include_allosomes: bool, default True
            Whether to also extract and map allosomal (X-chromosome) tract-length
            histograms. If False, ``allosome_bins`` and the other allosome-related
            attributes are left as None.

        Returns
        -------
        TractsData
            The extracted and mapped tract-length histogram data.
        """
        p_dict = dict(p_dict)

        autosome_bins, autosome_data = population.get_global_tractlengths(npts=npts,
                                                                        exclude_tracts_below_cM=exclude_tracts_below_cM)
        n_autosome_bins = len(autosome_bins)
        autosome_data_mapped = [np.zeros(n_autosome_bins, dtype='int64').tolist() for _ in p_dict]
        for k, v in autosome_data.items():
            autosome_data_mapped[p_dict[k]] = v

        if not include_allosomes:
            return cls(
                population=population,
                autosome_bins=autosome_bins,
                autosome_data_mapped=autosome_data_mapped,
            )

        allosome_bins, allosome_data = population.get_global_allosome_tractlengths(allosome='X',
                                                                                npts=npts,
                                                                                exclude_tracts_below_cM=exclude_tracts_below_cM)
        
        n_allosome_bins = len(allosome_bins)
        female_data = allosome_data[SexType.FEMALE]
        male_data = allosome_data[SexType.MALE]

        female_data_mapped = [np.zeros(n_allosome_bins, dtype='int64').tolist() for _ in p_dict]
        for k, v in female_data.items():
            female_data_mapped[p_dict[k]] = v

        male_data_mapped = [np.zeros(n_allosome_bins, dtype='int64').tolist() for _ in p_dict]
        for k, v in male_data.items():
            male_data_mapped[p_dict[k]] = v

        return cls(
            population=population,
            autosome_bins=autosome_bins,
            autosome_data_mapped=autosome_data_mapped,
            allosome_bins=allosome_bins,
            allosome_length=population.allosome_lengths['X'],
            female_data_mapped=female_data_mapped,
            male_data_mapped=male_data_mapped,
            num_females=population.num_females,
            num_males=population.num_males,
        )
