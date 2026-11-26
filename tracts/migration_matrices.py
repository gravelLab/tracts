"""
Computations on migration matrices that do not depend on the demographic parametrization or on the tract
length model: used both by :mod:`tracts.demography` (predicted ancestry proportions, ancestry fixing) and by
:mod:`tracts.phase_type` (tract length distributions).
"""
from __future__ import annotations

import numpy as np


def get_survival_factors(migration_matrix):
    r"""
    Takes a migration matrix of :math:`T` generations and returns a list of length :math:`T`,
    which is the probability of a migrant allele from that generation surviving to the present.
    Valid only under the monoecious model, that is, assuming unbiased migration and recombination rates for autosomal admixture.

    Parameters
    ----------
    migration_matrix: npt.ArrayLike
        A :math:`T\times P` array, where :math:`T` is the number of generations, and :math:`P` is the number of populations.
    
    Returns
    -------
    npt.ArrayLike
        A vector of length :math:`T`, where the entry at index :math:`t` is the probability of a migrant allele from generation :math:`t` surviving to the present.
    """
    survival_factors = np.ones(len(migration_matrix))
    for generation_number in range(1, len(migration_matrix)):
        survival_factors[generation_number] = survival_factors[generation_number - 1] * (
                1 - sum(migration_matrix[generation_number - 1]))
    return survival_factors


def ancestry_proportions_by_generation(mig_matrix_f: np.ndarray, mig_matrix_m: np.ndarray, X_chr: bool):
    r"""
    Computes, for every generation :math:`g`, the expected ancestry proportions carried by a female and by a male
    individual at generation :math:`g` in the past (Article equation labels: predicted_autosome_proportions, predicted_allosome_proportions).

    Parameters
    ----------
    mig_matrix_f : np.ndarray
        The female-specific migration matrix, as described in :func:`~tracts.phase_type.hybrid_pedigree.hybrid_pedigree_distribution`.
    mig_matrix_m : np.ndarray
        The male-specific migration matrix.
    X_chr : bool
        Whether the proportions are computed along the X chromosome (a male inherits his X chromosome from his mother)
        or along autosomes (a copy is inherited from either parent with equal probabilities).

    Returns
    -------
    np.ndarray
        An array of the shape of `mig_matrix_f`, whose row :math:`g` contains the ancestry proportions of a female individual at generation :math:`g`.
        Row 0 is not used.
    np.ndarray
        The counterpart for male individuals.
    """
    T = int(np.shape(mig_matrix_f)[0] - 1)
    proportions_f = np.zeros(np.shape(mig_matrix_f))
    proportions_m = np.zeros(np.shape(mig_matrix_m))
    proportions_f[T, ] = mig_matrix_f[T, ]
    proportions_m[T, ] = mig_matrix_m[T, ]
    for generation_number in range(T - 1, 0, -1):
        proportions_parents = (proportions_f[generation_number + 1, ] + proportions_m[generation_number + 1, ]) / 2
        proportions_mother = proportions_f[generation_number + 1, ]
        proportions_f[generation_number, ] = mig_matrix_f[generation_number, ] + (1 - mig_matrix_f[generation_number, ].sum())*proportions_parents
        proportions_m[generation_number, ] = mig_matrix_m[generation_number, ] + (1 - mig_matrix_m[generation_number, ].sum())*(
            proportions_mother if X_chr else proportions_parents)
    return proportions_f, proportions_m
