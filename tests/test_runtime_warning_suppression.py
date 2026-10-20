"""
Tests verifying that RuntimeWarnings from degenerate parameter values during
optimization are suppressed by ``np.errstate`` blocks, while genuine model
errors (negative / complex CDF) still raise exceptions and are NOT masked.

Background
----------
During likelihood optimisation with COBYLA / fsolve, the solver explores
parameter regions that produce degenerate migration matrices (e.g. a population
with zero founding proportion but non-empty states, or row-1 proportions that
sum to 1 with none going to a particular population). These lead to
indeterminate-form divisions such as ``0.0 / 0.0 = nan`` ("invalid value" in
NumPy) that would otherwise flood stderr with thousands of ``RuntimeWarning``
messages.

Key invariant
~~~~~~~~~~~~~
``np.errstate`` only suppresses the *printed warning*; arithmetic still
produces NaN / inf and the result propagates naturally.  The explicit
``raise Exception(...)`` calls in the CDF validation code are **completely
independent** of ``np.errstate`` and therefore still propagate correctly.

Test strategy
~~~~~~~~~~~~~
1. A baseline test confirms that the unprotected arithmetic (``0.0 / 0.0``)
   *does* emit a ``RuntimeWarning``, so the suppression tests are meaningful.
2. Suppression tests inject degenerate internal state into valid model objects
   (rather than constructing with degenerate matrices, which causes singular-
   matrix errors) and call the relevant public methods.
3. Exception tests monkeypatch ``PhT_CDF_windowed`` to return an obviously
   invalid CDF and assert the Exception propagates despite the errstate blocks.
"""

import warnings
from unittest.mock import patch
import numpy as np
import pytest
from tracts.phase_type.monoecious import PhTMonoecious
from tracts.phase_type.dioecious import PhTDioecious

# ------------------------- Shared constants -------------------------

# Chromosome length and evaluation grid used throughout.
_L = 1.5
_BINS = np.linspace(0.02, _L, 50)

# Simple 2-population migration matrices that construct without error.
_MG_VALID = np.array([[0.0, 0.0], [0.0, 0.0], [0.3, 0.7]])


# ------------------------- Helper -------------------------

def _runtime_warnings(caught):
    """Return only the RuntimeWarning entries from a catch_warnings list."""
    return [w for w in caught if issubclass(w.category, RuntimeWarning)]


# ------------------------- Baseline: show that the underlying arithmetic DOES warn without protection -------------------------

class TestBaseline:
    def test_zero_over_zero_emits_runtime_warning(self):
        """
        Confirm that ``np.float64(0.0) / np.float64(0.0)`` produces a
        RuntimeWarning under normal conditions.  This makes the subsequent
        suppression tests meaningful: if this test passes (warning IS emitted),
        we know the suppression tests are not vacuously true.
        """
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            np.float64(0.0) / np.float64(0.0)
        assert len(_runtime_warnings(w)) == 1, (
            "Expected exactly one RuntimeWarning from 0.0/0.0; "
            "got none — baseline arithmetic has changed."
        )


# ------------------------- Monoecious model -------------------------

class TestMonoeciousWarningSuppression:
    """
    Verify RuntimeWarning suppression in :class:`PhTMonoecious` and the
    underlying :class:`PhaseTypeDistribution.initialize_CDF_values`.

    Degenerate scenario reproduced
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    During optimisation the solver can produce a parameter vector where
    ``t0_proportions[pop] = 0`` (no ancestry from population *pop*) while
    ``alpha_list[pop]`` is all-zero (no flow from other populations into
    *pop*'s state space).  This causes::

        distribution_scaling_factor   →  -2 × 0 / dot([0,…], inv_S0) = 0/0
        tractlength_histogram_windowed →  scale = 2 × 0 × L / 0       = 0/0

    Both evaluate to NaN ("invalid value"), which is exactly the warning that
    should be suppressed.
    """

    @pytest.fixture
    def degenerate_mono(self):
        """Valid PhTMonoecious with pop-0 state injected as all-zero."""
        m = PhTMonoecious(_MG_VALID.copy())
        # Simulate a degenerate parameter point: pop 0 has zero ancestry and
        # zero alpha (all mass has drifted to pop 1 during optimisation).
        m.alpha_list[0][:] = 0.0
        m.t0_proportions[0] = 0.0
        return m

    def test_distribution_scaling_factor_no_warning(self, degenerate_mono):
        """
        ``distribution_scaling_factor`` computes ``-2 * t0 / dot(alpha, S0_inv)``.
        When both t0 and the dot product are zero the result is 0/0 = nan
        ("invalid value").  The surrounding ``np.errstate`` must suppress this.
        """
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = degenerate_mono.distribution_scaling_factor(population_number=0)
        assert _runtime_warnings(w) == [], (
            "RuntimeWarning emitted from distribution_scaling_factor despite "
            "np.errstate; the errstate block at that site may be missing or broken."
        )

    def test_tractlength_histogram_windowed_scale_no_warning(self, degenerate_mono):
        """
        ``tractlength_histogram_windowed`` computes ``scale = 2 * t0 * L / ETL``.
        With zero alpha, ETL = 0 and t0 = 0, giving 0/0 = nan.
        The surrounding ``np.errstate`` must suppress this.
        """
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            degenerate_mono.tractlength_histogram_windowed(
                population_number=0, bins=_BINS, L=_L
            )
        assert _runtime_warnings(w) == [], (
            "RuntimeWarning emitted from tractlength_histogram_windowed scale "
            "computation despite np.errstate."
        )

    def test_negative_cdf_raises_exception_not_suppressed(self):
        """
        When ``PhT_CDF_windowed`` returns a CDF with negative values (a genuine
        numerical instability), ``tractlength_histogram_windowed`` must raise an
        ``Exception``.  The ``np.errstate`` blocks must NOT suppress this.
        """
        mono = PhTMonoecious(_MG_VALID.copy())
        neg_cdf = np.full(len(_BINS), -0.5)

        def _bad_cdf(*args, **kwargs):
            return neg_cdf, 1.0, _L, 0.5

        with patch.object(mono, "PhT_CDF_windowed", _bad_cdf):
            with pytest.raises(Exception, match="CDF not positive and real"):
                mono.tractlength_histogram_windowed(
                    population_number=0, bins=_BINS, L=_L
                )


# ------------------------- Dioecious model -------------------------

class TestDioeciousWarningSuppression:
    """
    Verify RuntimeWarning suppression in :class:`PhTDioecious`.

    Two degenerate scenarios are exercised:

    1. **Zero ETL scale** – same as the monoecious case but for male/female
       CDFs separately (``scale_m = t0_m * L / ETL_m``).

    2. **Zero normalisation denominator** – when ``f_prop_at_1[pop] = 0`` and
       ``sum(f_prop_at_1) = 1``, the denominator ``norm_f_1 = 0``, giving
       ``0/0 = nan`` inside ``submodel_probabilities``.  This returns NaN
       probabilities, which the optimizer treats as an infeasible point rather
       than causing a crash.
    """

    @pytest.fixture
    def degenerate_dio(self):
        """Valid PhTDioecious with pop-0 state injected as all-zero."""
        mg = _MG_VALID.copy()
        d = PhTDioecious(mg, mg, rho_f=1.0, rho_m=1.0)
        # Inject degenerate parameters for pop 0.
        for alpha in (d.alpha_list_f[0], d.alpha_list_m[0]):
            alpha[:] = 0.0
        d.t0_proportions_f[0] = 0.0
        d.t0_proportions_m[0] = 0.0
        return d

    def test_histogram_scale_degenerate_no_warning(self, degenerate_dio):
        """
        ``tractlength_histogram_windowed`` computes male and female scales
        ``t0 * L / ETL``.  With zero alpha, ETL = 0 and t0 = 0, giving 0/0.
        The surrounding ``np.errstate`` must suppress both male and female
        RuntimeWarnings.
        """
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            degenerate_dio.tractlength_histogram_windowed(
                population_number=0, bins=_BINS, L=_L
            )
        assert _runtime_warnings(w) == [], (
            "RuntimeWarning emitted from dioecious tractlength_histogram_windowed "
            "scale computation despite np.errstate."
        )

    def test_zero_norm_submodel_probabilities_no_warning(self):
        """
        ``submodel_probabilities`` computes normalised isolation / connection
        probabilities via ``prob / norm``.  When row-1 female migration sums to
        1 with none allocated to population 0, ``norm_f_1 = 0`` for that
        population, giving ``0/0 = nan``.  The returned NaN propagates to the
        likelihood (where the optimizer discards the point) but must not emit a
        RuntimeWarning.
        """
        mg = _MG_VALID.copy()
        dio = PhTDioecious(mg, mg, rho_f=1.0, rho_m=1.0)

        # Inject degenerate row-1 proportions: pop 1 gets everything, pop 0 gets
        # nothing.  sum = 1.0  →  prob_ad_f_1 = 0  →  norm_f_1 = 0.
        dio.f_prop_at_1 = np.array([0.0, 1.0])
        dio.m_prop_at_1 = np.array([0.0, 1.0])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            prop_iso, prop_conn = dio.submodel_probabilities(
                population_number=0, s1=1
            )

        # The result is NaN (expected degenerate output).
        assert np.isnan(prop_iso) and np.isnan(prop_conn), (
            "Expected NaN probabilities for zero-norm case."
        )
        # No RuntimeWarning should have been emitted.
        assert _runtime_warnings(w) == [], (
            "RuntimeWarning emitted from submodel_probabilities despite "
            "np.errstate protecting the norm_* division."
        )

    def test_negative_male_cdf_raises_exception_not_suppressed(self):
        """
        A genuinely negative male CDF (real numerical error) must raise an
        Exception from ``tractlength_histogram_windowed``.  ``np.errstate``
        must NOT mask it.
        """
        mg = _MG_VALID.copy()
        dio = PhTDioecious(mg, mg, rho_f=1.0, rho_m=1.0)
        neg_cdf = np.full(len(_BINS), -0.5)

        def _bad_cdf(*args, **kwargs):
            return neg_cdf, 1.0, _L, 0.5

        with patch.object(dio, "PhT_CDF_windowed", _bad_cdf):
            with pytest.raises(Exception, match="type-m CDF is not positive and real"):
                dio.tractlength_histogram_windowed(
                    population_number=0, bins=_BINS, L=_L
                )
