"""
Bundles a demographic model together with the admixture and phase-type model
configuration used to evaluate its likelihood.

This exists to avoid threading a long, repeated list of related parameters
(``ad_model_autosomes``, ``ad_model_allosomes``, ``rho_f``, ``rho_m``, ``TP``,
``N_cores``, plus the demographic model itself) through :mod:`tracts.core`'s
optimization functions (``compute_objective``,
:func:`~tracts.core.optimize_cob_sex_biased_single_step`,
:func:`~tracts.core.optimize_cob_sex_biased_two_steps`) and their callers in
:mod:`tracts.driver`.
"""
from __future__ import annotations
import copy
import warnings
from dataclasses import dataclass
import numpy as np
from tracts.demography.parametrized_demography import ParametrizedDemography
from tracts.demography.parametrized_demography_sex_biased import ParametrizedDemographySexBiased
from tracts.phase_type import hybrid_pedigree as HP
from tracts.phase_type import PhTMonoecious, PhTDioecious
from tracts.phase_type.base_phase_type import _GenerationZeroContributionWarning
from tracts.tracts_data import TractsData
from tracts.likelihood_options import LikelihoodOptions

_VALID_AUTOSOME_MODELS = ('DC', 'DF', 'M', 'H-DC', 'H-DF')
_VALID_ALLOSOME_MODELS = ('DC', 'DF', 'H-DC', 'H-DF')


@dataclass
class LoglikBreakdown:
    """
    Log-likelihood contributions from each genome component included in a
    :meth:`GeneticModel.loglik` evaluation. Fields are None for components that
    were not requested (``include_autosomes``/``include_allosomes`` False).

    Attributes
    ----------
    autosomes: float | None
        The autosomal log-likelihood, or None if not computed.
    female_allosomes: float | None
        The female allosomal (X-chromosome) log-likelihood, or None if not computed.
    male_allosomes: float | None
        The male allosomal (X-chromosome) log-likelihood, or None if not computed.
    """
    autosomes: float | None = None
    female_allosomes: float | None = None
    male_allosomes: float | None = None

    @property
    def total(self) -> float:
        """
        The sum of all computed (non-None) components.
        """
        return sum(v for v in (self.autosomes, self.female_allosomes, self.male_allosomes) if v is not None)


@dataclass
class PhaseTypeModelConfig:
    """
    Configuration for the admixture and phase-type model used to compute the
    likelihood of a demographic model's migration matrices.

    Parameters
    ----------
    ad_model_autosomes: str
        The admixture model used for autosomes. Must be one of ``'DC'``
        (Dioecious-Coarse), ``'DF'`` (Dioecious-Fine), ``'M'`` (Monoecious),
        ``'H-DC'`` or ``'H-DF'`` (the hybrid-pedigree refinements of DC and DF,
        respectively). Defaults to ``'DC'``.
    ad_model_allosomes: str | None
        The admixture model used for allosomes. Must be one of ``'DC'``,
        ``'DF'``, ``'H-DC'``, or ``'H-DF'``. If None, allosomal admixture is not
        modelled. Defaults to ``'DC'``.
    rho_f: float
        The female-specific recombination rate. Defaults to 1.
    rho_m: float
        The male-specific recombination rate. Defaults to 1.
    TP: int
        The number of pedigree generations under the hybrid-pedigree
        refinements of the Dioecious models (``'H-DC'``, ``'H-DF'``). Ignored
        otherwise. Defaults to 2.
    N_cores: int
        The number of CPU cores to use for parallel processing under the
        hybrid-pedigree refinements. Ignored otherwise. Defaults to 1.
    """
    ad_model_autosomes: str = 'DC'
    ad_model_allosomes: str | None = 'DC'
    rho_f: float = 1
    rho_m: float = 1
    TP: int = 2
    N_cores: int = 1

    def __post_init__(self):
        if self.ad_model_autosomes not in _VALID_AUTOSOME_MODELS:
            raise ValueError(
                f"ad_model_autosomes must be one of {_VALID_AUTOSOME_MODELS}, got {self.ad_model_autosomes!r}."
            )
        if self.ad_model_allosomes is not None and self.ad_model_allosomes not in _VALID_ALLOSOME_MODELS:
            raise ValueError(
                f"ad_model_allosomes must be one of {_VALID_ALLOSOME_MODELS} or None, got {self.ad_model_allosomes!r}."
            )
        if self.N_cores < 1:
            raise ValueError(f"N_cores must be at least 1, got {self.N_cores}.")
        if self.TP < 1:
            raise ValueError(f"TP must be at least 1, got {self.TP}.")

    @property
    def models_allosomes(self) -> bool:
        """
        Whether allosomal admixture is modelled (``ad_model_allosomes`` is not None).
        """
        return self.ad_model_allosomes is not None

    @property
    def uses_hybrid_pedigree_autosomes(self) -> bool:
        """
        Whether the autosomal admixture model is a hybrid-pedigree refinement.
        """
        return self.ad_model_autosomes in ('H-DC', 'H-DF')

    @property
    def uses_hybrid_pedigree_allosomes(self) -> bool:
        """
        Whether the allosomal admixture model is a hybrid-pedigree refinement.
        """
        return self.ad_model_allosomes in ('H-DC', 'H-DF')


class GeneticModel:
    """
    Bundles a demographic model with the admixture and phase-type model
    configuration used to evaluate its likelihood against tract-length data.

    Attributes
    ----------
    demographic_model: ParametrizedDemography | ParametrizedDemographySexBiased
        The demographic model whose parameters are being fit.
    phase_type_config: PhaseTypeModelConfig
        The admixture and phase-type model configuration used to compute the
        likelihood of ``demographic_model``'s migration matrices.
    """

    def __init__(
        self,
        demographic_model: ParametrizedDemography | ParametrizedDemographySexBiased,
        phase_type_config: PhaseTypeModelConfig | None = None,
        **phase_type_kwargs,
    ):
        """
        Parameters
        ----------
        demographic_model: ParametrizedDemography | ParametrizedDemographySexBiased
            The demographic model whose parameters are being fit.
        phase_type_config: PhaseTypeModelConfig | None
            The admixture and phase-type model configuration. If None, a new
            :class:`PhaseTypeModelConfig` is constructed from ``phase_type_kwargs``.
        **phase_type_kwargs
            Keyword arguments forwarded to :class:`PhaseTypeModelConfig` when
            ``phase_type_config`` is None (e.g. ``ad_model_autosomes='DC'``).
            Ignored if ``phase_type_config`` is provided.
        """
        if not isinstance(demographic_model, (ParametrizedDemography, ParametrizedDemographySexBiased)):
            raise TypeError(
                "demographic_model must be a ParametrizedDemography or ParametrizedDemographySexBiased "
                f"instance, got {type(demographic_model).__name__}."
            )
        if phase_type_config is not None and phase_type_kwargs:
            raise ValueError("Provide either phase_type_config or phase_type_kwargs, not both.")

        self.demographic_model = demographic_model
        self.phase_type_config = (
            phase_type_config if phase_type_config is not None else PhaseTypeModelConfig(**phase_type_kwargs)
        )

    # ------------------ Convenience passthroughs to the wrapped demographic model ------------------

    @property
    def is_sex_biased(self) -> bool:
        """
        Whether ``demographic_model`` is a :class:`ParametrizedDemographySexBiased` instance.
        """
        return isinstance(self.demographic_model, ParametrizedDemographySexBiased)

    @property
    def parameter_handler(self):
        """
        The demographic model's ``FixedParametersHandler`` (see ``BaseParametrizedDemography.parameter_handler``).
        """
        return self.demographic_model.parameter_handler

    @property
    def model_base_params(self):
        """
        The demographic model's free base parameters (see ``BaseParametrizedDemography.model_base_params``).
        """
        return self.demographic_model.model_base_params

    @property
    def population_indices(self):
        """
        The demographic model's population-name-to-index mapping.
        """
        return self.demographic_model.population_indices

    def get_migration_matrices(self, params):
        """
        Computes the migration matrices for ``params`` via ``demographic_model.get_migration_matrices``.
        """
        return self.demographic_model.get_migration_matrices(params)

    def set_up_fixed_parameters(self, params_to_fix_by_ancestry: list | None = None,
                                proportions: dict | None = None,
                                user_params_to_fix_by_value: dict | None = None) -> None:
        """
        Sets up fixed parameters (by ancestry proportions and/or by user-provided value) on
        this instance's ``demographic_model`` (see ``BaseParametrizedDemography.set_up_fixed_parameters``).

        Provided as a GeneticModel method — rather than requiring callers to reach into
        ``genetic_model.demographic_model`` themselves — so that fixing parameters through a
        GeneticModel always mutates the exact demographic_model that GeneticModel's own
        ``parameter_handler``/``model_func``/``outofbounds_fun``/``loglik`` read from, and the
        change is guaranteed visible on any later use of this same GeneticModel instance.

        Parameters
        ----------
        params_to_fix_by_ancestry: list[str] | None
            Names of parameters to fix from ``proportions``. Defaults to none.
        proportions: dict[str, list[float]] | None
            Ancestry proportions used to fix ``params_to_fix_by_ancestry``. Defaults to none.
        user_params_to_fix_by_value: dict[str, float] | None
            A dict mapping parameter names to the values they should be fixed at. Defaults to none.
        """
        self.demographic_model.set_up_fixed_parameters(
            params_to_fix_by_ancestry=params_to_fix_by_ancestry or [],
            proportions=proportions or {},
            user_params_to_fix_by_value=user_params_to_fix_by_value or {},
        )

    def model_func(self, params):
        """
        Converts optimizer-space ``params`` to physical parameters via ``demographic_model.parameter_handler``
        and returns the resulting migration matrices. Equivalent to the function previously returned by
        :func:`~tracts.driver_utils.get_time_scaled_model_func`, but always bound to this instance's
        ``demographic_model`` — including after :meth:`copy`, unlike a stored closure over the original model.
        """
        return self.demographic_model.get_migration_matrices(
            self.demographic_model.parameter_handler.convert_to_physical_params(params)
        )

    def outofbounds_fun(self, params, verbose: bool = False):
        """
        Converts optimizer-space ``params`` to physical parameters via ``demographic_model.parameter_handler``
        and returns the resulting violation score. Equivalent to the function previously returned by
        :func:`~tracts.driver_utils.get_time_scaled_model_bounds`, but always bound to this instance's
        ``demographic_model`` — including after :meth:`copy`, unlike a stored closure over the original model.
        """
        return self.demographic_model.get_violation_score(
            self.demographic_model.parameter_handler.convert_to_physical_params(params), verbose=verbose
        )

    # ------------------ Likelihood ------------------
   
    def loglik(
        self,
        male_matrix: np.ndarray,
        female_matrix: np.ndarray,
        tracts_data: TractsData,
        likelihood_options: LikelihoodOptions,
    ) -> LoglikBreakdown:
        """
        Computes the log-likelihood of ``tracts_data`` given the migration matrices
        ``male_matrix``/``female_matrix``, dispatching to the phase-type or
        hybrid-pedigree admixture model selected by ``self.phase_type_config``
        (``ad_model_autosomes`` for the autosomal component, ``ad_model_allosomes``
        for the allosomal components) so that callers do not need to branch on the
        admixture model choice themselves.

        Parameters
        ----------
        male_matrix : np.ndarray
            The male migration matrix.
        female_matrix : np.ndarray
            The female migration matrix. For autosome-only (non sex-biased)
            evaluations, pass the same (averaged) matrix as ``male_matrix``.
        tracts_data : TractsData
            The population and mapped autosomal/allosomal tract-length histogram
            data used to compute the likelihood. Its allosome-related fields are
            required when ``likelihood_options.include_allosomes=True``.
        likelihood_options : LikelihoodOptions
            Its ``include_autosomes``/``include_allosomes`` flags determine which
            log-likelihood components are computed. ``verbose_log``/``verbose_screen``
            are not used here.

        Returns
        -------
        LoglikBreakdown
            The log-likelihood contributions for each requested component. Use
            ``.total`` for the combined log-likelihood.
        
        Notes
        -----
        Does not catch exceptions: constructing a phase-type model can raise
        ``np.linalg.LinAlgError`` or ``ValueError`` for infeasible migration
        matrices (e.g. singular matrices); callers should catch these the same way
        they already do around obtaining ``male_matrix``/``female_matrix`` in the
        first place (see ``compute_objective`` in ``tracts.core``).
        """
        config = self.phase_type_config
        result = LoglikBreakdown()

        if likelihood_options.include_autosomes:
            if config.uses_hybrid_pedigree_autosomes:
                result.autosomes = HP.HP_loglik(
                    mig_matrix_f=female_matrix,
                    mig_matrix_m=male_matrix,
                    rho_f=config.rho_f,
                    rho_m=config.rho_m,
                    TP=config.TP,
                    Dioecious_model=config.ad_model_autosomes[2:],
                    X_chr=False,
                    X_chr_male=False,
                    N_cores=config.N_cores,
                    bins=tracts_data.autosome_bins,
                    Ls=tracts_data.population.Ls,
                    data=[mat for mat in tracts_data.autosome_data_mapped],
                    num_samples=len(tracts_data.population.indivs),
                    cutoff=0,
                )
            else:
                autosome_model = self._build_autosome_model(female_matrix=female_matrix, male_matrix=male_matrix)
                result.autosomes = autosome_model.loglik(
                    bins=tracts_data.autosome_bins,
                    Ls=tracts_data.population.Ls,
                    data=[mat for mat in tracts_data.autosome_data_mapped],
                    num_samples=len(tracts_data.population.indivs),
                )

        if likelihood_options.include_allosomes:
            if config.uses_hybrid_pedigree_allosomes:
                result.female_allosomes = HP.HP_loglik(
                    mig_matrix_f=female_matrix,
                    mig_matrix_m=male_matrix,
                    rho_f=config.rho_f,
                    rho_m=config.rho_m,
                    TP=config.TP,
                    Dioecious_model=config.ad_model_allosomes[2:],
                    X_chr=True,
                    X_chr_male=False,
                    N_cores=config.N_cores,
                    bins=tracts_data.allosome_bins,
                    Ls=[tracts_data.allosome_length],
                    data=[mat for mat in tracts_data.female_data_mapped],
                    num_samples=tracts_data.num_females,
                    cutoff=0,
                )
                result.male_allosomes = HP.HP_loglik(
                    mig_matrix_f=female_matrix,
                    mig_matrix_m=male_matrix,
                    rho_f=config.rho_f,
                    rho_m=config.rho_m,
                    TP=config.TP,
                    Dioecious_model=config.ad_model_allosomes[2:],
                    X_chr=True,
                    X_chr_male=True,
                    N_cores=config.N_cores,
                    bins=tracts_data.allosome_bins,
                    Ls=[tracts_data.allosome_length],
                    data=[mat for mat in tracts_data.male_data_mapped],
                    num_samples=tracts_data.num_males,
                    cutoff=0,
                )
            else:
                result.female_allosomes = self._build_female_allosome_model(
                    female_matrix=female_matrix, male_matrix=male_matrix
                ).loglik(
                    bins=tracts_data.allosome_bins,
                    Ls=[tracts_data.allosome_length],
                    data=[mat for mat in tracts_data.female_data_mapped],
                    num_samples=tracts_data.num_females,
                )
                result.male_allosomes = self._build_male_allosome_model(
                    female_matrix=female_matrix, male_matrix=male_matrix
                ).loglik(
                    bins=tracts_data.allosome_bins,
                    Ls=[tracts_data.allosome_length],
                    data=[mat for mat in tracts_data.male_data_mapped],
                    num_samples=tracts_data.num_males,
                )

        return result

    def _build_autosome_model(self, female_matrix: np.ndarray, male_matrix: np.ndarray):
        """
        Constructs the Monoecious or Dioecious phase-type model used for the autosomal component
        of :meth:`loglik`, given ``self.phase_type_config.ad_model_autosomes``. Only called when
        ``not self.phase_type_config.uses_hybrid_pedigree_autosomes``.
        """
        config = self.phase_type_config
        if config.ad_model_autosomes == 'M':
            return PhTMonoecious(migration_matrix=0.5 * (female_matrix + male_matrix), rho=1)
        assert male_matrix.shape[0] < 20, "PhTDioecious currently only supports less than 20 generations for autosomes."
        return PhTDioecious(
            migration_matrix_f=female_matrix,
            migration_matrix_m=male_matrix,
            rho_f=config.rho_f,
            rho_m=config.rho_m,
            sex_model=config.ad_model_autosomes,
        )

    def _build_female_allosome_model(self, female_matrix: np.ndarray, male_matrix: np.ndarray) -> PhTDioecious:
        """
        Constructs the Dioecious phase-type model used for the female allosomal component of
        :meth:`loglik`. Only called when ``not self.phase_type_config.uses_hybrid_pedigree_allosomes``.
        """
        config = self.phase_type_config
        return PhTDioecious(
            migration_matrix_f=female_matrix,
            migration_matrix_m=male_matrix,
            rho_f=config.rho_f,
            rho_m=config.rho_m,
            sex_model=config.ad_model_allosomes,
            X_chromosome=True,
        )

    def _build_male_allosome_model(self, female_matrix: np.ndarray, male_matrix: np.ndarray) -> PhTDioecious:
        """
        Constructs the Dioecious phase-type model used for the male allosomal component of
        :meth:`loglik`. Only called when ``not self.phase_type_config.uses_hybrid_pedigree_allosomes``.
        """
        config = self.phase_type_config
        return PhTDioecious(
            migration_matrix_f=female_matrix,
            migration_matrix_m=male_matrix,
            rho_f=config.rho_f,
            rho_m=config.rho_m,
            sex_model=config.ad_model_allosomes,
            X_chromosome=True,
            X_chromosome_male=True,
        )

    def split_migration_matrices(self, matrices: dict, include_allosomes: bool) -> tuple[np.ndarray, np.ndarray]:
        """
        Splits ``matrices`` (as returned by :meth:`model_func`/``demographic_model.get_migration_matrices``)
        into ``(male_matrix, female_matrix)``. When ``include_allosomes`` is True, ``matrices`` must contain
        exactly the male and female matrices. Otherwise, all matrices in ``matrices`` are averaged into a
        single autosome-only matrix, used for both.
        """
        matrix_list = list(matrices.values())
        if include_allosomes:
            male_matrix, female_matrix = matrix_list
        else:
            avg_matrix = np.mean(matrix_list, axis=0)
            male_matrix = female_matrix = avg_matrix
        return male_matrix, female_matrix

    def check_generation_zero_migration_warning(self, male_matrix: np.ndarray, female_matrix: np.ndarray,
                                                include_autosomes: bool = True, include_allosomes: bool = False) -> bool:
        """
        Returns whether constructing the phase-type model(s) that :meth:`loglik` would use for
        ``male_matrix``/``female_matrix`` raises a ``_GenerationZeroContributionWarning`` (i.e. a nonzero
        source-population contribution at generation 0, which the model silently ignores). Does not compute
        the likelihood itself, and does not mutate ``male_matrix``/``female_matrix``.

        Used to report this warning once for a step's optimal parameters, rather than on every
        objective-function evaluation during optimization — see :func:`~tracts.driver.run_optimization`.
        """
        config = self.phase_type_config
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if include_autosomes and not config.uses_hybrid_pedigree_autosomes:
                self._build_autosome_model(female_matrix=female_matrix.copy(), male_matrix=male_matrix.copy())
            if include_allosomes and not config.uses_hybrid_pedigree_allosomes:
                self._build_female_allosome_model(female_matrix=female_matrix.copy(), male_matrix=male_matrix.copy())
        return any(issubclass(w.category, _GenerationZeroContributionWarning) for w in caught)

    # ------------------ Copying ------------------

    def copy(self) -> "GeneticModel":
        """
        Returns a deep copy of this GeneticModel, including its demographic model and
        phase-type configuration. Useful for optimization routines that need their own
        local copy of the demographic model's fixed-parameter state (e.g. a
        ``FixedParametersHandler`` mutated during optimization) without affecting the original.
        """
        return GeneticModel(
            demographic_model=copy.deepcopy(self.demographic_model),
            phase_type_config=copy.deepcopy(self.phase_type_config),
        )

    def __repr__(self) -> str:
        return (
            f"GeneticModel(demographic_model={type(self.demographic_model).__name__}"
            f"(name={self.demographic_model.name!r}), phase_type_config={self.phase_type_config!r})"
        )
