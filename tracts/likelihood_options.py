"""
Bundles the logging verbosity and autosome/allosome inclusion flags used when
evaluating a demographic model's likelihood, to avoid threading them as
separate parameters through :mod:`tracts.core`'s optimization functions.
"""
from __future__ import annotations
import dataclasses
from dataclasses import dataclass


@dataclass
class LikelihoodOptions:
    """
    Logging verbosity and autosome/allosome inclusion flags for a likelihood evaluation.

    Attributes
    ----------
    verbose_log: int
        Log optimization status every ``verbose_log`` iterations (0 = never). Defaults to 0.
    verbose_screen: int
        Print optimization status every ``verbose_screen`` iterations (0 = never). Defaults to 10.
    include_autosomes: bool
        Whether to include the autosomal log-likelihood in the objective. Defaults to True.
    include_allosomes: bool
        Whether to include the allosomal log-likelihood in the objective. Defaults to True.
    """
    verbose_log: int = 0
    verbose_screen: int = 10
    include_autosomes: bool = True
    include_allosomes: bool = True

    def __post_init__(self):
        if not self.include_autosomes and not self.include_allosomes:
            raise ValueError("At least one of include_autosomes or include_allosomes must be True.")

    def with_overrides(self, **overrides) -> "LikelihoodOptions":
        """
        Returns a copy of this LikelihoodOptions with the given fields overridden.

        Used where ``include_autosomes``/``include_allosomes`` vary per evaluation within a
        single optimization run (e.g. step 1 evaluates autosomes only, step 2 evaluates
        allosomes and optionally autosomes), while ``verbose_log``/``verbose_screen`` stay
        fixed for the whole run.
        """
        return dataclasses.replace(self, **overrides)
