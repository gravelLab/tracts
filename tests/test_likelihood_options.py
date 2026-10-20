"""
Tests for tracts/likelihood_options.py: LikelihoodOptions.
"""

import pytest

from tracts.likelihood_options import LikelihoodOptions


class TestLikelihoodOptionsDefaults:

    def test_defaults(self):
        opts = LikelihoodOptions()
        assert opts.verbose_log == 0
        assert opts.verbose_screen == 10
        assert opts.include_autosomes is True
        assert opts.include_allosomes is True


class TestLikelihoodOptionsValidation:

    def test_both_components_disabled_raises(self):
        with pytest.raises(ValueError, match="include_autosomes.*include_allosomes|include_allosomes.*include_autosomes"):
            LikelihoodOptions(include_autosomes=False, include_allosomes=False)

    def test_only_autosomes_enabled_is_valid(self):
        opts = LikelihoodOptions(include_autosomes=True, include_allosomes=False)
        assert opts.include_autosomes is True
        assert opts.include_allosomes is False

    def test_only_allosomes_enabled_is_valid(self):
        opts = LikelihoodOptions(include_autosomes=False, include_allosomes=True)
        assert opts.include_allosomes is True


class TestLikelihoodOptionsWithOverrides:

    def test_with_overrides_returns_new_instance(self):
        opts = LikelihoodOptions(verbose_log=1, verbose_screen=2)
        overridden = opts.with_overrides(include_autosomes=False)
        assert overridden is not opts

    def test_with_overrides_preserves_unset_fields(self):
        opts = LikelihoodOptions(verbose_log=1, verbose_screen=2)
        overridden = opts.with_overrides(include_allosomes=False)
        assert overridden.verbose_log == 1
        assert overridden.verbose_screen == 2
        assert overridden.include_autosomes is True
        assert overridden.include_allosomes is False

    def test_with_overrides_does_not_mutate_original(self):
        opts = LikelihoodOptions(include_autosomes=True, include_allosomes=True)
        opts.with_overrides(include_autosomes=False)
        assert opts.include_autosomes is True

    def test_with_overrides_can_set_multiple_fields(self):
        opts = LikelihoodOptions()
        overridden = opts.with_overrides(verbose_log=5, verbose_screen=6)
        assert overridden.verbose_log == 5
        assert overridden.verbose_screen == 6

    def test_with_overrides_still_validates(self):
        opts = LikelihoodOptions()
        with pytest.raises(ValueError):
            opts.with_overrides(include_autosomes=False, include_allosomes=False)
