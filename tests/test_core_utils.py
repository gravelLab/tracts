"""
Tests for tracts/core_utils.py: console/logging helpers used by tracts/core.py's
optimization functions.
"""

import logging
from types import SimpleNamespace
import pytest
from unittest.mock import MagicMock

from tracts import core_utils


# --------------- _print_and_log ---------------

class TestPrintAndLog:

    def test_prints_and_logs_each_message(self, capsys, caplog):
        with caplog.at_level(logging.INFO, logger="tracts.core_utils"):
            core_utils._print_and_log("first", "second")

        out = capsys.readouterr().out
        assert "first" in out
        assert "second" in out
        assert [r.message for r in caplog.records] == ["first", "second"]

    def test_no_arguments_prints_nothing(self, capsys, caplog):
        with caplog.at_level(logging.INFO, logger="tracts.core_utils"):
            core_utils._print_and_log()
        assert capsys.readouterr().out == ""
        assert caplog.records == []


# --------------- _print_verbose ---------------

class TestPrintVerbose:

    def test_verbose_log_positive_logs_lines(self, capsys, caplog):
        with caplog.at_level(logging.INFO, logger="tracts.core_utils"):
            core_utils._print_verbose(["a", "b"], verbose_log=1, verbose_screen=0)
        assert [r.message for r in caplog.records] == ["a", "b"]
        assert capsys.readouterr().out == ""

    def test_verbose_screen_positive_prints_lines(self, capsys, caplog):
        with caplog.at_level(logging.INFO, logger="tracts.core_utils"):
            core_utils._print_verbose(["a", "b"], verbose_log=0, verbose_screen=1)
        assert caplog.records == []
        assert capsys.readouterr().out == "a\nb\n"

    def test_both_zero_does_nothing(self, capsys, caplog):
        with caplog.at_level(logging.INFO, logger="tracts.core_utils"):
            core_utils._print_verbose(["a"], verbose_log=0, verbose_screen=0)
        assert caplog.records == []
        assert capsys.readouterr().out == ""

    def test_both_positive_does_both(self, capsys, caplog):
        with caplog.at_level(logging.INFO, logger="tracts.core_utils"):
            core_utils._print_verbose(["a"], verbose_log=1, verbose_screen=1)
        assert [r.message for r in caplog.records] == ["a"]
        assert capsys.readouterr().out == "a\n"


# --------------- _print_periodic ---------------

class TestPrintPeriodic:

    @pytest.mark.parametrize("counter", [0, 3, 6])
    def test_logs_on_checkpoint(self, counter, capsys, caplog):
        with caplog.at_level(logging.INFO, logger="tracts.core_utils"):
            core_utils._print_periodic(["a"], verbose_log=3, verbose_screen=0, counter=counter)
        assert [r.message for r in caplog.records] == ["a"]

    @pytest.mark.parametrize("counter", [1, 2, 4, 5])
    def test_silent_off_checkpoint(self, counter, capsys, caplog):
        with caplog.at_level(logging.INFO, logger="tracts.core_utils"):
            core_utils._print_periodic(["a"], verbose_log=3, verbose_screen=0, counter=counter)
        assert caplog.records == []

    def test_verbose_log_zero_never_logs(self, caplog):
        with caplog.at_level(logging.INFO, logger="tracts.core_utils"):
            core_utils._print_periodic(["a"], verbose_log=0, verbose_screen=0, counter=0)
        assert caplog.records == []

    def test_verbose_screen_checkpoint_prints(self, capsys):
        core_utils._print_periodic(["a", "b"], verbose_log=0, verbose_screen=2, counter=4)
        assert capsys.readouterr().out == "a\nb\n"

    def test_verbose_screen_off_checkpoint_silent(self, capsys):
        core_utils._print_periodic(["a"], verbose_log=0, verbose_screen=2, counter=3)
        assert capsys.readouterr().out == ""


# --------------- _print_single_step_header ---------------

class TestPrintSingleStepHeader:

    def _make_handler(self, free_indices=(0, 1), labels=("t", "rate")):
        handler = MagicMock()
        handler.free_parameters_indices = list(free_indices)
        handler.indices_to_labels.return_value = list(labels)
        return handler

    def test_prints_title_when_print_step_header_true(self, capsys):
        handler = self._make_handler()
        core_utils._print_single_step_header(handler, print_step_header=True, verbose_log=0, verbose_screen=1, counter=0)
        out = capsys.readouterr().out
        assert "Optimizing model likelihood over parameters" in out
        assert "['t', 'rate']" in out

    def test_suppresses_title_when_print_step_header_false(self, capsys):
        handler = self._make_handler()
        core_utils._print_single_step_header(handler, print_step_header=False, verbose_log=0, verbose_screen=1, counter=0)
        out = capsys.readouterr().out
        assert "Optimizing model likelihood" not in out

    def test_table_header_gated_by_periodic_checkpoint(self, capsys):
        handler = self._make_handler()
        core_utils._print_single_step_header(handler, print_step_header=False, verbose_log=0, verbose_screen=2, counter=1)
        assert "Iter." not in capsys.readouterr().out

        core_utils._print_single_step_header(handler, print_step_header=False, verbose_log=0, verbose_screen=2, counter=2)
        assert "Iter." in capsys.readouterr().out


# --------------- _get_steps ---------------

class TestGetSteps:

    def test_none_runs_both_steps(self):
        assert core_utils._get_steps(None, ad_model_allosomes="DC") == (True, True)

    @pytest.mark.parametrize("steps,expected", [
        ([1], (True, False)),
        (["step1"], (True, False)),
        ([2], (False, True)),
        (["step2"], (False, True)),
        ([1, 2], (True, True)),
        (["step1", "step2"], (True, True)),
    ])
    def test_valid_combinations(self, steps, expected):
        assert core_utils._get_steps(steps, ad_model_allosomes="DC") == expected

    def test_not_a_list_raises_type_error(self):
        with pytest.raises(TypeError):
            core_utils._get_steps(1, ad_model_allosomes="DC")

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="empty"):
            core_utils._get_steps([], ad_model_allosomes="DC")

    def test_invalid_step_value_raises(self):
        with pytest.raises(ValueError, match="Invalid step value"):
            core_utils._get_steps([3], ad_model_allosomes="DC")

    def test_duplicate_step_raises(self):
        with pytest.raises(ValueError, match="duplicate"):
            core_utils._get_steps([1, "step1"], ad_model_allosomes="DC")

    def test_both_steps_without_allosome_model_downgrades(self, caplog):
        with caplog.at_level(logging.INFO, logger="tracts.core_utils"):
            result = core_utils._get_steps(None, ad_model_allosomes=None)
        assert result == (True, False)
        assert any("Forcing step 2 to False" in r.message for r in caplog.records)

    def test_step1_only_without_allosome_model_allowed(self):
        assert core_utils._get_steps([1], ad_model_allosomes=None) == (True, False)

    def test_step2_only_without_allosome_model_raises(self):
        with pytest.raises(ValueError, match="ad_model_allosomes"):
            core_utils._get_steps([2], ad_model_allosomes=None)


# --------------- _flush_final_result ---------------

class TestFlushFinalResult:

    def _make_handler(self):
        handler = MagicMock()
        handler.enable_time_param_logging = True
        handler.convert_to_physical_params.return_value = [1.0, 2.0]
        return handler

    def test_no_params_does_nothing(self, capsys, caplog):
        with caplog.at_level(logging.INFO, logger="tracts.core_utils"):
            core_utils._flush_final_result({"params": None, "objective": 0.0}, self._make_handler(), 1, 1, 5)
        assert caplog.records == []
        assert capsys.readouterr().out == ""

    def test_on_checkpoint_both_needs_false_does_nothing(self, capsys, caplog):
        best_state = {"params": [1.0], "objective": -2.0}
        with caplog.at_level(logging.INFO, logger="tracts.core_utils"):
            core_utils._flush_final_result(best_state, self._make_handler(), verbose_log=3, verbose_screen=0, counter=3)
        assert caplog.records == []

    def test_off_checkpoint_logs(self, caplog):
        best_state = {"params": [1.0], "objective": -2.0}
        handler = self._make_handler()
        with caplog.at_level(logging.INFO, logger="tracts.core_utils"):
            core_utils._flush_final_result(best_state, handler, verbose_log=3, verbose_screen=0, counter=4, note="Autosomes")
        assert len(caplog.records) == 1
        assert "iter=" in caplog.records[0].message

    def test_off_checkpoint_prints_with_note(self, capsys):
        # _flush_final_result's screen output goes through eprint(), which writes to stderr.
        best_state = {"params": [1.0], "objective": -2.0}
        handler = self._make_handler()
        core_utils._flush_final_result(best_state, handler, verbose_log=0, verbose_screen=3, counter=4, note="Autosomes")
        err = capsys.readouterr().err
        assert "Autosomes" in err

    def test_restores_enable_time_param_logging(self):
        best_state = {"params": [1.0], "objective": -2.0}
        handler = self._make_handler()
        handler.enable_time_param_logging = True
        core_utils._flush_final_result(best_state, handler, verbose_log=0, verbose_screen=3, counter=4)
        assert handler.enable_time_param_logging is True

    def test_restores_enable_time_param_logging_even_on_exception(self):
        best_state = {"params": [1.0], "objective": -2.0}
        handler = self._make_handler()
        handler.enable_time_param_logging = True
        handler.convert_to_physical_params.side_effect = ValueError("boom")
        with pytest.raises(ValueError):
            core_utils._flush_final_result(best_state, handler, verbose_log=0, verbose_screen=3, counter=4)
        assert handler.enable_time_param_logging is True

    def test_reports_negated_objective(self, capsys):
        best_state = {"params": [1.0], "objective": -2.0}
        handler = self._make_handler()
        core_utils._flush_final_result(best_state, handler, verbose_log=0, verbose_screen=3, counter=4)
        err = capsys.readouterr().err
        assert "2" in err  # -best_state['objective'] == 2.0

    def test_breakdown_reports_female_and_male_allosomes_separately(self, capsys):
        # Step-2-style breakdown: no autosomes, separate female/male allosomal log-likelihoods.
        loglik = SimpleNamespace(autosomes=None, female_allosomes=-225.8, male_allosomes=-156.4)
        best_state = {"params": [1.0], "objective": 382.2, "loglik": loglik}
        handler = self._make_handler()
        core_utils._flush_final_result(best_state, handler, verbose_log=0, verbose_screen=3,
                                       counter=4, note="Allosomes")
        lines = [l for l in capsys.readouterr().err.splitlines() if l.strip()]
        # One row per computed component, not a single summed "Allosomes" row.
        assert len(lines) == 2
        assert "Female allosomes" in lines[0] and "-225.8" in lines[0]
        assert "Male allosomes" in lines[1] and "-156.4" in lines[1]
        # The summed note is not used when a breakdown is present.
        assert not any(line.rstrip().endswith("Allosomes") and "allosomes" not in line for line in lines)

    def test_breakdown_reports_all_three_components(self, caplog):
        loglik = SimpleNamespace(autosomes=-532.6, female_allosomes=-225.8, male_allosomes=-156.4)
        best_state = {"params": [1.0], "objective": 914.8, "loglik": loglik}
        handler = self._make_handler()
        with caplog.at_level(logging.INFO, logger="tracts.core_utils"):
            core_utils._flush_final_result(best_state, handler, verbose_log=3, verbose_screen=0, counter=4)
        messages = [r.message for r in caplog.records]
        assert len(messages) == 3
        assert "Autosomes" in messages[0]
        assert "Female allosomes" in messages[1]
        assert "Male allosomes" in messages[2]

    def test_none_breakdown_falls_back_to_summed_note(self, capsys):
        best_state = {"params": [1.0], "objective": -2.0, "loglik": None}
        handler = self._make_handler()
        core_utils._flush_final_result(best_state, handler, verbose_log=0, verbose_screen=3,
                                       counter=4, note="Allosomes")
        lines = [l for l in capsys.readouterr().err.splitlines() if l.strip()]
        assert len(lines) == 1
        assert "Allosomes" in lines[0] and "2" in lines[0]


# --------------- _print_step2_header ---------------

class TestPrintStep2Header:

    def test_no_free_params_does_nothing(self, capsys):
        core_utils._print_step2_header(
            step_1=True, autosomes_in_step_2=True, free_sex_bias_parameters={"sb": 0},
            table_header="H", line_header="-", print_step_header=True,
            ad_model_allosomes="DC", has_free_params=False, verbose_log=0, verbose_screen=1,
        )
        assert capsys.readouterr().out == ""

    def test_prints_step2_message_with_free_params(self, capsys):
        core_utils._print_step2_header(
            step_1=True, autosomes_in_step_2=True, free_sex_bias_parameters={"sb_eur": 0, "sb_afr": 0},
            table_header="TABLE_HEADER", line_header="----", print_step_header=True,
            ad_model_allosomes="DC", has_free_params=True, verbose_log=0, verbose_screen=1,
        )
        out = capsys.readouterr().out
        assert "Step 2" in out
        assert "sb_eur" in out and "sb_afr" in out
        assert "autosomal + allosomal" in out
        assert "TABLE_HEADER" in out

    def test_allosomal_only_message_when_autosomes_in_step_2_false(self, capsys):
        core_utils._print_step2_header(
            step_1=True, autosomes_in_step_2=False, free_sex_bias_parameters={"sb": 0},
            table_header="H", line_header="-", print_step_header=True,
            ad_model_allosomes="DC", has_free_params=True, verbose_log=0, verbose_screen=1,
        )
        assert "allosomal" in capsys.readouterr().out

    def test_message_mentions_p0_when_step_1_false(self, capsys):
        core_utils._print_step2_header(
            step_1=False, autosomes_in_step_2=True, free_sex_bias_parameters={"sb": 0},
            table_header="H", line_header="-", print_step_header=True,
            ad_model_allosomes="DC", has_free_params=True, verbose_log=0, verbose_screen=1,
        )
        assert "fixed at initial values" in capsys.readouterr().out

    def test_message_mentions_previous_step_when_step_1_true(self, capsys):
        core_utils._print_step2_header(
            step_1=True, autosomes_in_step_2=True, free_sex_bias_parameters={"sb": 0},
            table_header="H", line_header="-", print_step_header=True,
            ad_model_allosomes="DC", has_free_params=True, verbose_log=0, verbose_screen=1,
        )
        assert "fixed at values from previous optimization step" in capsys.readouterr().out

    def test_print_step_header_false_suppresses_step2_message_but_shows_table(self, capsys):
        core_utils._print_step2_header(
            step_1=True, autosomes_in_step_2=True, free_sex_bias_parameters={"sb": 0},
            table_header="TABLE_HEADER", line_header="-", print_step_header=False,
            ad_model_allosomes="DC", has_free_params=True, verbose_log=0, verbose_screen=1,
        )
        out = capsys.readouterr().out
        assert "Step 2" not in out
        assert "TABLE_HEADER" in out

    def test_ad_model_allosomes_none_suppresses_table_header(self, capsys):
        core_utils._print_step2_header(
            step_1=True, autosomes_in_step_2=True, free_sex_bias_parameters={"sb": 0},
            table_header="TABLE_HEADER", line_header="-", print_step_header=True,
            ad_model_allosomes=None, has_free_params=True, verbose_log=0, verbose_screen=1,
        )
        assert "TABLE_HEADER" not in capsys.readouterr().out
