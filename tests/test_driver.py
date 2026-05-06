import unittest
import os
from tracts.driver import run_tracts
from pathlib import Path

class TestDriver(unittest.TestCase):

    def _run_tracts_test(self, driver_file: str, script_dir: str, output_dir: str, log_name: str, expected_files: list):
        """
        Helper method to test run_tracts with specific driver file and expected outputs.
        
        Parameters
        ----------
        driver_file: str
            Name of the driver file
        script_dir: str
            Directory containing the driver file
        output_dir: str
            Expected output directory
        log_name: str
            Expected log file name
        expected_files: list
            List of expected output files (can be empty list if no specific files expected)
        """

        # Run tracts
        run_tracts(driver_file, script_dir=script_dir)

        # ------------ log file checks ------------

        # Verify output directory exists
        self.assertTrue(os.path.exists(output_dir), f"Output directory not created. Current output_dir is: {output_dir}")
        
        # Verify output files exist (check for expected outputs)
        output_files = os.listdir(output_dir)
        self.assertGreater(len(output_files), 0, "No output files created.")

        # Verify there is one .log file
        log_files = [f for f in output_files if f.endswith('.log')]
        self.assertEqual(len(log_files), 1, f"One .log file should be created in output directory. There are currently {len(log_files)} .log files.")
        
        # Verify log file name matches log_name
        self.assertEqual(log_files[0], log_name, f"Log file name {log_files[0]} does not match expected {log_name}")
        
        # Verify log file contains expected content
        log_path = os.path.join(output_dir, log_name)
        with open(log_path, "r") as f:
            log_content = f.read()
            self.assertGreater(len(log_content), 0, "Log file is empty.")

        # Delete log file after checks
        os.remove(log_path)
        
        # ------------ output files checks ------------
        
        # Verify expected output files exist (if any expected)
        for expected_file in expected_files:
            file_path = os.path.join(output_dir, expected_file)
            self.assertTrue(os.path.exists(file_path), f"Expected file '{expected_file}' not found in output directory.")
        
        # Clean up output directory
        for entry in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, entry))
        os.rmdir(output_dir)
        


    def test_run_tracts(self):
        """
        Test that run_tracts creates all outputs and log file in specified directory, for a set of driver files covering all possible
        configurations regarding optimization.            
        """
        
        script_dir = Path(__file__).parent / "drivers"
        driver_files_autosomes = sorted(
            [f.name for f in script_dir.iterdir() if "autosomes" in f.name]
        )
        driver_files_allosomes = sorted(
            [f.name for f in script_dir.iterdir() if "allosomes" in f.name]
        )
        output_dir = Path(__file__).parent / "drivers" / "test_output"
        log_name = "test_logfile.log"
        

        expected_files_with_allosomes = [
            "test_output_optimal_parameters.txt",
            "test_output_autosomes_all_populations.png",
            "test_output_female_allosomes_all_populations.png",
            "test_output_male_allosomes_all_populations.png",
            "test_output_male_allosome_predicted_tract_distribution",
            "test_output_female_allosome_predicted_tract_distribution",
            "test_output_male_allosome_sample_tract_distribution",
            "test_output_female_allosome_sample_tract_distribution",
            "test_output_tract_length_allosome_bins",
            "test_output_autosome_predicted_tract_distribution",
            "test_output_male_migration_matrix",
            "test_output_female_migration_matrix",
            "test_output_autosome_sample_tract_distribution",
            "test_output_tract_length_autosome_bins"
        ]

        expected_files_without_allosomes = [
            "test_output_optimal_parameters.txt",
            "test_output_autosomes_all_populations.png",
            "test_output_autosome_predicted_tract_distribution",
            "test_output_male_migration_matrix",
            "test_output_female_migration_matrix",
            "test_output_autosome_sample_tract_distribution",
            "test_output_tract_length_autosome_bins"
        ]
        
        for driver_file in driver_files_autosomes:
            self._run_tracts_test(driver_file, script_dir, output_dir, log_name, expected_files_without_allosomes)
        
        for driver_file in driver_files_allosomes:
            self._run_tracts_test(driver_file, script_dir, output_dir, log_name, expected_files_with_allosomes)

        



