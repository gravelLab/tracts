from tracts.driver_utils import locate_file_path, load_driver_file
from tracts.driver_utils import load_population, load_model_from_driver
import numpy as np
from pathlib import Path
current_dir = Path(__file__).resolve().parent

def test_ancestry_proportions(driver_filename = "./drivers/driver_test.yaml", script_path = current_dir):

    """
    This test checks the calculation of ancestry proportions and allosome proportions in a population based on a driver file. 
    It verifies that the calculated proportions match expected values for a given model and population setup.
    The test performs the following steps:
    1. Loads the driver file and extracts the allosome labels,
    2. Loads the population using the driver specifications and smooths unknown labels,
    3. Loads the model from the driver specifications,
    4. Calculates the ancestry proportions and allosome proportions,
    5. Asserts that the calculated proportions are close to the expected values of [0.4, 0.6] for ancestry proportions and [0.7, 0.3] for allosome proportions.
    """

    driver_path = locate_file_path(filename = driver_filename, script_dir=script_path)
    driver_spec = load_driver_file(driver_path)
    
    allosome_labels = driver_spec.samples.allosomes
    allosome_label = allosome_labels[0] if len(allosome_labels) > 0 else None

    pop = load_population(driver_path, driver_spec, script_dir=script_path, allosome_labels = allosome_labels) 
    pop.unknown_labels = driver_spec.optim.unknown_labels_for_smoothing
    
    pop.smooth_unknowns(allosome_labels = allosome_labels)
    model = load_model_from_driver(driver_spec=driver_spec, script_dir=script_path, 
    driver_path=driver_path, allosome_label=allosome_label)
    ancestor_labels = model.population_indices.keys()
    ancestry_proportions = pop.calculate_ancestry_proportions(ancestor_labels)
    allosome_proportions = pop.calculate_allosome_proportions(ancestor_labels, allosome_label)
    
    assert np.allclose(ancestry_proportions, np.array([0.4, 0.6])), f"Expected {np.array([0.4, 0.6])}, got {ancestry_proportions}"
    assert np.allclose(allosome_proportions, np.array([0.7, 0.3])), f"Expected {np.array([0.7, 0.3])}, got {allosome_proportions}"
