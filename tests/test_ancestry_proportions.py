from tracts.driver import locate_file_path, load_driver_file
from tracts.driver import load_population, load_model_from_driver
import numpy as np
from pathlib import Path

current_dir = Path(__file__).resolve().parent

def test_ancestry_proportions(driver_filename = "driver_test.yaml", script_path = current_dir):

    driver_path = locate_file_path(filename = driver_filename, script_dir=script_path)
    driver_spec = load_driver_file(driver_path)
    
    allosome_labels = driver_spec['samples']['allosomes'] if 'allosomes' in driver_spec['samples'] else []
    allosome_label = allosome_labels[0] if len(allosome_labels) > 0 else None

    pop = load_population(driver_path, driver_spec, script_dir=script_path, allosome_labels = allosome_labels) 
    pop.unknown_labels = driver_spec['unknown_labels_for_smoothing'] if 'unknown_labels_for_smoothing' in driver_spec else [] 
    
    pop.smooth_unknowns(allosome_labels = allosome_labels)
    model = load_model_from_driver(driver_spec=driver_spec, script_dir=script_path, driver_path=driver_path, allosome_label=allosome_label)
    ancestor_labels = model.population_indices.keys()
    ancestry_proportions = pop.calculate_ancestry_proportions(ancestor_labels)
    allosome_proportions = pop.calculate_allosome_proportions(ancestor_labels, allosome_label)
    
    assert np.allclose(ancestry_proportions, np.array([0.4, 0.6])), f"Expected {np.array([0.4, 0.6])}, got {ancestry_proportions}"
    assert np.allclose(allosome_proportions, np.array([0.7, 0.3])), f"Expected {np.array([0.7, 0.3])}, got {allosome_proportions}"
