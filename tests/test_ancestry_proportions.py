from tracts.driver_utils import locate_file_path, load_driver_file
from tracts.driver_utils import load_population, load_demographic_model_from_driver
import numpy as np
from pathlib import Path
from tracts.demography.parametrized_demography_sex_biased import ParametrizedDemographySexBiased, SexType
from tracts.phase_type.base_phase_type import ancestry_proportions_by_generation
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
    model, _, _, _ = load_demographic_model_from_driver(driver_spec=driver_spec, script_dir=script_path,
    driver_path=driver_path, allosome_label=allosome_label)
    ancestor_labels = model.population_indices.keys()
    ancestry_proportions = pop.calculate_ancestry_proportions(ancestor_labels)
    allosome_proportions = pop.calculate_allosome_proportions(ancestor_labels, allosome_label)
    
    assert np.allclose(ancestry_proportions, np.array([0.4, 0.6])), f"Expected {np.array([0.4, 0.6])}, got {ancestry_proportions}"
    assert np.allclose(allosome_proportions, np.array([0.7, 0.3])), f"Expected {np.array([0.7, 0.3])}, got {allosome_proportions}"


def test_proportions_from_matrices_sex_biased_founding_and_migration():
    """
    X and autosomal proportions for a hand-solvable model with populations (A, B), sampled 0, 1 and 2
    generations ago (rows 0, 1, 2):

        generation 2 (founding): females all from A, males all from B
        generation 1:            half of the females are migrants from B, no male migrants

    X, following lineages back in time from an equal number of sampled females and males:
      - 2/3 of X lineages start in females, 1/3 in males. A female's X is from her mother or father
        (1/2 each), a male's from his mother, so at generation 1 females carry 1/3 + 1/3 = 2/3, males 1/3.
      - Half of the female lineages at generation 1 are migrants: 1/3 of all lineages go to B.
      - The remaining 1/3 (females) and 1/3 (males) reach generation 2: females 1/6 + 1/3 = 1/2 (-> A),
        males 1/6 (-> B).
      X proportions: A = 1/2, B = 1/3 + 1/6 = 1/2.
    Autosomes (mean matrix): 1/4 -> B at generation 1; the other 3/4 split evenly. A = 3/8, B = 5/8.

    The previous implementation returned (1/3, 2/3) for the X: it requires both a sex-biased founding pulse
    and unequal female/male migration rates afterwards.
    """
    female = np.array([[0.0, 0.0],
                       [0.0, 0.5],
                       [1.0, 0.0]])
    male = np.array([[0.0, 0.0],
                     [0.0, 0.0],
                     [0.0, 1.0]])
    model = ParametrizedDemographySexBiased(allosome_label="X")
    model.parametrized_populations.append("ADM")
    proportions = model.proportions_from_matrices({f"ADM{SexType.FEMALE.suffix}": female,
                                                   f"ADM{SexType.MALE.suffix}": male})

    assert np.allclose(proportions["ADM_X"], [1 / 2, 1 / 2])
    assert np.allclose(proportions["ADM_autosomal"], [3 / 8, 5 / 8])

    # Same answer as the phase-type code: X proportions = (2 p_f[1] + p_m[1]) / 3.
    p_f, p_m = ancestry_proportions_by_generation(female, male, X_chr=True)
    assert np.allclose(proportions["ADM_X"], (2 * p_f[1] + p_m[1]) / 3)
