from tracts.driver_utils import locate_file_path, load_driver_file
from tracts.driver_utils import load_population, load_demographic_model_from_driver
import numpy as np
import pytest
from pathlib import Path
from tracts.demography.parametrized_demography_sex_biased import ParametrizedDemographySexBiased, SexType
from tracts.demography.base_parametrized_demography import BaseParametrizedDemography
from tracts.migration_matrices import ancestry_proportions_by_generation
from tracts.phase_type.monoecious import PhTMonoecious
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


# --------------- Reference implementations and random matrices ---------------
# The tests below pin down the current behaviour of the proportion computations, so that they can be
# refactored into a single implementation without changing results. The references are written out
# independently of the tracts code.

def _reference_proportions(migration_matrix):
    """Ancestry proportions of a single migration matrix, from the founding row (T) to the present (row 0)."""
    proportions = migration_matrix[-1, :]
    for row in migration_matrix[-2::-1, :]:
        proportions = proportions * (1 - row.sum()) + row
    return proportions


def _reference_x_proportions(female_matrix, male_matrix):
    """
    X proportions for sex-specific matrices: M and F are the X ancestry of a male and of a female individual
    of each generation. A male's X comes from his mother, a female's from either parent. The sample has equal
    numbers of females and males, so 2/3 of X chromosomes are carried by females.
    """
    male, female = male_matrix[-1, :], female_matrix[-1, :]
    for male_row, female_row in zip(male_matrix[-2::-1, :], female_matrix[-2::-1, :]):
        male, female = (male_row + (1 - male_row.sum()) * female,
                        female_row + (1 - female_row.sum()) * (male + female) / 2)
    return (male + 2 * female) / 3


def _random_migration_matrix(rng, n_generations, n_populations):
    """
    A random (n_generations + 1) x n_populations migration matrix: row 0 (present) is zero, the founding row
    sums to 1, and the rows in between sum to less than 1 (some are zero).
    """
    matrix = np.zeros((n_generations + 1, n_populations))
    for g in range(1, n_generations):
        if rng.random() < 0.6:
            matrix[g] = rng.dirichlet(np.ones(n_populations)) * rng.uniform(0, 0.6)
    matrix[-1] = rng.dirichlet(np.ones(n_populations))
    return matrix


def _sex_biased_model(population="ADM"):
    model = ParametrizedDemographySexBiased(allosome_label="X")
    model.parametrized_populations.append(population)
    return model


def _sex_biased_input(female_matrix, male_matrix, population="ADM"):
    return {f"{population}{SexType.FEMALE.suffix}": female_matrix,
            f"{population}{SexType.MALE.suffix}": male_matrix}


SEEDS = range(20)


# --------------- Regression tests ---------------

@pytest.mark.parametrize("seed", SEEDS)
def test_proportions_from_matrix_matches_reference(seed):
    """BaseParametrizedDemography.proportions_from_matrix on random matrices with a zero row 0."""
    rng = np.random.default_rng(seed)
    matrix = _random_migration_matrix(rng, n_generations=int(rng.integers(1, 20)),
                                      n_populations=int(rng.integers(2, 5)))
    assert np.allclose(BaseParametrizedDemography.proportions_from_matrix(matrix), _reference_proportions(matrix))


def test_proportions_from_matrix_rejects_rows_not_summing_to_one():
    """The founding row must sum to 1; otherwise the proportions do not sum to 1 and a ValueError is raised."""
    matrix = np.array([[0.0, 0.0],
                       [0.1, 0.1],
                       [0.5, 0.3]])
    with pytest.raises(ValueError):
        BaseParametrizedDemography.proportions_from_matrix(matrix)


@pytest.mark.parametrize("seed", SEEDS)
def test_proportions_from_matrices_sex_biased_matches_reference(seed):
    """Autosomal and X entries of ParametrizedDemographySexBiased.proportions_from_matrices on random matrices."""
    rng = np.random.default_rng(seed)
    n_generations, n_populations = int(rng.integers(1, 20)), int(rng.integers(2, 5))
    female = _random_migration_matrix(rng, n_generations, n_populations)
    male = _random_migration_matrix(rng, n_generations, n_populations)
    proportions = _sex_biased_model().proportions_from_matrices(_sex_biased_input(female, male))

    assert set(proportions) == {"ADM_autosomal", "ADM_X"}
    assert np.allclose(proportions["ADM_autosomal"], _reference_proportions((female + male) / 2))
    assert np.allclose(proportions["ADM_X"], _reference_x_proportions(female, male))


# --------------- Consistency between demography and phase-type computations ---------------

@pytest.mark.parametrize("seed", SEEDS)
def test_demography_and_phase_type_proportions_agree(seed):
    """
    proportions_from_matrix / proportions_from_matrices agree with ancestry_proportions_by_generation (used by
    the likelihood), with the sample taken at generation 0: autosomal proportions (p_f[1] + p_m[1]) / 2, X
    proportions (2 p_f[1] + p_m[1]) / 3.
    """
    rng = np.random.default_rng(seed)
    n_generations, n_populations = int(rng.integers(1, 20)), int(rng.integers(2, 5))
    female = _random_migration_matrix(rng, n_generations, n_populations)
    male = _random_migration_matrix(rng, n_generations, n_populations)
    proportions = _sex_biased_model().proportions_from_matrices(_sex_biased_input(female, male))

    p_f, p_m = ancestry_proportions_by_generation(female, male, X_chr=False)
    assert np.allclose(proportions["ADM_autosomal"], (p_f[1] + p_m[1]) / 2)
    p_f, p_m = ancestry_proportions_by_generation(female, male, X_chr=True)
    assert np.allclose(proportions["ADM_X"], (2 * p_f[1] + p_m[1]) / 3)

    p, _ = ancestry_proportions_by_generation(female, female, X_chr=False)
    assert np.allclose(BaseParametrizedDemography.proportions_from_matrix(female), p[1])


@pytest.mark.parametrize("seed", SEEDS)
def test_monoecious_t0_proportions_agree(seed):
    """
    PhTMonoecious.t0_proportions (closed form: sum over generations of migration rates times survival factors)
    equals the ancestry proportions from ancestry_proportions_by_generation and from proportions_from_matrix.
    The comparison uses the matrix as preprocessed by PhTMonoecious (entries below 1e-3 zeroed, founding row
    renormalized). PhTMonoecious stores that matrix with row 1 set to zero after computing t0_proportions
    (generation 1 is handled separately in the density), so row 1 is restored here.
    """
    rng = np.random.default_rng(seed)
    matrix = _random_migration_matrix(rng, n_generations=int(rng.integers(2, 20)),
                                      n_populations=int(rng.integers(2, 5)))
    model = PhTMonoecious(matrix)
    processed = model.migration_matrix.copy()
    processed[1] = np.where(matrix[1] < 1e-3, 0, matrix[1])

    p, _ = ancestry_proportions_by_generation(processed, processed, X_chr=False)
    assert np.allclose(model.t0_proportions, p[1])
    assert np.allclose(model.t0_proportions, BaseParametrizedDemography.proportions_from_matrix(processed))

