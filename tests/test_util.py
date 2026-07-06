import numpy as np
import pytest

from tracts.util import (
    time_to_physical_function,
    rate_to_physical_function,
    sex_bias_to_physical_function,
    time_to_optimizer_function,
    rate_to_optimizer_function,
    sex_bias_to_optimizer_function,
    sex_bias_founder_to_physical_function,
    sex_bias_founder_to_optimizer_function,
)

"""
This test suite verifies the correctness of the transformations between optimizer-space and physical-space parameters for time, rate, and sex bias. 
The tests check that the transformations are inverses of each other, that they map to the correct ranges, and that they produce expected values for known inputs.
Additionally, the tests ensure that vectorized and scalar inputs are both supported by the transformation functions.
"""

def test_time_transform_inverse_roundtrip():
    """
    Test that time transformations are inverse to each other.

    Optimizer-space time parameters are unconstrained real values, while
    physical-space time parameters are strictly positive. The exponential and
    logarithm should therefore define inverse transformations.
    """
    x = np.array([-10.0, -1.0, 0.0, 1.0, 10.0])

    physical = time_to_physical_function(x)
    recovered = time_to_optimizer_function(physical)

    np.testing.assert_allclose(recovered, x)


def test_time_to_physical_is_positive():
    """
    Test that optimizer-space time values are mapped to strictly positive
    physical-space values.
    """
    x = np.linspace(-100, 100, 1000)
    physical = time_to_physical_function(x)

    assert np.all(physical > 0)


def test_time_to_optimizer_known_values():
    """
    Test known values of the physical-to-optimizer time transformation.
    """
    x = np.array([1.0, np.e, np.e**2])

    expected = np.array([0.0, 1.0, 2.0])
    result = time_to_optimizer_function(x)

    np.testing.assert_allclose(result, expected)


def test_rate_transform_inverse_roundtrip():
    """
    Test that rate transformations are inverse to each other.

    Optimizer-space rate parameters are unconstrained real values, while
    physical-space rates lie in the interval (0, 1). The logistic sigmoid and
    logit should therefore define inverse transformations.
    """
    x = np.array([-20.0, -2.0, 0.0, 2.0, 20.0])

    physical = rate_to_physical_function(x)
    recovered = rate_to_optimizer_function(physical)

    np.testing.assert_allclose(recovered, x)


def test_rate_to_physical_is_between_zero_and_one():
    """
    Test that moderate optimizer-space rate values are mapped to (0, 1).

    Extremely large positive inputs may round numerically to exactly 1.0 in
    double precision, so this test uses a moderate range.
    """
    x = np.linspace(-30, 30, 1000)
    physical = rate_to_physical_function(x)

    assert np.all(physical > 0)
    assert np.all(physical < 1)


def test_rate_to_physical_known_values():
    """
    Test known values of the optimizer-to-physical rate transformation.
    """
    assert rate_to_physical_function(0.0) == pytest.approx(0.5)


def test_rate_to_optimizer_known_values():
    """
    Test known values of the physical-to-optimizer rate transformation.
    """
    assert rate_to_optimizer_function(0.5) == pytest.approx(0.0)


def test_sex_bias_transform_inverse_roundtrip():
    """
    Test that sex-bias transformations are inverse to each other on a moderate
    optimizer range where floating-point saturation is negligible.
    """
    x = np.array([-10.0, -2.0, 0.0, 2.0, 10.0])

    physical = sex_bias_to_physical_function(x)
    recovered = sex_bias_to_optimizer_function(physical)

    np.testing.assert_allclose(recovered, x, rtol=1e-10, atol=1e-10)


def test_sex_bias_to_physical_is_between_minus_one_and_one():
    """
    Test that moderate optimizer-space sex-bias values are mapped to (-1, 1).

    Extremely large positive or negative inputs may round numerically to the
    exact boundaries in double precision.
    """
    x = np.linspace(-30, 30, 1000)
    physical = sex_bias_to_physical_function(x)

    assert np.all(physical > -1)
    assert np.all(physical < 1)


def test_sex_bias_to_physical_known_values():
    """
    Test known values of the optimizer-to-physical sex-bias transformation.
    """
    assert sex_bias_to_physical_function(0.0) == pytest.approx(0.0)


def test_sex_bias_to_optimizer_boundaries():
    """
    Test exact boundary behavior of the physical-to-optimizer sex-bias
    transformation.

    At y = 1 the transformation diverges to +infinity, and at y = -1 it
    diverges to -infinity. The implementation replaces these infinities by
    finite sentinel values.
    """
    assert sex_bias_to_optimizer_function(1.0) == pytest.approx(1e32)
    assert sex_bias_to_optimizer_function(-1.0) == pytest.approx(-1e32)
    assert np.isnan(sex_bias_to_optimizer_function(np.nan))


def test_sex_bias_to_optimizer_near_boundaries_is_finite():
    """
    Test that values strictly inside (-1, 1), but close to the boundaries,
    remain finite.
    """
    eps = 1e-12

    result_pos = sex_bias_to_optimizer_function(1.0 - eps)
    result_neg = sex_bias_to_optimizer_function(-1.0 + eps)

    assert np.isfinite(result_pos)
    assert np.isfinite(result_neg)
    assert result_pos > 0
    assert result_neg < 0


def test_sex_bias_to_optimizer_zero():
    """
    Test that zero sex bias maps to zero in optimizer space.
    """
    assert sex_bias_to_optimizer_function(0.0) == pytest.approx(0.0)


def test_sex_bias_to_optimizer_symmetry():
    """
    Test odd symmetry of the sex-bias optimizer transformation.

    The transformation satisfies f(-y) = -f(y) for y in (-1, 1).
    """
    y = np.array([0.1, 0.25, 0.5, 0.75, 0.95])

    result_pos = sex_bias_to_optimizer_function(y)
    result_neg = sex_bias_to_optimizer_function(-y)

    np.testing.assert_allclose(result_neg, -result_pos)


def test_sex_bias_to_optimizer_is_monotone_increasing():
    """
    Test that the physical-to-optimizer sex-bias transformation is strictly
    increasing on (-1, 1).
    """
    y = np.linspace(-0.99, 0.99, 1000)
    result = sex_bias_to_optimizer_function(y)

    assert np.all(np.diff(result) > 0)


def test_sex_bias_to_optimizer_outside_domain_is_finite():
    """
    Test behavior slightly outside the mathematical domain.

    Values outside [-1, 1] produce NaNs in the raw logarithmic formula. The
    implementation replaces them by finite sentinel values according to the
    sign of the input.
    """
    y = np.array([-1.01, 1.01])

    result = sex_bias_to_optimizer_function(y)
    expected = np.array([-1e32, 1e32])

    np.testing.assert_allclose(result, expected)


def test_vectorized_inputs_are_supported():
    """
    Test that all transformations support NumPy array inputs.
    """
    x = np.array([-1.0, 0.0, 1.0])

    assert time_to_physical_function(x).shape == x.shape
    assert rate_to_physical_function(x).shape == x.shape
    assert sex_bias_to_physical_function(x).shape == x.shape


@pytest.mark.parametrize("x", [
    np.array([-1.0, 0.0, 1.0, -2.0, 2.0, 0.5]),          # 3 pops, mixed signs
    np.array([0.0, 0.0, 0.0, 0.0]),                        # 2 pops, all zeros
    np.array([-5.0, 3.0, 1.0, -1.0, -3.0, 5.0]),          # 3 pops, large values
    np.array([0.1, -0.1]),                                  # 1 pop
    np.array([2.0, -2.0, 1.0, -1.0, 0.5, -0.5, 3.0, -3.0]),  # 4 pops
])
def test_sex_bias_founder_transform_inverse_roundtrip_optimizer_to_physical(x):
    """
    Test that converting from optimizer space to physical space and back recovers
    the original optimizer-space parameters.
    """
    physical = sex_bias_founder_to_physical_function(x)
    recovered = sex_bias_founder_to_optimizer_function(physical)

    np.testing.assert_allclose(recovered, x, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("x", [
    np.array([0.2, 0.3, 0.1, 0.0, 0.5, -0.5]),            # 3 pops, rates sum <1
    np.array([0.4, 0.4, 0.0, 0.0]),                        # 2 pops, zero sex bias
    np.array([0.05, 0.1, 0.15, 0.2, 0.8, -0.8, 0.5, -0.5]),  # 4 pops
    np.array([0.3, 0.2]),                                   # 1 pop, no bias

])
def test_sex_bias_founder_transform_inverse_roundtrip_physical_to_optimizer(x):
    """
    Test that converting from physical space to optimizer space and back recovers
    the original physical-space parameters.

    Input x has first half as total rates and second half as sex biases.
    Rates must sum to strictly less than 1 (remainder population absorbs the rest).
    """
    if len(x) % 2 != 0:
        pytest.skip("Parametrize entry has odd length, skipping.")
    n = len(x) // 2
    rates = x[:n]
    if np.sum(rates) >= 1:
        pytest.skip("Rates sum to >= 1, outside domain.")

    optimizer = sex_bias_founder_to_optimizer_function(x)
    recovered = sex_bias_founder_to_physical_function(optimizer)

    np.testing.assert_allclose(recovered, x, rtol=1e-10, atol=1e-10)


def test_sex_bias_founder_rates_sum_less_than_one():
    """
    Test that physical-space rates (first half of output) sum to strictly less
    than 1, leaving room for the remainder population.
    """
    for x in [
        np.array([-1.0, 0.0, 1.0, -2.0, 2.0, 0.5]),
        np.array([0.0, 0.0, 0.0, 0.0]),
        np.array([-5.0, 3.0, 1.0, -1.0, -3.0, 5.0]),
    ]:
        physical = sex_bias_founder_to_physical_function(x)
        n = len(physical) // 2
        rates = physical[:n]
        assert np.sum(rates) < 1


def test_sex_bias_founder_sex_biases_in_minus_one_to_one():
    """
    Test that physical-space sex biases (second half of output) lie in (-1, 1).
    """
    for x in [
        np.array([-1.0, 0.0, 1.0, -2.0, 2.0, 0.5]),
        np.array([0.0, 0.0, 0.0, 0.0]),
        np.array([-5.0, 3.0, 1.0, -1.0, -3.0, 5.0]),
    ]:
        physical = sex_bias_founder_to_physical_function(x)
        n = len(physical) // 2
        sex_biases = physical[n:]
        assert np.all(sex_biases > -1)
        assert np.all(sex_biases < 1)


def test_sex_bias_founder_zero_bias_at_equal_rates():
    """
    Test that equal female and male optimizer rates produce zero sex bias.
    """
    x = np.array([1.0, -0.5, 1.0, -0.5])  # same female and male rates for 2 pops
    physical = sex_bias_founder_to_physical_function(x)
    n = len(physical) // 2
    sex_biases = physical[n:]

    np.testing.assert_allclose(sex_biases, 0.0, atol=1e-14)


def test_scalar_inputs_are_supported():
    """
    Test that all transformations support scalar inputs.
    """
    assert np.isscalar(time_to_physical_function(0.0))
    assert np.isscalar(rate_to_physical_function(0.0))
    assert np.isscalar(sex_bias_to_physical_function(0.0))
    assert np.isscalar(time_to_optimizer_function(1.0))
    assert np.isscalar(rate_to_optimizer_function(0.5))