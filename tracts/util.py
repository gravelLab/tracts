import sys
from numbers import Number
import numpy as np
from scipy.special import logit, expit


def eprint(*args, **kwargs):
    """
    Print to the standard error stream. This is a convenience wrapper around :func:`print` that redirects output
    to :data:`sys.stderr`.

    Parameters
    ----------
    args
        Positional arguments passed to :func:`print`.
    kwargs
        Keyword arguments passed to :func:`print`.
    """
    print(*args, file=sys.stderr, **kwargs)


def all_same_sign(lst: list[Number]) -> bool:
    """
    Check whether all elements in a list have the same sign. 

    Parameters
    ----------
    lst: list[numbers.Number]
        List of numeric values. Raises error if empty.
    
    Returns
    -------
    bool
        ``True`` if all elements have the same sign as the first element, ``False`` otherwise.
    """
    first_sign = np.sign(lst[0])
    return all(np.sign(x) == first_sign for x in lst)


# ----- Helper functions to convert between optimizer space and physical space -----


def time_to_physical_function(x):
    """
    Convert a time parameter from optimizer space to physical space. This transformation maps unconstrained real values to strictly positive values
    using the exponential function.

    Parameters
    ----------
    x: float | numpy.ndarray
        Time parameter in optimizer space.
    
    Returns
    -------
    float | numpy.ndarray
        Time parameter in physical space.
    """
    return np.exp(x)


def sex_bias_founder_to_physical_function(x):
    """
    Convert a founder event parameters from optimizer space to physical space. 
    This transformation maps unconstrained real values to strictly positive values
    using the exponential function, so that the ancestry of all founder populations sums to one.
    We assume that there is exactly one remainder population

    Parameters
    ----------
    x: float | numpy.ndarray
        the list of parameters in optimizer space. 
        The first half of parameters is the rates, second half is the sex biases. 
    
    Returns
    -------
    float | numpy.ndarray
        same parameters in physical space.
    """
    
    
    # optimizer parameters will be female_rates for populations 1, 2, k, male rates for populations 1,2,k
    # physical parameteres wil be rates for populations 1,2,k, then sex biases for each popualtino ((m_f- m_m)/(m_f + m_m)) 
    n = len(x) // 2
    x_female = x[:n]
    x_male = x[n:]

    # softmax with implicit remainder fixed at logit 0, so source rates sum to <1
    r_f = np.exp(x_female) / (1 + np.sum(np.exp(x_female)))
    r_m = np.exp(x_male) / (1 + np.sum(np.exp(x_male)))

    rates = (r_f + r_m) / 2
    sex_biases = (r_f - r_m) / (r_f + r_m)

    return np.concatenate([rates, sex_biases])


def sex_bias_founder_to_optimizer_function(x):
    """
    Convert founder event parameters from physical space to optimizer space.
    This is the inverse of :func:`sex_bias_founder_to_physical_function`.

    Parameters
    ----------
    x: numpy.ndarray
        The list of parameters in physical space.
        The first half of parameters is the total rates, second half is the sex biases ((r_f - r_m) / (r_f + r_m)).

    Returns
    -------
    numpy.ndarray
        Same parameters in optimizer space.
        The first half is the female rates in optimizer space, second half is the male rates in optimizer space.
    """
    n = len(x) // 2
    rates = x[:n]
    sex_biases = x[n:]

    r_f = rates * (1 + sex_biases)
    r_m = rates * (1 - sex_biases)

    x_female = np.log(r_f / (1 - np.sum(r_f)))
    x_male = np.log(r_m / (1 - np.sum(r_m)))

    return np.concatenate([x_female, x_male])


def rate_to_physical_function(x):
    """
    Convert a rate parameter from optimizer space to physical space. This transformation maps unconstrained real values to the interval ``(0, 1)``
    using the logistic sigmoid function.

    Parameters
    ----------
    x: float | numpy.ndarray
        Rate parameter in optimizer space.
    
    Returns
    -------
    float | numpy.ndarray
        Rate parameter in physical space.
    """
    return expit(x)


def sex_bias_to_physical_function(x):
    """
    Convert a sex-bias parameter from optimizer space to physical space. This transformation maps unconstrained real values to the interval ``(-1, 1)``.

    Parameters
    ----------
    x: float | numpy.ndarray
        Sex-bias parameter in optimizer space.
    
    Returns
    -------
    float | numpy.ndarray
        Sex-bias parameter in physical space.
    """
    return 2 * expit(x) - 1


def time_to_optimizer_function(x):
    """
    Convert a time parameter from physical space to optimizer space. This transformation maps strictly positive values to the real line
    using the natural logarithm.

    Parameters
    ----------
    x: float | numpy.ndarray
        Time parameter in physical space.
    
    Returns
    -------
    float | numpy.ndarray
        Time parameter in optimizer space.
    """
    return np.log(x)


def rate_to_optimizer_function(x):
    """
    Convert a rate parameter from physical space to optimizer space. This transformation maps values in the interval ``(0, 1)`` to the real line
    using the logit function.

    Parameters
    ----------
    x: float | numpy.ndarray
        Rate parameter in physical space.
    
    Returns
    -------
    float | numpy.ndarray
        Rate parameter in optimizer space.
    """
    return logit(x)


def sex_bias_to_optimizer_function(y):
    """
    Convert a sex-bias parameter from physical space to optimizer space. This transformation maps values in the interval ``[-1, 1]`` to the real line
    using the log-ratio transformation:

    .. math::

       \\log(1 + y) - \\log(1 - y).

    Non-finite values (e.g., at the boundaries ``y = ±1``) are replaced by ``±1e32``.

    Parameters
    ----------
    y: float | numpy.ndarray
        Sex-bias parameter in physical space.
    
    Returns
    -------
    float | numpy.ndarray
        Sex-bias parameter in optimizer space.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        log_result = np.log1p(y) - np.log1p(-y)
        log_result = np.where(
            np.isfinite(log_result),
            log_result,
            np.where(
            np.isnan(y), np.nan,
            np.where(y >= 0, 1e32, -1e32)
                )
            )
        return log_result