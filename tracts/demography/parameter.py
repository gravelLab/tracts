from __future__ import annotations
from enum import Enum
from collections.abc import Callable
from typing import TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from tracts.demography.base_parametrized_demography import BaseParametrizedDemography

logger = logging.getLogger(__name__)

small = 10**-9  
large = 1/small
class ParamType(Enum):
    """
    A class representing the type of a parameter in a demographic model.

    Attributes
    ----------
    TIME : tuple[float, float]
        The bounds for time parameters, which must be positive and can be arbitrarily large.
    RATE : tuple[float, float]
        The bounds for rate parameters, which must be between 0 and 1.
    SEX_BIAS : tuple[float, float]
        The bounds for sex bias parameters, which must be between -1 and 1.
    UNTYPED : tuple[float, float]
        The bounds for untyped parameters, which can be arbitrarily large in either direction.
    bounds : tuple[float, float]
        The lower and upper bounds for the parameter type.
    """
    TIME=(small,large)
    RATE=(small,1-small)
    SEX_BIAS=(-1,1)
    UNTYPED=(-large,large)

    def __init__(self, lower_bound: float, upper_bound: float):
        """
        Initializes a ParamType object.

        Parameters
        ----------
        lower_bound : float
            The lower bound for the parameter type.
        upper_bound : float
            The upper bound for the parameter type.
        """
        self.bounds=(lower_bound, upper_bound)

class Parameter:
    """
    A class representing a parameter in a demographic model.

    Attributes
    ----------
    name : str
        The name of the parameter.
    type : ParamType
        The type of the parameter, which determines its bounds.
    bounds : tuple[float, float]
        The lower and upper bounds of the parameter, determined by its type.
    index : int | None
        The index of the parameter in the list of parameters to optimize, or None if the parameter is fixed by value or ancestry proportion.
    """
    def __init__(self, name: str, param_type: ParamType, bounds: tuple[float, float], index: int | None = None):
        """
        Initializes a Parameter object.

        Parameters
        ----------
        name : str
            The name of the parameter.
        param_type : ParamType
            The type of the parameter.
        bounds : tuple[float, float]
            The lower and upper bounds of the parameter.
        index : int | None, optional
            The index of the parameter in the list of parameters to optimize, or None if the parameter is fixed by value or ancestry proportion. Default is None.
        """
        self.name=name
        self.type=param_type
        self.bounds=bounds
        self.index=index
        return

class DependentParameter(Parameter, Callable):
    """
    A class representing a parameter in a demographic model that is dependent on other parameters.

    Attributes
    ----------
    name : str
        The name of the parameter.
    type : ParamType
        The type of the parameter, which determines its bounds.
    bounds : tuple[float, float]
        The lower and upper bounds of the parameter, determined by its type.
    expression : Callable[[BaseParametrizedDemography, list[float]], float]
        A function that computes the value of the parameter given a demography and a list of its dependent parameter values.
    index : int | None
        The index of the parameter in the list of parameters to optimize, or None if the parameter is fixed by value or ancestry proportion.
    """
    def __init__(self, name: str, expression: Callable, param_type: ParamType, bounds: tuple[float, float], index: int | None = None):
        """
        Initializes a DependentParameter object.

        Parameters
        ----------
        name : str
            The name of the parameter.
        expression : Callable[[BaseParametrizedDemography, list[float]], float]
            A function that computes the value of the parameter given a demography and a list of its dependent parameter values.
        param_type : ParamType
            The type of the parameter, which determines its bounds.
        bounds : tuple[float, float]
            The lower and upper bounds of the parameter, determined by its type.
        index : int | None, optional
            The index of the parameter in the list of parameters to optimize, or None if the parameter is fixed by value or ancestry proportion. Default is None.
        """
        super().__init__(name=name,
                        param_type=param_type,
                        bounds=bounds,
                        index=index)
        self.expression = expression
        return
        
    def __call__(self, demography: BaseParametrizedDemography, params: list[float]):
        """
        Computes the value of the parameter given a demography and a list of its dependent parameter values.
        Parameters
        ----------
        demography : BaseParametrizedDemography
            The demographic model that this parameter is associated with.
        params : list[float]
            The list of parameter values for the parameters that this parameter is dependent on, in the same order as they are defined in the demographic model.
        
        Returns
        -------
        float
            The value of the parameter computed from its expression given the demography and the values of its dependent parameters.
        """
        return self.expression(demography, params)


