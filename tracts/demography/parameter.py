from enum import Enum
import numpy
from collections.abc import Callable

class ParamType(Enum):
    TIME=(0,numpy.inf)
    RATE=(0,1)
    SEX_BIAS=(-1,1)
    UNTYPED=(-numpy.inf,numpy.inf)

    def __init__(self, lower_bound, upper_bound):
        self.bounds=(lower_bound, upper_bound)


class Parameter:
    def __init__(self, name: str, param_type: ParamType, bound: tuple[float, float], index=None):
        self.name=name
        self.param_type=param_type
        self.bound=bound
        self.index=index
        return

class DependentParameter(Parameter, Callable):
    def __init__(self, name: str, expression: Callable, param_type: ParamType, bound: tuple[float, float], index=None):
        super().__init__(name, param_type, bound, index)
        self.expression = expression
        return
        
    def __call__(self, demography, params):
        return self.expression(demography, params)


