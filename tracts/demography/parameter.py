from enum import Enum
import numpy
from collections.abc import Callable

small = 10**-9  
large = 1/small

class ParamType(Enum):
    
    TIME=(small,large)
    RATE=(small,1-small)
    SEX_BIAS=(-1+small,1-small)
    UNTYPED=(-large,large)

    def __init__(self, lower_bound, upper_bound):
        self.bounds=(lower_bound, upper_bound)


class Parameter:
    def __init__(self, name: str, param_type: ParamType, bounds: tuple[float, float], index=None):
        self.name=name
        self.type=param_type
        self.bounds=bounds
        self.index=index
        return

class DependentParameter(Parameter, Callable):
    def __init__(self, name: str, expression: Callable, param_type: ParamType, bounds: tuple[float, float], index=None):
        super().__init__(name, param_type, bounds, index)
        self.expression = expression
        return
        
    def __call__(self, demography, params):
        return self.expression(demography, params)


