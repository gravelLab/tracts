from enum import Enum

class ParamType(Enum):
    TIME=(None,None)
    RATE=(0,1)
    SEX_BIAS=(-1,1)

    def __init__(self, lower_bound, upper_bound):
        self.bounds=(lower_bound, upper_bound)