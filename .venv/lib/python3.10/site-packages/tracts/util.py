import sys
from numbers import Number

import numpy as np


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def all_same_sign(lst: list[Number]) -> bool:
    first_sign = np.sign(lst[0])
    return all(np.sign(x) == first_sign for x in lst)
