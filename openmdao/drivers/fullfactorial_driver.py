"""
OpenMDAO design-of-experiments driver implementing the Full Factorial method.
"""

from openmdao.drivers.predeterminedruns_driver import PredeterminedRunsDriver
from six import moves, iteritems
import numpy as np
import itertools


class FullFactorialDriver(PredeterminedRunsDriver):
    """Design-of-experiments Driver implementing the Full Factorial method.

    Args
    ----
    num_levels : int, optional
        The number of evenly spaced levels between each design variable
        lower and upper bound. Defaults to 1.

    num_par_doe : int, optional
        The number of DOE cases to run concurrently.  Defaults to 1.

    load_balance : bool, Optional
        If True, use rank 0 as master and load balance cases among all of the
        other ranks. Defaults to False.

    """

    def __init__(self, num_levels=1, num_par_doe=1, load_balance=False):
        super(FullFactorialDriver, self).__init__(num_par_doe=num_par_doe,
                                                  load_balance=load_balance)
        self.num_levels = num_levels

    def _build_runlist(self):
        value_arrays = dict()
        for name, value in iteritems(self.get_desvar_metadata()):
            low = value['lower']
            high = value['upper']
            value_arrays[name] = np.linspace(low, high,
                                             num=self.num_levels).tolist()

        keys = list(value_arrays.keys())
        for combination in itertools.product(*value_arrays.values()):
            yield moves.zip(keys, combination)
