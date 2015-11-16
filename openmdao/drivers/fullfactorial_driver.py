"""
OpenMDAO design-of-experiments driver implementing the Full Factorial method.
"""

from openmdao.drivers.predeterminedruns_driver import PredeterminedRunsDriver
from six import moves, iteritems
import numpy as np
import itertools


class FullFactorialDriver(PredeterminedRunsDriver):
    """Design-of-experiments Driver implementing the Full Factorial method.
    """

    def __init__(self, num_levels=1):
        super(FullFactorialDriver, self).__init__()
        self.num_levels = num_levels

    def _build_runlist(self):
        value_arrays = dict()
        for name, value in iteritems(self.get_desvar_metadata()):
            low = value['lower']
            high = value['upper']
            value_arrays[name] = np.linspace(low, high, num=self.num_levels).tolist()

        keys = list(value_arrays.keys())
        for combination in itertools.product(*value_arrays.values()):
            yield dict(moves.zip(keys, combination))
