"""
OpenMDAO design-of-experiments driver implementing the Full Factorial method.
"""

from openmdao.drivers.predeterminedruns_driver import PredeterminedRunsDriver
from six import moves, iteritems
import numpy
import itertools

class FullFactorialDriver(PredeterminedRunsDriver):
    def __init__(self, *args, **kwargs):
        super(FullFactorialDriver, self).__init__(*args, **kwargs)

    def _build_runlist(self):
        value_arrays = dict()
        for name, value in iteritems(self.get_desvar_metadata()):
            low = value["low"]
            high = value["high"]
            value_arrays[name] = numpy.linspace(low, high, num=self.num_steps).tolist()
        
        keys = list(value_arrays.keys())
        for combination in itertools.product(*value_arrays.values()):
            yield dict(moves.zip(keys, combination))
