"""
OpenMDAO design-of-experiments driver implementing the Full Factorial method.
"""

from openmdao.drivers.predeterminedruns_driver import PredeterminedRunsDriver
import six.moves
import numpy
import itertools

class FullFactorialDriver(PredeterminedRunsDriver):
    def __init__(self, *args, **kwargs):
        super(FullFactorialDriver, self).__init__(*args, **kwargs)

    def _build_runlist(self):
        # Set up Uniform distribution arrays
        value_arrays = dict()
        for name, value in self.get_desvar_metadata().iteritems():
            low = value["low"]
            high = value["high"]
            value_arrays[name] = numpy.linspace(low, high, num=self.num_steps).tolist()
        # log["arrays"] = value_arrays

        keys = list(value_arrays.keys())
        for combination in itertools.product(*value_arrays.values()):
            yield dict(six.moves.zip(keys, combination))
