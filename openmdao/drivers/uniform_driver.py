"""
OpenMDAO design-of-experiments driver implementing the Uniform method.
"""

from openmdao.drivers.predeterminedruns_driver import PredeterminedRunsDriver
from six import moves, iteritems
import numpy

class UniformDriver(PredeterminedRunsDriver):
    def __init__(self, *args, **kwargs):
        super(UniformDriver, self).__init__(*args, **kwargs)

    def _build_runlist(self):
        for i in moves.xrange(self.num_steps):
            yield dict(((key, numpy.random.uniform(bound['low'], bound['high'])) for key, bound in iteritems(self.get_desvar_metadata())))
