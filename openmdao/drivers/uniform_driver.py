"""
OpenMDAO design-of-experiments driver implementing the Uniform method.
"""

from openmdao.drivers.predeterminedruns_driver import PredeterminedRunsDriver
import six.moves
import numpy

class UniformDriver(PredeterminedRunsDriver):
    def __init__(self, *args, **kwargs):
        super(UniformDriver, self).__init__(*args, **kwargs)

    def _build_runlist(self):
        for i in six.moves.xrange(self.num_steps):
            yield dict(((key, numpy.random.uniform(bound['low'], bound['high'])) for key, bound in self.get_desvar_metadata().iteritems()))
