"""
OpenMDAO design-of-experiments driver implementing the Uniform method.
"""

from openmdao.drivers.predeterminedruns_driver import PredeterminedRunsDriver
from six import moves, iteritems
import numpy as np


class UniformDriver(PredeterminedRunsDriver):
    """Design-of-experiments Driver implementing the Uniform method.
    """

    def __init__(self, num_samples=1):
        super(UniformDriver, self).__init__()
        self.num_samples = num_samples

    def _build_runlist(self):
        """Build a runlist based on a uniform distribution."""

        for i in moves.xrange(self.num_samples):
            yield dict(((key, np.random.uniform(bound['lower'], bound['upper']))
                        for key, bound in iteritems(self.get_desvar_metadata())))
