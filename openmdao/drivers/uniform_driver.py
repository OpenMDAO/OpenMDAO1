"""
OpenMDAO design-of-experiments driver implementing the Uniform method.
"""

from openmdao.drivers.predeterminedruns_driver import PredeterminedRunsDriver
from six import moves, iteritems
import numpy as np


class UniformDriver(PredeterminedRunsDriver):
    """Design-of-experiments Driver implementing the Uniform method.

    Args
    ----
    num_samples : int, optional
        The number of samples to run. Defaults to 1.

    num_par_doe : int, optional
        The number of DOE cases to run concurrently.  Defaults to 1.

    load_balance : bool, Optional
        If True, use rank 0 as master and load balance cases among all of the
        other ranks. Defaults to False.

    """

    def __init__(self, num_samples=1, num_par_doe=1, load_balance=False):
        super(UniformDriver, self).__init__(num_par_doe=num_par_doe,
                                            load_balance=load_balance)
        self.num_samples = num_samples

    def _build_runlist(self):
        """Build a runlist based on a uniform distribution."""

        bounds = dict()
        for name, meta in iteritems(self.get_desvar_metadata()):

            # Support for array desvars
            val = self.root.unknowns._dat[name].val
            nval = len(val)

            for k in range(nval):

                low = meta['lower']
                high = meta['upper']
                if isinstance(low, np.ndarray):
                    low = low[k]
                if isinstance(high, np.ndarray):
                    high = high[k]

                bounds[(name, k)] = np.linspace(low, high)

        for i in moves.range(self.num_samples):
            yield ((key, np.random.uniform(bound[0], bound[1]))
                        for key, bound in iteritems(bounds))
