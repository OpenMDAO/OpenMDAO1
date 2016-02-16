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

    seed : iint, optional
        Seed for randon number generator.

    num_par_doe : int, optional
        The number of DOE cases to run concurrently.  Defaults to 1.

    load_balance : bool, Optional
        If True, use rank 0 as master and load balance cases among all of the
        other ranks. Defaults to False.

    """

    def __init__(self, num_samples=1, seed=None, num_par_doe=1, load_balance=False):
        super(UniformDriver, self).__init__(num_par_doe=num_par_doe,
                                            load_balance=load_balance)
        self.num_samples = num_samples
        self.seed = seed

    def _build_runlist(self):
        """Build a runlist based on a uniform distribution."""

        if self.seed is not None:
            np.random.seed(self.seed)

        for i in moves.range(self.num_samples):
            sample = []
            for key, meta in iteritems(self.get_desvar_metadata()):
                nval = meta['size']
                values = []
                for k in range(nval):

                    low = meta['lower']
                    high = meta['upper']
                    if isinstance(low, np.ndarray):
                        low = low[k]
                    if isinstance(high, np.ndarray):
                        high = high[k]

                    values.append(np.random.uniform(low, high))
                sample.append([key, np.array(values)])

            yield sample
