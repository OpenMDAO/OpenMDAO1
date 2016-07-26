"""
OpenMDAO driver that runs a user-specified list of cases.
"""

from openmdao.drivers.predeterminedruns_driver import PredeterminedRunsDriver


class CaseDriver(PredeterminedRunsDriver):
    """OpenMDAO driver that runs a sequence of cases.

    Args
    ----
    cases : sequence of cases
        A sequence of cases, where each case is a sequence of (name, value) tuples.

    num_par_doe : int, optional
        The number of cases to run concurrently.  Defaults to 1.

    load_balance : bool, Optional
        If True, use rank 0 as master and load balance cases among all of the
        other ranks. Defaults to False.

    """

    def __init__(self, cases=(), num_par_doe=1, load_balance=False):
        super(CaseDriver, self).__init__(num_par_doe=num_par_doe,
                                         load_balance=load_balance)
        self.cases = cases

    def _build_runlist(self):
        """Yield cases from our sequence of cases."""

        for case in self.cases:
            yield case
