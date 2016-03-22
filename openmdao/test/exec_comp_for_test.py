from __future__ import print_function

import time

from openmdao.components.exec_comp import ExecComp
from openmdao.core.system import AnalysisError
from openmdao.core.mpi_wrap import MPI

class ExecComp4Test(ExecComp):
    """
    A version of ExecComp for benchmarking and testing.

    Args
    ----
    exprs : str or list of str
        The expressions that determine the inputs and outputs of this component.

    nl_delay : float(0.01)
        The sleep time in seconds that will occur when solve_nonlinear is called.

    lin_delay : float(0.01)
        The sleep time in seconds that will occur when apply_linear is called.

    trace : bool(False)
        If True, print some mininal trace information while running.

    rec_procs : tuple of the form (minprocs, maxprocs)
        Minimum and maximun MPI processes usable by this component.

    fails : list or tuple of int
        If the current self.num_nl_solves matches any of these, then this
        component will raise an exception.

    critical : bool(False)
        If True and fails is not empty, this component will raise a
        RuntimeError when a failure is induced. Otherwise, an AnalysisError
        will be raised.
    """
    def __init__(self, exprs, nl_delay=0.01, lin_delay=0.01,
                 trace=False, req_procs=(1,1), fails=(),
                 critical=False, **kwargs):
        super(ExecComp4Test, self).__init__(exprs, **kwargs)
        self.nl_delay = nl_delay
        self.lin_delay = lin_delay
        self.trace = trace
        self.num_nl_solves = 0
        self.num_apply_lins = 0
        self.req_procs = req_procs
        self.fails = fails
        self.critical = critical

        # make a case_rank output so that we can see in tests which rank
        # ran a case
        self.add_output('case_rank', 0, pass_by_obj=True)

    def get_req_procs(self):
        return self.req_procs

    def solve_nonlinear(self, params, unknowns, resids):
        if MPI:
            unknowns['case_rank'] = MPI.COMM_WORLD.rank

        if self.trace:
            print(self.pathname, "solve_nonlinear")
        try:
            super(ExecComp4Test, self).solve_nonlinear(params, unknowns, resids)
            time.sleep(self.nl_delay)
            if self.num_nl_solves in self.fails:
                if self.critical:
                    raise RuntimeError("OMG, a critical error!")
                else:
                    raise AnalysisError("just an analysis error")
        finally:
            self.num_nl_solves += 1

    def apply_linear(self, params, unknowns, dparams, dunknowns, dresids, mode):
        if self.trace:
            print(self.pathname, "apply_linear")
        self._apply_linear_jac(params, unknowns, dparams, dunknowns, dresids, mode)
        time.sleep(self.lin_delay)
        self.num_apply_lins += 1
