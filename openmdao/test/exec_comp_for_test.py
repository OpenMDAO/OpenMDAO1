from __future__ import print_function

import time

from openmdao.components.exec_comp import ExecComp

class ExecComp4Test(ExecComp):
    """
    A version of ExecComp for benchmarking.
    """
    def __init__(self, exprs, nl_delay=0.01, lin_delay=0.01,
                 trace=False, req_procs=(1,1), **kwargs):
        super(ExecComp4Test, self).__init__(exprs, **kwargs)
        self.nl_delay = nl_delay
        self.lin_delay = lin_delay
        self.trace = trace
        self.num_nl_solves = 0
        self.num_apply_lins = 0
        self.req_procs = req_procs

    def get_req_procs(self):
        return self.req_procs

    def solve_nonlinear(self, params, unknowns, resids):
        if self.trace:
            print(self.pathname, "solve_nonlinear")
        super(ExecComp4Test, self).solve_nonlinear(params, unknowns, resids)
        time.sleep(self.nl_delay)
        self.num_nl_solves += 1

    def apply_linear(self, params, unknowns, dparams, dunknowns, dresids, mode):
        if self.trace:
            print(self.pathname, "apply_linear")
        self._apply_linear_jac(params, unknowns, dparams, dunknowns, dresids, mode)
        time.sleep(self.lin_delay)
        self.num_apply_lins += 1
