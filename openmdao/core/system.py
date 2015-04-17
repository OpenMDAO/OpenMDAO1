
class System(object):
    def __init__(self):
        self.name = ''

    def preconditioner(self):
        pass

    def jacobian(self, params, unknowns):
        pass

    def solve_nonlinear(self, params, unknowns, resids):
        pass

    def apply_nonlinear(self, params, unknowns, resids):
        pass

    def solve_linear(self, mode="fwd", params, unknowns, resids,
                                       dparams, dunknowns, dresids):
        pass

    def apply_linear(self, mode="fwd", params, unknowns, resids,
                                       dparams, dunknowns, dresids):
        pass
