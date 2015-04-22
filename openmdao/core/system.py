
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

    def solve_linear(self, params, unknowns, resids, dparams, dunknowns, 
        dresids, mode="fwd"):
        pass

    def apply_linear(self, params, unknowns, resids, dparams, dunknowns, 
        dresids, mode="fwd"):
        pass
