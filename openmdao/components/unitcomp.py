from openmdao.core.component import Component
import numpy as np

class UnitComp(Component):
    """
    A Component that converts the input into the requested units.
    """

    def __init__(self, shape, param_name, out_name, units):
        super(UnitComp, self).__init__()

        self.param_name = param_name
        self.out_name = out_name

        if param_name == out_name:
            msg = "UnitComp param_name cannot match out_name: '{name}'"
            raise ValueError(msg.format(name=param_name))

        self.add_param(param_name, shape=shape, units=units)
        self.add_output(out_name, shape=shape, units=units)

    def solve_nonlinear(self, params, unknowns, resids):
        """ No action."""
        unknowns[self.out_name] = params[self.param_name]

    def jacobian(self, params, unknowns, resids):
        """ Derivative is 1.0"""
        J = {}
        J[(self.out_name, self.param_name)] = np.array([1.0])
        return J
