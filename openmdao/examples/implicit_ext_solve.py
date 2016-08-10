# Simple implicit component example. OpenMDAO solves it.

from __future__ import print_function

import numpy as np

from openmdao.api import Component, Group, Problem, ScipyGMRES, Newton, ExecComp, IndepVarComp

class SimpleImplicitComp(Component):
    """ A Simple Implicit Component with an additional output equation.

    f(x,z) = xz + z - 4
    y = x + 2z

    Sol: when x = 0.5, z = 2.666

    Coupled derivs:

    y = x + 8/(x+1)
    dy_dx = 1 - 8/(x+1)**2 = -2.5555555555555554

    z = 4/(x+1)
    dz_dx = -4/(x+1)**2 = -1.7777777777777777
    """

    def __init__(self):
        super(SimpleImplicitComp, self).__init__()

        # Params
        self.add_param('x', 0.5)

        # Unknowns
        self.add_output('y', 0.0)

        # States
        self.add_state('z', 0.0)

    def solve_nonlinear(self, params, unknowns, resids):
        """ Simple iterative solve. (Babylonian method)."""

        x = params['x']
        z = unknowns['z']
        unknowns['y'] = x + 2.0*z

    def apply_nonlinear(self, params, unknowns, resids):
        """ Don't solve; just calculate the residual."""

        x = params['x']
        z = unknowns['z']
        resids['z'] = x*z + z - 4.0

        # Output equations need to evaluate a residual just like an explicit comp.
        resids['y'] = x + 2.0*z - unknowns['y']

    def linearize(self, params, unknowns, resids):
        """Analytical derivatives."""

        J = {}

        # Output equation
        J[('y', 'x')] = np.array([1.0])
        J[('y', 'z')] = np.array([2.0])

        # State equation
        J[('z', 'z')] = np.array([params['x'] + 1.0])
        J[('z', 'x')] = np.array([unknowns['z']])

        return J

if __name__ == '__main__':

    top = Problem()
    root = top.root = Group()
    root.add('p1', IndepVarComp('x', 0.5))
    root.add('comp', SimpleImplicitComp())
    root.add('comp2', ExecComp('zz = 2.0*z'))

    root.connect('p1.x', 'comp.x')
    root.connect('comp.z', 'comp2.z')

    root.ln_solver = ScipyGMRES()
    root.nl_solver = Newton()
    top.setup()
    top.print_all_convergence()

    top.run()

    print('Solution: x = %f, z = %f, y = %f' % (top['comp.x'], top['comp.z'], top['comp.y']))
