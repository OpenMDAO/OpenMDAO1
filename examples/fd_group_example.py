""" This example shows how to finite difference a group of components."""

from __future__ import print_function

from openmdao.components.paramcomp import ParamComp
from openmdao.core.component import Component
from openmdao.core.group import Group
from openmdao.core.problem import Problem


class SimpleComp(Component):
    """ A simple component that provides derivatives. """

    def __init__(self):
        super(SimpleComp, self).__init__()

        # Params
        self.add_param('x', 2.0)

        # Unknowns
        self.add_output('y', 0.0)

    def solve_nonlinear(self, params, unknowns, resids):
        """ Doesn't do much.  Just multiply by 3"""
        unknowns['y'] = 3.0*params['x']
        print('Execute', self.name)

    def jacobian(self, params, unknowns, resids):
        """Analytical derivatives."""

        J = {}
        J[('y', 'x')] = 3.0
        print('Calculate Derivatives:', self.name)
        return J


class Model(Group):
    """ Simple model to experiment with finite difference."""

    def __init__(self):
        super(Model, self).__init__()

        self.add('px', ParamComp('x', 2.0))

        self.add('comp1', SimpleComp())

        # 2 and 3 are in a sub Group
        sub = self.add('sub', Group())
        sub.add('comp2', SimpleComp())
        sub.add('comp3', SimpleComp())

        self.add('comp4', SimpleComp())

        self.connect('px.x', 'comp1.x')
        self.connect('comp1.y', 'sub.comp2.x')
        self.connect('sub.comp2.y', 'sub.comp3.x')
        self.connect('sub.comp3.y', 'comp4.x')

        # Tell the group with comps 2 and 3 to finite difference
        self.sub.fd_options['force_fd'] = True
        self.sub.fd_options['step_size'] = 1.0e-4


if __name__ == '__main__':
    # Setup and run the model.

    top = Problem()
    top.root = Model()

    top.setup()
    top.run()

    print('\n\nStart Calc Gradient')
    print ('-'*25)

    J = top.calc_gradient(['px.x'], ['comp4.y'])
    print(J)