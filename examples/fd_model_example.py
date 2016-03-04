""" This example shows how to finite difference a single component."""

from __future__ import print_function

from openmdao.api import IndepVarComp, Component, Group, Problem

class SimpleComp(Component):
    """ A simple component that provides derivatives. """

    def __init__(self):
        super(SimpleComp, self).__init__()

        # Params
        self.add_param('x', 2.0)

        # Unknowns
        self.add_output('y', 0.0)

        self.print_output = True

    def solve_nonlinear(self, params, unknowns, resids):
        """ Doesn't do much.  Just multiply by 3"""
        unknowns['y'] = 3.0*params['x']
        if self.print_output:
            print('Execute', self.name)

    def linearize(self, params, unknowns, resids):
        """Analytical derivatives."""

        J = {}
        J[('y', 'x')] = 3.0
        if self.print_output:
            print('Calculate Derivatives:', self.name)
        return J


class Model(Group):
    """ Simple model to experiment with finite difference."""

    def __init__(self):
        super(Model, self).__init__()

        self.add('px', IndepVarComp('x', 2.0))

        self.add('comp1', SimpleComp())
        self.add('comp2', SimpleComp())
        self.add('comp3', SimpleComp())
        self.add('comp4', SimpleComp())

        self.connect('px.x', 'comp1.x')
        self.connect('comp1.y', 'comp2.x')
        self.connect('comp2.y', 'comp3.x')
        self.connect('comp3.y', 'comp4.x')

        # Tell these whole model to finite difference
        self.fd_options['force_fd'] = True

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
