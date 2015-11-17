""" Constrained optimization of the paraboloid component."""

from __future__ import print_function

from openmdao.api import IndepVarComp, Component, Problem, Group, ExecComp, ScipyOptimizer

class Paraboloid(Component):
    """ Evaluates the equation f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3 """

    def __init__(self):
        super(Paraboloid, self).__init__()

        self.add_param('x', val=0.0)
        self.add_param('y', val=0.0)

        self.add_output('f_xy', val=0.0)

    def solve_nonlinear(self, params, unknowns, resids):
        """f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3"""

        x = params['x']
        y = params['y']

        unknowns['f_xy'] = (x-3.0)**2 + x*y + (y+4.0)**2 - 3.0

    def linearize(self, params, unknowns, resids):
        """ Jacobian for our paraboloid."""

        x = params['x']
        y = params['y']
        J = {}

        J['f_xy', 'x'] = 2.0*x - 6.0 + y
        J['f_xy', 'y'] = 2.0*y + 8.0 + x
        return J

if __name__ == "__main__":

    top = Problem()

    root = top.root = Group()

    root.add('p1', IndepVarComp('x', 3.0))
    root.add('p2', IndepVarComp('y', -4.0))
    root.add('p', Paraboloid())

    # Constraint Equation
    root.add('con', ExecComp('c = x-y'))

    root.connect('p1.x', 'p.x')
    root.connect('p2.y', 'p.y')
    root.connect('p.x', 'con.x')
    root.connect('p.y', 'con.y')

    top.driver = ScipyOptimizer()
    top.driver.options['optimizer'] = 'SLSQP'

    top.driver.add_desvar('p1.x', lower=-50, upper=50)
    top.driver.add_desvar('p2.y', lower=-50, upper=50)
    top.driver.add_objective('p.f_xy')
    top.driver.add_constraint('con.c', lower=15.0)

    top.setup()
    top.run()

    print('\n')
    print('Minimum of %f found at (%f, %f)' % (top['p.f_xy'], top['p.x'], top['p.y']))

    # Expected Output
    # Minimum of -27.083333 found at (7.166667, -7.833333)
