# Solves the intersection of a line with a parabola

from __future__ import print_function

from openmdao.api import Component, Group, Problem, Newton, ScipyGMRES


class Line(Component):
    """Evaluates y = -2x + 4."""

    def __init__(self):
        super(Line, self).__init__()

        self.add_param('x', 1.0)
        self.add_output('y', 0.0)

        # User can change these.
        self.slope = -2.0
        self.intercept = 4.0

    def solve_nonlinear(self, params, unknowns, resids):
        """ y = -2x + 4 """

        x = params['x']
        m = self.slope
        b = self.intercept

        unknowns['y'] = m*x + b

    def linearize(self, params, unknowns, resids):
        """ Jacobian for our line."""

        m = self.slope
        J = {}

        J['y', 'x'] = m
        return J


class Parabola(Component):
    """Evaluates y = 3x^2 - 5"""

    def __init__(self):
        super(Parabola, self).__init__()

        self.add_param('x', 1.0)
        self.add_output('y', 0.0)

        # User can change these.
        self.a = 3.0
        self.b = 0.0
        self.c = -5.0

    def solve_nonlinear(self, params, unknowns, resids):
        """ y = 3x^2 - 5 """

        x = params['x']
        a = self.a
        b = self.b
        c = self.c

        unknowns['y'] = a*x**2 + b*x + c

    def linearize(self, params, unknowns, resids):
        """ Jacobian for our parabola."""

        x = params['x']
        a = self.a
        b = self.b
        J = {}

        J['y', 'x'] = 2.0*a*x + b
        return J


class Balance(Component):
    """Evaluates the residual y1-y2"""

    def __init__(self):
        super(Balance, self).__init__()

        self.add_param('y1', 0.0)
        self.add_param('y2', 0.0)
        self.add_state('x', 5.0)

    def solve_nonlinear(self, params, unknowns, resids):
        """This component does no calculation on its own. It mainly holds the
        initial value of the state. An OpenMDAO solver outside of this
        component varies it to drive the residual to zero."""
        pass

    def apply_nonlinear(self, params, unknowns, resids):
        """ Report the residual y1-y2 """

        y1 = params['y1']
        y2 = params['y2']

        resids['x'] = y1 - y2

    def linearize(self, params, unknowns, resids):
        """ Jacobian for our parabola."""

        J = {}
        J['x', 'y1'] = 1.0
        J['x', 'y2'] = -1.0
        return J

if __name__ == '__main__':

    top = Problem()
    root = top.root = Group()
    root.add('line', Line())
    root.add('parabola', Parabola())
    root.add('bal', Balance())

    root.connect('line.y', 'bal.y1')
    root.connect('parabola.y', 'bal.y2')
    root.connect('bal.x', 'line.x')
    root.connect('bal.x', 'parabola.x')

    root.nl_solver = Newton()
    root.ln_solver = ScipyGMRES()

    top.setup()

    # Positive solution
    top['bal.x'] = 7.0
    root.list_states()
    top.run()
    print('Positive Solution x=%f, line.y=%f, parabola.y=%f' % (top['bal.x'], top['line.y'], top['parabola.y']))

    # Negative solution
    top['bal.x'] = -7.0
    root.list_states()
    top.run()
    print('Negative Solution x=%f, line.y=%f, parabola.y=%f' % (top['bal.x'], top['line.y'], top['parabola.y']))
