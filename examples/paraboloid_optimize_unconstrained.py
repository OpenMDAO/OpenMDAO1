""" Simple example with the paraboloid component."""

from openmdao.components.paramcomp import ParamComp
from openmdao.core.component import Component
from openmdao.core.problem import Problem, Group


class Paraboloid(Component):
    """ Evaluates the equation f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3 """

    def __init__(self):
        super(Paraboloid, self).__init__()

        self.add_param('x', val=0.0)
        self.add_param('y', val=0.0)

        self.add_output('f_xy', val=0.0)

    def solve_nonlinear(self, params, unknowns, resids):
        """f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3
        Optimal solution (minimum): x = 6.6667; y = -7.3333
        """

        x = params['x']
        y = params['y']

        unknowns['f_xy'] = (x-3.0)**2 + x*y + (y+4.0)**2 - 3.0

    def jacobian(self, params, unknowns, resids):
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

    root.add('p1', ParamComp('x', 3.0))
    root.add('p2', ParamComp('y', -4.0))
    root.add('p', Paraboloid())

    root.connect('p1.x', 'p.x')
    root.connect('p2.y', 'p.y')

    top.setup()
    top.run()

    print root.p.unknowns['f_xy']