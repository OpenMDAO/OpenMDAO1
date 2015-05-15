""" Formerly the one-two-one-two-one test. A model that diverges, converges,
diverges, then converges again. """

from openmdao.components.paramcomp import ParamComp
from openmdao.core.component import Component
from openmdao.core.group import Group


class Comp1(Component):

    def __init__(self):
        super(Comp1, self).__init__()
        self.add_param('x1', 1.0)
        self.add_output('y1', 1.0)
        self.add_output('y2', 1.0)

    def solve_nonlinear(self, params, unknowns, resids):
        """ Runs the component"""
        unknowns['y1'] = 2.0*params['x1']**2
        unknowns['y2'] = 3.0*params['x1']

    def jacobian(self, params, unknowns):
        """Returns the Jacobian."""
        J = {}
        J[('y1', 'x1')] = 4.0*params['x1']
        J[('y2', 'x1')] = 3.0
        return J

class Comp2(Component):

    def __init__(self):
        super(Comp2, self).__init__()
        self.add_param('x1', 1.0)
        self.add_output('y1', 1.0)

    def solve_nonlinear(self, params, unknowns, resids):
        """ Runs the component"""
        unknowns['y1'] = 0.5*params['x1']

    def jacobian(self, params, unknowns):
        """Returns the Jacobian."""
        J = {}
        J[('y1', 'x1')] = 0.5
        return J

class Comp3(Component):

    def __init__(self):
        super(Comp3, self).__init__()
        self.add_param('x1', 1.0)
        self.add_output('y1', 1.0)

    def solve_nonlinear(self, params, unknowns, resids):
        """ Runs the component"""
        unknowns['y1'] = 3.5*params['x1']

    def jacobian(self, params, unknowns):
        """Returns the Jacobian."""
        J = {}
        J[('y1', 'x1')] = 3.5
        return J

class Comp4(Component):

    def __init__(self):
        super(Comp4, self).__init__()
        self.add_param('x1', 1.0)
        self.add_param('x2', 1.0)
        self.add_output('y1', 1.0)
        self.add_output('y2', 1.0)

    def solve_nonlinear(self, params, unknowns, resids):
        """ Runs the component"""
        unknowns['y1'] = params['x1'] + 2.0*params['x2']
        unknowns['y2'] = 3.0*params['x1'] - 5.0*params['x2']

    def jacobian(self, params, unknowns):
        """Returns the Jacobian."""
        J = {}
        J[('y1', 'x1')] = 1.0
        J[('y1', 'x2')] = 2.0
        J[('y2', 'x1')] = 3.0
        J[('y2', 'x2')] = -5.0
        return J

class Comp5(Component):

    def __init__(self):
        super(Comp5, self).__init__()
        self.add_param('x1', 1.0)
        self.add_output('y1', 1.0)

    def solve_nonlinear(self, params, unknowns, resids):
        """ Runs the component"""
        unknowns['y1'] = 0.8*params['x1']

    def jacobian(self, params, unknowns):
        """Returns the Jacobian."""
        J = {}
        J[('y1', 'x1')] = 0.8
        return J

class Comp6(Component):

    def __init__(self):
        super(Comp6, self).__init__()
        self.add_param('x1', 1.0)
        self.add_output('y1', 1.0)

    def solve_nonlinear(self, params, unknowns, resids):
        """ Runs the component"""
        unknowns['y1'] = 0.5*params['x1']

    def jacobian(self, params, unknowns):
        """Returns the Jacobian."""
        J = {}
        J[('y1', 'x1')] = 0.5
        return J

class Comp7(Component):

    def __init__(self):
        super(Comp7, self).__init__()
        self.add_param('x1', 1.0)
        self.add_param('x2', 1.0)
        self.add_output('y1', 1.0)

    def solve_nonlinear(self, params, unknowns, resids):
        """ Runs the component"""
        unknowns['y1'] = params['x1'] + 3.0*params['x2']

    def jacobian(self, params, unknowns):
        """Returns the Jacobian."""
        J = {}
        J[('y1', 'x1')] = 1.0
        J[('y1', 'x2')] = 3.0
        return J


class ConvergeDiverge(Group):
    """ Topology one - two - one - two - one. This model was critical in
    testing parallel reverse scatters."""

    def __init__(self):
        super(ConvergeDiverge, self).__init__()

        self.add('comp1', Comp1())
        self.add('comp2', Comp2())
        self.add('comp3', Comp3())
        self.add('comp4', Comp4())
        self.add('comp5', Comp5())
        self.add('comp6', Comp6())
        self.add('comp7', Comp7())

        self.add('p', ParamComp('x', 2.0))

        self.connect("p:x", "comp1:x1")
        self.connect('comp1:y1', 'comp2:x1')
        self.connect('comp1:y2', 'comp3:x1')
        self.connect('comp2:y1', 'comp4:x1')
        self.connect('comp3:y1', 'comp4:x2')
        self.connect('comp4:y1', 'comp5:x1')
        self.connect('comp4:y2', 'comp6:x1')
        self.connect('comp5:y1', 'comp7:x1')
        self.connect('comp6:y1', 'comp7:x2')