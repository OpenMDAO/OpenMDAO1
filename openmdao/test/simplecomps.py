""" Some simple test components. """

import numpy as np
import scipy.sparse

from openmdao.components.paramcomp import ParamComp
from openmdao.core.component import Component
from openmdao.core.group import Group


class SimpleComp(Component):
    """ The simplest component you can imagine. """

    def __init__(self):
        super(SimpleComp, self).__init__()

        # Params
        self.add_param('x', 3.0)

        # Unknowns
        self.add_output('y', 5.5)

    def solve_nonlinear(self, params, unknowns, resids):
        """ Doesn't do much. """

        unknowns['y'] = 2.0*params['x']


class SimpleCompDerivMatVec(SimpleComp):
    """ The simplest component you can imagine, this time with derivatives
    defined using apply_linear. """

    def apply_linear(self, params, unknowns, dparams, dunknowns, dresids,
                     mode):
        """Returns the product of the incoming vector with the Jacobian."""

        if mode == 'fwd':
            dresids['y'] = 2.0*dparams['x']

        elif mode == 'rev':
            dparams['x'] = 2.0*dresids['y']



class SimpleCompDerivJac(SimpleComp):
    """ The simplest component you can imagine, this time with derivatives
    defined using Jacobian to return a jacobian. """

    def jacobian(self, params, unknowns, resids):
        """Returns the Jacobian."""

        J = {}
        J[('y', 'x')] = np.array([2.0])
        return J


class SimpleArrayComp(Component):
    '''A fairly simple array component'''

    def __init__(self):
        super(SimpleArrayComp, self).__init__()

        # Params
        self.add_param('x', np.zeros([2]))

        # Unknowns
        self.add_output('y', np.zeros([2]))

    def solve_nonlinear(self, params, unknowns, resids):
        """ Doesn't do much. """

        unknowns['y'][0] = 2.0*params['x'][0] + 7.0*params['x'][1]
        unknowns['y'][1] = 5.0*params['x'][0] - 3.0*params['x'][1]
        # print(self.name, "ran", params['x'], unknowns['y'])

    def jacobian(self, params, unknowns, resids):
        """Analytical derivatives"""

        dy1_dx1 = 2.0
        dy1_dx2 = 7.0
        dy2_dx1 = 5.0
        dy2_dx2 = -3.0
        J = {}
        J[('y', 'x')] = np.array([[dy1_dx1, dy1_dx2], [dy2_dx1, dy2_dx2]])

        return J


class ArrayComp2D(Component):
    '''2D Array component'''

    def __init__(self):
        super(ArrayComp2D, self).__init__()

        # Params
        self.add_param('x', np.zeros((2, 2)))

        # Unknowns
        self.add_output('y', np.zeros((2, 2)))

    def solve_nonlinear(self, params, unknowns, resids):
        """ Doesn't do much. """

        x = params['x']
        y = np.zeros((2, 2))

        y[0][0] = 2.0*x[0][0] + 1.0*x[0][1] + \
                  3.0*x[1][0] + 7.0*x[1][1]

        y[0][1] = 4.0*x[0][0] + 2.0*x[0][1] + \
                  6.0*x[1][0] + 5.0*x[1][1]

        y[1][0] = 3.0*x[0][0] + 6.0*x[0][1] + \
                  9.0*x[1][0] + 8.0*x[1][1]

        y[1][1] = 1.0*x[0][0] + 3.0*x[0][1] + \
                  2.0*x[1][0] + 4.0*x[1][1]

        unknowns['y'] = y

    def jacobian(self, params, unknowns, resids):
        """Analytical derivatives"""

        J = {}
        J['y', 'x'] = np.array([[2.0, 1.0, 3.0, 7.0],
                                [4.0, 2.0, 6.0, 5.0],
                                [3.0, 6.0, 9.0, 8.0],
                                [1.0, 3.0, 2.0, 4.0]])
        return J


class SimpleSparseArrayComp(Component):
    '''A fairly simple sparse array component'''

    def __init__(self):
        super(SimpleSparseArrayComp, self).__init__()

        # Params
        self.add_param('x', np.zeros([4]))

        # Unknowns
        self.add_output('y', np.zeros([4]))

    def solve_nonlinear(self, params, unknowns, resids):
        """ Doesn't do much. """

        unknowns['y'][0] = 2.0*params['x'][0] + 7.0*params['x'][3]
        unknowns['y'][2] = 5.0*params['x'][1] - 3.0*params['x'][2]
        # print(self.name, "ran", params['x'], unknowns['y'])

    def jacobian(self, params, unknowns, resids):
        """Analytical derivatives"""

        dy1_dx1 = 2.0
        dy1_dx2 = 7.0
        dy2_dx1 = 5.0
        dy2_dx2 = -3.0
        data = [dy1_dx1, dy1_dx2, dy2_dx1, dy2_dx2]
        row = [0, 0, 2, 2]
        col = [0, 3, 1, 2]
        J = {}
        J[('y', 'x')] = scipy.sparse.csc_matrix((data, (row, col)),
                                                shape=(4, 4))

        return J


class SimpleImplicitComp(Component):
    """ A Simple Implicit Component with an additional output equation.

    f(x,z) = xz + z - 4
    y = x + 2z

    Sol: when x = 0.5, z = 2.666
    """

    def __init__(self):
        super(SimpleImplicitComp, self).__init__()

        # Params
        self.add_param('x', 0.5, low=0.01, high=1.0)

        # Unknowns
        self.add_output('y', 0.0)

        # States
        self.add_state('z', 0.0)

        self.maxiter = 10
        self.atol = 1.0e-6

    def solve_nonlinear(self, params, unknowns, resids):
        """ Simple iterative solve. (Babylonian method) """

        x = params['x']
        z = unknowns['z']
        znew = z

        iter = 0
        eps = 1.0e99
        while iter < self.maxiter and abs(eps) > self.atol:
            z = znew
            znew = 4.0 - x*z

            eps = x*znew + znew - 4.0

        unknowns['z'] = znew
        unknowns['y'] = x + 2.0*znew

        resids['z'] = eps

    def apply_nonlinear(self, params, unknowns, resids):
        """ Don't solve; just calculate the redisual. """

        x = params['x']
        z = unknowns['z']
        resids['z'] = x*z + z - 4.0

        # Output equations need to evaluate a residual just like an explicit comp.
        resids['y'] = x + 2.0*z - unknowns['y']

    def jacobian(self, params, unknowns, resids):
        """Analytical derivatives"""

        J = {}

        # Output equation
        J[('y', 'x')] = np.array([1.0])
        J[('y', 'z')] = np.array([2.0])

        # State equation
        J[('z', 'z')] = np.array([params['x'] + 1.0])
        J[('z', 'x')] = np.array([unknowns['z']])

        return J


class SimpleNoflatComp(Component):
    """ The simplest component you can imagine. """

    def __init__(self):
        super(SimpleNoflatComp, self).__init__()

        # Params
        self.add_param('x', '')

        # Unknowns
        self.add_output('y', '')

    def solve_nonlinear(self, params, unknowns, resids):
        """ Doesn't do much. """

        unknowns['y'] = params['x']+self.name


class FanOutComp1(Component):

    def __init__(self):
        super(FanOutComp1, self).__init__()

        # Params
        self.add_param('x', 0.0)

        # Unknowns
        self.add_output('y', 0.0)

    def solve_nonlinear(self, params, unknowns, resids):
        """ Doesn't do much. """

        unknowns['y'] = 3.0*params['x']

    def jacobian(self, params, unknowns, resids):
        """Analytical derivatives"""
        J = {}
        J[('y', 'x')] = np.array([3.0])
        return J


class FanOutComp2(Component):

    def __init__(self):
        super(FanOutComp2, self).__init__()

        # Params
        self.add_param('x', 0.0)

        # Unknowns
        self.add_output('y', 0.0)

    def solve_nonlinear(self, params, unknowns, resids):
        """ Doesn't do much. """

        unknowns['y'] = -2.0*params['x']

    def jacobian(self, params, unknowns, resids):
        """Analytical derivatives"""
        J = {}
        J[('y', 'x')] = np.array([-2.0])
        return J


class FanOutComp3(Component):

    def __init__(self):
        super(FanOutComp3, self).__init__()

        # Params
        self.add_param('x', 0.0)

        # Unknowns
        self.add_output('y', 0.0)

    def solve_nonlinear(self, params, unknowns, resids):
        """ Doesn't do much. """

        unknowns['y'] = 5.0*params['x']

    def jacobian(self, params, unknowns, resids):
        """Analytical derivatives"""
        J = {}
        J[('y', 'x')] = np.array([5.0])
        return J

class FanInTarget(Component):

    def __init__(self):
        super(FanInTarget, self).__init__()

        # Params
        self.add_param('x1', 0.0)
        self.add_param('x2', 0.0)

        # Unknowns
        self.add_output('y', 0.0)

    def solve_nonlinear(self, params, unknowns, resids):
        """ Doesn't do much. """

        unknowns['y'] = 3.0*params['x1'] * 7.0*params['x2']

    def jacobian(self, params, unknowns, resids):
        """Analytical derivatives"""
        J = {}
        J[('y', 'x1')] = np.array([3.0])
        J[('y', 'x2')] = np.array([7.0])
        return J


class FanOut(Group):
    """ Topology where one comp broadcasts an output to two target
    components."""

    def __init__(self):
        super(FanOut, self).__init__()

        self.add('comp1', FanOutComp1())
        self.add('comp2', FanOutComp2())
        self.add('comp3', FanOutComp3())
        self.add('p', ParamComp('x', 1.0))

        self.connect("comp1:y", "comp2:x")
        self.connect("comp1:y", "comp3:x")
        self.connect("p:x", "comp1:x")


class FanOutGrouped(Group):
    """ Topology where one comp broadcasts an output to two target
    components."""

    def __init__(self):
        super(FanOutGrouped, self).__init__()

        sub = self.add('sub', Group())
        self.add('comp1', FanOutComp1())
        sub.add('comp2', FanOutComp2())
        sub.add('comp3', FanOutComp3())
        self.add('p', ParamComp('x', 1.0))

        self.connect("comp1:y", "sub:comp2:x")
        self.connect("comp1:y", "sub:comp3:x")
        self.connect("p:x", "comp1:x")


class FanIn(Group):
    """ Topology where two comp feed a single comp."""

    def __init__(self):
        super(FanIn, self).__init__()

        self.add('comp1', FanOutComp2())
        self.add('comp2', FanOutComp3())
        self.add('comp3', FanInTarget())
        self.add('p1', ParamComp('x1', 1.0))
        self.add('p2', ParamComp('x2', 1.0))

        self.connect("comp1:y", "comp3:x1")
        self.connect("comp2:y", "comp3:x2")
        self.connect("p1:x1", "comp1:x")
        self.connect("p2:x2", "comp2:x")


class FanInGrouped(Group):
    """ Topology where two comp feed a single comp."""

    def __init__(self):
        super(FanInGrouped, self).__init__()

        sub = self.add('sub', Group()
                       )
        sub.add('comp1', FanOutComp2())
        sub.add('comp2', FanOutComp3())
        self.add('comp3', FanInTarget())
        self.add('p1', ParamComp('x1', 1.0))
        self.add('p2', ParamComp('x2', 1.0))

        self.connect("sub:comp1:y", "comp3:x1")
        self.connect("sub:comp2:y", "comp3:x2")
        self.connect("p1:x1", "sub:comp1:x")
        self.connect("p2:x2", "sub:comp2:x")

