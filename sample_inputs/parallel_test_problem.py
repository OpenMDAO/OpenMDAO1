import numpy as np

from openmdao.core.component import Component
from openmdao.components import ExprComp, ParamComp
from openmdao.solvers import Newton, Krylov, ScipyGMRes


class Simple(Component):

    def __init__(self):
        super(Simple, self).__init__()
        self.add_param('x', val=0.)
        self.add_param('y', val=2.)
        self.add_param('A', val=np.zeros(75, dtype=np.float))

        self.add_unknown('z', val=0.)

        self.dz_dA = np.ones(75, dtype=np.float)

    def solve_nonlinear(self, params, unknowns, resids):
        unknowns['z'] = params['x']**2 + params['y'] + np.sum(params['A'] - 3.0)

    def jacobian(self, params, unknowns, resids):
        J = {}

        J['z','x'] = 2*params['x'] * params['y']
        J['z','y'] = params['x']**2
        J['z','A'] = self.dz_dA

        return J

class SimpleProblem(Assembly):

    def __init__(self):
        super(SimpleProblem, self).__init__()

        for i in xrange(4):
            self.add('a%i' % i, Simple())

        for i in xrange(2):
            self.add('b%i' % i, Simple())

        self.connect("a0:z", "b0:x")
        self.connect("a0:z", "b0:A[0]")
        self.connect("a1:z", "b0:y")
        self.connect("a1:z", "b0:A[1]")
        self.connect("a2:z", "b1:x")
        self.connect("a2:z", "b1:A[0]")
        self.connect("a3:z", "b1:y")
        self.connect("a3:z", "b1:A[1]")

        for i in xrange(2):
            self.add('c%i' % i, Simple())
            self.connect("b%i:z" % i, "c%i:x" % i)
            self.connect("b%i:z" % i, "c%i:A[5]" % i)

        for i in xrange(4):
            self.add("d%i" % i, Simple())
            self.connect("c0.z", "d%i.x" % i)
            self.connect("c0.z", "d%i.A[0]" % i)
            self.connect("c1.z", "d%i.y" % i)
            self.connect("c1.z", "d%i.A[50]" % i)

        for i in xrange(4):
            self.add("e%i" % i, SimpleComp())
            self.connect("d%i.z" % i, "e%i.x" % i)
            self.connect("d%i.z" % i, "e%i.y" % i)
            self.connect("d%i.z" % i, "e%i.A[0]" % i)

        expr = "log(" + ('+'.join(["e%i:z" % i for i in xrange(4)])) + ") + a0:z"

        self.add('obj', ExprComp(expr))
