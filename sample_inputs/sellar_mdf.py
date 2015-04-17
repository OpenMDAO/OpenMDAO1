import numpy as np

from openmdao.core import Component, Assembly, Group,
from openmdao.lib import ExprComp, ParamComp, Newton, Krylov, ScipyGMRes

class SellarDis1(Component):

    def __init__(self):
        super(Component, self).__init__()
        self.add_input('z', value=np.zeros(2))
        self.add_input('x', value=0.)
        self.add_input('y2', value=0.)

        self.add_output('y1', value=0.)

    def solve_nonlinear(self, params, unknowns, residuals):
        unknowns['y1'] = params['z'][0]**2 + params['z'][1] + params['x'] - .2*params['y2']

    def jacobian(self, params, unknowns):
        J = {}

        J['y1','y2'] = -.2
        J['y1','z'] = np.array([2*ins['z'][0], 1.])
        J['y1','x'] = 1

        return J

    # def apply_nonlinear(self, params, unknowns, residuals):
    #     pass

    #    pvec, uvec, rvec, dpvec, duvec, drvec
    # def solve_linear(self, mode="fwd", ins, outs, residuals, dins, douts, dresiduals):
    #     pass

    # def apply_linear(self, mode="fwd", ins, outs, residuals, dins, douts, dresiduals):
    #     pass

    # def preconditioner(self):
    #     pass


class SellarDis2(Component):

    def __init__(self):
        super(Component, self).__init__()

        self.add_input('z', value=np.zeros(2))
        self.add_input('y1', value=0.)

        self.add_output('y2', value=0.)

        def apply_nonlinear(self, params, unknowns):
            unknowns['y2'] = params['y1']**.5 + np.sum(params['z'])


        def jacobian(self, params, unknowns):
            J = {}

            J['y2', 'y1'] = .5*params['y1']**-.5
            J['y2', 'z'] = np.ones(2)

            return J

class SellarProblem(Assembly):

    def __init__(self):
        super(SellarProblem, self).__init__()

        self.add(ParamComp('x', 1.))
        self.add(ParamComp('z', np.ones(2)))

        d1 = self.add("s1", SellarDis1(), promotes=('y1','y2','z'))
        d2 = self.add("s2", SellarDis2(), promotes=('z', 'y1', 'y2'))

        #objective value is f:val
        self.add(ExprComp('x**2 + sum(z) + y1 + exp(-y2)'), name='f')

        #constraint value is c1:val
        self.add(ConstraintComp('3.16 < y1'), name="c1")
        self.add(ConstraintComp('y2 < 24.0'), name="c2")

        # self.connect('x', ('s1:x',))
        # self.connect('z', ('s1:z', 's2:z', ))
        # self.connect('s1:y1', ('s2:y1', 'c1:y1'))
        # self.connect('s2:y2', ('s1:y2', 'f:y2', 'c2:y2'))
        # self.connect('s2:y2', ('f:y2', 'c2:y2'))

        # resid = self.add(ResidComp('y2-y2','s1:y2'))

        mda = self.add(Group(d1, d2))
        mda.nl_solver = Newton()
        mda.lin_solver = ScipyGMRes()
