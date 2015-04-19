import numpy as np

from openmdao.core import Component, Assembly, Group
from openmdao.components import ExprComp, ParamComp
from openmdao.solvers import Newton, Krylov, ScipyGMRes

class SellarDis1(Component):

    def __init__(self):
        super(SellarDis1, self).__init__()
        self.add_input('z', val=np.zeros(2))
        self.add_input('x', val=0.)
        self.add_input('y2', val=0.)

        self.add_output('y1', val=0.)

    def solve_nonlinear(self, params, unknowns, resids):
        unknowns['y1'] = params['z'][0]**2 + params['z'][1] + params['x'] - .2*params['y2']

    def jacobian(self, params, unknowns):
        J = {}

        J['y1','y2'] = -.2
        J['y1','z'] = np.array([2*params['z'][0], 1.])
        J['y1','x'] = 1

        return J

    # def apply_nonlinear(self, params, unknowns, resids):
    #     pass

    #    pvec, uvec, rvec, dpvec, duvec, drvec
    # def solve_linear(self, mode="fwd", params, unknowns, resids, vec_params, vec_unknowns, vec_resids):
    #     pass

    # def apply_linear(self, mode="fwd", params, unknowns, resids, vec_params, vec_unknowns, vec_resids):
    #     pass

    # def preconditioner(self):
    #     pass


class SellarDis2(Component):

    def __init__(self):
        super(SellarDis2, self).__init__()

        self.add_input('z', val=np.zeros(2))
        self.add_input('y1', val=0.)

        self.add_output('y2', val=0.)

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

        self.add('x_param', ParamComp('x', 1.), promotes=None)
        self.add('z_param',ParamComp('z', np.ones(2)))

        d1 = self.add("s1", SellarDis1(), promotes=('y1','y2','z'))
        d2 = self.add("s2", SellarDis2(), promotes=('z', 'y1','y2'))

        #objective val is f:val
        self.add('obj', ExprComp('x**2 + sum(z) + y1 + exp(-y2)'))

        #Expr Comps of any type have default promoting of all variables.
        #regular components have default promoting = False
        self.add("c1", ConstraintComp('3.16 < y1'))
        self.add("c2", ConstraintComp('y2 < 24.0'))

        self.connect('x_param:x', 's1:x')

        mda = self.add(Group(d1, d2))
        mda.solve_nonlinear = Newton()
        mda.solve_linear = ScipyGMRes()


if __name__ == "__main__":

    from openmdao.core import Problem
    from openmdao.drivers import SLSQP

    prob = Problem()
    prob.root = SellarProblem()
    prob.driver = SLSQP()
    prob.driver.add_param('x_param:x', low=-10, high=10)
    prob.driver.add_param('y', low=-10, high=10)
    prob.driver.add_objective('obj')
    prob.driver.add_constraint('c1')
    prob.driver.add_constraint('c2')

    prob.run()
