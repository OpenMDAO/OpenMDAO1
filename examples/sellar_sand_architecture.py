""" Sellar problem using SAND architecture."""

import time

import numpy as np

from openmdao.api import Component, Group, Problem, IndepVarComp, ExecComp, NLGaussSeidel, ScipyGMRES, \
     ScipyOptimizer


class SellarDis1(Component):

    def __init__(self):
        super(SellarDis1, self).__init__()

        self.add_param('z', val=np.zeros(2))
        self.add_param('x', val=0.0)
        self.add_param('y2', val=1.0)
        self.add_param('y1', val=1.0)

        self.add_output('resid1', val=1.0)

    def solve_nonlinear(self, params, unknowns, resids):

        z1 = params['z'][0]
        z2 = params['z'][1]
        x1 = params['x']
        y2 = params['y2']
        y1 = params['y1']

        unknowns['resid1'] = z1**2 + z2 + x1 - 0.2*y2 - y1

    def linearize(self, params, unknowns, resids):
        J = {}

        J['resid1','y1'] = -1.0
        J['resid1','y2'] = -0.2
        J['resid1','z'] = np.array([[2*params['z'][0], 1.0]])
        J['resid1','x'] = 1.0

        return J


class SellarDis2(Component):

    def __init__(self):
        super(SellarDis2, self).__init__()

        self.add_param('z', val=np.zeros(2))
        self.add_param('y1', val=1.0)
        self.add_param('y2', val=1.0)

        self.add_output('resid2', val=1.0)

    def solve_nonlinear(self, params, unknowns, resids):

        z1 = params['z'][0]
        z2 = params['z'][1]
        y1 = params['y1']
        y1 = abs(y1)
        y2 = params['y2']

        unknowns['resid2'] = y1**.5 + z1 + z2 - y2

    def linearize(self, params, unknowns, resids):
        J = {}

        J['resid2', 'y2'] = -1.0
        J['resid2', 'y1'] = 0.5*params['y1']**-0.5
        J['resid2', 'z'] = np.array([[1.0, 1.0]])

        return J

class SellarSAND(Group):

    def __init__(self):
        super(SellarSAND, self).__init__()

        self.add('px', IndepVarComp('x', 1.0), promotes=['x'])
        self.add('pz', IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])
        self.add('py1', IndepVarComp('y1', 1.0), promotes=['y1'])
        self.add('py2', IndepVarComp('y2', 1.0), promotes=['y2'])

        self.add('d1', SellarDis1(), promotes=['resid1', 'z', 'x', 'y1', 'y2'])
        self.add('d2', SellarDis2(), promotes=['resid2','z', 'y1', 'y2'])

        self.add('obj_cmp', ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                     z=np.array([0.0, 0.0]), x=0.0, y1=0.0, y2=0.0),
                 promotes=['obj', 'z', 'x', 'y1', 'y2'])

        self.add('con_cmp1', ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        self.add('con_cmp2', ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])


if __name__ == '__main__':
    top = Problem()
    top.root = SellarSAND()

    top.driver = ScipyOptimizer()
    top.driver.options['optimizer'] = 'SLSQP'
    top.driver.options['tol'] = 1.0e-12

    top.driver.add_desvar('z', lower=np.array([-10.0, 0.0]),upper=np.array([10.0, 10.0]))
    top.driver.add_desvar('x', lower=0.0, upper=10.0)
    top.driver.add_desvar('y1', lower=-10.0, upper=10.0)
    top.driver.add_desvar('y2', lower=-10.0, upper=10.0)

    top.driver.add_objective('obj')
    top.driver.add_constraint('con1', upper=0.0)
    top.driver.add_constraint('con2', upper=0.0)
    top.driver.add_constraint('resid1', equals=0.0)
    top.driver.add_constraint('resid2', equals=0.0)

    top.setup()
    tt = time.time()
    top.run()


    print("\n")
    print( "Minimum found at (z1,z2,x) = (%3.4f, %3.4f, %3.4f)" % (top['z'][0], \
                                             top['z'][1], \
                                             top['x']))
    print("Coupling vars: %3.4f, %3.4f" % (top['d1.y1'], top['d1.y2']))
    print("Minimum objective: %3.4f" % top['obj'])
