from __future__ import print_function
from six.moves import range
import numpy as np

from openmdao.core.component import Component
from openmdao.core.group import Group

from openmdao.components.indep_var_comp import IndepVarComp
from openmdao.components.exec_comp import ExecComp


class Plus(Component):
    def __init__(self, adder):
        super(Plus, self).__init__()
        self.add_param('x', np.random.random())
        self.add_output('f1', shape=1)
        self.adder = float(adder)

    def solve_nonlinear(self, params, unknowns, resids):
        unknowns['f1'] = params['x'] + self.adder

class Times(Component):
    def __init__(self, scalar):
        super(Times, self).__init__()
        self.add_param('f1', np.random.random())
        self.add_output('f2', shape=1)
        self.scalar = float(scalar)

    def solve_nonlinear(self, params, unknowns, resids):
        unknowns['f2'] = params['f1'] + self.scalar

class Point(Group):

    def __init__(self, adder, scalar):
        super(Point, self).__init__()

        # self.add('plus', ExecComp('f1 = x + %f'%adder, x=np.random.random()), promotes=['*'])
        # self.add('times', ExecComp('f2 = %f*f1'%scalar), promotes=['*'])

        self.add('plus', Plus(adder), promotes=['*'])
        self.add('times', Times(scalar), promotes=['*'])
        self.set_order(('plus','times'))

class Summer(Component):

    def __init__(self, size):
        super(Summer, self).__init__()
        self.size = size
        for i in range(size):
            self.add_param('y%d'%i, 0.)

        self.add_output('total', shape=1)

    def solve_nonlinear(self, params, unknowns, resids):
        tot = 0
        for i in range(self.size):
            tot += params['y%d'%i]
        unknowns['total'] = tot

class MultiPoint(Group):

    def __init__(self, adders, scalars):
        super(MultiPoint, self).__init__()

        size = len(adders)
        # self.add('desvar', IndepVarComp('X', val=10*np.arange(size, dtype=float)), promotes=['*'])

        for i,(a,s) in enumerate(zip(adders, scalars)):
            c_name = 'p%d'%i
            self.add(c_name, Point(a,s))
            # self.connect('X', c_name+'.x', src_indices=[i])
            self.connect(c_name+'.f2','aggregate.y%d'%i)

        self.add('aggregate', Summer(size))

if __name__ == '__main__':
    import time
    from openmdao.core.problem import Problem

    prob = Problem()

    size = 10000
    print ("SIZE: %d" % size)

    adders =  np.random.random(size)
    scalars = np.random.random(size)

    prob.root = MultiPoint(adders, scalars)

    st = time.time()
    print("setup started")
    prob.setup(check=False)
    print("setup time", time.time() - st)

    print("num connections:",len(prob.root.connections))
    print("num unknowns:", len(prob.root._unknowns_dict),
          "size:", prob.root.unknowns.vec.size)
    print("num params:", len(prob.root._params_dict),
          "size:", prob.root.params.vec.size)

    # prob['p0.x'] = 10
    # prob['p1.x'] = 13
    # prob['p2.x'] = 15
    st = time.time()
    print("run started")
    prob.run()
    print("run time", time.time() - st)

    print(prob['aggregate.total'])
