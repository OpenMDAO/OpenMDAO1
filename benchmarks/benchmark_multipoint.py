from __future__ import print_function
import unittest
from six.moves import range
import numpy as np

import time
from openmdao.api import Problem, Group, Component, IndepVarComp, ExecComp


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

        self.add('plus', Plus(adder), promotes=['*'])
        self.add('times', Times(scalar), promotes=['*'])
        self.set_order(('plus','times')) # helps speed up setup

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

        for i,(a,s) in enumerate(zip(adders, scalars)):
            c_name = 'p%d'%i
            self.add(c_name, Point(a,s))
            self.connect(c_name+'.f2','aggregate.y%d'%i)

        self.add('aggregate', Summer(size))

class BenchmarkMultipoint(unittest.TestCase):
    """A few 'brute force' multipoint cases (1K, 2K, 5K)"""

    def _setup_bm(self, npts):

        prob = Problem()

        size = npts

        adders =  np.random.random(size)
        scalars = np.random.random(size)

        prob.root = MultiPoint(adders, scalars)

        st = time.time()
        prob.setup(check=False)

        # print("num connections:",len(prob.root.connections))
        # print("num unknowns:", len(prob.root._unknowns_dict),
        #       "size:", prob.root.unknowns.vec.size)
        # print("num params:", len(prob.root._params_dict),
        #       "size:", prob.root.params.vec.size)
        #
        return prob

    def benchmark_setup_5K(self):
        self._setup_bm(5000)

    def benchmark_setup_2K(self):
        self._setup_bm(2000)

    def benchmark_setup_1K(self):
        self._setup_bm(1000)

    def benchmark_run_5K(self):
        p = self._setup_bm(5000)
        p.run()

    def benchmark_run_2K(self):
        p = self._setup_bm(2000)
        p.run()

    def benchmark_run_1K(self):
        p = self._setup_bm(1000)
        p.run()
