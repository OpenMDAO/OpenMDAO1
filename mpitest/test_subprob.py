
import sys
import unittest

from six import text_type, PY3
from six.moves import cStringIO

import numpy as np
from numpy.testing import assert_almost_equal
import random

from openmdao.api import Component, Problem, Group, IndepVarComp, ExecComp, \
                         Driver, ScipyOptimizer, CaseDriver, SubProblem, \
                         SqliteRecorder, pyOptSparseDriver, NLGaussSeidel, \
                         ScipyGMRES

from openmdao.core.mpi_wrap import MPI
from openmdao.test.mpi_util import MPITestCase
from openmdao.test.sellar import SellarNoDerivatives, SellarDis1withDerivatives, SellarDis2withDerivatives

if MPI:
    from openmdao.core.petsc_impl import PetscImpl as impl
else:
    from openmdao.api import BasicImpl as impl

from openmdao.test.util import assert_rel_error


class UQTestDriver(CaseDriver):
    def __init__(self, nsamples, num_par_doe=1, load_balance=False):
        super(UQTestDriver, self).__init__(num_par_doe=num_par_doe,
                                             load_balance=load_balance)
        self.nsamples = nsamples
        self.dist = None
        self.std_devs = {}

    def add_desvar(self, name, **kwargs):
        if 'std_dev' in kwargs:
            self.std_devs[name] = kwargs.pop('std_dev')

        super(UQTestDriver, self).add_desvar(name, **kwargs)

    def run(self, problem):
        if self.dist is None:
            self.dist = {}
            for dv in sorted(self._desvars):
                if dv in self.std_devs:
                    dval = problem[dv]
                    size = dval.size if isinstance(dval, np.ndarray) else 1
                    self.dist[dv] = self.std_devs[dv]*np.random.normal(0.0, 1.0,
                                      self.nsamples*size).reshape(self.nsamples, size)
        self.cases = []
        for i in range(self.nsamples):
            case = []
            for dv in self._desvars:
                dval = problem[dv]
                if dv in self.dist:
                    case.append((dv, dval + self.dist[dv][i]))
                else:
                    case.append((dv, dval))

            #print("case: ",case)
            self.cases.append(case)

        super(UQTestDriver, self).run(problem)

        uncertain_outputs = {}

        # collect the responses and find the mean
        for responses, _, _ in self.get_all_responses():
            for name, val in responses:
                if name not in uncertain_outputs:
                    uncertain_outputs[name] = data = [0.0, 0.0]
                else:
                    data = uncertain_outputs[name]
                data[0] += 1.
                data[1] += val

        # now, set response values in unknowns to the mean value of our
        # uncertain outputs
        for name in uncertain_outputs:
            if name in self.root.unknowns:
                data = uncertain_outputs[name]
                self.root.unknowns[name] = data[1]/data[0]

class SellarDerivatives(Group):
    """ Group containing the Sellar MDA. This version uses the disciplines
    with derivatives."""

    def __init__(self):
        super(SellarDerivatives, self).__init__()

        # params will be provided by parent group
        self.add('px', IndepVarComp('x', 1.0), promotes=['x'])
        self.add('pz', IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])

        self.add('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
        self.add('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])

        self.add('obj_cmp', ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                     z=np.array([0.0, 0.0]), x=0.0),
                 promotes=['obj', 'x', 'z', 'y1', 'y2'])

        self.add('con_cmp1', ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        self.add('con_cmp2', ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

        self.nl_solver = NLGaussSeidel()
        self.ln_solver = ScipyGMRES()

class TestSubProblemMPI(MPITestCase):
    N_PROCS = 4

    def test_opt_over_doe_uq(self):
        np.random.seed(42)

        prob = Problem(impl=impl, root=Group())
        prob.root.deriv_options['type'] = 'fd'

        subprob = Problem(impl=impl, root=SellarDerivatives())
        subprob.root.deriv_options['type'] = 'fd'
        subprob.driver = UQTestDriver(nsamples=100, num_par_doe=self.N_PROCS)
        subprob.driver.add_desvar('z', std_dev=1e-2)
        subprob.driver.add_desvar('x', std_dev=1e-2)
        subprob.driver.add_response('obj')
        subprob.driver.add_response('con1')
        subprob.driver.add_response('con2')

        #subprob.driver.recorders.append(SqliteRecorder("subsellar.db"))

        prob.root.add("indeps", IndepVarComp([('x', 1.0),
                                              ('z', np.array([5.0, 2.0]))]),
                      promotes=['x', 'z'])
        prob.root.add("sub", SubProblem(subprob, params=['z','x'],
                                                 unknowns=['obj', 'con1', 'con2']))

        prob.root.connect('x', 'sub.x')
        prob.root.connect('z', 'sub.z')

        # top level driver setup
        prob.driver = ScipyOptimizer()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['tol'] = 1.0e-8
        prob.driver.options['maxiter'] = 50
        #prob.driver.options['disp'] = False

        prob.driver.add_desvar('z', lower=np.array([-10.0,  0.0]),
                                    upper=np.array([ 10.0, 10.0]))
        prob.driver.add_desvar('x', lower=0.0, upper=10.0)

        prob.driver.add_objective('sub.obj')
        prob.driver.add_constraint('sub.con1', upper=0.0)
        prob.driver.add_constraint('sub.con2', upper=0.0)

        #prob.driver.recorders.append(SqliteRecorder("sellar.db"))

        prob.setup(check=False)

        prob.run()

        tol = 1.e-3
        assert_rel_error(self, prob['sub.obj'], 3.1833940, tol)
        assert_rel_error(self, prob['z'][0], 1.977639, tol)
        assert_rel_error(self, prob['z'][1], 0.0, tol)
        assert_rel_error(self, prob['x'], 0.0, tol)
