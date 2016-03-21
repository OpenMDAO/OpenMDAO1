
import unittest
import numpy as np
from six import text_type, PY3
from six.moves import cStringIO
import warnings

from openmdao.api import Component, Problem, Group, IndepVarComp, ExecComp, ScipyGMRES
from openmdao.test.sellar import StateConnection

class StateConnWithSolveLinear(StateConnection):
    def solve_linear(self, dumat, drmat, vois, mode=None):
        pass

class TestLayout(unittest.TestCase):

    def test_state_single(self):

        prob = Problem(root=Group())
        root = prob.root
        root.add('statecomp', StateConnection())

        try:
            prob.setup(check=False)
        except Exception as err:
            self.assertTrue("Group '' has a LinearGaussSeidel solver with maxiter==1 "
                             "but it contains implicit states ['statecomp.y2_command']. "
                             "To fix this error, change to a different linear solver, e.g. "
                             "ScipyGMRES or PetscKSP, or increase maxiter (not recommended)." in
                             str(err))
        else:
            self.fail("Exception expected")

    def test_state_single_w_ancestor_iter(self):
        prob = Problem(root=Group())
        root = prob.root
        G1 = root.add("G1", Group())
        G1.add('statecomp', StateConnection())
        root.ln_solver.options['maxiter'] = 5
        # should be no exception here since top level solver has maxiter > 1
        resultes = prob.setup(check=False)

    def test_state_not_single(self):

        prob = Problem(root=Group())
        root = prob.root
        root.ln_solver = ScipyGMRES()

        root.add('statecomp', StateConnection())
        root.add('C1', ExecComp('y=2.0*x'))

        s = cStringIO()
        results = prob.setup(out_stream=s) # should be no exception here

    def test_state_single_maxiter_gt_1(self):

        prob = Problem(root=Group())
        root = prob.root
        root.ln_solver.options['maxiter'] = 2

        root.add('statecomp', StateConnection())

        # this should not raise an exception because maxiter > 1
        prob.setup(check=False)

    def test_state_single_solve_linear(self):
        # this comp has its own solve_linear method, so there should be
        # no exceptions or layout recommendations made here.
        prob = Problem(root=Group())
        root = prob.root
        root.add('statecomp', StateConnWithSolveLinear())

        s = cStringIO()
        prob.setup(out_stream=s)
        self.assertTrue('has implicit states' not in s.getvalue())

    def test_cycle(self):
        prob = Problem(root=Group())
        root = prob.root

        C1 = root.add("C1", ExecComp('y=2.0*x'))
        C2 = root.add("C2", ExecComp('y=2.0*x'))
        C3 = root.add("C3", ExecComp('y=2.0*x'))

        root.connect('C1.y','C2.x')
        root.connect('C2.y','C3.x')
        root.connect('C3.y','C1.x')

        try:
            prob.setup(check=False)
        except Exception as err:
            self.assertTrue("Group '' has a LinearGaussSeidel solver with maxiter==1 "
                             "but it contains cycles [['C1', 'C2', 'C3']]. To fix this "
                             "error, change to a different linear solver, e.g. ScipyGMRES "
                             "or PetscKSP, or increase maxiter (not recommended)." in str(err))
        else:
            self.fail("Exception expected")

    def test_cycle_maxiter_gt_1(self):
        prob = Problem(root=Group())
        root = prob.root
        root.ln_solver.options['maxiter'] = 2

        C1 = root.add("C1", ExecComp('y=2.0*x'))
        C2 = root.add("C2", ExecComp('y=2.0*x'))
        C3 = root.add("C3", ExecComp('y=2.0*x'))

        root.connect('C1.y','C2.x')
        root.connect('C2.y','C3.x')
        root.connect('C3.y','C1.x')

        # this should not raise an exception because maxiter > 1
        s = cStringIO()
        results = prob.setup(out_stream=s)

    def test_cycle_maxiter_gt_1_subgroup(self):
        prob = Problem(root=Group())
        root = prob.root
        root.ln_solver.options['maxiter'] = 2

        G1 = root.add("G1", Group())
        C1 = G1.add("C1", ExecComp('y=2.0*x'))
        C2 = G1.add("C2", ExecComp('y=2.0*x'))
        C3 = G1.add("C3", ExecComp('y=2.0*x'))

        G1.connect('C1.y','C2.x')
        G1.connect('C2.y','C3.x')
        G1.connect('C3.y','C1.x')

        # this should not raise an exception because maxiter > 1 in an ancestor group
        s = cStringIO()
        results = prob.setup(out_stream=s)



if __name__ == "__main__":
    unittest.main()
