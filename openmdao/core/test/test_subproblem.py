
import sys
import unittest
import warnings

from six import text_type, PY3
from six.moves import cStringIO

import numpy as np
from numpy.testing import assert_almost_equal

from openmdao.api import Component, Problem, Group, IndepVarComp, ExecComp, \
                         LinearGaussSeidel, ScipyGMRES, Driver
from openmdao.core.mpi_wrap import MPI
from openmdao.test.example_groups import ExampleGroup, ExampleGroupWithPromotes, ExampleByObjGroup
from openmdao.test.sellar import SellarStateConnection
from openmdao.test.simple_comps import SimpleComp, SimpleImplicitComp, RosenSuzuki, FanIn
from openmdao.util.options import OptionsDictionary

if PY3:
    def py3fix(s):
        return s.replace('<type', '<class')
else:
    def py3fix(s):
        return s

#
# expected jacobian
#
expectedJ = {
    'subprob.comp.f': {
        'desvars.x': np.array([
            [ -3., -3., -17.,  9.]
        ])
    },
    'subprob.comp.g': {
        'desvars.x': np.array([
            [ 3.,   1.,   3.,  1.],
            [ 1.,   4.,   2.,  3.],
            [ 6.,   1.,   2., -1.],
        ])
    }
}

expectedJ_array = np.concatenate((
    expectedJ['subprob.comp.f']['desvars.x'],
    expectedJ['subprob.comp.g']['desvars.x']
))

class TestSubProblem(unittest.TestCase):

    def test_general_access(self):
        sprob = Problem(root=Group())
        sroot = sprob.root
        sroot.add('Indep', IndepVarComp('x', 7.0))
        sroot.add('C1', ExecComp(['y1=x1*2.0', 'y2=x2*3.0']))
        sroot.connect('Indep.x', 'C1.x1')

        prob = Problem(root=Group())
        prob.add_subproblem('subprob', sprob,
                            params=['Indep.x', 'C1.x2'],
                            unknowns=['C1.y1', 'C1.y2'])

        prob.setup(check=False)

        prob['subprob.Indep.x'] = 99.0 # set a param that maps to an unknown in subproblem
        prob['subprob.C1.x2'] = 5.0  # set a dangling param

        prob.run()

        self.assertEqual(prob['subprob.C1.y1'], 198.0)
        self.assertEqual(prob['subprob.C1.y2'], 15.0)

    def test_simplest_run_w_promote(self):
        subprob = Problem(root=Group())
        subprob.root.add('x_param', IndepVarComp('x', 7.0), promotes=['x'])
        subprob.root.add('mycomp', ExecComp('y=x*2.0'), promotes=['x','y'])

        prob = Problem(root=Group())
        prob.add_subproblem('subprob', subprob, params=['x'], unknowns=['y'])

        prob.setup(check=False)
        prob.run()
        result = prob.root.unknowns['subprob.y']
        self.assertAlmostEqual(14.0, result, 3)

    def test_basic_run(self):
        subprob = Problem(root=ExampleGroup())

        prob = Problem(root=Group())
        prob.add_subproblem('subprob', subprob,
                            params=['G3.C3.x'], unknowns=['G3.C4.y'])

        prob.setup(check=False)
        prob.run()

        self.assertAlmostEqual(prob['subprob.G3.C4.y'], 40.)

        stream = cStringIO()

        # get test coverage for list_connections and make sure it doesn't barf
        prob.root.subprob.list_connections(stream=stream)

    def test_byobj_run(self):
        subprob = Problem(root=ExampleByObjGroup())

        prob = Problem(root=Group())
        prob.add_subproblem('subprob', subprob,
                            params=['G2.G1.C2.y'], unknowns=['G3.C4.y'])

        prob.setup(check=False)
        prob.run()

        self.assertEqual(prob['subprob.G3.C4.y'], 'fooC2C3C4')

    def test_calc_gradient(self):
        root = Group()
        root.add('parm', IndepVarComp('x', np.array([1., 1., 1., 1.])))
        root.add('comp', RosenSuzuki())

        root.connect('parm.x', 'comp.x')

        subprob = Problem(root)
        subprob.driver.add_desvar('parm.x', lower=-10, upper=99)
        subprob.driver.add_objective('comp.f')
        subprob.driver.add_constraint('comp.g', upper=0.)

        prob = Problem(root=Group())
        prob.root.add('desvars', IndepVarComp('x', np.ones(4)))
        prob.add_subproblem('subprob', subprob,
                            params=['parm.x'], unknowns=['comp.f', 'comp.g'])
        prob.root.connect('desvars.x', 'subprob.parm.x')

        prob.setup(check=False)
        prob.run()

        indep_list = ['desvars.x']
        unknown_list = ['subprob.comp.f', 'subprob.comp.g']

        # check that calc_gradient returns proper dict value when mode is 'fwd'
        J = prob.calc_gradient(indep_list, unknown_list, mode='fwd', return_format='dict')
        assert_almost_equal(J['subprob.comp.f']['desvars.x'],
                           expectedJ['subprob.comp.f']['desvars.x'])
        assert_almost_equal(J['subprob.comp.g']['desvars.x'],
                            expectedJ['subprob.comp.g']['desvars.x'])

        # check that calc_gradient returns proper array value when mode is 'fwd'
        J = prob.calc_gradient(indep_list, unknown_list, mode='fwd', return_format='array')
        assert_almost_equal(J, expectedJ_array)

        # check that calc_gradient returns proper dict value when mode is 'rev'
        J = prob.calc_gradient(indep_list, unknown_list, mode='rev', return_format='dict')
        assert_almost_equal(J['subprob.comp.f']['desvars.x'], expectedJ['subprob.comp.f']['desvars.x'])
        assert_almost_equal(J['subprob.comp.g']['desvars.x'], expectedJ['subprob.comp.g']['desvars.x'])

        # check that calc_gradient returns proper array value when mode is 'rev'
        J = prob.calc_gradient(indep_list, unknown_list, mode='rev', return_format='array')
        assert_almost_equal(J, expectedJ_array)

        # check that calc_gradient returns proper dict value when mode is 'fd'
        J = prob.calc_gradient(indep_list, unknown_list, mode='fd', return_format='dict')
        assert_almost_equal(J['subprob.comp.f']['desvars.x'], expectedJ['subprob.comp.f']['desvars.x'], decimal=5)
        assert_almost_equal(J['subprob.comp.g']['desvars.x'], expectedJ['subprob.comp.g']['desvars.x'], decimal=5)

        # check that calc_gradient returns proper array value when mode is 'fd'
        J = prob.calc_gradient(indep_list, unknown_list, mode='fd', return_format='array')
        assert_almost_equal(J, expectedJ_array, decimal=5)

if __name__ == "__main__":
    unittest.main()
