
import sys
import unittest
import warnings

from six import text_type, PY3
from six.moves import cStringIO

import numpy as np

from openmdao.api import Component, Problem, Group, IndepVarComp, ExecComp, \
                         LinearGaussSeidel, ScipyGMRES, Driver, ProblemSystem
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


class TestSubProblem(unittest.TestCase):

    def test_general_access(self):
        sprob = Problem(root=Group())
        sroot = sprob.root
        sroot.add('Indep', IndepVarComp('x', 7.0))
        sroot.add('C1', ExecComp(['y1=x1*2.0', 'y2=x2*3.0']))
        sroot.connect('Indep.x', 'C1.x1')

        ps = ProblemSystem(sprob,
                           params=['Indep.x', 'C1.x2'],
                           unknowns=['C1.y1', 'C1.y2'])

        prob = Problem(root=Group())
        root = prob.root
        root.add('subprob', ps)

        prob.setup(check=False)

        prob['subprob.Indep.x'] = 99.0 # set a param that maps to an unknown in subproblem
        prob['subprob.C1.x2'] = 5.0  # set a dangling param

        prob.run()

        self.assertEqual(prob['subprob.C1.y1'], 198.0)
        self.assertEqual(prob['subprob.C1.y2'], 15.0)

    def test_simplest_run_w_promote(self):
        prob = Problem(root=Group())
        root = prob.root

        root.add('x_param', IndepVarComp('x', 7.0), promotes=['x'])
        root.add('mycomp', ExecComp('y=x*2.0'), promotes=['x','y'])

        ps = ProblemSystem(prob, params=['x'], unknowns=['y'])

        prob = Problem(root=Group())
        root = prob.root
        root.add('subprob', ps)

        prob.setup(check=False)
        prob.run()
        result = root.unknowns['subprob.y']
        self.assertAlmostEqual(14.0, result, 3)

    def test_basic_run(self):
        prob = Problem(root=ExampleGroup())

        ps = ProblemSystem(prob, params=['G3.C3.x'], unknowns=['G3.C4.y'])

        prob = Problem(root=Group())
        root = prob.root
        root.add('subprob', ps)

        prob.setup(check=False)
        prob.run()

        self.assertAlmostEqual(prob['subprob.G3.C4.y'], 40.)

        stream = cStringIO()

        # get test coverage for list_connections and make sure it doesn't barf
        prob.root.subprob.list_connections(stream=stream)

    def test_byobj_run(self):
        prob = Problem(root=ExampleByObjGroup())

        ps = ProblemSystem(prob, params=['G2.G1.C2.y'], unknowns=['G3.C4.y'])

        prob = Problem(root=Group())
        root = prob.root
        root.add('subprob', ps)

        prob.setup(check=False)
        prob.run()

        self.assertEqual(prob['subprob.G3.C4.y'], 'fooC2C3C4')

if __name__ == "__main__":
    unittest.main()
