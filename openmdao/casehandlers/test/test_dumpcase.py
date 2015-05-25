""" Unit test for the DumpCaseRecorder. """

import unittest
import StringIO

from openmdao.core.problem import Problem
from openmdao.casehandlers.dumpcase import DumpCaseRecorder
from openmdao.solvers.nl_gauss_seidel import NLGaussSeidel
from openmdao.test.sellar import SellarNoDerivatives
from openmdao.test.testutil import assert_rel_error

# class TestNLGaussSeidel(unittest.TestCase):

#     def test_sellar(self):

#         top = Problem()


#         sout = StringIO.StringIO()
#         top.driver.recorders = [DumpCaseRecorder(sout)]


#         top.root = SellarNoDerivatives()
#         top.root.nl_solver = NLGaussSeidel()

#         top.setup()
#         top.run()

#         assert_rel_error(self, top['y1'], 25.58830273, .00001)
#         assert_rel_error(self, top['y2'], 12.05848819, .00001)

#         # Make sure we aren't iterating like crazy
#         self.assertLess(top.root.nl_solver.iter_count, 8)

# if __name__ == "__main__":
#     unittest.main()



import unittest
from unittest import SkipTest
import StringIO


from openmdao.components.paramcomp import ParamComp
from openmdao.core.group import Group
from openmdao.core.problem import Problem
from openmdao.solvers.scipy_gmres import ScipyGMRES
from openmdao.test.converge_diverge import ConvergeDiverge, SingleDiamond, \
                                           ConvergeDivergeGroups, SingleDiamondGrouped
from openmdao.test.simplecomps import SimpleCompDerivMatVec, FanOut, FanIn, \
                                      SimpleCompDerivJac, FanOutGrouped, \
                                      FanInGrouped
from openmdao.test.testutil import assert_rel_error


class TestDumpCaseRecorder(unittest.TestCase):

    def test_basic(self):

        top = Problem()
        top.root = ConvergeDiverge()
        top.root.lin_solver = ScipyGMRES()

        sout = StringIO.StringIO()
        recorder = DumpCaseRecorder(top.driver, sout)
        top.driver.add_recorder(recorder)
        top.setup()
        top.run()

        expected = \
'''Simulation Info:
  OpenMDAO Version: 1.0
Driver Info:
  Driver Class: Driver
Case:
  Params:
comp1:x1: [ 2.]
comp2:x1: [ 8.]
comp3:x1: [ 6.]
comp4:x1: [ 4.]
comp4:x2: [ 21.]
comp5:x1: [ 46.]
comp6:x1: [-93.]
comp7:x1: [ 36.8]
comp7:x2: [-46.5]
  Unknowns:
comp1:y1: [ 8.]
comp1:y2: [ 6.]
comp2:y1: [ 4.]
comp3:y1: [ 21.]
comp4:y1: [ 46.]
comp4:y2: [-93.]
comp5:y1: [ 36.8]
comp6:y1: [-46.5]
comp7:y1: [-102.7]
p:x: [ 2.]
  Resids:
comp1:y1: [ 0.]
comp1:y2: [ 0.]
comp2:y1: [ 0.]
comp3:y1: [ 0.]
comp4:y1: [ 0.]
comp4:y2: [ 0.]
comp5:y1: [ 0.]
comp6:y1: [ 0.]
comp7:y1: [ 0.]
p:x: [ 0.]
'''

        self.assertEqual( sout.getvalue(), expected )

    def test_excludes(self):

        top = Problem()
        top.root = ConvergeDiverge()
        top.root.lin_solver = ScipyGMRES()

        sout = StringIO.StringIO()

        recorder = DumpCaseRecorder(top.driver, sout)
        recorder.options['excludes'] = ['comp4:y1']
        top.driver.add_recorder(recorder)
        top.setup()
        top.run()

        expected = \
'''Simulation Info:
  OpenMDAO Version: 1.0
Driver Info:
  Driver Class: Driver
Case:
  Params:
comp1:x1: [ 2.]
comp2:x1: [ 8.]
comp3:x1: [ 6.]
comp4:x1: [ 4.]
comp4:x2: [ 21.]
comp5:x1: [ 46.]
comp6:x1: [-93.]
comp7:x1: [ 36.8]
comp7:x2: [-46.5]
  Unknowns:
comp1:y1: [ 8.]
comp1:y2: [ 6.]
comp2:y1: [ 4.]
comp3:y1: [ 21.]
comp4:y2: [-93.]
comp5:y1: [ 36.8]
comp6:y1: [-46.5]
comp7:y1: [-102.7]
p:x: [ 2.]
  Resids:
comp1:y1: [ 0.]
comp1:y2: [ 0.]
comp2:y1: [ 0.]
comp3:y1: [ 0.]
comp4:y2: [ 0.]
comp5:y1: [ 0.]
comp6:y1: [ 0.]
comp7:y1: [ 0.]
p:x: [ 0.]
'''

        self.assertEqual( sout.getvalue(), expected )

    def test_includes(self):

        top = Problem()
        top.root = ConvergeDiverge()
        top.root.lin_solver = ScipyGMRES()

        sout = StringIO.StringIO()

        recorder = DumpCaseRecorder(top.driver, sout)
        recorder.options['includes'] = ['comp4:y1']
        top.driver.add_recorder(recorder)
        top.setup()
        top.run()

        expected = \
'''Simulation Info:
  OpenMDAO Version: 1.0
Driver Info:
  Driver Class: Driver
Case:
  Params:
  Unknowns:
comp4:y1: [ 46.]
  Resids:
comp4:y1: [ 0.]
'''

        self.assertEqual( sout.getvalue(), expected )

    def test_includes_and_excludes(self):

        top = Problem()
        top.root = ConvergeDiverge()
        top.root.lin_solver = ScipyGMRES()

        sout = StringIO.StringIO()
        recorder = DumpCaseRecorder(top.driver, sout)
        recorder.options['includes'] = ['comp4:*']
        recorder.options['excludes'] = ['*:y2']
        top.driver.add_recorder(recorder)
        top.setup()
        top.run()

        expected = \
'''Simulation Info:
  OpenMDAO Version: 1.0
Driver Info:
  Driver Class: Driver
Case:
  Params:
comp4:x1: [ 4.]
comp4:x2: [ 21.]
  Unknowns:
comp4:y1: [ 46.]
  Resids:
comp4:y1: [ 0.]
'''

        self.assertEqual( sout.getvalue(), expected )

if __name__ == "__main__":
    unittest.main()
