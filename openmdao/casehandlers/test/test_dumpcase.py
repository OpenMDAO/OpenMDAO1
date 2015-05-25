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


class TestScipyGMRES(unittest.TestCase):

    def test_converge_diverge(self):

        top = Problem()
        top.root = ConvergeDiverge()
        top.root.lin_solver = ScipyGMRES()

        sout = StringIO.StringIO()
        top.driver.add_recorder(DumpCaseRecorder(sout))
        top.setup()
        top.run()

        param_list = ['p:x']
        unknown_list = ['comp7:y1']

        J = top.calc_gradient(param_list, unknown_list, mode='fwd', return_format='dict')
        assert_rel_error(self, J['comp7:y1']['p:x'][0][0], -40.75, 1e-6)

        J = top.calc_gradient(param_list, unknown_list, mode='rev', return_format='dict')
        assert_rel_error(self, J['comp7:y1']['p:x'][0][0], -40.75, 1e-6)


    # def test_single_diamond(self):

    #     top = Problem()
    #     top.root = SingleDiamond()
    #     top.root.lin_solver = ScipyGMRES()
    #     top.setup()
    #     top.run()

    #     param_list = ['p:x']
    #     unknown_list = ['comp4:y1', 'comp4:y2']

    #     J = top.calc_gradient(param_list, unknown_list, mode='fwd', return_format='dict')
    #     assert_rel_error(self, J['comp4:y1']['p:x'][0][0], 25, 1e-6)
    #     assert_rel_error(self, J['comp4:y2']['p:x'][0][0], -40.5, 1e-6)

    #     J = top.calc_gradient(param_list, unknown_list, mode='rev', return_format='dict')
    #     assert_rel_error(self, J['comp4:y1']['p:x'][0][0], 25, 1e-6)
    #     assert_rel_error(self, J['comp4:y2']['p:x'][0][0], -40.5, 1e-6)



if __name__ == "__main__":
    unittest.main()
