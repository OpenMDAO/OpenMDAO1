
import unittest

import numpy as np

from openmdao.api import IndepVarComp, Group, Problem, CaseDriver, InMemoryRecorder
from openmdao.test.exec_comp_for_test import ExecComp4Test


class TestCaseDriver(unittest.TestCase):

    def test_case_driver(self):
        problem = Problem()
        root = problem.root = Group()
        root.add('indep_var', IndepVarComp('x', val=1.0))
        root.add('const', IndepVarComp('c', val=2.0))
        root.add('mult', ExecComp4Test("y=c*x"))

        root.connect('indep_var.x', 'mult.x')
        root.connect('const.c', 'mult.c')

        cases = [
            [('indep_var.x', 3.0), ('const.c', 1.5)],
            [('indep_var.x', 4.0), ('const.c', 2.)],
            [('indep_var.x', 5.5), ('const.c', 3.0)],
        ]

        problem.driver = CaseDriver(cases)

        problem.driver.add_desvar('indep_var.x')
        problem.driver.add_desvar('const.c')

        problem.driver.add_recorder(InMemoryRecorder())

        problem.setup(check=False)
        problem.run()

        for i, data in enumerate(problem.driver.recorders[0].iters):
            data['unknowns'] = dict(data['unknowns'])
            self.assertEqual(data['unknowns']['indep_var.x']*data['unknowns']['const.c'],
                             data['unknowns']['mult.y'])
            self.assertEqual(cases[i][0][1]*cases[i][1][1],
                             data['unknowns']['mult.y'])

        self.assertEqual(len(problem.driver.recorders[0].iters), 3)

if __name__ == "__main__":
    unittest.main()
