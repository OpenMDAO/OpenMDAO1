
import unittest

from openmdao.api import IndepVarComp, Component, Group, Problem, \
                         FullFactorialDriver, InMemoryRecorder, AnalysisError
from openmdao.test.exec_comp_for_test import ExecComp4Test

class LBParallelDOETestCase6(unittest.TestCase):

    def test_multiproc_doe(self):

        problem = Problem()
        root = problem.root = Group()
        root.add('indep_var', IndepVarComp('x', val=1.0))
        root.add('const', IndepVarComp('c', val=2.0))
        root.add('mult', ExecComp4Test("y=c*x"))

        root.connect('indep_var.x', 'mult.x')
        root.connect('const.c', 'mult.c')

        num_levels = 25
        problem.driver = FullFactorialDriver(num_levels=num_levels,
                                       num_par_doe=6,
                                       load_balance=True)
        problem.driver.add_desvar('indep_var.x',
                                  lower=1.0, upper=float(num_levels))
        problem.driver.add_objective('mult.y')

        problem.driver.add_recorder(InMemoryRecorder())

        problem.driver.add_response('indep_var.x')
        problem.driver.add_response('const.c')
        problem.driver.add_response('mult.y')

        problem.setup(check=False)
        problem.run()

        for data in problem.driver.recorders[0].iters:
            self.assertEqual(data['unknowns']['indep_var.x']*2.0,
                             data['unknowns']['mult.y'])

        num_cases = len(problem.driver.recorders[0].iters)
        self.assertEqual(num_cases, num_levels)

if __name__ == '__main__':
    unittest.main()
