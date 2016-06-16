
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
        root.add('mult', ExecComp4Test("y=c*x", nl_delay=1.0))

        root.connect('indep_var.x', 'mult.x')
        root.connect('const.c', 'mult.c')

        num_levels = 25
        problem.driver = FullFactorialDriver(num_levels=num_levels,
                                             num_par_doe=7,
                                             load_balance=True)
        problem.driver.add_desvar('indep_var.x',
                                  lower=1.0, upper=float(num_levels))
        problem.driver.add_objective('mult.y')

        rec = problem.driver.add_recorder(InMemoryRecorder())
        rec.options['record_derivs'] = False
        rec.options['includes'] = ['indep_var.x', 'const.c', 'mult.y']

        problem.setup(check=False)
        problem.run()

        for data in problem.driver.recorders[0].iters:
            self.assertEqual(data['unknowns']['indep_var.x']*2.0,
                             data['unknowns']['mult.y'])

        num_cases = len(problem.driver.recorders[0].iters)
        self.assertEqual(num_cases, num_levels)

    def test_load_balanced_doe_crit_fail(self):

        problem = Problem()
        root = problem.root = Group()
        root.add('indep_var', IndepVarComp('x', val=1.0))
        root.add('const', IndepVarComp('c', val=2.0))

        root.add('mult', ExecComp4Test("y=c*x", fail_rank=(0,1,2,3,4),
                 fails=[3], fail_hard=True))

        root.connect('indep_var.x', 'mult.x')
        root.connect('const.c', 'mult.c')

        num_levels = 25
        problem.driver = FullFactorialDriver(num_levels=num_levels,
                                       num_par_doe=5,
                                       load_balance=True)
        problem.driver.add_desvar('indep_var.x',
                                  lower=1.0, upper=float(num_levels))
        problem.driver.add_objective('mult.y')

        problem.driver.add_recorder(InMemoryRecorder())

        problem.setup(check=False)
        problem.run()

        for data in problem.driver.recorders[0].iters:
            self.assertEqual(data['unknowns']['indep_var.x']*2.0,
                             data['unknowns']['mult.y'])

        num_cases = len(problem.driver.recorders[0].iters)

        # in load balanced mode, we can't really predict how many cases
        # will actually run before we terminate, so just check to see if
        # we at least have less than the full set we'd have if nothing
        # went wrong.
        self.assertTrue(num_cases < num_levels and num_cases >= 3,
                "Cases run (%d) should be less than total cases (%d)" %
                (num_cases, num_levels))

    def test_load_balanced_doe_soft_fail(self):

        problem = Problem()
        root = problem.root = Group()
        root.add('indep_var', IndepVarComp('x', val=1.0))
        root.add('const', IndepVarComp('c', val=2.0))

        fail_rank = 1
        root.add('mult', ExecComp4Test("y=c*x", fail_rank=fail_rank, fails=[3,4,5]))

        root.connect('indep_var.x', 'mult.x')
        root.connect('const.c', 'mult.c')

        num_levels = 25
        num_par_doe = 5
        problem.driver = FullFactorialDriver(num_levels=num_levels,
                                             num_par_doe=num_par_doe,
                                             load_balance=True)
        problem.driver.add_desvar('indep_var.x',
                                  lower=1.0, upper=float(num_levels))
        problem.driver.add_objective('mult.y')

        problem.driver.add_recorder(InMemoryRecorder())

        problem.setup(check=False)
        problem.run()

        num_cases = len(problem.driver.recorders[0].iters)

        self.assertEqual(num_cases, num_levels)

        nfails = [0]*num_par_doe
        nsuccs = [0]*num_par_doe

        for data in problem.driver.recorders[0].iters:
            rank = data['unknowns']['mult.case_rank']
            if data['success']:
                self.assertEqual(data['unknowns']['indep_var.x']*2.0,
                                 data['unknowns']['mult.y'])
                nsuccs[rank] += 1
            else:
                nfails[rank] += 1

        # there's a chance that the fail rank didn't get enough
        # cases to actually fail 3 times, so we need to check
        # how many cases it actually got.

        cases_in_fail_rank = nsuccs[fail_rank] + nfails[fail_rank]

        if cases_in_fail_rank > 5:
            self.assertEqual(nfails[fail_rank], 3)
        elif cases_in_fail_rank > 4:
            self.assertEqual(nfails[fail_rank], 2)
        elif cases_in_fail_rank > 3:
            self.assertEqual(nfails[fail_rank], 1)
        else:
            self.assertEqual(nfails[fail_rank], 0)

if __name__ == '__main__':
    unittest.main()
