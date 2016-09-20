from __future__ import print_function

""" Unit test for the SqliteCaseReader. """

import errno
import os
import unittest
from shutil import rmtree
from tempfile import mkdtemp
import time

from openmdao.api import Problem, SqliteRecorder, ScipyOptimizer, Group, \
    IndepVarComp, read_cases
from openmdao.recorders.sqlite_reader import SqliteCaseReader
from openmdao.examples.paraboloid_example import Paraboloid


def run_problem(problem):
    t0 = time.time()
    problem.run()
    t1 = time.time()
    return t0, t1


class TestSqliteReader(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dir = mkdtemp()
        cls.filename = os.path.join(cls.dir, "sqlite_test")
        cls.tablename_metadata = 'metadata'
        cls.tablename_iterations = 'iterations'
        cls.tablename_derivs = 'derivs'
        cls.recorder = SqliteRecorder(cls.filename)
        cls.recorder.options['record_metadata'] = False
        cls.eps = 1e-5

        prob = Problem()

        root = prob.root = Group()

        root.add('p1', IndepVarComp('x', 3.0))
        root.add('p2', IndepVarComp('y', -4.0))
        root.add('p', Paraboloid())

        root.connect('p1.x', 'p.x')
        root.connect('p2.y', 'p.y')

        prob.driver = ScipyOptimizer()

        prob.driver.add_desvar('p1.x', lower=-1, upper=10,
                               scaler=1.0, adder=0.0)
        prob.driver.add_desvar('p2.y', lower=-1, upper=10,
                               scaler=1.0, adder=0.0)
        prob.driver.add_objective('p.f_xy', scaler=1.0, adder=0.0)

        prob.driver.add_recorder(cls.recorder)
        cls.recorder.options['record_params'] = True
        cls.recorder.options['record_resids'] = True
        cls.recorder.options['record_unknowns'] = True
        cls.recorder.options['record_metadata'] = True
        prob.setup(check=False)

        prob['p1.x'] = 10.0
        prob['p2.y'] = 10.0

        t0, t1 = run_problem(prob)
        prob.cleanup()  # closes recorders

    @classmethod
    def tearDownClass(cls):
        try:
            rmtree(cls.dir)
        except OSError as e:
            # If directory already deleted, keep going
            if e.errno not in (errno.ENOENT, errno.EACCES, errno.EPERM):
                raise e

    def test_sqlite_reader_instantiates(self):
        cr = read_cases(self.filename)
        assert(isinstance(cr, SqliteCaseReader))

    def test_num_cases(self):
        cr = read_cases(self.filename)
        self.assertEqual(cr.num_cases, 3, 'Number of cases read is incorrect')

    def test_list_cases(self):
        cr = read_cases(self.filename)
        expected = ['rank0:SLSQP|1', 'rank0:SLSQP|2', 'rank0:SLSQP|3']
        for i, case_id in enumerate(cr.list_cases()):
            self.assertEqual(case_id, expected[i],
                             'List cases returns incorrect case id')

    def test_get_case(self):
        cr = read_cases(self.filename)
        last_case1 = cr.get_case(-1)
        last_case2 = cr.get_case('rank0:SLSQP|3')

        self.assertEqual(last_case1['p.f_xy'], last_case2['p.f_xy'],
                         'get_case returned incorrect results')

    def test_derivs(self):
        pass

if __name__ == "__main__":
    unittest.main()
