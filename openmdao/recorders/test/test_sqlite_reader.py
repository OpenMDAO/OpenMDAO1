""" Unit tests for the SqliteCaseReader. """
from __future__ import print_function

import errno
import os
import unittest
from shutil import rmtree
from tempfile import mkdtemp
import time

from sqlitedict import SqliteDict

from openmdao.api import Problem, SqliteRecorder, ScipyOptimizer, Group, \
    IndepVarComp, read_cases
from openmdao.recorders.sqlite_reader import SqliteCaseReader
from openmdao.examples.paraboloid_example import Paraboloid


try:
    from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver
except ImportError:
    pyOptSparseDriver = None


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
        cls.recorder = SqliteRecorder(cls.filename)
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
        cls.recorder.options['record_derivs'] = True
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

    def test_format_version(self):
        cr = read_cases(self.filename)
        self.assertIsNotNone(cr.format_version,
                             'format_version not retrieved from file!')

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

    def test_unknowns(self):
        cr = read_cases(self.filename)
        last_case = cr.get_case(-1)
        n = cr.num_cases

        with SqliteDict(self.filename, 'iterations', flag='r') as db:
            f_xy = db['rank0:SLSQP|{0}'.format(n)]['Unknowns']['p.f_xy']
            self.assertAlmostEqual(last_case['p.f_xy'], f_xy,
                                   msg='case reader returned incorrect value')

    @unittest.skip('Skipped until ScipyOptimizer returns a keyed Jacobian')
    def test_derivs(self):
        cr = read_cases(self.filename)
        derivs = cr.get_case(-1).derivs
        n = cr.num_cases

        with SqliteDict(self.filename, 'derivs', flag='r') as db:
            derivs_table = db['rank0:SLSQP|{0}'.format(n)]['Derivatives']
            df_dx = derivs_table['p.f_xy']['p1.x']
            df_dy = derivs_table['p.f_xy']['p2.y']
            self.assertAlmostEqual(derivs['p.f_xy']['p1.x'], df_dx)
            self.assertAlmostEqual(derivs['p.f_xy']['p2.y'], df_dy)


class TestSqliteReaderNoDerivs(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dir = mkdtemp()
        cls.filename = os.path.join(cls.dir, "sqlite_test")
        cls.recorder = SqliteRecorder(cls.filename)
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
        cls.recorder.options['record_derivs'] = False
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

    def test_format_version(self):
        cr = read_cases(self.filename)
        self.assertIsNotNone(cr.format_version, 'format_version not '
                                                'retrieved from file!')

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
        cr = read_cases(self.filename)
        self.assertIsNone(cr.get_case(-1).derivs)


class TestSqliteReaderNoMetadata(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dir = mkdtemp()
        cls.filename = os.path.join(cls.dir, "sqlite_test")
        cls.recorder = SqliteRecorder(cls.filename)
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
        cls.recorder.options['record_metadata'] = False
        cls.recorder.options['record_derivs'] = True
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

    @unittest.skip('recorders currently do not save format_version '
                   'when record_metadata option is False')
    def test_format_version(self):
        cr = read_cases(self.filename)
        self.assertIsNotNone(cr.format_version, 'format_version not '
                                                'retrieved from file!')

    def test_parameter_metadata(self):
        cr = read_cases(self.filename)
        self.assertIsNone(cr.parameters, 'parameter metadata should be None')

    def test_unknown_metadata(self):
        cr = read_cases(self.filename)
        self.assertIsNone(cr.unknowns, 'unknown metadata should be None')

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

    @unittest.skip('Skipped until ScipyOptimizer returns a keyed Jacobian')
    def test_derivs(self):
        cr = read_cases(self.filename)
        derivs = cr.get_case(-1).derivs
        n = cr.num_cases

        with SqliteDict(self.filename, 'derivs', flag='r') as derivs_db:
            db_derivs = derivs_db['rank0:SLSQP|{0}'.format(n)]['Derivatives']
            df_dx = db_derivs['p.f_xy']['p1.x']
            df_dy = db_derivs['p.f_xy']['p2.y']
            self.assertAlmostEqual(derivs['p.f_xy']['p1.x'], df_dx)
            self.assertAlmostEqual(derivs['p.f_xy']['p2.y'], df_dy)


class TestSqliteReaderNoUnknowns(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dir = mkdtemp()
        cls.filename = os.path.join(cls.dir, "sqlite_test")
        cls.tablename_metadata = 'metadata'
        cls.tablename_iterations = 'iterations'
        cls.tablename_derivs = 'derivs'
        cls.recorder = SqliteRecorder(cls.filename)
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
        cls.recorder.options['record_unknowns'] = False
        cls.recorder.options['record_metadata'] = False
        cls.recorder.options['record_derivs'] = True
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

    @unittest.skip('recorders currently do not save format_version '
                   'when record_metadata option is False')
    def test_format_version(self):
        cr = read_cases(self.filename)
        self.assertIsNotNone(cr.format_version, 'format_version not retrieved '
                                                'from file!')

    def test_parameter_metadata(self):
        cr = read_cases(self.filename)
        self.assertIsNone(cr.parameters, 'parameter metadata should be None')

    def test_unknown_metadata(self):
        cr = read_cases(self.filename)
        self.assertIsNone(cr.unknowns, 'unknown metadata should be None')

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

    def test_no_unknowns(self):
        cr = read_cases(self.filename)
        last_case = cr.get_case(-1)
        with self.assertRaises(ValueError) as exception_context_manager:
            last_case['p.f_xy']

    @unittest.skip('Skipped until ScipyOptimizer returns a keyed Jacobian')
    def test_derivs(self):
        cr = read_cases(self.filename)
        derivs = cr.get_case(-1).derivs
        n = cr.num_cases

        with SqliteDict(self.filename, 'derivs', flag='r') as derivs_db:
            db_derivs = derivs_db['rank0:SLSQP|{0}'.format(n)]['Derivatives']
            df_dx = db_derivs['p.f_xy']['p1.x']
            df_dy = db_derivs['p.f_xy']['p2.y']
            self.assertAlmostEqual(derivs['p.f_xy']['p1.x'], df_dx)
            self.assertAlmostEqual(derivs['p.f_xy']['p2.y'], df_dy)


class TestSqliteReaderNoResids(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dir = mkdtemp()
        cls.filename = os.path.join(cls.dir, "sqlite_test")
        cls.recorder = SqliteRecorder(cls.filename)
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
        cls.recorder.options['record_resids'] = False
        cls.recorder.options['record_unknowns'] = True
        cls.recorder.options['record_metadata'] = True
        cls.recorder.options['record_derivs'] = True
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

    def test_resids(self):
        cr = read_cases(self.filename)
        self.assertIsNone(cr.get_case(-1).residuals, 'Case erroneously '
                                                     'contains resids.')


@unittest.skipIf(pyOptSparseDriver is None, 'pyoptsparse unavailble')
class TestSqliteReaderPyOptSparse(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dir = mkdtemp()
        cls.filename = os.path.join(cls.dir, "sqlite_test")
        cls.recorder = SqliteRecorder(cls.filename)
        cls.eps = 1e-5

        prob = Problem()

        root = prob.root = Group()

        root.add('p1', IndepVarComp('x', 3.0))
        root.add('p2', IndepVarComp('y', -4.0))
        root.add('p', Paraboloid())

        root.connect('p1.x', 'p.x')
        root.connect('p2.y', 'p.y')

        prob.driver = pyOptSparseDriver()

        prob.driver.options['optimizer'] = 'SLSQP'

        prob.driver.add_desvar('p1.x', lower=-1, upper=10,
                               scaler=1.0, adder=0.0)
        prob.driver.add_desvar('p2.y', lower=-1, upper=10,
                               scaler=1.0, adder=0.0)
        prob.driver.add_objective('p.f_xy', scaler=1.0, adder=0.0)

        prob.driver.add_recorder(cls.recorder)
        cls.recorder.options['record_params'] = True
        cls.recorder.options['record_resids'] = True
        cls.recorder.options['record_unknowns'] = False
        cls.recorder.options['record_metadata'] = False
        cls.recorder.options['record_derivs'] = True
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

    @unittest.skip('recorders currently do not save format_version '
                   'when record_metadata option is False')
    def test_format_version(self):
        cr = read_cases(self.filename)
        self.assertIsNotNone(cr.format_version, 'format_version not retrieved '
                                                'from file!')

    def test_parameter_metadata(self):
        cr = read_cases(self.filename)
        self.assertIsNone(cr.parameters, 'parameter metadata should be None')

    def test_unknown_metadata(self):
        cr = read_cases(self.filename)
        self.assertIsNone(cr.unknowns, 'unknown metadata should be None')

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

    def test_no_unknowns(self):
        cr = read_cases(self.filename)
        last_case = cr.get_case(-1)
        with self.assertRaises(ValueError) as exception_context_manager:
            last_case['p.f_xy']

    def test_derivs(self):
        cr = read_cases(self.filename)
        derivs = cr.get_case(-1).derivs
        n = cr.num_cases

        with SqliteDict(self.filename, 'derivs', flag='r') as derivs_db:
            db_derivs = derivs_db['rank0:SLSQP|{0}'.format(n)]['Derivatives']
            df_dx = db_derivs['p.f_xy']['p1.x']
            df_dy = db_derivs['p.f_xy']['p2.y']
            self.assertAlmostEqual(derivs['p.f_xy']['p1.x'], df_dx)
            self.assertAlmostEqual(derivs['p.f_xy']['p2.y'], df_dy)


if __name__ == "__main__":
    unittest.main()
