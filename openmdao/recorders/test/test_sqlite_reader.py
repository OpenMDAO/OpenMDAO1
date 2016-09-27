""" Unit tests for the SqliteCaseReader. """
from __future__ import print_function

import errno
import os
import unittest
from shutil import rmtree
from tempfile import mkdtemp


from sqlitedict import SqliteDict

from openmdao.api import Problem, ScipyOptimizer, Group, \
    IndepVarComp, CaseReader
from openmdao.recorders.sqlite_recorder import SqliteRecorder, format_version
from openmdao.recorders.sqlite_reader import SqliteCaseReader
from openmdao.examples.paraboloid_example import Paraboloid

try:
    from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver
except ImportError:
    pyOptSparseDriver = None


optimizers = {'scipy': ScipyOptimizer,
              'pyoptsparse': pyOptSparseDriver}


def _setup_test_case(case, record_params=True, record_resids=True,
                     record_unknowns=True, record_derivs=True,
                     record_metadata=True, optimizer='scipy'):
    case.dir = mkdtemp()
    case.filename = os.path.join(case.dir, "sqlite_test")
    case.recorder = SqliteRecorder(case.filename)

    prob = Problem()

    root = prob.root = Group()

    root.add('p1', IndepVarComp('x', 3.0))
    root.add('p2', IndepVarComp('y', -4.0))
    root.add('p', Paraboloid())

    root.connect('p1.x', 'p.x')
    root.connect('p2.y', 'p.y')

    prob.driver = optimizers[optimizer]()

    prob.driver.add_desvar('p1.x', lower=-1, upper=10,
                           scaler=1.0, adder=0.0)
    prob.driver.add_desvar('p2.y', lower=-1, upper=10,
                           scaler=1.0, adder=0.0)
    prob.driver.add_objective('p.f_xy', scaler=1.0, adder=0.0)

    prob.driver.add_recorder(case.recorder)
    case.recorder.options['record_params'] = record_params
    case.recorder.options['record_resids'] = record_resids
    case.recorder.options['record_unknowns'] = record_unknowns
    case.recorder.options['record_metadata'] = record_metadata
    case.recorder.options['record_derivs'] = record_derivs
    prob.setup(check=False)

    prob['p1.x'] = 10.0
    prob['p2.y'] = 10.0
    
    case.original_path = os.getcwd()
    os.chdir(case.dir)

    prob.run()
    prob.cleanup()  # closes recorders


class TestSqliteCaseReader(unittest.TestCase):

    def setUp(self):
        _setup_test_case(self, record_params=True, record_metadata=True,
                         record_derivs=True, record_resids=True,
                         record_unknowns=True, optimizer='scipy')

    def tearDown(self):
        os.chdir(self.original_path)
        try:
            rmtree(self.dir)
        except OSError as e:
            # If directory already deleted, keep going
            if e.errno not in (errno.ENOENT, errno.EACCES, errno.EPERM):
                raise e

    def test_format_version(self):
        cr = CaseReader(self.filename)
        self.assertEqual(cr.format_version, format_version,
                         msg='format version not read correctly')

    def test_reader_instantiates(self):
        """ Test that CaseReader returns an HDF5CaseReader. """
        cr = CaseReader(self.filename)
        self.assertTrue(isinstance(cr, SqliteCaseReader), msg='CaseReader not'
                        ' returning the correct subclass.')

    def test_params(self):
        """ Tests that the reader returns params correctly. """
        cr = CaseReader(self.filename)
        last_case = cr.get_case(-1)
        last_case_id = cr.list_cases()[-1]
        n = cr.num_cases
        with SqliteDict(self.filename, 'iterations', flag='r') as db:
            for key in db[last_case_id]['Parameters'].keys():
                val = db[last_case_id]['Parameters'][key]
                self.assertAlmostEqual(last_case.parameters[key], val,
                                       msg='Case reader gives incorrect '
                                       'Parameter value for {0}'.format(key))

    def test_unknowns(self):
        """ Tests that the reader returns unknowns correctly. """
        cr = CaseReader(self.filename)
        last_case = cr.get_case(-1)
        last_case_id = cr.list_cases()[-1]
        with SqliteDict(self.filename, 'iterations', flag='r') as db:
            for key in db[last_case_id]['Unknowns'].keys():
                val = db[last_case_id]['Unknowns'][key][()]
                self.assertAlmostEqual(last_case[key], val,
                                       msg='Case reader gives incorrect '
                                       'Unknown value for {0}'.format(key))

    def test_resids(self):
        """ Tests that the reader returns resids correctly. """
        cr = CaseReader(self.filename)
        last_case = cr.get_case(-1)
        last_case_id = cr.list_cases()[-1]
        with SqliteDict(self.filename, 'iterations', flag='r') as db:
            for key in db[last_case_id]['Residuals'].keys():
                val = db[last_case_id]['Residuals'][key][()]
                self.assertAlmostEqual(last_case.resids[key], val,
                                       msg='Case reader gives incorrect '
                                       'Unknown value for {0}'.format(key))

    @unittest.skip('Skipped until ScipyOptimizer returns a keyed Jacobian')
    def test_derivs(self):
        """ Test that the reader returns the derivs correctly. """
        cr = CaseReader(self.filename)
        derivs = cr.get_case(-1).derivs
        last_case_id = cr.list_cases()[-1]
        with SqliteDict(self.filename, 'derivs', flag='r') as derivs_db:
            derivs_table = derivs_db[last_case_id]
            df_dx = derivs_table['p.f_xy']['p1.x'][()]
            df_dy = derivs_table['p.f_xy']['p2.y'][()]
            self.assertAlmostEqual(derivs['p.f_xy']['p1.x'], df_dx)
            self.assertAlmostEqual(derivs['p.f_xy']['p2.y'], df_dy)


class TestSqliteCaseReaderNoParams(TestSqliteCaseReader):

    def setUp(self):
        _setup_test_case(self, record_params=False, record_metadata=True,
                         record_derivs=True, record_resids=True,
                         record_unknowns=True, optimizer='scipy')

    def test_params(self):
        """ Test that params is None if not provided in the recording. """
        cr = CaseReader(self.filename)
        last_case = cr.get_case(-1)
        self.assertIsNone(last_case.parameters,
                          "Case erroneously contains parameters.")


class TestSqliteCaseReaderNoResids(TestSqliteCaseReader):

    def setUp(self):
        _setup_test_case(self, record_params=True, record_metadata=True,
                         record_derivs=True, record_resids=False,
                         record_unknowns=True, optimizer='scipy')

    def test_resids(self):
        """ Test that params is None if not provided in the recording. """
        cr = CaseReader(self.filename)
        last_case = cr.get_case(-1)
        self.assertIsNone(last_case.resids,
                          "Case erroneously contains resids.")


@unittest.skip('Skipped until format_version is always recorded')
class TestSqliteCaseReaderNoMetadata(TestSqliteCaseReader):
    def setUp(self):
        _setup_test_case(self, record_params=True, record_metadata=False,
                         record_derivs=True, record_resids=True,
                         record_unknowns=True, optimizer='scipy')

    def test_metadata(self):
        """ Test that metadata is correctly read.

        format_version should always be present.
         """
        cr = CaseReader(self.filename)
        self.assertEqual(cr.format_version, format_version,
                         msg='incorrect format version: '
                             '{0} vs. {1}'.format(cr.format_version,
                                                  format_version))
        self.assertIsNone(cr.parameters,
                          msg='parameter metadata should be None')
        self.assertIsNone(cr.unknowns, msg='unknown metadata should be None')


class TestSqliteCaseReaderNoUnknowns(TestSqliteCaseReader):

    def setUp(self):
        _setup_test_case(self, record_params=True, record_metadata=True,
                         record_derivs=True, record_resids=True,
                         record_unknowns=False, optimizer='scipy')

    def test_unknowns(self):
        """ Test that unknowns is None if not provided in the recording. """
        cr = CaseReader(self.filename)
        last_case = cr.get_case(-1)
        self.assertIsNone(last_case.unknowns,
                          "Case erroneously contains unknowns.")


class TestSqliteCaseReaderNoDerivs(TestSqliteCaseReader):

    def setUp(self):
        _setup_test_case(self, record_params=True, record_metadata=True,
                         record_derivs=False, record_resids=True,
                         record_unknowns=True, optimizer='scipy')

    def test_derivs(self):
        """ Test that derivs is None if not provided in the recording. """
        cr = CaseReader(self.filename)
        last_case = cr.get_case(-1)
        self.assertIsNone(last_case.derivs,
                          "Case erroneously contains derivs.")


@unittest.skipIf(pyOptSparseDriver is None, 'pyOptSparse not available.')
class TestSqliteCaseReaderPyOptSparse(TestSqliteCaseReader):

    def setUp(self):
        _setup_test_case(self, record_params=True, record_metadata=True,
                         record_derivs=True, record_resids=True,
                         record_unknowns=True, optimizer='pyoptsparse')


if __name__ == "__main__":
    unittest.main()
