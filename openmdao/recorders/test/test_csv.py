"""Unit test for the CsvRecorder."""

from tempfile import mkdtemp
import csv
import unittest
import time

from six import StringIO, iteritems

import numpy as np
from numpy import array

from openmdao.api import Problem, Group, ScipyOptimizer, IndepVarComp, \
                         FullFactorialDriver
from openmdao.recorders.csv_recorder import CsvRecorder
from openmdao.test.converge_diverge import ConvergeDiverge
from openmdao.test.example_groups import ExampleGroup
from openmdao.test.sellar import SellarDerivativesGrouped
from openmdao.test.exec_comp_for_test import ExecComp4Test
from openmdao.test.util import assert_rel_error
from openmdao.util.record_util import format_iteration_coordinate
from collections import OrderedDict

# check that pyoptsparse is installed
# if it is, try to use SLSQP
OPT = None
OPTIMIZER = None

try:
    from pyoptsparse import OPT
    try:
        OPT('SLSQP')
        OPTIMIZER = 'SLSQP'
    except:
        pass
except:
    pass

if OPTIMIZER:
    from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver


def run_problem(problem):
    t0 = time.time()
    problem.run()
    t1 = time.time()

    return t0, t1


class TestCsvRecorder(unittest.TestCase):
    def setUp(self):
        self.dir = mkdtemp()
        self.io = StringIO()
        self.recorder = CsvRecorder(self.io)
        self.eps = 1e-5

    def assertMetadataRecorded(self, metadata):
        # TODO: implement this
        return

    def assertIterationDataRecorded(self, expected, tolerance):

        saved_results = {}

        self.io.seek(0)
        csv_reader = csv.DictReader(self.io)
        for row in csv_reader:
            for header_name in row:

                if header_name == 'Derivatives':
                    continue

                if header_name not in saved_results:
                    saved_results[header_name] = [header_name]

                try:
                    value = float(row[header_name])
                except TypeError:
                    value = row[header_name]

                saved_results[header_name].append(value)

        for coord, (t0, t1), params, unknowns, resids in expected:
            icoord = format_iteration_coordinate(coord)

            if params is not None:
                for param in params:
                    self.assertTrue(param[0] in saved_results)
                    self.assertTrue(tuple(saved_results[param[0]]) == param, tuple(saved_results[param[0]]) + param)

            if unknowns is not None:
                for unknown in unknowns:
                    self.assertTrue(unknown[0] in saved_results)
                    self.assertEqual(tuple(saved_results[unknown[0]]), unknown)
                    self.assertTrue(tuple(saved_results[unknown[0]]) == unknown, tuple(saved_results[unknown[0]]) + unknown)

        return saved_results

    def test_only_resids_recorded(self):
        prob = Problem()
        prob.root = ConvergeDiverge()
        prob.driver.add_recorder(self.recorder)
        self.recorder.options['record_params'] = False
        self.recorder.options['record_unknowns'] = False
        self.recorder.options['record_resids'] = True
        prob.setup(check=False)

        t0, t1 = run_problem(prob)

        coordinate = [0, 'Driver', (1, )]

        expected_resids = [
            ("comp1.y1", 0.0),
            ("comp1.y2", 0.0),
            ("comp2.y1", 0.0),
            ("comp3.y1", 0.0),
            ("comp4.y1", 0.0),
            ("comp4.y2", 0.0),
            ("comp5.y1", 0.0),
            ("comp6.y1", 0.0),
            ("comp7.y1", 0.0),
            ("p.x", 0.0)
        ]

        self.assertIterationDataRecorded(((coordinate, (t0, t1), None, None,
                                          expected_resids),), self.eps)

    def test_only_unknowns_recorded(self):
        prob = Problem()
        prob.root = ConvergeDiverge()
        prob.driver.add_recorder(self.recorder)
        prob.setup(check=False)

        t0, t1 = run_problem(prob)
        prob.cleanup()

        coordinate = [0, 'Driver', (1, )]

        expected_unknowns = [
            ("comp1.y1", 8.0),
            ("comp1.y2", 6.0),
            ("comp2.y1", 4.0),
            ("comp3.y1", 21.0),
            ("comp4.y1", 46.0),
            ("comp4.y2", -93.0),
            ("comp5.y1", 36.8),
            ("comp6.y1", -46.5),
            ("comp7.y1", -102.7),
            ("p.x", 2.0)
        ]

        self.assertIterationDataRecorded(((coordinate, (t0, t1), None,
                                         expected_unknowns, None),), self.eps)

    def test_only_params_recorded(self):
        prob = Problem()
        prob.root = ConvergeDiverge()
        prob.driver.add_recorder(self.recorder)
        self.recorder.options['record_params'] = True
        self.recorder.options['record_resids'] = False
        self.recorder.options['record_unknowns'] = False
        prob.setup(check=False)

        t0, t1 = run_problem(prob)
        prob.cleanup()

        coordinate = [0, 'Driver', (1,)]
        expected_params = [
            ("comp1.x1", 2.0),
            ("comp2.x1", 8.0),
            ("comp3.x1", 6.0),
            ("comp4.x1", 4.0),
            ("comp4.x2", 21.0),
            ("comp5.x1", 46.0),
            ("comp6.x1", -93.0),
            ("comp7.x1", 36.8),
            ("comp7.x2", -46.5)
        ]

        self.assertIterationDataRecorded(((coordinate, (t0, t1), expected_params,
                                         None, None),), self.eps)

    def test_basic(self):
        prob = Problem()
        prob.root = ConvergeDiverge()
        prob.driver.add_recorder(self.recorder)
        self.recorder.options['record_params'] = True
        self.recorder.options['record_resids'] = True
        self.recorder.options['record_resids'] = False
        prob.setup(check=False)

        t0, t1 = run_problem(prob)
        prob.cleanup()

        coordinate = [0, 'Driver', (1, )]

        expected_params = [
            ("comp1.x1", 2.0),
            ("comp2.x1", 8.0),
            ("comp3.x1", 6.0),
            ("comp4.x1", 4.0),
            ("comp4.x2", 21.0),
            ("comp5.x1", 46.0),
            ("comp6.x1", -93.0),
            ("comp7.x1", 36.8),
            ("comp7.x2", -46.5)
        ]

        expected_unknowns = [
            ("comp1.y1", 8.0),
            ("comp1.y2", 6.0),
            ("comp2.y1", 4.0),
            ("comp3.y1", 21.0),
            ("comp4.y1", 46.0),
            ("comp4.y2", -93.0),
            ("comp5.y1", 36.8),
            ("comp6.y1", -46.5),
            ("comp7.y1", -102.7),
            ("p.x", 2.0)
        ]

        expected_resids = [
            ("comp1.y1", 0.0),
            ("comp1.y2", 0.0),
            ("comp2.y1", 0.0),
            ("comp3.y1", 0.0),
            ("comp4.y1", 0.0),
            ("comp4.y2", 0.0),
            ("comp5.y1", 0.0),
            ("comp6.y1", 0.0),
            ("comp7.y1", 0.0),
            ("p.x", 0.0)
        ]

        self.assertIterationDataRecorded(((coordinate, (t0, t1), expected_params,
                                          expected_unknowns, expected_resids),),
                                          self.eps)

    def test_includes(self):
        prob = Problem()
        prob.root = ConvergeDiverge()
        prob.driver.add_recorder(self.recorder)
        self.recorder.options['includes'] = ['comp1.*']
        self.recorder.options['record_params'] = True
        self.recorder.options['record_resids'] = False
        prob.setup(check=False)
        t0, t1 = run_problem(prob)
        prob.cleanup()

        coordinate = [0, 'Driver', (1,)]

        expected_params = [
            ("comp1.x1", 2.0)
        ]
        expected_unknowns = [
            ("comp1.y1", 8.0),
            ("comp1.y2", 6.0)
        ]
        expected_resids = [
            ("comp1.y1", 0.0),
            ("comp1.y2", 0.0)
        ]

        self.assertIterationDataRecorded(((coordinate, (t0, t1), expected_params,
                                         expected_unknowns, expected_resids),),
                                         self.eps)

    def test_includes_and_excludes(self):
        prob = Problem()
        prob.root = ConvergeDiverge()
        prob.driver.add_recorder(self.recorder)
        self.recorder.options['includes'] = ['comp1.*']
        self.recorder.options['excludes'] = ["*.y2"]
        self.recorder.options['record_params'] = True
        self.recorder.options['record_resids'] = False
        prob.setup(check=False)
        t0, t1 = run_problem(prob)
        prob.cleanup()

        coordinate = [0, 'Driver', (1,)]

        expected_params = [
            ("comp1.x1", 2.0)
        ]
        expected_unknowns = [
            ("comp1.y1", 8.0)
        ]
        expected_resids = [
            ("comp1.y1", 0.0)
        ]

        self.assertIterationDataRecorded(((coordinate, (t0, t1), expected_params,
                                          expected_unknowns, expected_resids),),
                                          self.eps)

    def test_solver_record(self):
        prob = Problem()
        prob.root = ConvergeDiverge()
        prob.root.nl_solver.add_recorder(self.recorder)
        self.recorder.options['record_params'] = True
        self.recorder.options['record_resids'] = False
        prob.setup(check=False)
        t0, t1 = run_problem(prob)
        prob.cleanup()

        coordinate = [0, 'Driver', (1,), "root", (1,)]

        expected_params = [
            ("comp1.x1", 2.0),
            ("comp2.x1", 8.0),
            ("comp3.x1", 6.0),
            ("comp4.x1", 4.0),
            ("comp4.x2", 21.0),
            ("comp5.x1", 46.0),
            ("comp6.x1", -93.0),
            ("comp7.x1", 36.8),
            ("comp7.x2", -46.5)
        ]
        expected_unknowns = [
            ("comp1.y1", 8.0),
            ("comp1.y2", 6.0),
            ("comp2.y1", 4.0),
            ("comp3.y1", 21.0),
            ("comp4.y1", 46.0),
            ("comp4.y2", -93.0),
            ("comp5.y1", 36.8),
            ("comp6.y1", -46.5),
            ("comp7.y1", -102.7),
            ("p.x", 2.0)
        ]
        expected_resids = [
            ("comp1.y1", 0.0),
            ("comp1.y2", 0.0),
            ("comp2.y1", 0.0),
            ("comp3.y1", 0.0),
            ("comp4.y1", 0.0),
            ("comp4.y2", 0.0),
            ("comp5.y1", 0.0),
            ("comp6.y1", 0.0),
            ("comp7.y1", 0.0),
            ("p.x", 0.0)
        ]

        self.assertIterationDataRecorded(((coordinate, (t0, t1), expected_params,
                                         expected_unknowns, expected_resids),),
                                         self.eps)

    def test_sublevel_record(self):

        prob = Problem()
        prob.root = ExampleGroup()
        prob.root.G2.G1.nl_solver.add_recorder(self.recorder)
        self.recorder.options['record_params'] = True
        self.recorder.options['record_resids'] = False
        prob.setup(check=False)
        t0, t1 = run_problem(prob)
        prob.cleanup()

        coordinate = [0, 'Driver', (1,), "root", (1,), "G2", (1,), "G1", (1,)]

        expected_params = [
            ("C2.x", 5.0)
        ]
        expected_unknowns = [
            ("C2.y", 10.0)
        ]
        expected_resids = [
            ("C2.y", 0.0)
        ]

        self.assertIterationDataRecorded(((coordinate, (t0, t1), expected_params,
                                         expected_unknowns, expected_resids),),
                                         self.eps)

    def test_multilevel_record(self):
        # FIXME: this test fails with the csv recorder
        self.recorder.close()
        raise unittest.SkipTest('This is not supported by the csv recorder yet.')

        prob = Problem()
        prob.root = ExampleGroup()
        prob.root.G2.G1.nl_solver.add_recorder(self.recorder)
        prob.driver.add_recorder(self.recorder)
        self.recorder.options['record_params'] = True
        self.recorder.options['record_resids'] = True
        prob.setup(check=False)
        t0, t1 = run_problem(prob)
        prob.cleanup()

        solver_coordinate = ['Driver', (1,), "root", (1,), "G2", (1,), "G1", (1,)]

        g1_expected_params = [
            ("C2.x", 5.0)
        ]
        g1_expected_unknowns = [
            ("C2.y", 10.0)
        ]
        g1_expected_resids = [
            ("C2.y", 0.0)
        ]

        # g1_expected = (g1_expected_params, g1_expected_unknowns, g1_expected_resids)

        driver_coordinate = ['Driver', (1,)]

        driver_expected_params = [
            ("G3.C3.x", 10.0)
        ]

        driver_expected_unknowns = [
            ("G2.C1.x", 5.0),
            ("G2.G1.C2.y", 10.0),
            ("G3.C3.y", 20.0),
            ("G3.C4.y", 40.0),
        ]

        driver_expected_resids = [
            ("G2.C1.x", 0.0),
            ("G2.G1.C2.y", 0.0),
            ("G3.C3.y", 0.0),
            ("G3.C4.y", 0.0),
        ]

        expected = []
        expected.append((solver_coordinate, (t0, t1), g1_expected_params,
                         g1_expected_unknowns, g1_expected_resids))
        expected.append((driver_coordinate, (t0, t1), driver_expected_params,
                         driver_expected_unknowns, driver_expected_resids))

        self.assertIterationDataRecorded(expected, self.eps)

    def test_driver_records_metadata(self):
        prob = Problem()
        prob.root = ConvergeDiverge()
        prob.driver.add_recorder(self.recorder)
        self.recorder.options['record_metadata'] = True
        try:
            prob.setup(check=False)
        except Exception as err:
            self.assertEqual(str(err),
              "Recording of metadata is not supported by CsvRecorder.")

    def test_driver_doesnt_record_metadata(self):
        prob = Problem()
        prob.root = ConvergeDiverge()
        prob.driver.add_recorder(self.recorder)
        self.recorder.options['record_metadata'] = False
        prob.setup(check=False)
        prob.cleanup()

        self.assertMetadataRecorded(None)

    def test_root_solver_records_metadata(self):
        prob = Problem()
        prob.root = ConvergeDiverge()
        prob.root.nl_solver.add_recorder(self.recorder)
        self.recorder.options['record_metadata'] = True
        try:
            prob.setup(check=False)
        except Exception as err:
            self.assertEqual(str(err),
                "Recording of metadata is not supported by CsvRecorder.")

    def test_root_solver_doesnt_record_metadata(self):
        prob = Problem()
        prob.root = ConvergeDiverge()
        prob.root.nl_solver.add_recorder(self.recorder)
        self.recorder.options['record_metadata'] = False
        prob.setup(check=False)
        prob.cleanup()

        self.assertMetadataRecorded(None)

    def test_subsolver_records_metadata(self):
        prob = Problem()
        prob.root = ExampleGroup()
        prob.root.G2.G1.nl_solver.add_recorder(self.recorder)
        self.recorder.options['record_metadata'] = True
        try:
            prob.setup(check=False)
        except Exception as err:
            self.assertEqual(str(err),
               "Recording of metadata is not supported by CsvRecorder.")

    def test_subsolver_doesnt_record_metadata(self):
        prob = Problem()
        prob.root = ExampleGroup()
        prob.root.G2.G1.nl_solver.add_recorder(self.recorder)
        self.recorder.options['record_metadata'] = False
        prob.setup(check=False)
        prob.cleanup()

        self.assertMetadataRecorded(None)

    def test_root_derivs_dict(self):

        if OPT is None:
            raise unittest.SkipTest("pyoptsparse is not installed")

        if OPTIMIZER is None:
            raise unittest.SkipTest("pyoptsparse is not providing SNOPT or SLSQP")

        prob = Problem()
        prob.root = SellarDerivativesGrouped()

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.opt_settings['ACC'] = 1e-9
        prob.driver.options['print_results'] = False

        prob.driver.add_desvar('z', lower=np.array([-10.0, 0.0]),
                             upper=np.array([10.0, 10.0]))
        prob.driver.add_desvar('x', lower=0.0, upper=10.0)

        prob.driver.add_objective('obj')
        prob.driver.add_constraint('con1', upper=0.0)
        prob.driver.add_constraint('con2', upper=0.0)

        prob.driver.add_recorder(self.recorder)
        self.recorder.options['record_metadata'] = False
        self.recorder.options['record_derivs'] = True
        prob.setup(check=False)

        prob.run()

        prob.cleanup()

        self.io.seek(0)
        csv_reader = csv.DictReader(self.io)
        rows = [row for row in csv_reader]

        # execution
        row = rows[0]
        self.assertEqual(row['Derivatives'], '')

        # derivatives
        row = rows[1]
        self.assertEqual(row['obj'], '')
        J1 = eval(row['Derivatives'])[0]

        Jbase = {}
        Jbase['con1'] = {}
        Jbase['con1']['x'] = -0.98061433
        Jbase['con1']['z'] = np.array([-9.61002285, -0.78449158])
        Jbase['con2'] = {}
        Jbase['con2']['x'] = 0.09692762
        Jbase['con2']['z'] = np.array([1.94989079, 1.0775421 ])
        Jbase['obj'] = {}
        Jbase['obj']['x'] = 2.98061392
        Jbase['obj']['z'] = np.array([9.61001155, 1.78448534])

        for key1, val1 in Jbase.items():
            for key2, val2 in val1.items():
                assert_rel_error(self, J1[key1][key2], val2, .00001)

    def test_root_derivs_array(self):
        prob = Problem()
        prob.root = SellarDerivativesGrouped()

        prob.driver = ScipyOptimizer()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['tol'] = 1.0e-8
        prob.driver.options['disp'] = False

        prob.driver.add_desvar('z', lower=np.array([-10.0, 0.0]),
                             upper=np.array([10.0, 10.0]))
        prob.driver.add_desvar('x', lower=0.0, upper=10.0)

        prob.driver.add_objective('obj')
        prob.driver.add_constraint('con1', upper=0.0)
        prob.driver.add_constraint('con2', upper=0.0)

        prob.driver.add_recorder(self.recorder)
        self.recorder.options['record_metadata'] = False
        self.recorder.options['record_derivs'] = True
        prob.setup(check=False)

        prob.run()

        prob.cleanup()

        self.io.seek(0)
        csv_reader = csv.DictReader(self.io)
        rows = [row for row in csv_reader]

        # execution
        row = rows[0]
        self.assertEqual(row['Derivatives'], '')

        # derivatives
        row = rows[1]
        self.assertEqual(row['obj'], '')
        J1 = eval('array(' + row['Derivatives'] + ')')[0]

        assert_rel_error(self, J1[0][0], 9.61001155, .00001)
        assert_rel_error(self, J1[0][1], 1.78448534, .00001)
        assert_rel_error(self, J1[0][2], 2.98061392, .00001)
        assert_rel_error(self, J1[1][0], -9.61002285, .00001)
        assert_rel_error(self, J1[1][1], -0.78449158, .00001)
        assert_rel_error(self, J1[1][2], -0.98061433, .00001)
        assert_rel_error(self, J1[2][0], 1.94989079, .00001)
        assert_rel_error(self, J1[2][1], 1.0775421, .00001)
        assert_rel_error(self, J1[2][2], 0.09692762, .00001)

    def test_csv_failures(self):
        problem = Problem()
        root = problem.root = Group()
        root.add('indep_var', IndepVarComp('x', val=1.0))
        root.add('const', IndepVarComp('c', val=2.0))

        root.add('mult', ExecComp4Test("y=c*x", fails=[3, 4]))

        root.connect('indep_var.x', 'mult.x')
        root.connect('const.c', 'mult.c')

        num_levels = 15
        problem.driver = FullFactorialDriver(num_levels=num_levels)
        problem.driver.add_desvar('indep_var.x',
                                  lower=1.0, upper=float(num_levels))
        problem.driver.add_objective('mult.y')

        problem.driver.add_recorder(self.recorder)

        problem.setup(check=False)

        problem.run()

        lines = self.io.getvalue().split('\n')
        vnames = lines[0].split(',')
        succ_idx = vnames.index('success')

        fails = []
        for i, line in enumerate(lines[1:]):
            succ = line.split(',',1)[0]
            if succ == '0':
                fails.append(i)

        self.assertEqual(fails, [3,4])


if __name__ == "__main__":
    unittest.main()
