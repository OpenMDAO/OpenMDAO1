""" Unit test for the DumpRecorder. """

from shutil import rmtree
from tempfile import mkdtemp
import errno
import os
import unittest
import time

from six import StringIO, iteritems

import numpy as np

from openmdao.api import IndepVarComp, Group, ScipyOptimizer, Problem
from openmdao.recorders.dump_recorder import DumpRecorder
from openmdao.test.converge_diverge import ConvergeDiverge
from openmdao.test.example_groups import ExampleGroup
from openmdao.test.paraboloid import Paraboloid
from openmdao.test.sellar import SellarDerivativesGrouped
from openmdao.test.util import assert_rel_error
from openmdao.util.record_util import format_iteration_coordinate

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

class TestDumpRecorder(unittest.TestCase):
    def setUp(self):
        self.dir = mkdtemp()
        self.filename = os.path.join(self.dir, "dump_test")
        self.recorder = DumpRecorder(self.filename)
        self.recorder.options['record_metadata'] = False
        self.eps = 1e-5

    def tearDown(self):
        try:
            rmtree(self.dir)
        except OSError as e:
            # If directory already deleted, keep going
            if e.errno not in (errno.ENOENT, errno.EACCES, errno.EPERM):
                raise e

    def assertMetadataRecorded(self, metadata):
        sout = open(self.filename)

        if metadata is None:
            for line in sout.readlines():
                self.assertTrue("Metadata:\n" not in line)

        else:
            line = sout.readline()
            self.assertEqual("Metadata:\n", line)
            groupings = zip(("Params:\n", "Unknowns:\n"), metadata)

            for header, expected in groupings:
                line = sout.readline()
                self.assertEqual(header, line)

                for name, metadata in expected:
                    line = sout.readline()
                    name = str(name)
                    metadata = str(metadata)

                    self.assertEqual("  {0}: {1}\n".format(name, metadata), line)

    def assertIterationDataRecorded(self, expected, tolerance):
        sout = open(self.filename)

        for coord, (t0, t1), params, unknowns, resids in expected:
            icoord = format_iteration_coordinate(coord)

            line = sout.readline()
            self.assertTrue('Timestamp: ' in line)
            timestamp = float(line[11:-1])
            self.assertTrue(t0 <= timestamp and timestamp <= t1)

            line = sout.readline()
            self.assertEqual("Iteration Coordinate: {0}\n".format(icoord), line)

            line = sout.readline()
            self.assertEqual("Iteration succeeded: yes\n", line)

            groupings = []

            if params is not None:
                groupings.append(("Params:\n", params))

            if unknowns is not None:
                groupings.append(("Unknowns:\n", unknowns))

            if resids is not None:
                groupings.append(("Resids:\n", resids))

            for header, exp in groupings:
                line = sout.readline()
                self.assertEqual(header, line)
                for key, val in exp:
                    line = sout.readline()
                    self.assertEqual("  {0}: {1}\n".format(key, str(val)), line)

        sout.close()

    def test_only_resids_recorded(self):
        prob = Problem()
        prob.root = ConvergeDiverge()
        prob.driver.add_recorder(self.recorder)
        self.recorder.options['record_params'] = False
        self.recorder.options['record_unknowns'] = False
        self.recorder.options['record_resids'] = True
        prob.setup(check=False)

        t0, t1 = run_problem(prob)
        prob.cleanup() # close recorders

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

        self.assertIterationDataRecorded(((coordinate, (t0, t1), None, None, expected_resids),), self.eps)

    def test_only_unknowns_recorded(self):
        prob = Problem()
        prob.root = ConvergeDiverge()
        prob.driver.add_recorder(self.recorder)
        prob.setup(check=False)

        t0, t1 = run_problem(prob)
        prob.cleanup() # close recorders

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

        self.assertIterationDataRecorded(((coordinate, (t0, t1), None, expected_unknowns, None),), self.eps)

    def test_only_params_recorded(self):
        prob = Problem()
        prob.root = ConvergeDiverge()
        prob.driver.add_recorder(self.recorder)
        self.recorder.options['record_params'] = True
        self.recorder.options['record_resids'] = False
        self.recorder.options['record_unknowns'] = False
        prob.setup(check=False)

        t0, t1 = run_problem(prob)
        prob.cleanup() # close recorders

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

        self.assertIterationDataRecorded(((coordinate, (t0, t1), expected_params, None, None),), self.eps)

    def test_basic(self):
        prob = Problem()
        prob.root = ConvergeDiverge()
        prob.driver.add_recorder(self.recorder)
        self.recorder.options['record_params'] = True
        self.recorder.options['record_resids'] = True
        prob.setup(check=False)

        t0, t1 = run_problem(prob)
        prob.cleanup() # close recorders

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

        self.assertIterationDataRecorded(((coordinate, (t0, t1), expected_params, expected_unknowns, expected_resids),), self.eps)

    def test_includes(self):
        prob = Problem()
        prob.root = ConvergeDiverge()
        prob.driver.add_recorder(self.recorder)
        self.recorder.options['includes'] = ['comp1.*']
        self.recorder.options['record_params'] = True
        self.recorder.options['record_resids'] = True
        prob.setup(check=False)
        t0, t1 = run_problem(prob)
        prob.cleanup() # close recorders

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

        self.assertIterationDataRecorded(((coordinate, (t0, t1), expected_params, expected_unknowns, expected_resids),), self.eps)

    def test_includes_and_excludes(self):
        prob = Problem()
        prob.root = ConvergeDiverge()
        prob.driver.add_recorder(self.recorder)
        self.recorder.options['includes'] = ['comp1.*']
        self.recorder.options['excludes'] = ["*.y2"]
        self.recorder.options['record_params'] = True
        self.recorder.options['record_resids'] = True
        prob.setup(check=False)
        t0, t1 = run_problem(prob)
        prob.cleanup() # close recorders

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

        self.assertIterationDataRecorded(((coordinate, (t0, t1), expected_params, expected_unknowns, expected_resids),), self.eps)

    def test_solver_record(self):
        prob = Problem()
        prob.root = ConvergeDiverge()
        prob.root.nl_solver.add_recorder(self.recorder)
        self.recorder.options['record_params'] = True
        self.recorder.options['record_resids'] = True
        prob.setup(check=False)
        t0, t1 = run_problem(prob)
        prob.cleanup() # close recorders

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

        self.assertIterationDataRecorded(((coordinate, (t0, t1), expected_params, expected_unknowns, expected_resids),), self.eps)

    def test_sublevel_record(self):

        prob = Problem()
        prob.root = ExampleGroup()
        prob.root.G2.G1.nl_solver.add_recorder(self.recorder)
        self.recorder.options['record_params'] = True
        self.recorder.options['record_resids'] = True
        prob.setup(check=False)
        t0, t1 = run_problem(prob)
        prob.cleanup() # close recorders

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

        self.assertIterationDataRecorded(((coordinate, (t0, t1), expected_params, expected_unknowns, expected_resids),), self.eps)

    def test_multilevel_record(self):
        prob = Problem()
        prob.root = ExampleGroup()
        prob.root.G2.G1.nl_solver.add_recorder(self.recorder)
        prob.driver.add_recorder(self.recorder)
        self.recorder.options['record_params'] = True
        self.recorder.options['record_resids'] = True
        prob.setup(check=False)
        t0, t1 = run_problem(prob)
        prob.cleanup() # close recorders

        solver_coordinate = [0, 'Driver', (1,), "root", (1,), "G2", (1,), "G1", (1,)]

        g1_expected_params = [
            ("C2.x", 5.0)
        ]
        g1_expected_unknowns = [
            ("C2.y", 10.0)
        ]
        g1_expected_resids = [
            ("C2.y", 0.0)
        ]

        g1_expected = (g1_expected_params, g1_expected_unknowns, g1_expected_resids)

        driver_coordinate = [0, 'Driver', (1,)]

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
        expected.append((solver_coordinate, (t0, t1), g1_expected_params, g1_expected_unknowns, g1_expected_resids))
        expected.append((driver_coordinate, (t0, t1), driver_expected_params, driver_expected_unknowns, driver_expected_resids))

        self.assertIterationDataRecorded(expected, self.eps)

    def test_driver_records_metadata(self):
        prob = Problem()
        prob.root = ConvergeDiverge()
        prob.driver.add_recorder(self.recorder)
        self.recorder.options['record_metadata'] = True
        prob.setup(check=False)
        prob.cleanup() # close recorders

        expected_params = list(iteritems(prob.root.params))
        expected_unknowns = list(iteritems(prob.root.unknowns))
        expected_resids = list(iteritems(prob.root.resids))

        self.assertMetadataRecorded((expected_params, expected_unknowns, expected_resids))

    def test_driver_records_unknown_types_metadata(self):

        prob = Problem()
        root = prob.root = Group()

        # Need an optimization problem to test to make sure
        #   the is_desvar, is_con, is_obj metadata is being
        #   recorded for the Unknowns
        root.add('p1', IndepVarComp('x', 50.0))
        root.add('p2', IndepVarComp('y', 50.0))
        root.add('comp', Paraboloid())

        root.connect('p1.x', 'comp.x')
        root.connect('p2.y', 'comp.y')

        prob.driver = ScipyOptimizer()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.add_desvar('p1.x', lower=-50.0, upper=50.0)
        prob.driver.add_desvar('p2.y', lower=-50.0, upper=50.0)

        prob.driver.add_objective('comp.f_xy')
        prob.driver.options['disp'] = False

        prob.driver.add_recorder(self.recorder)
        self.recorder.options['record_metadata'] = True
        prob.setup(check=False)
        prob.cleanup() # close recorders

        expected_params = list(iteritems(prob.root.params))
        expected_unknowns = list(iteritems(prob.root.unknowns))
        expected_resids = list(iteritems(prob.root.resids))

        self.assertMetadataRecorded((expected_params, expected_unknowns, expected_resids))


    def test_driver_doesnt_record_metadata(self):
        prob = Problem()
        prob.root = ConvergeDiverge()
        prob.driver.add_recorder(self.recorder)
        self.recorder.options['record_metadata'] = False
        prob.setup(check=False)
        prob.cleanup() # close recorders

        self.assertMetadataRecorded(None)

    def test_root_solver_records_metadata(self):
        prob = Problem()
        prob.root = ConvergeDiverge()
        prob.root.nl_solver.add_recorder(self.recorder)
        self.recorder.options['record_metadata'] = True
        prob.setup(check=False)
        prob.cleanup() # close recorders

        expected_params = list(iteritems(prob.root.params))
        expected_unknowns = list(iteritems(prob.root.unknowns))
        expected_resids = list(iteritems(prob.root.resids))

        self.assertMetadataRecorded((expected_params, expected_unknowns, expected_resids))

    def test_root_solver_doesnt_record_metadata(self):
        prob = Problem()
        prob.root = ConvergeDiverge()
        prob.root.nl_solver.add_recorder(self.recorder)
        self.recorder.options['record_metadata'] = False
        prob.setup(check=False)
        prob.cleanup() # close recorders

        self.assertMetadataRecorded(None)

    def test_subsolver_records_metadata(self):
        prob = Problem()
        prob.root = ExampleGroup()
        prob.root.G2.G1.nl_solver.add_recorder(self.recorder)
        self.recorder.options['record_metadata'] = True
        prob.setup(check=False)
        prob.cleanup() # close recorders

        expected_params = list(iteritems(prob.root.params))
        expected_unknowns = list(iteritems(prob.root.unknowns))
        expected_resids = list(iteritems(prob.root.resids))

        self.assertMetadataRecorded((expected_params, expected_unknowns, expected_resids))

    def test_subsolver_doesnt_record_metadata(self):
        prob = Problem()
        prob.root = ExampleGroup()
        prob.root.G2.G1.nl_solver.add_recorder(self.recorder)
        self.recorder.options['record_metadata'] = False
        prob.setup(check=False)
        prob.cleanup() # close recorders

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
        self.recorder.options['record_unknowns'] = True

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

        sout = open(self.filename)
        lines = sout.readlines()

        self.assertEqual(lines[14].rstrip(), 'Derivatives:')
        self.assertTrue('  con1 wrt x:' in lines[15])
        self.assertTrue('  con1 wrt z:' in lines[16])
        self.assertTrue('  con2 wrt x:' in lines[17])
        self.assertTrue('  con2 wrt z:' in lines[18])
        self.assertTrue('  obj wrt x:' in lines[19])
        self.assertTrue('  obj wrt z:' in lines[20])
        self.assertTrue('1.784' in lines[20])

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

        sout = open(self.filename)
        lines = sout.readlines()

        self.assertEqual(lines[14].rstrip(), 'Derivatives:')
        self.assertTrue('9.61' in lines[15])
        self.assertTrue('0.784' in lines[16])
        self.assertTrue('1.077' in lines[17])


if __name__ == "__main__":
    unittest.main()
