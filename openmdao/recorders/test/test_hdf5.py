""" Unit test for the HDF5Recorder. """

import errno
import os
from shutil import rmtree
from tempfile import mkdtemp
import unittest
import time

from six.moves import zip
from six import iteritems

from openmdao.core.problem import Problem
from openmdao.test.converge_diverge import ConvergeDiverge
from openmdao.test.example_groups import ExampleGroup
from openmdao.util.record_util import format_iteration_coordinate
from openmdao.test.util import assert_rel_error

SKIP = False

def run_problem(problem):
    t0 = time.time()
    problem.run()
    t1 = time.time()

    return t0, t1

try:
    from openmdao.recorders.hdf5_recorder import HDF5Recorder
    import h5py
except ImportError:
    # Necessary for the file to parse
    from openmdao.recorders.base_recorder import BaseRecorder
    HDF5Recorder = BaseRecorder
    SKIP = True

class TestHDF5Recorder(unittest.TestCase):
    def setUp(self):
        if SKIP:
            raise unittest.SkipTest("Could not import HDF5Recorder. Is h5py installed?")

        self.dir = mkdtemp()
        self.filename = os.path.join(self.dir, "tmp.hdf5")
        self.recorder = HDF5Recorder(self.filename)
        self.recorder.options['record_metadata'] = False
        self.eps = 1e-5

    def tearDown(self):
        try:
            rmtree(self.dir)
        except OSError as e:
            # If directory already deleted, keep going
            if e.errno not in (errno.ENOENT, errno.EACCES, errno.EPERM):
                raise e

    def assertMetadataRecorded(self, expected):
        sentinel = object()
        hdf = h5py.File(self.filename, 'r')

        metadata = hdf.get('metadata', None)

        if expected is None:
            self.assertIsNone(metadata)
            return

        self.assertEquals(len(metadata), 2)

        pairings = zip(expected, (metadata[x] for x in ('Parameters', 'Unknowns')))

        for expected, actual in pairings:
            self.assertEqual(len(expected), len(actual))

            for key, val in expected:
                found_val = actual.get(key, sentinel)

                if found_val is sentinel:
                    self.fail("Did not find key '{0}'".format(key))

                for mkey, mval in iteritems(val):
                    found_val = actual[key].get(mkey, sentinel)

                    if found_val is sentinel:
                        self.fail("Did not find metadata key '{0}'".format(mkey))

                    self.assertEqual(found_val.value, mval)

    def assertIterationDataRecorded(self, expected, tolerance):
        sentinel = object()
        hdf = h5py.File(self.filename, 'r')

        for coord, (t0, t1), params, unknowns, resids in expected:
            icoord = format_iteration_coordinate(coord)
            actual_group = hdf[icoord]

            groupings = {
                    "Parameters" :  params,
                    "Unknowns" :  unknowns,
                    "Residuals" :  resids,
            }

            self.assertEqual(actual_group.attrs['success'], 1)
            self.assertEqual(actual_group.attrs['msg'], '')

            if params is None:
                self.assertIsNone(actual_group.get('Parameters', None))
                del groupings['Parameters']

            if unknowns is None:
                self.assertIsNone(actual_group.get('Unknowns', None))
                del groupings['Unknowns']

            if resids is None:
                self.assertIsNone(actual_group.get('Residuals', None))
                del groupings['Residuals']

            timestamp = actual_group.attrs['timestamp']
            self.assertTrue(t0 <= timestamp and timestamp <= t1)

            for label, values in iteritems(groupings):
                actual = actual_group[label]

                # If len(actual) == len(expected) and actual <= expected, then
                # actual == expected.
                self.assertEqual(len(actual), len(values))

                for key, val in values:
                    found_val = actual.get(key, sentinel)

                    if found_val is sentinel:
                        self.fail("Did not find key '{0}'.".format(key))

                    assert_rel_error(self, found_val.value, val, tolerance)

        hdf.close()

    def test_only_resids_recorded(self):
        prob = Problem()
        prob.root = ConvergeDiverge()
        prob.driver.add_recorder(self.recorder)
        self.recorder.options['record_params'] = False
        self.recorder.options['record_unknowns'] = False
        self.recorder.options['record_resids'] = True
        prob.setup(check=False)

        t0, t1 = run_problem(prob)
        prob.cleanup() # closes recorders

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
        prob.cleanup() # closes recorders

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
        prob.cleanup() # closes recorders

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
        prob.cleanup() # closes recorders

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
        prob.cleanup() # closes recorders

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
        prob.cleanup() # closes recorders

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
        prob.cleanup() # closes recorders

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
                                expected_unknowns, expected_resids),), self.eps)

    def test_sublevel_record(self):

        prob = Problem()
        prob.root = ExampleGroup()
        prob.root.G2.G1.nl_solver.add_recorder(self.recorder)
        self.recorder.options['record_params'] = True
        self.recorder.options['record_resids'] = True
        prob.setup(check=False)
        t0, t1 = run_problem(prob)
        prob.cleanup() # closes recorders

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
                                expected_unknowns, expected_resids),), self.eps)

    def test_multilevel_record(self):
        prob = Problem()
        prob.root = ExampleGroup()
        prob.root.G2.G1.nl_solver.add_recorder(self.recorder)
        prob.driver.add_recorder(self.recorder)
        self.recorder.options['record_params'] = True
        self.recorder.options['record_resids'] = True
        prob.setup(check=False)
        t0, t1 = run_problem(prob)
        prob.cleanup() # closes recorders

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
        prob.setup(check=False)
        prob.cleanup() # closes recorders

        expected_params = list(iteritems(prob.root.params))
        expected_unknowns = list(iteritems(prob.root.unknowns))
        expected_resids = list(iteritems(prob.root.resids))

        self.assertMetadataRecorded((expected_params, expected_unknowns,
                                    expected_resids))

    def test_driver_doesnt_record_metadata(self):
        prob = Problem()
        prob.root = ConvergeDiverge()
        prob.driver.add_recorder(self.recorder)
        self.recorder.options['record_metadata'] = False
        prob.setup(check=False)
        prob.cleanup() # closes recorders

        self.assertMetadataRecorded(None)

    def test_root_solver_records_metadata(self):
        prob = Problem()
        prob.root = ConvergeDiverge()
        prob.root.nl_solver.add_recorder(self.recorder)
        self.recorder.options['record_metadata'] = True
        prob.setup(check=False)
        prob.cleanup() # closes recorders

        expected_params = list(iteritems(prob.root.params))
        expected_unknowns = list(iteritems(prob.root.unknowns))
        expected_resids = list(iteritems(prob.root.resids))

        self.assertMetadataRecorded((expected_params, expected_unknowns,
                                     expected_resids))

    def test_root_solver_doesnt_record_metadata(self):
        prob = Problem()
        prob.root = ConvergeDiverge()
        prob.root.nl_solver.add_recorder(self.recorder)
        self.recorder.options['record_metadata'] = False
        prob.setup(check=False)
        prob.cleanup() # closes recorders

        self.assertMetadataRecorded(None)

    def test_subsolver_records_metadata(self):
        prob = Problem()
        prob.root = ExampleGroup()
        prob.root.G2.G1.nl_solver.add_recorder(self.recorder)
        self.recorder.options['record_metadata'] = True
        prob.setup(check=False)
        prob.cleanup() # closes recorders

        expected_params = list(iteritems(prob.root.params))
        expected_unknowns = list(iteritems(prob.root.unknowns))
        expected_resids = list(iteritems(prob.root.resids))

        self.assertMetadataRecorded((expected_params, expected_unknowns,
                                    expected_resids))

    def test_subsolver_doesnt_record_metadata(self):
        prob = Problem()
        prob.root = ExampleGroup()
        prob.root.G2.G1.nl_solver.add_recorder(self.recorder)
        self.recorder.options['record_metadata'] = False
        prob.setup(check=False)
        prob.cleanup() # closes recorders

        self.assertMetadataRecorded(None)

if __name__ == "__main__":
    unittest.main()
