""" Unit test for the JsonRecorder. """

from shutil import rmtree
from tempfile import mkdtemp
import errno
import os
import unittest
import time
import json

from six import StringIO, iteritems

from openmdao.core.problem import Problem
from openmdao.test.converge_diverge import ConvergeDiverge
from openmdao.test.example_groups import ExampleGroup
from openmdao.recorders.json_recorder import JsonRecorder
from openmdao.util.record_util import format_iteration_coordinate

def run_problem(problem):
    t0 = time.time()
    problem.run()
    t1 = time.time()

    return t0, t1

def _json_encode_fallback(object):
    if isinstance(object, array):
        return object.tolist()
    elif isinstance(object, numpy.ndarray):
        return object.tolist()
    else:
        raise TypeError("No fallback encoding for type {0}".format(type(object).__name__))

# Roundtrips an object through the JSON parser to make it easier
# to test for equality
def jsonize_object(obj):
    return json.loads(json.dumps(obj, default=_json_encode_fallback))

class TestJsonRecorder(unittest.TestCase):
    def setUp(self):
        self.dir = mkdtemp()
        self.filename = os.path.join(self.dir, "dump_test")
        self.recorder = JsonRecorder(self.filename)
        self.eps = 1e-5

    def tearDown(self):
        try:
            rmtree(self.dir)
        except OSError as e:
            # If directory already deleted, keep going
            if e.errno != errno.ENOENT:
                raise e

    def assertMetadataRecorded(self, metadata):
        sout = open(self.filename)
        jsonData = json.load(sout)

        if metadata is None:
            self.assertTrue("metadata" not in jsonData)
        else:
            self.assertTrue("metadata" in jsonData)
            
            metadataObject = jsonData["metadata"]
            self.assertTrue("params" in metadataObject)
            self.assertTrue("unknowns" in metadataObject)

            # Might want to check the data inside the metadata objects? This
            # is more complicated than it looks, since metadata's a pretty
            # complex type

    def assertIterationDataRecorded(self, expected, tolerance):
        sout = open(self.filename)
        jsonData = json.load(sout)

        self.assertTrue("iterations" in jsonData)

        iteration = 0

        for coord, (t0, t1), params, unknowns, resids in expected:
            iterationData = jsonData["iterations"][iteration]

            # Validate the timestamp
            self.assertTrue(t0 <= iterationData["timestamp"] and iterationData["timestamp"] <= t1)

            # Validate the iteration coordinate
            self.assertListEqual(iterationData["coord"], jsonize_object(coord))

            # Validate the recorded data
            if params is None:
                self.assertTrue("params" not in iterationData)
            else:
                self.assertTrue("params" in iterationData)

                for param in params:
                    #These are formatted as tuples: ("name", value)
                    self.assertTrue(param[0] in iterationData["params"])
                    self.assertAlmostEqual(iterationData["params"][param[0]], param[1], delta=tolerance)

            if unknowns is None:
                self.assertTrue("unknowns" not in iterationData)
            else:
                self.assertTrue("unknowns" in iterationData)

                for unknown in unknowns:
                    #These are formatted as tuples: ("name", value)
                    self.assertTrue(unknown[0] in iterationData["unknowns"])
                    self.assertAlmostEqual(iterationData["unknowns"][unknown[0]], unknown[1], delta=tolerance)

            if resids is None:
                self.assertTrue("resids" not in iterationData)
            else:
                self.assertTrue("resids" in iterationData)

                for resid in resids:
                    #These are formatted as tuples: ("name", value)
                    self.assertTrue(resid[0] in iterationData["resids"])
                    self.assertAlmostEqual(iterationData["resids"][resid[0]], resid[1], delta=tolerance)

            iteration += 1

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
        self.recorder.close()
        
        coordinate = ['Driver', (1, )]

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
        self.recorder.close()
        
        coordinate = ['Driver', (1, )]

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
        self.recorder.close()

        coordinate = ['Driver', (1,)]
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
        self.recorder.close()
        
        coordinate = ['Driver', (1, )]

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
        self.recorder.close()
        
        coordinate = ['Driver', (1,)]

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
        self.recorder.close()

        coordinate = ['Driver', (1,)]

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
        self.recorder.close()
        
        coordinate = ['Driver', (1,), "root", (1,)]

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
        self.recorder.close()

        coordinate = ['Driver', (1,), "root", (1,), "G2", (1,), "G1", (1,)]

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
        self.recorder.close()

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

        g1_expected = (g1_expected_params, g1_expected_unknowns, g1_expected_resids)

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
        expected.append((solver_coordinate, (t0, t1), g1_expected_params, g1_expected_unknowns, g1_expected_resids))
        expected.append((driver_coordinate, (t0, t1), driver_expected_params, driver_expected_unknowns, driver_expected_resids))

        self.assertIterationDataRecorded(expected, self.eps)

    def test_driver_records_metadata(self):
        prob = Problem()
        prob.root = ConvergeDiverge()
        prob.driver.add_recorder(self.recorder)
        self.recorder.options['record_metadata'] = True
        prob.setup(check=False)
        self.recorder.close()

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
        self.recorder.close()

        self.assertMetadataRecorded(None)

    def test_root_solver_records_metadata(self):
        prob = Problem()
        prob.root = ConvergeDiverge()
        prob.root.nl_solver.add_recorder(self.recorder)
        self.recorder.options['record_metadata'] = True
        prob.setup(check=False)
        self.recorder.close()

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
        self.recorder.close()

        self.assertMetadataRecorded(None)

    def test_subsolver_records_metadata(self):
        prob = Problem()
        prob.root = ExampleGroup()
        prob.root.G2.G1.nl_solver.add_recorder(self.recorder)
        self.recorder.options['record_metadata'] = True
        prob.setup(check=False)
        self.recorder.close()

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
        self.recorder.close()

        self.assertMetadataRecorded(None)

if __name__ == "__main__":
    unittest.main()
