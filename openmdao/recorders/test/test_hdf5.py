""" Unit test for the HDF5Recorder. """

import unittest
from six import assertCountEqual
from openmdao.recorders.hdf5recorder import HDF5Recorder
from openmdao.core.problem import Problem
from openmdao.test.converge_diverge import ConvergeDiverge
from openmdao.test.examplegroups import ExampleGroup
from openmdao.test.testutil import assert_rel_error

class TestHDF5Recorder(unittest.TestCase):
    eps = 1e-5

    def assertDatasetEquals(self, actual, expected, tolerance):
        # If len(actual) == len(expected) and actual < expected, then
        # actual == expected.
        self.assertEqual(len(actual), len(expected))

        sentinel = object()
        for key, val in expected.items():
            found_val = actual.get(key, sentinel)
            if found_val is sentinel:
                self.fail("Did not find key '{0}'.".format(key))
            assert_rel_error(self, found_val[()], expected[key], tolerance)

    def test_basic(self):

        top = Problem()
        top.root = ConvergeDiverge()

        recorder = HDF5Recorder('tmp.hdf5', driver='core', backing_store=False)
        top.driver.add_recorder(recorder)
        top.setup()
        top.run()

        f = recorder.out['Driver/1']
        params = f['Parameters']
        unknowns = f['Unknowns']
        resids = f['Residuals']

        expected_params = {
            "comp1.x1": 2.0,
            "comp2.x1": 8.0,
            "comp3.x1": 6.0,
            "comp4.x1": 4.0,
            "comp4.x2": 21.0,
            "comp5.x1": 46.0,
            "comp6.x1": -93.0,
            "comp7.x1": 36.8,
            "comp7.x2": -46.5}
        expected_unknowns = {
            "comp1.y1": 8.0,
            "comp1.y2": 6.0,
            "comp2.y1": 4.0,
            "comp3.y1": 21.0,
            "comp4.y1": 46.0,
            "comp4.y2": -93.0,
            "comp5.y1": 36.8,
            "comp6.y1": -46.5,
            "comp7.y1": -102.7,
            "p.x": 2.0}
        expected_resids = {
            "comp1.y1": 0.0,
            "comp1.y2": 0.0,
            "comp2.y1": 0.0,
            "comp3.y1": 0.0,
            "comp4.y1": 0.0,
            "comp4.y2": 0.0,
            "comp5.y1": 0.0,
            "comp6.y1": 0.0,
            "comp7.y1": 0.0,
            "p.x": 0.0}

        self.assertDatasetEquals(params, expected_params, self.eps)
        self.assertDatasetEquals(unknowns, expected_unknowns, self.eps)
        self.assertDatasetEquals(resids, expected_resids, self.eps)


    def test_excludes(self):

        top = Problem()
        top.root = ConvergeDiverge()

        recorder = HDF5Recorder('tmp.hdf5', driver='core', backing_store=False)
        recorder.options['excludes'] = ['comp4.*']
        top.driver.add_recorder(recorder)
        top.setup()
        top.run()

        f = recorder.out['Driver/1']
        params = f['Parameters']
        unknowns = f['Unknowns']
        resids = f['Residuals']

        expected_params = {
            "comp1.x1": 2.0,
            "comp2.x1": 8.0,
            "comp3.x1": 6.0,
            "comp5.x1": 46.0,
            "comp6.x1": -93.0,
            "comp7.x1": 36.8,
            "comp7.x2": -46.5}
        expected_unknowns = {
            "comp1.y1": 8.0,
            "comp1.y2": 6.0,
            "comp2.y1": 4.0,
            "comp3.y1": 21.0,
            "comp5.y1": 36.8,
            "comp6.y1": -46.5,
            "comp7.y1": -102.7,
            "p.x": 2.0}
        expected_resids = {
            "comp1.y1": 0.0,
            "comp1.y2": 0.0,
            "comp2.y1": 0.0,
            "comp3.y1": 0.0,
            "comp5.y1": 0.0,
            "comp6.y1": 0.0,
            "comp7.y1": 0.0,
            "p.x": 0.0}

        self.assertDatasetEquals(params, expected_params, self.eps)
        self.assertDatasetEquals(unknowns, expected_unknowns, self.eps)
        self.assertDatasetEquals(resids, expected_resids, self.eps)

    def test_includes(self):

        top = Problem()
        top.root = ConvergeDiverge()

        recorder = HDF5Recorder('tmp.hdf5', driver='core', backing_store=False)
        recorder.options['includes'] = ['comp1.*']
        top.driver.add_recorder(recorder)
        top.setup()
        top.run()

        f = recorder.out['Driver/1']
        params = f['Parameters']
        unknowns = f['Unknowns']
        resids = f['Residuals']

        expected_params = {
            "comp1.x1": 2.0,}
        expected_unknowns = {
            "comp1.y1": 8.0,
            "comp1.y2": 6.0,}
        expected_resids = {
            "comp1.y1": 0.0,
            "comp1.y2": 0.0,}

        self.assertDatasetEquals(params, expected_params, self.eps)
        self.assertDatasetEquals(unknowns, expected_unknowns, self.eps)
        self.assertDatasetEquals(resids, expected_resids, self.eps)

    def test_includes_and_excludes(self):

        top = Problem()
        top.root = ConvergeDiverge()

        recorder = HDF5Recorder('tmp.hdf5', driver='core', backing_store=False)
        recorder.options['includes'] = ['comp1.*']
        recorder.options['excludes'] = ['*.y2']
        top.driver.add_recorder(recorder)
        top.setup()
        top.run()

        f = recorder.out['Driver/1']
        params = f['Parameters']
        unknowns = f['Unknowns']
        resids = f['Residuals']

        expected_params = {
            "comp1.x1": 2.0, }
        expected_unknowns = {
            "comp1.y1": 8.0, }
        expected_resids = {
            "comp1.y1": 0.0, }

        self.assertDatasetEquals(params, expected_params, self.eps)
        self.assertDatasetEquals(unknowns, expected_unknowns, self.eps)
        self.assertDatasetEquals(resids, expected_resids, self.eps)

    def test_solver_record(self):

        top = Problem()
        top.root = ConvergeDiverge()
        recorder = HDF5Recorder('tmp.hdf5', driver='core', backing_store=False)
        top.root.nl_solver.add_recorder(recorder)

        top.setup()
        top.run()

        f = recorder.out['Driver/1/root/1']
        params = f['Parameters']
        unknowns = f['Unknowns']
        resids = f['Residuals']

        expected_params = {
            "comp1.x1": 2.0,
            "comp2.x1": 8.0,
            "comp3.x1": 6.0,
            "comp4.x1": 4.0,
            "comp4.x2": 21.0,
            "comp5.x1": 46.0,
            "comp6.x1": -93.0,
            "comp7.x1": 36.8,
            "comp7.x2": -46.5}
        expected_unknowns = {
            "comp1.y1": 8.0,
            "comp1.y2": 6.0,
            "comp2.y1": 4.0,
            "comp3.y1": 21.0,
            "comp4.y1": 46.0,
            "comp4.y2": -93.0,
            "comp5.y1": 36.8,
            "comp6.y1": -46.5,
            "comp7.y1": -102.7,
            "p.x": 2.0}
        expected_resids = {
            "comp1.y1": 0.0,
            "comp1.y2": 0.0,
            "comp2.y1": 0.0,
            "comp3.y1": 0.0,
            "comp4.y1": 0.0,
            "comp4.y2": 0.0,
            "comp5.y1": 0.0,
            "comp6.y1": 0.0,
            "comp7.y1": 0.0,
            "p.x": 0.0}

        self.assertDatasetEquals(params, expected_params, self.eps)
        self.assertDatasetEquals(unknowns, expected_unknowns, self.eps)
        self.assertDatasetEquals(resids, expected_resids, self.eps)

    def test_sublevel_record(self):

        top = Problem()
        top.root = ExampleGroup()
        recorder = HDF5Recorder('tmp.hdf5', driver='core', backing_store=False)
        top.root.G2.G1.nl_solver.add_recorder(recorder)

        top.setup()
        top.run()

        f = recorder.out['Driver/1/root/1/G2/1/G1/1']
        params = f['Parameters']
        unknowns = f['Unknowns']
        resids = f['Residuals']

        expected_params = {
            "C2.x": 5.0}
        expected_unknowns = {
            "C2.y": 10.0}
        expected_resids = {
            "C2.y": 0.0}

        self.assertDatasetEquals(params, expected_params, self.eps)
        self.assertDatasetEquals(unknowns, expected_unknowns, self.eps)
        self.assertDatasetEquals(resids, expected_resids, self.eps)

if __name__ == "__main__":
    unittest.main()
