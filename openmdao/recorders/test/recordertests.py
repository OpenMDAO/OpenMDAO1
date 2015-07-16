import unittest
from openmdao.core.problem import Problem
from openmdao.recorders.baserecorder import BaseRecorder
from openmdao.test.converge_diverge import ConvergeDiverge
from openmdao.test.examplegroups import ExampleGroup

class RecorderTests(object):
    class Tests(unittest.TestCase):
        recorder = BaseRecorder()
        eps = 1e-5

        def assertDatasetEquals(self, expected, tolerance):
            self.fail("assertDatasetEquals not implemented!")

        def tearDown(self):
            self.recorder.close()

        def test_basic(self):
            prob = Problem()
            prob.root = ConvergeDiverge()
            prob.driver.add_recorder(self.recorder)
            prob.setup(check=False)
            prob.run()

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

            expected = (expected_params, expected_unknowns, expected_resids)

            self.assertDatasetEquals(
                [(['Driver', (1,)], expected)],
                self.eps
            )

        def test_includes(self):
            prob = Problem()
            prob.root = ConvergeDiverge()
            prob.driver.add_recorder(self.recorder)
            self.recorder.options['includes'] = ['comp1.*']
            prob.setup(check=False)
            prob.run()

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

            expected = (expected_params, expected_unknowns, expected_resids)

            self.assertDatasetEquals(
                [(['Driver', (1,)], expected)],
                self.eps
            )

        def test_includes_and_excludes(self):
            prob = Problem()
            prob.root = ConvergeDiverge()
            prob.driver.add_recorder(self.recorder)
            self.recorder.options['includes'] = ['comp1.*']
            self.recorder.options['excludes'] = ["*.y2"]
            prob.setup(check=False)
            prob.run()

            expected_params = [
                ("comp1.x1", 2.0)
            ]
            expected_unknowns = [
                ("comp1.y1", 8.0)
            ]
            expected_resids = [
                ("comp1.y1", 0.0)
            ]

            expected = (expected_params, expected_unknowns, expected_resids)

            self.assertDatasetEquals(
                [(['Driver', (1,)], expected)],
                self.eps
            )

        def test_solver_record(self):
            prob = Problem()
            prob.root = ConvergeDiverge()
            prob.root.nl_solver.add_recorder(self.recorder)
            prob.setup(check=False)
            prob.run()

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

            expected = (expected_params, expected_unknowns, expected_resids)

            self.assertDatasetEquals(
                [(['Driver', (1,), "root", (1,)], expected)],
                self.eps
            )

        def test_sublevel_record(self):

            prob = Problem()
            prob.root = ExampleGroup()
            prob.root.G2.G1.nl_solver.add_recorder(self.recorder)
            prob.setup(check=False)
            prob.run()

            expected_params = [
                ("C2.x", 5.0)
            ]
            expected_unknowns = [
                ("C2.y", 10.0)
            ]
            expected_resids = [
                ("C2.y", 0.0)
            ]

            expected = (expected_params, expected_unknowns, expected_resids)

            self.assertDatasetEquals(
                [(['Driver', (1,), "root", (1,), "G2", (1,), "G1", (1,)], expected)],
                self.eps
            )

        def test_multilevel_record(self):
            prob = Problem()
            prob.root = ExampleGroup()
            prob.root.G2.G1.nl_solver.add_recorder(self.recorder)
            prob.driver.add_recorder(self.recorder)
            prob.setup(check=False)
            prob.run()

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

            driver_expected = (driver_expected_params, driver_expected_unknowns, driver_expected_resids)

            self.assertDatasetEquals(
                [(['Driver', (1,), "root", (1,), "G2", (1,), "G1", (1,)], g1_expected),
                 (['Driver', (1,)], driver_expected)],
                self.eps
            )
