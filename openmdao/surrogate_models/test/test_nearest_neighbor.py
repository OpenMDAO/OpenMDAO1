import numpy as np
import unittest

from openmdao.surrogate_models import NearestNeighbor
from openmdao.test.util import assert_rel_error


class TestNearestNeighbor(unittest.TestCase):

    def test_unrecognized_type(self):
        with self.assertRaises(ValueError) as cm:
            NearestNeighbor(interpolant_type='junk')

        expected_msg = "NearestNeighbor: interpolant_type 'junk' not supported." \
                       " interpolant_type must be one of ['linear', 'weighted'," \
                       " 'cosine', 'hermite', 'rbf']."

        self.assertEqual(expected_msg, str(cm.exception))


class TestLinearInterpolator1D(unittest.TestCase):
    def setUp(self):
        self.surrogate = NearestNeighbor(interpolant_type='linear')
        self.x = np.array([[0.], [1.], [2.], [3.]])
        self.y = np.array([[0.], [1.], [1.], [0.]])
        self.surrogate.train(self.x, self.y)

    def test_training(self):
        for x0, y0 in zip(self.x, self.y):
            mu = self.surrogate.predict(x0)
            assert_rel_error(self, mu, y0, 1e-9)

    def test_prediction(self):
        test_x = np.array([[0.5], [1.5], [2.5]])
        expected_y = np.array([[0.5], [1.], [0.5]])

        for x0, y0 in zip(test_x, expected_y):
            mu = self.surrogate.predict(x0)
            assert_rel_error(self, mu, y0, 1e-9)

    def test_bulk_prediction(self):
        test_x = np.array([[0.5], [1.5], [2.5]])
        expected_y = np.array([[0.5], [1.], [0.5]])
        mu = self.surrogate.predict(test_x)
        assert_rel_error(self, mu, expected_y, 1e-9)

    def test_jacobian(self):
        test_x = np.array([[0.5], [1.5], [2.5]])
        expected_deriv = np.array([[1.], [0.], [-1.]])

        for x0, y0 in zip(test_x, expected_deriv):
            jac = self.surrogate.jacobian(x0)
            assert_rel_error(self, jac, y0, 1e-9)


class TestLinearInterpolatorND(unittest.TestCase):
    def setUp(self):
        self.surrogate = NearestNeighbor(interpolant_type='linear')
        self.x = np.array([[0., 0.], [2., 0.], [2., 2.], [0., 2.], [1., 1.]])
        self.y = np.array([[1., 0., .5, 1.],
                           [1., 0., .5, 1.],
                           [1., 0., .5, 1.],
                           [1., 0., .5, 1.],
                           [0., 1., .5, 0.]])
        self.surrogate.train(self.x, self.y)

    def test_training(self):
        for x0, y0 in zip(self.x, self.y):
            mu = self.surrogate.predict(x0)
            assert_rel_error(self, mu, y0, 1e-9)

    def test_prediction(self):
        test_x = np.array([[1., 0.5],
                           [0.5, 1.0],
                           [1.0, 1.5],
                           [1.5, 1.],
                           [0., 1.],
                           [.5, .5]
                           ])
        expected_y = np.array([[0.5, 0.5, 0.5, 0.5],
                               [0.5, 0.5, 0.5, 0.5],
                               [0.5, 0.5, 0.5, 0.5],
                               [0.5, 0.5, 0.5, 0.5],
                               [1., 0., 0.5, 1.],
                               [0.5, 0.5, 0.5, 0.5]
                               ])

        for x0, y0 in zip(test_x, expected_y):
            mu = self.surrogate.predict(x0)
            assert_rel_error(self, mu, y0, 1e-9)

    def test_bulk_prediction(self):
        test_x = np.array([[1., 0.5],
                           [0.5, 1.0],
                           [1.0, 1.5],
                           [1.5, 1.],
                           [0., 1.],
                           [.5, .5]
                           ])
        expected_y = np.array([[0.5, 0.5, 0.5, 0.5],
                               [0.5, 0.5, 0.5, 0.5],
                               [0.5, 0.5, 0.5, 0.5],
                               [0.5, 0.5, 0.5, 0.5],
                               [1., 0., 0.5, 1.],
                               [0.5, 0.5, 0.5, 0.5]
                               ])

        mu = self.surrogate.predict(test_x)
        assert_rel_error(self, mu, expected_y, 1e-9)

    def test_jacobian(self):
        test_x = np.array([[1., 0.5],
                           [0.5, 1.],
                           [1., 1.5],
                           [1.5, 1.]
                           ])
        expected_deriv = list(map(np.array, [
            [[0., -1.], [0., 1.], [0., 0.], [0., -1.]],
            [[-1., 0.], [1., 0.], [0., 0.], [-1., 0.]],
            [[0., 1.], [0., -1.], [0., 0.], [0., 1.]],
            [[1., 0.], [-1., 0.], [0., 0.], [1., 0.]]
            ]))

        for x0, y0 in zip(test_x, expected_deriv):
            mu = self.surrogate.jacobian(x0)
            assert_rel_error(self, mu, y0, 1e-9)
