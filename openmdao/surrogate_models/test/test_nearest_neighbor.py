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


class TestWeightedInterpolator1D(unittest.TestCase):
    def setUp(self):
        self.surrogate = NearestNeighbor(interpolant_type='weighted')
        self.x = np.array([[0.], [1.], [2.], [3.]])
        self.y = np.array([[0.], [1.], [1.], [0.]])
        self.surrogate.train(self.x, self.y)

    def test_insufficient_points(self):
        with self.assertRaises(ValueError) as cm:
            self.surrogate.predict(self.x[0], n=100)

        expected_msg = ('WeightedInterpolant does not have sufficient '
            'training data to use n=100, only 4 points available.')

        self.assertEqual(str(cm.exception), expected_msg)

    def test_training(self):
        for x0, y0 in zip(self.x, self.y):
            mu = self.surrogate.predict(x0, n=3)
            assert_rel_error(self, mu, y0, 1e-9)

    def test_prediction(self):
        test_x = np.array([[0.5], [1.5], [2.5]])
        expected_y = np.array([[0.52631579], [0.94736842], [0.52631579]])

        for x0, y0 in zip(test_x, expected_y):
            mu = self.surrogate.predict(x0, n=3)
            assert_rel_error(self, mu, y0, 1e-8)

    def test_bulk_prediction(self):

        test_x = np.array([[0.5], [1.5], [2.5]])
        expected_y = np.array([[0.52631579], [0.94736842], [0.52631579]])

        mu = self.surrogate.predict(test_x, n=3)
        assert_rel_error(self, mu, expected_y, 1e-8)

    def test_jacobian(self):
        test_x = np.array([[0.5], [1.5], [2.5]])
        expected_deriv = np.array([[1.92797784], [0.06648199], [-1.92797784]])

        for x0, y0 in zip(test_x, expected_deriv):
            jac = self.surrogate.jacobian(x0, n=3)
            assert_rel_error(self, jac, y0, 1e-6)


class TestWeightedInterpolatorND(unittest.TestCase):
    def setUp(self):
        self.surrogate = NearestNeighbor(interpolant_type='weighted')
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
        a = ((16. / (5 * np.sqrt(5.)) + 16. / (13. * np.sqrt(13.))) /
             (16. / (5 * np.sqrt(5.)) + 16. / (13. * np.sqrt(13.)) + 8.))

        b = 8. / (8. + 16. / (5 * np.sqrt(5.)) + 16. / (13. * np.sqrt(13.)))
        c = (2. + 2. / (5. * np.sqrt(5))) / (3. + 2. / (5. * np.sqrt(5)))
        d = 1. / (3. + 2. / (5. * np.sqrt(5)))

        expected_y = np.array([[a, b, 0.5, a],
                               [a, b, 0.5, a],
                               [a, b, 0.5, a],
                               [a, b, 0.5, a],
                               [c, d, 0.5, c],
                               [0.54872067, 0.45127933, 0.5, 0.54872067]
                               ])

        for x0, y0 in zip(test_x, expected_y):
            mu = self.surrogate.predict(x0, n=5, dist_eff=3)
            assert_rel_error(self, mu, y0, 1e-6)

    def test_bulk_prediction(self):
        test_x = np.array([[1., 0.5],
                           [0.5, 1.0],
                           [1.0, 1.5],
                           [1.5, 1.],
                           [0., 1.],
                           [.5, .5]
                           ])

        a = ((16. / (5 * np.sqrt(5.)) + 16. / (13. * np.sqrt(13.))) /
             (16./(5*np.sqrt(5.)) + 16. / (13. * np.sqrt(13.)) + 8.))
        b = 8. / (8. + 16. / (5 * np.sqrt(5.)) + 16. / (13. * np.sqrt(13.)))
        c = (2. + 2./(5.*np.sqrt(5))) / (3. + 2. / (5. * np.sqrt(5)))
        d = 1. / (3. + 2. / (5. * np.sqrt(5)))

        expected_y = np.array([[a, b, 0.5, a],
                               [a, b, 0.5, a],
                               [a, b, 0.5, a],
                               [a, b, 0.5, a],
                               [c, d, 0.5, c],
                               [0.54872067, 0.45127933, 0.5, 0.54872067]
                               ])

        mu = self.surrogate.predict(test_x, n=5, dist_eff=3)
        assert_rel_error(self, mu, expected_y, 1e-6)

    def test_jacobian(self):
        test_x = np.array([[1., 0.5],
                           [0.5, 1.],
                           [1., 1.5],
                           [1.5, 1.]
                           ])
        a = 0.99511746
        expected_deriv = list(map(np.array, [
            [[0., -a], [0., a], [0., 0.], [0., -a]],
            [[-a, 0], [a, 0.], [0., 0.], [-a, 0]],
            [[0., a], [0., -a], [0., 0.], [0., a]],
            [[a, 0.], [-a, 0.], [0., 0.], [a, 0.]]
        ]))

        for x0, y0 in zip(test_x, expected_deriv):
            mu = self.surrogate.jacobian(x0)
            assert_rel_error(self, mu, y0, 1e-6)
