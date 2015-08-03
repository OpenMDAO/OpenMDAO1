# pylint: disable-msg=C0111,C0103

import unittest
import random

from numpy import array, linspace, sin, cos, pi

from openmdao.surrogate_models import ResponseSurface
from openmdao.test.util import assert_rel_error
from six.moves import zip


def branin(x):
    y = (x[1] - (5.1 / (4. * pi ** 2.)) * x[0] ** 2. + 5. * x[0] / pi - 6.) ** 2. + 10. * (1. - 1. / (8. * pi)) * cos(
        x[0]) + 10.
    return y


def branin_1d(x):
    return branin(array([x[0], 2.275]))


class TestResponseSurfaceSurrogate(unittest.TestCase):
    def setUp(self):
        random.seed(10)

    def test_1d_training(self):

        x = array([[0.0], [2.0], [3.0]])
        y = array([[branin_1d(case)] for case in x])
        surrogate = ResponseSurface()
        surrogate.train(x, y)

        for x0, y0 in zip(x, y):
            mu = surrogate.predict(x0)
            assert_rel_error(self, mu, y0, 1e-9)

    def test_1d_predictor(self):
        x = array([[0.0], [2.0], [3.0], [4.0], [6.0]])
        y = array([[branin_1d(case)] for case in x])

        surrogate = ResponseSurface()
        surrogate.train(x, y)

        new_x = array([pi])
        mu = surrogate.predict(new_x)

        assert_rel_error(self, mu, 1.73114, 1e-4)

    def test_1d_ill_conditioned(self):
        # Test for least squares solver utilization when ill-conditioned
        x = array([[case] for case in linspace(0., 1., 40)])
        y = sin(x)
        surrogate = ResponseSurface()
        surrogate.train(x, y)
        new_x = array([0.5])
        mu = surrogate.predict(new_x)

        assert_rel_error(self, mu, sin(0.5), 1e-3)

    def test_2d(self):

        x = array([[-2., 0.], [-0.5, 1.5], [1., 1.], [0., .25]])
        y = array([[branin(case)] for case in x])

        surrogate = ResponseSurface()
        surrogate.train(x, y)

        for x0, y0 in zip(x, y):
            mu = surrogate.predict(x0)
            assert_rel_error(self, mu, y0, 1e-9)

        mu = surrogate.predict([.5, .5])

        assert_rel_error(self, mu, branin([.5, .5]), 1e-1)

    def test_no_training_data(self):
        surrogate = ResponseSurface()

        try:
            surrogate.predict([0., 1.])
        except RuntimeError as err:
            self.assertEqual(str(err),
                             "ResponseSurface has not been trained, so no prediction can be made.")
        else:
            self.fail("RuntimeError Expected")

    def test_one_pt(self):
        surrogate = ResponseSurface()
        x = [[0.]]
        y = [[1.]]

        surrogate.train(x,y)
        assert_rel_error(self, surrogate.betas, array([[1.], [0.], [0.]]), 1e-9)

    def test_vector_input(self):
        surrogate = ResponseSurface()

        x = array([[0., 0., 0.], [1., 1., 1.]])
        y = array([[0.], [3.]])

        surrogate.train(x, y)

        for x0, y0 in zip(x, y):
            mu = surrogate.predict(x0)
            assert_rel_error(self, mu, y0, 1e-9)

    def test_vector_output(self):
        surrogate = ResponseSurface()

        y = array([[0., 0.], [1., 1.], [2., 0.]])
        x = array([[0.], [2.], [4.]])

        surrogate.train(x, y)

        for x0, y0 in zip(x, y):
            mu = surrogate.predict(x0)
            assert_rel_error(self, mu, y0, 1e-9)


if __name__ == "__main__":
    unittest.main()
