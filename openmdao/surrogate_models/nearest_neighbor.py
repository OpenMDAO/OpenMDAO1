# Based off of the N-Dimensional Interpolation library by Stephen Marone.
# https://github.com/SMarone/NDInterp

import numpy as np

from collections import OrderedDict
from openmdao.surrogate_models.surrogate_model import SurrogateModel
from openmdao.surrogate_models.nn_interpolators.linear_interpolator import \
    LinearInterpolator


_interpolators = OrderedDict([('linear', LinearInterpolator),
                  ('weighted', None),
                  ('cosine', None),
                  ('hermite', None),
                  ('rbf', None)])


class NearestNeighbor(SurrogateModel):
    def __init__(self, interpolant_type='rbf'):
        super(NearestNeighbor, self).__init__()

        if interpolant_type not in _interpolators.keys():
            msg = "NearestNeighbor: interpolant_type '{0}' not supported." \
                  " interpolant_type must be one of {1}.".format(
                      interpolant_type, list(_interpolators.keys())
                  )
            raise ValueError(msg)

        self.interpolant_type = interpolant_type
        self.interpolant = None

    def train(self, x, y):
        super(NearestNeighbor, self).train(x, y)
        self.interpolant = _interpolators[self.interpolant_type](x, y)

    def predict(self, x):
        super(NearestNeighbor, self).predict(x)
        return self.interpolant(x)

    def jacobian(self, x):
        jac = self.interpolant.gradient(x)
        if jac.shape[0] == 1:
            return jac[0, ...]
        return jac
