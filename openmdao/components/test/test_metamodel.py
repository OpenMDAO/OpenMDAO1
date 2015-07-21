
import unittest
import math
from six import iteritems

import numpy as np
from math import sin

from openmdao.core.component import Component
from openmdao.core.group import Group
from openmdao.core.problem import Problem
from openmdao.components.metamodel import MetaModel
from openmdao.surrogatemodels.response_surface import ResponseSurface
from openmdao.surrogatemodels.kriging import FloatKrigingSurrogate


class TestMetaModel(unittest.TestCase):

    def test_sin_metamodel(self):
        class Sin(Component):
            """ Simple sine calculation. """
            def __init__(self):
                self.add_param('x', 0., units="rad")
                self.add_output('f_x', 0.)

            def solve_nonlinear(self, params, unknowns, resids):
                unknowns['f_x'] = .5*sin(params['x'])

        class Simulation(Group):
            """ Top level assembly for MetaModel of a sine component using a
            Kriging surrogate."""

            def __init__(self):
                super(Simulation, self).__init__()

                # Create meta_model for f_x as the response
                sin_mm = self.add("sin_mm", MetaModel(params = ('x', ),
                                                      responses = ('f_x', )))

                # this should force sin_mm to have a paramter `train:x`, and an output `train:f_x`
                # for training data, and an `x` and `f_x` variales for prediction

                # Use Kriging for the f_x output
                sin_mm.default_surrogate = FloatKrigingSurrogate()
                #sin_mm.surrogates = {'f_x': FloatKrigingSurrogate} (allowable to set one surrogate per response)

        prob = Problem(Simulation())
        prob.setup(check=False)

        prob['sin_mm.train:x'] = np.linspace(0,10,200)
        print(np.sin(prob['sin_mm.train:x']))
        prob['sin_mm.train:f_x'] = np.sin(prob['sin_mm.train:x'])

        prob['sin_mm.x'] = 2.1 #prediction should happen at this point

        prob.run()

        print(prob['sin_mm.f_x']) #predicted value

    def test_basics(self):

        meta = MetaModel()

        meta.add_param('x1', 0.)
        meta.add_param('x2', 0.)

        meta.add_unknown('y1', 0.)
        meta.add_unknown('y2', 0.)

        meta.params.x1 = [1.0, 2.0, 3.0]
        meta.params.x2 = [1.0, 3.0, 4.0]
        meta.responses.y1 = [3.0, 2.0, 1.0]
        meta.responses.y2 = [1.0, 4.0, 7.0]

        meta.default_surrogate = ResponseSurface()

        meta.x1 = 2.0
        meta.x2 = 3.0

        prob = Problem(root=Group())
        prob.add('meta', meta)
        prob.meta.run()

        assert_rel_error(self, prob.meta.y1, 2.0, .00001)
        assert_rel_error(self, prob.meta.y2, 4.0, .00001)

        prob.meta.x1 = 2.5
        prob.meta.x2 = 3.5
        prob.meta.run()

        assert_rel_error(self, prob.meta.y1, 1.5934, .001)

        # Slot a new surrogate.
        prob.meta.default_surrogate = FloatKrigingSurrogate()
        self.assertTrue(prob.meta._train == True)

        prob.meta.run()

        assert_rel_error(self, prob.meta.y1, 1.4609, .001)

    def test_train(self):
        # generate training data for paraboloid
        data = {}
        data['x'] = []
        data['y'] = []
        data['f_xy'] = []
        for x, y in [(1, 2), (3, 4), (5, 6), (7, 8), (9,10)]:
            data['x'].append(x)
            data['y'].append(y)
            data['f_xy'].append((x-3.0)**2 + x*y + (y+4.0)**2 - 3.0)

        # crate metamodel
        meta = MetaModel(params=('x', 'y'),
                         responses=('f_xy'))
        meta.default_surrogate = FloatKrigingSurrogate()

        meta.params.x = data['x']
        meta.params.y = data['y']
        meta.responses.f_xy = data['f_xy']

        prob = Problem(root=Group())
        prob.add('meta', meta)
        prob.setup(check=False)
        prob.run()


if __name__ == "__main__":
    unittest.main()
