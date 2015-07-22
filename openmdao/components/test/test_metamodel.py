
import unittest
import math
from six import iteritems
from six.moves import cStringIO

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

        # Create meta_model for f_x as the response
        sin_mm = MetaModel()

        # the following will cause sin_mm to have
        # a paramter `train:x`, and an output `train:f_x` for training data
        # and an `x` and `f_x` variables for prediction
        sin_mm.add_param('x', 0.)
        sin_mm.add_output('f_x', 0.)
        #sin_mm.add_output('f_x', 0., surrogate=FloatKrigingSurrogate())

        # Use Kriging for the f_x output
        #sin_mm.default_surrogate = FloatKrigingSurrogate()
        #sin_mm.surrogates = {'f_x': FloatKrigingSurrogate} (allowable to set one surrogate per response)

        prob = Problem(Group())
        prob.root.add('sin_mm', sin_mm)

        # check that missing surrogate is detected in check_setup
        stream = cStringIO()
        prob.setup(out_stream=stream)
        msg = ("No default surrogate model is defined and the "
               "following outputs do not have a surrogate model:\n"
               "['f_x']\n"
               "Either specify a default_surrogate, or specify a "
               "surrogate model for all outputs.")
        self.assertTrue(msg in stream.getvalue())

        sin_mm.default_surrogate = FloatKrigingSurrogate()
        prob.setup(check=False)

        prob['sin_mm.train:x'] = np.linspace(0,10,200)
        print('\nTraining Inputs:')
        print(prob['sin_mm.train:x'])

        prob['sin_mm.train:f_x'] = .5*np.sin(prob['sin_mm.train:x'])
        print('\nTraining Outputs:')
        print(prob['sin_mm.train:f_x'])

        prob['sin_mm.x'] = 2.22 #prediction should happen at this point

        prob.run()

        print('\nPredicted Values:')
        print(prob['sin_mm.f_x']) #predicted value
        expected_value = .5*np.sin(prob['sin_mm.x'])
        self.assertAlmostEqual(prob['sin_mm.f_x'], expected_value, places=5)

    def test_basics(self):

        mm = MetaModel()

        mm.add_param('x1', 0.)
        mm.add_param('x2', 0.)

        mm.add_unknown('y1', 0.)
        mm.add_unknown('y2', 0.)

        mm.default_surrogate = ResponseSurface()

        prob = Problem(root=Group())
        prob.add('mm', mm)
        prob.setup()

        prob['mm.train:x1'] = [1.0, 2.0, 3.0]
        prob['mm.train:x2'] = [1.0, 3.0, 4.0]
        prob['mm.train:y1'] = [3.0, 2.0, 1.0]
        prob['mm.train:y2'] = [1.0, 4.0, 7.0]

        prob['mm.x1'] = 2.0
        prob['mm.x2'] = 3.0

        prob.run()

        assert_rel_error(self, prob['mm.y1'], 2.0, .00001)
        assert_rel_error(self, prob['mm.y1'], 4.0, .00001)

        prob['mm.x1'] = 2.5
        prob['mm.x2'] = 3.5
        prob.mm.run()

        assert_rel_error(self, prob['mm.y1'], 1.5934, .001)

        # Slot a new surrogate.
        prob.mm.default_surrogate = FloatKrigingSurrogate()
        prob.setup()

        self.assertTrue(prob.mm._train == True)

        prob.mm.run()

        assert_rel_error(self, prob.mm.y1, 1.4609, .001)

    def test_train(self):
        #generate training data for paraboloid
        data = {}
        data['x'] = []
        data['y'] = []
        data['f_xy'] = []
        for x, y in [(1, 2), (3, 4), (5, 6), (7, 8), (9,10)]:
            data['x'].append(x)
            data['y'].append(y)
            data['f_xy'].append((x-3.0)**2 + x*y + (y+4.0)**2 - 3.0)

        # crate metamodel
        mm = MetaModel(params=('x', 'y'),
                         responses=('f_xy'))
        mm.default_surrogate = FloatKrigingSurrogate()

        mm.params.x = data['x']
        mm.params.y = data['y']
        mm.responses.f_xy = data['f_xy']

        prob = Problem(root=Group())
        prob.add('meta', meta)
        prob.setup(check=False)
        prob.run()


if __name__ == "__main__":
    unittest.main()
