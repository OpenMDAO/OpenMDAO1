
import unittest
import math
from six.moves import cStringIO

import numpy as np
from math import sin

from openmdao.core import Component, Group, Problem
from openmdao.components import MetaModel
from openmdao.surrogate_models import ResponseSurface, FloatKrigingSurrogate

from openmdao.test.util import assert_rel_error


class TestMetaModel(unittest.TestCase):

    def test_sin_metamodel(self):
        class Sin(Component):
            """ Simple sine calculation. """
            def __init__(self):
                self.add_param('x', 0., units="rad")
                self.add_output('f_x', 0.)

            def solve_nonlinear(self, params, unknowns, resids):
                unknowns['f_x'] = .5*sin(params['x'])

        # create a MetaModel for Sin and add it to a Problem
        sin_mm = MetaModel()
        sin_mm.add_param('x', 0.)
        sin_mm.add_output('f_x', 0.)

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

        # check that output with no specified surrogate gets the default
        sin_mm.default_surrogate = FloatKrigingSurrogate()
        prob.setup(check=False)
        surrogate = prob.root.unknowns.metadata('sin_mm.f_x').get('surrogate')
        self.assertTrue(isinstance(surrogate, FloatKrigingSurrogate),
                        'sin_mm.f_x should get the default surrogate')

        # train the surrogate and check predicted value
        prob['sin_mm.train:x'] = np.linspace(0,10,200)
        prob['sin_mm.train:f_x'] = .5*np.sin(prob['sin_mm.train:x'])

        prob['sin_mm.x'] = 2.22

        prob.run()

        self.assertAlmostEqual(prob['sin_mm.f_x'],
                               .5*np.sin(prob['sin_mm.x']),
                               places=5)

    def test_basics(self):
        # create a metamodel component
        mm = MetaModel()

        mm.add_param('x1', 0.)
        mm.add_param('x2', 0.)

        mm.add_output('y1', 0.)
        mm.add_output('y2', 0., surrogate=FloatKrigingSurrogate())

        mm.default_surrogate = ResponseSurface()

        # add metamodel to a problem
        prob = Problem(root=Group())
        prob.root.add('mm', mm)
        prob.setup(check=False)

        # check that surrogates were properly assigned
        surrogate = prob.root.unknowns.metadata('mm.y1').get('surrogate')
        self.assertTrue(isinstance(surrogate, ResponseSurface))

        surrogate = prob.root.unknowns.metadata('mm.y2').get('surrogate')
        self.assertTrue(isinstance(surrogate, FloatKrigingSurrogate))

        # populate training data
        prob['mm.train:x1'] = [1.0, 2.0, 3.0]
        prob['mm.train:x2'] = [1.0, 3.0, 4.0]
        prob['mm.train:y1'] = [3.0, 2.0, 1.0]
        prob['mm.train:y2'] = [1.0, 4.0, 7.0]

        # run problem for provided data point and check prediction
        prob['mm.x1'] = 2.0
        prob['mm.x2'] = 3.0

        self.assertTrue(mm.train)   # training will occur before 1st run
        prob.run()

        assert_rel_error(self, prob['mm.y1'], 2.0, .00001)
        assert_rel_error(self, prob['mm.y2'], 4.0, .00001)

        # run problem for interpolated data point and check prediction
        prob['mm.x1'] = 2.5
        prob['mm.x2'] = 3.5

        self.assertFalse(mm.train)  # training will not occur before 2nd run
        prob.run()

        assert_rel_error(self, prob['mm.y1'], 1.5934, .001)

        # change default surrogate, re-setup and check that metamodel re-trains
        mm.default_surrogate = FloatKrigingSurrogate()
        prob.setup(check=False)

        surrogate = prob.root.unknowns.metadata('mm.y1').get('surrogate')
        self.assertTrue(isinstance(surrogate, FloatKrigingSurrogate))

        self.assertTrue(mm.train)  # training will occur after re-setup
        mm.warm_restart = True     # use existing training data

        prob['mm.x1'] = 2.5
        prob['mm.x2'] = 3.5

        prob.run()

        assert_rel_error(self, prob['mm.y1'], 1.4609, .001)

    def test_warm_start(self):
        # create metamodel with warm_restart = True
        meta = MetaModel()
        meta.add_param('x1', 0.)
        meta.add_param('x2', 0.)
        meta.add_output('y1', 0.)
        meta.add_output('y2', 0.)
        meta.default_surrogate = ResponseSurface()
        meta.warm_restart = True

        # add to problem
        prob = Problem(Group())
        prob.root.add('meta', meta)
        prob.setup(check=False)

        # provide initial training data
        prob['meta.train:x1'] = [1.0, 3.0]
        prob['meta.train:x2'] = [1.0, 4.0]
        prob['meta.train:y1'] = [3.0, 1.0]
        prob['meta.train:y2'] = [1.0, 7.0]

        # run against a data point and check result
        prob['meta.x1'] = 2.0
        prob['meta.x2'] = 3.0
        prob.run()

        assert_rel_error(self, prob['meta.y1'], 1.9085, .001)
        assert_rel_error(self, prob['meta.y2'], 3.9203, .001)

        # Add 3rd training point, moves the estimate for that point
        # back to where it should be.
        prob['meta.train:x1'] = [2.0]
        prob['meta.train:x2'] = [3.0]
        prob['meta.train:y1'] = [2.0]
        prob['meta.train:y2'] = [4.0]

        meta.train = True  # currently need to tell meta to re-train

        prob.run()
        assert_rel_error(self, prob['meta.y1'], 2.0, .00001)
        assert_rel_error(self, prob['meta.y2'], 4.0, .00001)

    #def test_array_inputs(self):
        #raise unittest.SkipTest('MetaModel does not currently support array params')

        #meta = MetaModel()
        #meta.add_param('x', np.zeros((4)))
        #meta.add_output('y1', 0.)
        #meta.add_output('y2', 0.)
        #meta.default_surrogate = KrigingSurrogate()

        #prob = Problem(Group())
        #prob.root.add('meta', meta)
        #prob.setup(check=False)

        #prob['meta.train:x'] = [
            #[1.0, 1.0, 1.0, 1.0],
            #[2.0, 1.0, 1.0, 1.0],
            #[1.0, 2.0, 1.0, 1.0],
            #[1.0, 1.0, 2.0, 1.0],
            #[1.0, 1.0, 1.0, 2.0]
        #]
        #prob['meta.train:y1'] = [3.0, 2.0, 1.0, 6.0, -2.0]
        #prob['meta.train:y2'] = [1.0, 4.0, 7.0, -3.0, 3.0]

        #prob['meta.x'] = [1.0, 2.0, 1.0, 1.0]
        #prob.run()

        #assert_rel_error(self, prob['meta.y1'], 1.0, .00001)
        #assert_rel_error(self, prob['meta.y2'], 7.0, .00001)


if __name__ == "__main__":
    unittest.main()
