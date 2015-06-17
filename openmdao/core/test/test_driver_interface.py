""" Test for the Driver class -- basic driver interface."""

from pprint import pformat
import unittest

import numpy as np

from openmdao.core.driver import Driver
from openmdao.core.options import OptionsDictionary
from openmdao.core.problem import Problem
from openmdao.test.sellar import SellarDerivatives

class MySimpleDriver(Driver):

    def __init__(self):
        super(MySimpleDriver, self).__init__()

        # What we support
        self.supports['Inequality Constraints'] = True
        self.supports['Equality Constraints'] = False
        self.supports['Linear Constraints'] = False
        self.supports['Multiple Objectives'] = False

        # My driver options
        self.options = OptionsDictionary()
        self.options.add_option('tol', 1e-4)
        self.options.add_option('maxiter', 10)

        self.alpha = .01
        self.violated = []

    def run(self, problem):
        """ Mimic a very simplistic unconstrained optimization."""

        # Get dicts with pointers to our vectors
        params = self.get_params()
        objective = self.get_objectives()
        constraints = self.get_constraints()

        param_list = params.keys()
        objective_names = objective.keys()
        constraint_names = constraints.keys()
        unknown_list = objective_names + constraint_names

        itercount = 0
        while itercount < self.options['maxiter']:

            # Run the model
            problem.root.solve_nonlinear()
            #print('z1: %f, z2: %f, x1: %f, y1: %f, y2: %f' % (problem['z'][0],
                                                              #problem['z'][1],
                                                              #problem['x'],
                                                              #problem['y1'],
                                                              #problem['y2']))
            #print('obj: %f, con1: %f, con2: %f' % (problem['obj'], problem['con1'],
                                                   #problem['con2']))

            # Calculate gradient
            J = problem.calc_gradient(param_list, unknown_list, return_format='dict')

            objective = self.get_objectives()
            constraints = self.get_constraints()

            for key1 in objective_names:
                for key2 in param_list:

                    grad = J[key1][key2] * objective[key1]
                    new_val = params[key2] - self.alpha*grad

                    # Set parameter
                    self.set_param(key2, new_val)

            self.violated = []
            for name, val in constraints.items():
                if np.linalg.norm(val) > 0.0:
                    self.violated.append(name)

            itercount += 1

class TestDriver(unittest.TestCase):

    def test_mydriver(self):

        top = Problem()
        root = top.root = SellarDerivatives()

        top.driver = MySimpleDriver()
        top.driver.add_param('z', low=-100.0, high=100.0)

        top.driver.add_objective('obj')
        top.driver.add_constraint('con1')
        top.driver.add_constraint('con2')

        top.setup()
        top.run()

        obj = top['obj']
        self.assertLess(obj, 28.0)

if __name__ == "__main__":
    unittest.main()
