""" Tests the ins and outs of automatic unit conversion in OpenMDAO."""

import unittest

from openmdao.components.paramcomp import ParamComp
from openmdao.core.component import Component
from openmdao.core.group import Group
from openmdao.core.problem import Problem
from openmdao.test.testutil import assert_rel_error

class SrcComp(Component):

    def __init__(self):
        super(SrcComp, self).__init__()

        self.add_param('x1', 100.0)
        self.add_output('x2', 100.0, units='degC')

    def solve_nonlinear(self, params, unknowns, resids):
        """ No action."""
        unknowns['x2'] = params['x1']

class TgtComp(Component):

    def __init__(self):
        super(TgtComp, self).__init__()

        self.add_param('x2', 100.0, units='degF')
        self.add_output('x3', 100.0)

    def solve_nonlinear(self, params, unknowns, resids):
        """ No action."""
        unknowns['x3'] = params['x2']


class TestUnitConversion(unittest.TestCase):
    """ Testing automatic unit conversion."""

    def test_basic(self):

        prob = Problem()
        prob.root = Group()
        prob.root.add('src', SrcComp())
        prob.root.add('tgt', TgtComp())
        prob.root.add('px1', ParamComp('x1', 100.0), promotes=['x1'])
        prob.root.connect('x1', 'src:x1')
        prob.root.connect('src:x2', 'tgt:x2')

        prob.setup()
        prob.run()

        assert_rel_error(self, prob['src:x2'], 100.0, 1e-6)
        assert_rel_error(self, prob['tgt:x3'], 212.0, 1e-6)


if __name__ == "__main__":
    unittest.main()