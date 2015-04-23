""" Unit test for Groups. """

import unittest

import numpy as np

from openmdao.core.group import Group
from openmdao.test.testcomps import SimpleComp

class TestGroup(unittest.TestCase):

    def test_add(self):

        group = Group()
        comp = SimpleComp()
        group.add('mycomp', comp)

        subs = dict(group.subsystems())
        self.assertEqual(len(subs), 1)
        self.assertEqual(subs['mycomp'], comp)

        comp2 = SimpleComp()
        group.add("nextcomp", comp2)

        subs = dict(group.subsystems())
        self.assertEqual(len(subs), 2)
        self.assertEqual(subs['mycomp'], comp)
        self.assertEqual(subs['nextcomp'], comp2)

    def test_variables(self):
        group = Group()
        comp = SimpleComp()
        group.add('mycomp', comp)

        nextcomp = SimpleComp()
        group.add('nextcomp', nextcomp)

        params, unknowns, states = group.variables()

        expect_params = ['mycomp:x', 'nextcomp:x']
        expect_unknowns = ['mycomp:y', 'nextcomp:y']


        self.assertEquals(expect_params, params.keys())
        self.assertEquals(expect_unknowns, unknowns.keys())
        self.assertEquals([], states.keys())


    def test_setup(self):
        pass

    def test_solve(self):
        pass


if __name__ == "__main__":
    unittest.main()