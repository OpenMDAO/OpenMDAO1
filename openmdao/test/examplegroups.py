""" Some simple test components. """

import numpy as np

from openmdao.core.group import Group
from openmdao.components.paramcomp import ParamComp
from openmdao.test.simplecomps import SimpleComp

class ExampleGroup(Group):
    """A nested `Group` for testing"""
    def __init__(self):
        super(ExampleGroup, self).__init__()

        self.G2 = self.add('G2', Group())
        self.C1 = self.G2.add('C1', ParamComp('x', 5.))

        self.G1 = self.G2.add('G1', Group())
        self.C2 = self.G1.add('C2', SimpleComp())

        self.G3 = self.add('G3', Group())
        self.C3 = self.G3.add('C3', SimpleComp())
        self.C4 = self.G3.add('C4', SimpleComp())

        self.G2.connect('C1:x', 'G1:C2:x')
        self.connect('G2:G1:C2:y', 'G3:C3:x')
        self.G3.connect('C3:y', 'C4:x')


class ExampleGroupWithPromotes(Group):
    """A nested `Group` with implicit connections for testing"""
    def __init__(self):
        super(ExampleGroupWithPromotes, self).__init__()

        self.G2 = self.add('G2', Group())
        self.C1 = self.G2.add('C1', ParamComp('x', 5.), promotes=['x'])

        self.G1 = self.G2.add('G1', Group(), promotes=['x'])
        self.C2 = self.G1.add('C2', SimpleComp(), promotes=['x'])

        self.G3 = self.add('G3', Group(), promotes=['x'])
        self.C3 = self.G3.add('C3', SimpleComp())
        self.C4 = self.G3.add('C4', SimpleComp(), promotes=['x'])

        self.connect('G2:G1:C2:y', 'G3:C3:x')
        self.G3.connect('C3:y', 'x')

class SimplerGroup(Group):
    """A non-nested Group for basic full scatter test"""
    def __init__(self):
        super(SimplerGroup, self).__init__()

        self.C1 = self.add('C1', ParamComp('x', 5.))
        self.C2 = self.add('C2', SimpleComp())

        self.connect('C1:x', 'C2:x')
