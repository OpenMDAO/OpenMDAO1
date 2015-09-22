""" Some simple test components. """

import numpy as np

from openmdao.core.group import Group
from openmdao.components.indep_var_comp import IndepVarComp
from openmdao.components.exec_comp import ExecComp
from openmdao.test.simple_comps import SimplePassByObjComp

class ExampleGroup(Group):
    """A nested `Group` for testing"""
    def __init__(self):
        super(ExampleGroup, self).__init__()

        self.G2 = self.add('G2', Group())
        self.C1 = self.G2.add('C1', IndepVarComp('x', 5.))

        self.G1 = self.G2.add('G1', Group())
        self.C2 = self.G1.add('C2', ExecComp('y=x*2.0',x=3.,y=5.5))

        self.G3 = self.add('G3', Group())
        self.C3 = self.G3.add('C3', ExecComp('y=x*2.0',x=3.,y=5.5))
        self.C4 = self.G3.add('C4', ExecComp('y=x*2.0',x=3.,y=5.5))

        self.G2.connect('C1.x', 'G1.C2.x')
        self.connect('G2.G1.C2.y', 'G3.C3.x')
        self.G3.connect('C3.y', 'C4.x')

class ExampleByObjGroup(Group):
    """A nested `Group` for testing"""
    def __init__(self):
        super(ExampleByObjGroup, self).__init__()

        self.G2 = self.add('G2', Group())
        self.C1 = self.G2.add('C1', IndepVarComp('x', 'foo'))

        self.G1 = self.G2.add('G1', Group())
        self.C2 = self.G1.add('C2', SimplePassByObjComp())

        self.G3 = self.add('G3', Group())
        self.C3 = self.G3.add('C3', SimplePassByObjComp())
        self.C4 = self.G3.add('C4', SimplePassByObjComp())

        self.G2.connect('C1.x', 'G1.C2.x')
        self.connect('G2.G1.C2.y', 'G3.C3.x')
        self.G3.connect('C3.y', 'C4.x')


class ExampleGroupWithPromotes(Group):
    """A nested `Group` with implicit connections for testing"""
    def __init__(self):
        super(ExampleGroupWithPromotes, self).__init__()

        self.G2 = self.add('G2', Group())
        self.C1 = self.G2.add('C1', IndepVarComp('x', 5.), promotes=['x'])

        self.G1 = self.G2.add('G1', Group(), promotes=['x'])
        self.C2 = self.G1.add('C2', ExecComp('y=x*2.0',x=3.,y=5.5), promotes=['x'])

        self.G3 = self.add('G3', Group(), promotes=['x'])
        self.C3 = self.G3.add('C3', ExecComp('y=x*2.0',x=3.,y=5.5))
        self.C4 = self.G3.add('C4', ExecComp('y=x*2.0',x=3.,y=5.5), promotes=['x'])

        self.connect('G2.G1.C2.y', 'G3.C3.x')
        self.G3.connect('C3.y', 'x')
