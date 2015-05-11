""" OpenMDAO class definition for ParamComp"""

from openmdao.core.component import Component

class ParamComp(Component):
    """A Component that provides an output to connect to a parameter."""

    def __init__(self, name, val):
        super(ParamComp, self).__init__()

        self.add_output(name, val)
