""" OpenMDAO class definition for ParamComp"""

import warnings

from openmdao.components.indep_var_comp import IndepVarComp

class ParamComp(IndepVarComp):
    """A Component that provides an output to connect to a parameter."""

    def __init__(self, name, val=None, **kwargs):
        super(ParamComp, self).__init__(name, val, **kwargs)
        warnings.warn("ParamComp is deprecated. Please switch to IndepVarComp, "
                     "which can be found in openmdao.components.indep_var_comp.")
