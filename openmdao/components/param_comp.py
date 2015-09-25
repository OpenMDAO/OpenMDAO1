""" OpenMDAO class definition for ParamComp"""

import warnings

from openmdao.components.indep_var_comp import IndepVarComp

class ParamComp(IndepVarComp):
    """
    A Component that provides an independent variable as an output.

    Options
    -------
    fd_options['force_fd'] :  bool(False)
        Set to True to finite difference this system.
    fd_options['form'] :  str('forward')
        Finite difference mode. (forward, backward, central) You can also set to 'complex_step' to peform the complex step method if your components support it.
    fd_options['step_size'] :  float(1e-06)
        Default finite difference stepsize
    fd_options['step_type'] :  str('absolute')
        Set to absolute, relative

    """

    def __init__(self, name, val=None, **kwargs):
        super(ParamComp, self).__init__(name, val, **kwargs)
        warnings.simplefilter('always', DeprecationWarning)
        warnings.warn("ParamComp is deprecated. Please switch to IndepVarComp, "
                      "which can be found in openmdao.components.indep_var_comp.",
                      DeprecationWarning,stacklevel=2)
        warnings.simplefilter('ignore', DeprecationWarning)
