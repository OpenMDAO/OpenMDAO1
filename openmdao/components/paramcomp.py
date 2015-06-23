""" OpenMDAO class definition for ParamComp"""

from openmdao.core.component import Component

class ParamComp(Component):
    """A Component that provides an output to connect to a parameter."""

    def __init__(self, name, val, **kwargs):
        super(ParamComp, self).__init__()

        self.add_output(name, val, **kwargs)

    def apply_linear(self, mode, ls_inputs=None, vois=[None]):
        """For `ParamComp`, just pass on the incoming values.

        Parameters
        ----------
        mode : string
            Derivative mode, can be 'fwd' or 'rev'.

        ls_inputs : dict
            We can only solve derivatives for the inputs the instigating
            system has access to. (Not used here.)

        vois: list of strings
            List of all quantities of interest to key into the mats.
        """
        if mode=='fwd':
            sol_vec, rhs_vec = self.dumat, self.drmat
        else:
            sol_vec, rhs_vec = self.drmat, self.dumat

        for voi in vois:
            rhs_vec[voi].vec[:] += sol_vec[voi].vec[:]

    def solve_nonlinear(self, params, unknowns, resids):
        pass
