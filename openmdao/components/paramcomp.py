""" OpenMDAO class definition for ParamComp"""

from openmdao.core.component import Component

class ParamComp(Component):
    """A Component that provides an output to connect to a parameter."""

    def __init__(self, name, val, **kwargs):
        super(ParamComp, self).__init__()

        self.add_output(name, val, **kwargs)

    def apply_linear(self, params, unknowns, dparams, dunknowns, dresids, mode,
                     ls_inputs=None):
        """For `ParamComp`, just pass on the incoming values.

        Parameters
        ----------
        params : `VecWrapper`
            `VecWrapper` containing parameters. (p)

        unknowns : `VecWrapper`
            `VecWrapper` containing outputs and states. (u)

        dparams : `VecWrapper`
            `VecWrapper` containing either the incoming vector in forward mode
            or the outgoing result in reverse mode. (dp)

        dunknowns : `VecWrapper`
            In forward mode, this `VecWrapper` contains the incoming vector for
            the states. In reverse mode, it contains the outgoing vector for
            the states. (du)

        dresids : `VecWrapper`
            `VecWrapper` containing either the outgoing result in forward mode
            or the incoming vector in reverse mode. (dr)

        mode : string
            Derivative mode, can be 'fwd' or 'rev'.

        ls_inputs : list
            We can only solve derivatives for the inputs the instigating
            system has access to. (Not used here.)
        """
        if mode=='fwd':
            sol_vec, rhs_vec = dunknowns, dresids
        else:
            sol_vec, rhs_vec = dresids, dunknowns

        rhs_vec.vec[:] += sol_vec.vec[:]

    def solve_nonlinear(self, params, unknowns, resids):
        pass
