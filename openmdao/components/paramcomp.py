""" OpenMDAO class definition for ParamComp"""

from openmdao.core.component import Component

class ParamComp(Component):
    """A Component that provides an output to connect to a parameter."""

    def __init__(self, name, val):
        super(ParamComp, self).__init__()

        self.add_output(name, val)

    def apply_linear(self, params, unknowns, dparams, dunknowns, dresids, mode):
        """For `ParamComp`, just pass.

        Parameters
        ----------
        params : `VecwWrapper`
            `VecwWrapper` containing parameters (p)

        unknowns : `VecwWrapper`
            `VecwWrapper` containing outputs and states (u)

        dparams : `VecwWrapper`
            `VecwWrapper` containing either the incoming vector in forward mode
            or the outgoing result in reverse mode. (dp)

        dunknowns : `VecwWrapper`
            In forward mode, this `VecwWrapper` contains the incoming vector for
            the states. In reverse mode, it contains the outgoing vector for
            the states. (du)

        dresids : `VecwWrapper`
            `VecwWrapper` containing either the outgoing result in forward mode
            or the incoming vector in reverse mode. (dr)

        mode : string
            Derivative mode, can be 'fwd' or 'rev'
        """
        pass