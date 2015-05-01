""" Defines the base class for a Component in OpenMDAO."""

from collections import OrderedDict

from openmdao.core.system import System


class Component(System):
    """ Base class for a Component system. The Component can declare
    variables and operates on its inputs to produce unknowns, which can be
    excplicit outputs or implicit states."""

    def __init__(self):
        super(Component, self).__init__()
        self._params = OrderedDict()
        self._outputs = OrderedDict()
        self._states = OrderedDict()

        self.J = None

    def add_param(self, name, val, **kwargs):
        args = kwargs.copy()
        args['val'] = val
        self._params[name] = args

    def add_output(self, name, val, **kwargs):
        args = kwargs.copy()
        args['val'] = val
        self._outputs[name] = args

    def add_state(self, name, val, **kwargs):
        args = kwargs.copy()
        args['val'] = val
        self._states[name] = args

    def variables(self):
        return self._params, self._outputs, self._states

    def linearize(self, params, unknowns):
        """ Calculates the Jacobian of a component if it provides
        derivatives. Preconditioners will also be pre-calculated here if
        needed.

        params: vecwrapper
            VecWrapper containing parameters (p)

        unknowns: vecwrapper
            VecWrapper containing outputs and states (u)
        """
        self.J = self.jacobian(params, unknowns)

    def jacobian(self, params, unknowns):
        """ Returns Jacobian. Returns None unless component overides.
        J should be a dictionary whose keys are tuples of the form
        ('unknown', 'param') and whose values are ndarrays.

        params: vecwrapper
            VecWrapper containing parameters (p)

        unknowns: vecwrapper
            VecWrapper containing outputs and states (u)
        """
        return None

    def apply_linear(self, params, unknowns, resids, dparams, dunknowns,
        dstates, mode):
        """Multiplies incoming vector by the Jacobian (fwd mode) or the
        transpose Jacobian (rev mode). If the user doesn't provide this
        method, then we just multiply by self.J.

        params: vecwrapper
            VecWrapper containing parameters (p)

        unknowns: vecwrapper
            VecWrapper containing outputs and states (u)

        resids: vecwrapper
            VecWrapper containing residuals (f)

        dparams: vecwrapper
            VecWrapper containing either the incoming vector in forward mode
            or the outgoing result in reverse mode. (dp)

        dunknowns: vecwrapper
            VecWrapper containing either the outgoing result in forward mode
            or the incoming vector in reverse mode. (df)

        dstates: vecwrapper
            In forward mode, this VecWrapper contains the incoming vector for
            the states. In reverse mode, it contains the outgoing vector for
            the states. (du)

        mode: string
            Derivative mode, can be 'fwd' or 'rev'
        """

        for key, J in self.J.iteritems():
            unknown, param = key

            # States are never in dparams.
            if param in dparams:
                arg = dparams[param]
            elif param in dstates:
                arg = dstates[param]
            else:
                continue

            if unknown not in dunknowns:
                continue

            result = dunknowns[unknown]

            # Vectors are flipped during adjoint
            if mode == 'fwd':
                result[:] = J.dot(arg.flatten()).reshape(result.shape)
            else:
                arg[:] = J.T.dot(result.flatten()).reshape(arg.shape)

