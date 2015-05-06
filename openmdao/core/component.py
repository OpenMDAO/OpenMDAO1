""" Defines the base class for a Component in OpenMDAO."""

from collections import OrderedDict
from six import iteritems

from openmdao.core.system import System


class Component(System):
    """ Base class for a Component system. The Component can declare
    variables and operates on its inputs to produce unknowns, which can be
    excplicit outputs or implicit states.
    """

    def __init__(self):
        super(Component, self).__init__()
        self._post_setup = False

        self._jacobian_cache = {}

    def add_param(self, name, val, **kwargs):
        self._check_name(name)
        args = kwargs.copy()
        args['val'] = val
        self._params_dict[name] = args

    def add_output(self, name, val, **kwargs):
        self._check_name(name)
        args = kwargs.copy()
        args['val'] = val
        self._unknowns_dict[name] = args

    def add_state(self, name, val, **kwargs):
        self._check_name(name)
        args = kwargs.copy()
        args['val'] = val
        args['state'] = True
        self._unknowns_dict[name] = args

    @property
    def params(self):
        '''
        Returns `OrderedDict` of all parameters for the component
        '''
        return self._params_dict
        
    @property
    def unknowns(self):
        '''
        Returns `OrderedDict` of all unknowns (states and outputs) for the component
        '''
        return self._unknowns_dict
        
    def _check_name(self, name):
        if self._post_setup:
            raise RuntimeError("%s: can't add variable %s because setup has already been called",
                               (self.pathname, name))
        if name in self._params_dict or name in self._unknowns_dict:
            raise RuntimeError("%s: variable %s already exists" %
                               (self.pathname, name))

    def _setup_variables(self):
        """Returns our params and unknowns, and stores them
        as attributes of the component"""

        # rekey with absolute path names and add relative names

        _new_params = OrderedDict()
        for name, meta in self._params_dict.items():
            if not self.pathname:
                var_pathname = name
            else:
                var_pathname = ':'.join([self.pathname, name])
            _new_params[var_pathname] = meta
            meta['relative_name'] = name
        self._params_dict = _new_params

        _new_unknowns = OrderedDict()
        for name, meta in self._unknowns_dict.items():
            if not self.pathname:
                var_pathname = name
            else:
                var_pathname = ':'.join([self.pathname, name])
            _new_unknowns[var_pathname] = meta
            meta['relative_name'] = name
        self._unknowns_dict = _new_unknowns

        self._post_setup = True

        return self._params_dict, self._unknowns_dict

    def linearize(self, params, unknowns):
        """ Calculates the Jacobian of a component if it provides
        derivatives. Preconditioners will also be pre-calculated here if
        needed.

        params: vecwrapper
            VecWrapper containing parameters (p)

        unknowns: vecwrapper
            VecWrapper containing outputs and states (u)
        """
        self._jacobian_cache = self.jacobian(params, unknowns)

    def jacobian(self, params, unknowns):
        """ Returns Jacobian. Returns None unless component overides and
        returns something. J should be a dictionary whose keys are tuples of
        the form ('unknown', 'param') and whose values are ndarrays.

        params: vecwrapper
            VecWrapper containing parameters (p)

        unknowns: vecwrapper
            VecWrapper containing outputs and states (u)
        """
        return None

    def apply_linear(self, params, unknowns, dparams, dunknowns, dresids, mode):
        """Multiplies incoming vector by the Jacobian (fwd mode) or the
        transpose Jacobian (rev mode). If the user doesn't provide this
        method, then we just multiply by self._jacobian_cache.

        params: vecwrapper
            VecWrapper containing parameters (p)

        unknowns: vecwrapper
            VecWrapper containing outputs and states (u)

        dparams: vecwrapper
            VecWrapper containing either the incoming vector in forward mode
            or the outgoing result in reverse mode. (dp)

        dunknowns: vecwrapper
            In forward mode, this VecWrapper contains the incoming vector for
            the states. In reverse mode, it contains the outgoing vector for
            the states. (du)

        dresids: vecwrapper
            VecWrapper containing either the outgoing result in forward mode
            or the incoming vector in reverse mode. (dr)

        mode: string
            Derivative mode, can be 'fwd' or 'rev'
        """

        for key, J in iteritems(self._jacobian_cache):
            unknown, param = key

            # States are never in dparams.
            if param in dparams:
                arg = dparams[param]
            elif param in dunknowns:
                arg = dunknowns[param]
            else:
                continue

            if unknown not in dresids:
                continue

            result = dresids[unknown]

            # Vectors are flipped during adjoint
            if mode == 'fwd':
                result[:] += J.dot(arg.flatten()).reshape(result.shape)
            else:
                arg[:] += J.T.dot(result.flatten()).reshape(arg.shape)

    def applyJ(self, params, unknowns, resids, dparams, dunknowns, dstates,
               mode):
        """ This method wraps apply_linear and adds the additional 1.0 on the
        diagonal for explicit outputs.

        df = du - dGdp * dp or du = df and dp = -dGdp^T * df
        """

        # Forward Mode
        if self.mode == 'fwd':

            self.apply_linear(params, unknowns, dparams, dunknowns, dresids,
                              mode)
            dunknowns.vec[:] *= -1.0

            for var in dunknowns:
                dunknowns[var][:] += dparams[var][:]

        # Adjoint Mode
        elif self.mode == 'adjoint':

            # Sign on the local Jacobian needs to be -1 before
            # we add in the fake residual. Since we can't modify
            # the 'du' vector at this point without stomping on the
            # previous component's contributions, we can multiply
            # our local 'arg' by -1, and then revert it afterwards.
            dunknowns.vec[:] *= -1.0
            self.apply_linear(params, unknowns, dparams, dunknowns, dresids,
                              mode)
            dunknowns.vec[:] *= -1.0

            for var in dunknowns:
                dparams[var][:] += dunknowns[var][:]
