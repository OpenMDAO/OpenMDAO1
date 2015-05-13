""" Defines the base class for a Component in OpenMDAO."""
import functools
import numpy as np
from collections import OrderedDict
from six import iteritems

from openmdao.core.system import System

'''
Object to represent default value for `add_output`.
'''
_NotSet = object()

class Component(System):
    """ Base class for a Component system. The Component can declare
    variables and operates on its inputs to produce unknowns, which can be
    excplicit outputs or implicit states.
    """

    def __init__(self):
        super(Component, self).__init__()
        self._post_setup = False

        self._jacobian_cache = {}
    
    def _check_val(self, name, var_type, kargs):
        try:
            if kargs['val'] is _NotSet:
                kargs['val'] = np.zeros(kargs['shape'])
        except KeyError as error:
            msg = ("Shape of {var_type} '{name}' must be specified because "
                   "'val' is not set")
            msg = msg.format(var_type=var_type, name=name)
            raise ValueError(msg)
    
    def _check_args(var_type):
        def wrap(add_function):
            @functools.wraps(add_function)
            def wrapper(self, name, val=_NotSet, **kargs):
                kargs['val'] = val
                self._check_name(name)
                self._check_val(name, var_type, kargs)
                return add_function(self, name, **kargs)
            return wrapper
        return wrap
    
    @_check_args('param')
    def add_param(self, name, val=_NotSet, **kwargs):
        args = kwargs.copy()
        args['val'] = val
        self._params_dict[name] = args

    @_check_args('output')
    def add_output(self, name, val=_NotSet, **kwargs):
        args = kwargs.copy()
        args['val'] = val
        self._unknowns_dict[name] = args

    @_check_args('state')
    def add_state(self, name, val, **kwargs):
        args = kwargs.copy()
        args['val'] = val
        args['state'] = True
        self._unknowns_dict[name] = args

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

    def apply_nonlinear(self, params, unknowns, resids):
        """ Evaluates the residuals for this component. For explicit
        components, the residual is the output produced by the current params
        minus the previously calculated output. Thus, an explicit component
        must execute its solve nonlinear method. Implicit components should
        override this and calculate their residuals in place.

        Parameters
        ----------
        params : `VecWrapper`
            ``VecWrapper` ` containing parameters (p)

        unknowns : `VecWrapper`
            `VecWrapper`  containing outputs and states (u)

        resids : `VecWrapper`
            `VecWrapper`  containing residuals. (r)
        """

        # Since explicit comps don't put anything in resids, we can use it to
        # cache the old values of the unknowns.
        resids.vec[:] = unknowns.vec[:]

        self.solve_nonlinear(params, unknowns, resids)

        # Unknwons are restored to the old values too; apply_nonlinear does
        # not change the output vector.
        resids.vec[:] -= unknowns.vec[:]
        unknowns.vec[:] += resids.vec[:]

    def linearize(self, params, unknowns):
        """ Calculates the Jacobian of a component if it provides
        derivatives. Preconditioners will also be pre-calculated here if
        needed.

        Parameters
        ----------
        params : `VecwWapper`
            `VecwWapper` containing parameters (p)

        unknowns : `VecwWapper`
            `VecwWapper` containing outputs and states (u)
        """
        self._jacobian_cache = self.jacobian(params, unknowns)

    def jacobian(self, params, unknowns):
        """ Returns Jacobian. Returns None unless component overides and
        returns something. J should be a dictionary whose keys are tuples of
        the form ('unknown', 'param') and whose values are ndarrays.

        Parameters
        ----------
        params : `VecwWapper`
            `VecwWapper` containing parameters (p)

        unknowns : `VecwWapper`
            `VecwWapper` containing outputs and states (u)

        Returns
        -------
        dict
            Dictionary whose keys are tuples of the form ('unknown', 'param')
            and whose values are ndarrays
        """
        return None

    def apply_linear(self, params, unknowns, dparams, dunknowns, dresids, mode):
        """Multiplies incoming vector by the Jacobian (fwd mode) or the
        transpose Jacobian (rev mode). If the user doesn't provide this
        method, then we just multiply by self._jacobian_cache.

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
