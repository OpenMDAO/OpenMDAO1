""" Defines the base class for a Component in OpenMDAO."""
import functools
import copy
import numpy as np
from collections import OrderedDict
from six import iteritems
import numpy as np

from openmdao.core.system import System

'''
Object to represent default value for `add_output`.
'''
_NotSet = object()


#note, doing this because I don't know if its safe to deep copy the vectors
def _copy_vec(vec):
    """grabs all the keys and values from an unknowns vector and returns a copy of it"""

    new_vec = {}
    for var_name, val in iteritems(vec):
        new_vec[var_name] = copy.deepcopy(vec[var_name])

    return new_vec

class Component(System):
    """ Base class for a Component system. The Component can declare
    variables and operates on its inputs to produce unknowns, which can be
    excplicit outputs or implicit states.
    """

    def __init__(self):
        super(Component, self).__init__()
        self._post_setup = False

        self._jacobian_cache = {}

    def _get_initial_val(self, val, shape):
        if val is _NotSet:
            return np.zeros(shape)

        return val

    def _check_val(self, name, var_type, val, shape):
        if val is _NotSet and shape is None:
            msg = ("Shape of {var_type} '{name}' must be specified because "
                   "'val' is not set")
            msg = msg.format(var_type=var_type, name=name)
            raise ValueError(msg)

    def add_param(self, name, val=_NotSet, **kwargs):
        self._check_val(name, 'param', val, kwargs.get('shape'))
        self._check_name(name)
        args = kwargs.copy()
        args['val'] = self._get_initial_val(val, kwargs.get('shape'))
        self._params_dict[name] = args

    def add_output(self, name, val=_NotSet, **kwargs):
        self._check_val(name, 'output', val, kwargs.get('shape'))
        self._check_name(name)
        args = kwargs.copy()
        args['val'] = self._get_initial_val(val, kwargs.get('shape'))
        self._unknowns_dict[name] = args

    def add_state(self, name, val=_NotSet, **kwargs):
        self._check_val(name, 'state', val, kwargs.get('shape'))
        self._check_name(name)
        args = kwargs.copy()
        args['val'] = self._get_initial_val(val, kwargs.get('shape'))
        args['state'] = True
        self._unknowns_dict[name] = args

    def _check_name(self, name):
        if self._post_setup:
            raise RuntimeError("%s: can't add variable '%s' because setup has already been called",
                               (self.pathname, name))
        if name in self._params_dict or name in self._unknowns_dict:
            raise RuntimeError("%s: variable '%s' already exists" %
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
        for u_name in unknowns:
            #obtuse, but necessary to handle scalar and array cases and make sure we don't loose the pointer
            resids.__setitem__(u_name, -unknowns[u_name])

        self.solve_nonlinear(params, unknowns, resids)

        # Unknwons are restored to the old values too; apply_nonlinear does
        # not change the output vector.
        for u_name in unknowns:
            resids[u_name] += unknowns[u_name]
            unknowns[u_name] -= resids[u_name]

    def fd_jacobian(self, params, unknowns, resids):
        """Finite difference across all unknonws in component w.r.t. all params

        Returns
        -------
        dict
            jacobian dictionary
        """

        self.solve_nonlinear(params, unknowns, resids)
        jac = {}
        fd_r = _copy_vec(resids)


        outputs = []
        states = []
        for u_name, meta in iteritems(self._unknowns_dict):
            if meta.get('state'):
                states.append(meta['relative_name'])
            else:
                outputs.append(meta['relative_name'])

        #compute dUdP (outputs)
        for p_name in params:
            p_size = np.size(params[p_name])
            p_shape = np.shape(params[p_name])

            for u_name in unknowns:
                u_size = np.size(unknowns[u_name])
                jac[u_name, p_name] = np.ones((u_size, p_size))

            for idx in xrange(p_size):
                #make work vectors to take steps in
                fd_p = _copy_vec(params)
                fd_u = _copy_vec(unknowns)

                if p_shape: # array
                    nd_idx = np.unravel_index(idx, p_shape)
                    fd_p[p_name][nd_idx] *= 1.001
                    step = fd_p[p_name][nd_idx] - params[p_name][nd_idx]
                else: # scalar
                    fd_p[p_name] *= 1.001
                    step = fd_p[p_name] - params[p_name]

                self.apply_nonlinear(fd_p, fd_u, fd_r)

                for u_name in unknowns:
                    new_u_val = fd_r[u_name]
                    fd_deriv = (new_u_val - resids[u_name])/step

                    if p_shape:
                        jac[u_name, p_name][:,idx] = fd_deriv.flatten()
                    else:
                        jac[u_name, p_name][:,idx] = fd_deriv

        # compute dRdU (resids) and dRdP
        for s_name in states:
            #make a work vector to take steps in
            s_size = np.size(unknowns[s_name])
            s_shape = np.shape(unknowns[s_name])

            for u_name in unknowns:
                u_size = np.size(unknowns[u_name])
                jac[u_name, s_name] = np.ones((u_size, s_size))

            for idx in xrange(s_size):
                fd_u = _copy_vec(unknowns)
                if s_shape:
                    nd_idx = np.unravel_index(idx, s_shape)
                    fd_u[s_name][nd_idx] *= 1.001
                    step = fd_u[s_name][nd_idx] - unknowns[s_name][nd_idx]
                else:
                    fd_u[s_name] *= 1.001
                    step = fd_u[s_name] - unknowns[s_name]
                self.apply_nonlinear(params, fd_u, fd_r)
                for u_name in unknowns:
                    new_r_val = fd_r[u_name]
                    fd_deriv = (new_r_val - resids[u_name])/step

                    if s_shape:
                        jac[u_name, s_name][:,idx] = fd_deriv.flatten()
                    else:
                        jac[u_name, s_name][:,idx] = fd_deriv

        return jac

    def jacobian(self, params, unknowns, resids):
        """ Returns Jacobian. Returns None unless component overides and
        returns something. J should be a dictionary whose keys are tuples of
        the form ('unknown', 'param') and whose values are ndarrays.

        Parameters
        ----------
        params : `VecwWapper`
            `VecwWapper` containing parameters (p)

        unknowns : `VecwWapper`
            `VecwWapper` containing outputs and states (u)

        resids : `VecWrapper`
            `VecWrapper`  containing residuals. (r)

        Returns
        -------
        dict
            Dictionary whose keys are tuples of the form ('unknown', 'param')
            and whose values are ndarrays
        """
        return None

    def apply_linear(self, params, unknowns, dparams, dunknowns, dresids, mode):
        """
        Multiplies incoming vector by the Jacobian (fwd mode) or the
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
                arg_vec = dparams
            elif param in dunknowns:
                arg_vec = dunknowns
            else:
                continue

            if unknown not in dresids:
                continue

            result = dresids[unknown]

            # Vectors are flipped during adjoint

            if mode == 'fwd':
                dresids[unknown] += J.dot(arg_vec[param].flatten()).reshape(result.shape)
            else:
                arg_vec[param] += J.T.dot(result.flatten()).reshape(arg_vec[param].shape)
