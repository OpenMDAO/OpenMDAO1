""" Defines the base class for a Component in OpenMDAO."""
import numpy as np
from collections import OrderedDict
from six import iteritems
from six.moves import range
import numpy as np
from itertools import chain

from openmdao.core.system import System
from openmdao.core.basicimpl import BasicImpl
from openmdao.core.varmanager import create_views


'''
Object to represent default value for `add_output`.
'''
_NotSet = object()


class Component(System):
    """ Base class for a Component system. The Component can declare
    variables and operates on its params to produce unknowns, which can be
    explicit outputs or implicit states.
    """

    def __init__(self):
        super(Component, self).__init__()
        self._post_setup = False

        self._jacobian_cache = {}
        self._vecs = None

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

    @property
    def unknowns(self):
        return self._vecs.unknowns

    @property
    def dunknowns(self):
        return self._vecs.dunknowns

    @property
    def params(self):
        return self._vecs.params

    @property
    def dparams(self):
        return self._vecs.dparams

    @property
    def resids(self):
        return self._vecs.resids

    @property
    def dresids(self):
        return self._vecs.dresids

    def vectors(self):
        """
        Returns
        -------
        `VecTuple`
            A named tuple containing the unknowns, dunknowns, params, dparams, resids, and
            dresids `VecWrapper`s.
        """
        return self._vecs

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

    def _setup_vectors(self, param_owners, connections, parent_vm=None,
                       top_unknowns=None, impl=BasicImpl):
        """
        Set up local `VecWrapper`s to store this component's variables.

        Parameters
        ----------
        param_owners : dict
            a dictionary mapping `System` pathnames to the pathnames of parameters
            they are reponsible for propagating. (ignored)

        connections : dict
            a dictionary mapping the pathname of a target variable to the
            pathname of the source variable that it is connected to

        parent_vm : `VarManager`, optional
            the `VarManager` for the parent `Group`, if any.

        top_unknowns : `VecWrapper`, optional
            the `Problem` level unknowns `VecWrapper`

        impl : an implementation factory, optional
            Specifies the factory object used to create `VecWrapper` objects.
        """
        if not self.is_active():
            return

        self._vecs = create_views(top_unknowns, parent_vm, self.comm, self.pathname,
                                  self._params_dict, self._unknowns_dict, [], connections)

        params = self._vecs.params

        # create params vec entries for any unconnected params
        for pathname, meta in self._params_dict.items():
            name = params._scoped_abs_name(pathname)
            if name not in params:
                params._add_unconnected_var(pathname, meta)

    def apply_nonlinear(self, params, unknowns, resids):
        """
        Evaluates the residuals for this component. For explicit
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
        resids.vec[:] = -unknowns.vec[:]

        self.solve_nonlinear(params, unknowns, resids)

        # Unknowns are restored to the old values too. apply_nonlinear does
        # not change the output vector.
        resids.vec[:] += unknowns.vec[:]
        unknowns.vec[:] -= resids.vec[:]

    def fd_jacobian(self, params, unknowns, resids):
        """
        Finite difference across all unknonws in component w.r.t. all params.

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

        jac = {}
        resid_cache = resids.vec.copy()

        states = []
        for u_name, meta in iteritems(self._unknowns_dict):
            if meta.get('state'):
                states.append(meta['relative_name'])

        # Compute gradient for this param or state.
        for p_name in chain(params, states):

            if p_name in states:
                inputs = unknowns
            else:
                inputs = params

            # Size our Inputs
            p_size = np.size(inputs[p_name])

            # Size our Outputs
            for u_name in unknowns:
                u_size = np.size(unknowns[u_name])
                jac[u_name, p_name] = np.ones((u_size, p_size))

            # Finite Difference each index in array
            for idx in range(p_size):

                step = inputs.flat[p_name][idx] * 0.001
                inputs.flat[p_name][idx] += step

                self.apply_nonlinear(params, unknowns, resids)

                inputs.flat[p_name][idx] -= step

                # delta resid is delta unknown
                resids.vec[:] -= resid_cache
                resids.vec[:] *= (1.0/step)

                for u_name in unknowns:
                    jac[u_name, p_name][:, idx] = resids.flat[u_name]

                # Restore old residual
                resids.vec[:] = resid_cache

        return jac

    def jacobian(self, params, unknowns, resids):
        """
        Returns Jacobian. Returns None unless component overides and
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

        if self._jacobian_cache is None:
            msg = ("No derivatives defined for Component '{name}'")
            msg = msg.format(name=self.name)
            raise ValueError(msg)


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

