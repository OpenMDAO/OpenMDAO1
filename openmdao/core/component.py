""" Defines the base class for a Component in OpenMDAO."""

import sys
import re
from pprint import pformat
from collections import OrderedDict
import functools
from six import iteritems
from six.moves import range

# pylint: disable=E0611, F0401
import numpy as np

from openmdao.core.system import System
from openmdao.core.basicimpl import BasicImpl
from openmdao.util.types import is_differentiable

'''
Object to represent default value for `add_output`.
'''
_NotSet = object()

# regex to check for valid variable names.
namecheck_rgx = re.compile(
    '([_a-zA-Z][_a-zA-Z0-9]*)+(\:[_a-zA-Z][_a-zA-Z0-9]*)*')

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
            # Interpret a shape of 1 to mean scalar.
            if shape == 1:
                return 0.
            return np.zeros(shape)

        return val

    def _check_val(self, name, var_type, val, shape):
        if val is _NotSet and shape is None:
            msg = ("Shape of {var_type} '{name}' must be specified because "
                   "'val' is not set")
            msg = msg.format(var_type=var_type, name=name)
            raise ValueError(msg)

    def _add_variable(self, name, val, var_type, **kwargs):
        shape = kwargs.get('shape')
        self._check_val(name, var_type, val, shape)
        self._check_name(name)
        args = kwargs.copy()

        args['val'] = val = self._get_initial_val(val, shape)

        if is_differentiable(val) and not args.get('pass_by_obj'):
            if isinstance(val, np.ndarray):
                args['size'] = val.size
                args['shape'] = val.shape
            else:
                args['size'] = 1
                args['shape'] = 1
        else:
            args['size'] = 0
            args['pass_by_obj'] = True

        if isinstance(shape, int) and shape > 1:
            args['shape'] = (shape,)

        return args

    def add_param(self, name, val=_NotSet, **kwargs):
        self._params_dict[name] = self._add_variable(name, val, 'param', **kwargs)

    def add_output(self, name, val=_NotSet, **kwargs):
        self._unknowns_dict[name] = self._add_variable(name, val, 'output', **kwargs)

    def add_state(self, name, val=_NotSet, **kwargs):
        args = self._add_variable(name, val, 'state', **kwargs)
        args['state'] = True
        self._unknowns_dict[name] = args

    def _check_name(self, name):
        if self._post_setup:
            raise RuntimeError("%s: can't add variable '%s' because setup has already been called",
                               (self.pathname, name))
        if name in self._params_dict or name in self._unknowns_dict:
            raise RuntimeError("%s: variable '%s' already exists" %
                               (self.pathname, name))

        match = namecheck_rgx.match(name)
        if match is None or match.group() != name:
            raise NameError("%s: '%s' is not a valid variable name." % (self.pathname, name))

    def setup_param_indices(self):
        """
        Override this in your Component to set specific indices that will be pulled from
        source variables to fill your parameters.  This method should set the 'src_indices'
        metadata for any parameters that require it.
        """
        pass

    def _get_fd_params(self):
        """
        Get the list of parameters that are needed to perform a
        finite difference on this `Component`.

        Returns
        -------
        list of str
            List of names of params for this `Component` .
        """
        return [k for k,m in self.params.items() if not m.get('pass_by_obj')]

    def _get_fd_unknowns(self):
        """
        Get the list of unknowns that are needed to perform a
        finite difference on this `Group`.

        Returns
        -------
        list of str
            List of names of unknowns for this `Component`.
        """
        return [k for k,m in self.unknowns.items() if not m.get('pass_by_obj')]

    def _setup_variables(self):
        """Returns our params and unknowns dictionaries, re-keyed
        to use absolute variable names, and stores them
        as attributes of the component
        """

        self.setup_param_indices()

        # rekey with absolute path names and add relative names
        _new_params = OrderedDict()
        for name, meta in self._params_dict.items():
            if not self.pathname:
                var_pathname = name
            else:
                var_pathname = '.'.join([self.pathname, name])
            _new_params[var_pathname] = meta
            meta['relative_name'] = name

        self._params_dict = _new_params

        _new_unknowns = OrderedDict()
        for name, meta in self._unknowns_dict.items():
            if not self.pathname:
                var_pathname = name
            else:
                var_pathname = '.'.join([self.pathname, name])
            _new_unknowns[var_pathname] = meta
            meta['relative_name'] = name
        self._unknowns_dict = _new_unknowns

        self._post_setup = True

        # set 'remote' attribute if this comp is not active
        if not self.is_active():
            self._set_vars_as_remote()

        return self._params_dict, self._unknowns_dict

    def _setup_vectors(self, param_owners, parent,
                       top_unknowns=None, relevance=None, impl=BasicImpl):
        """
        Set up local `VecWrappers` to store this component's variables.

        Parameters
        ----------
        param_owners : dict
            a dictionary mapping `System` pathnames to the pathnames of parameters
            they are reponsible for propagating. (ignored)

        parent : `Group`
            The parent `Group`.

        top_unknowns : `VecWrapper`, optional
            the `Problem` level unknowns `VecWrapper`

        relevance : `Relevance`
            An object containing relevance info for each variable of interest.

        impl : an implementation factory, optional
            Specifies the factory object used to create `VecWrapper` objects.
        """
        if not self.is_active():
            return

        self._relevance = relevance
        self._impl_factory = impl

        # create storage for the relevant vecwrappers, keyed by variable_of_interest
        for group, vois in relevance.groups.items():
            if group is not None:
                for voi in vois:
                    self._create_views(top_unknowns, parent, [], relevance, voi)

        # we don't get non deriv vecs (u, p, r) unless we have a None group, so force
        # their creation here
        self._create_views(top_unknowns, parent, [], relevance, None)

        # create params vec entries for any unconnected params
        for pathname, meta in self._params_dict.items():
            name = self.params._scoped_abs_name(pathname)
            if name not in self.params:
                self.params._add_unconnected_var(pathname, meta)

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
            `VecWrapper` containing parameters (p)

        unknowns : `VecWrapper`
            `VecWrapper` containing outputs and states (u)

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

    def jacobian(self, params, unknowns, resids):
        """
        Returns Jacobian. Returns None unless component overides and
        returns something. J should be a dictionary whose keys are tuples of
        the form ('unknown', 'param') and whose values are ndarrays.

        Parameters
        ----------
        params : `VecWrapper`
            `VecWrapper` containing parameters. (p)

        unknowns : `VecWrapper`
            `VecWrapper` containing outputs and states. (u)

        resids : `VecWrapper`
            `VecWrapper`  containing residuals. (r)

        Returns
        -------
        dict
            Dictionary whose keys are tuples of the form ('unknown', 'param')
            and whose values are ndarrays.
        """
        return None

    def apply_linear(self, params, unknowns, dparams, dunknowns, dresids, mode):
        """
        Multiplies incoming vector by the Jacobian (fwd mode) or the
        transpose Jacobian (rev mode). If the user doesn't provide this
        method, then we just multiply by self._jacobian_cache.

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
        """
        self._apply_linear_jac(params, unknowns, dparams, dunknowns, dresids,
                              mode)

    def solve_linear(self, dumat, drmat, vois, mode=None):
        """
        Single linear solution applied to whatever input is sitting in
        the rhs vector.

        Parameters
        ----------
        dumat : dict of `VecWrappers`
            In forward mode, each `VecWrapper` contains the incoming vector
            for the states. There is one vector per quantity of interest for
            this problem. In reverse mode, it contains the outgoing vector for
            the states. (du)

        drmat : `dict of VecWrappers`
            `VecWrapper` containing either the outgoing result in forward mode
            or the incoming vector in reverse mode. There is one vector per
            quantity of interest for this problem. (dr)

        vois: list of strings
            List of all quantities of interest to key into the mats.

        mode : string
            Derivative mode, can be 'fwd' or 'rev', but generally should be
            called without mode so that the user can set the mode in this
            system's ln_solver.options.
        """

        if mode=='fwd':
            sol_vec, rhs_vec = self.dumat, self.drmat
        else:
            sol_vec, rhs_vec = self.drmat, self.dumat

        for voi in vois:
            sol_vec[voi].vec[:] = rhs_vec[voi].vec[:]

    def dump(self, nest=0, out_stream=sys.stdout, verbose=True, dvecs=False):
        """
        Writes a formated dump of this `Component` to file.

        Parameters
        ----------
        nest : int, optional
            Starting nesting level.  Defaults to 0.

        out_stream : an open file, optional
            Where output is written.  Defaults to sys.stdout.

        verbose : bool, optional
            If True (the default), output additional info beyond
            just the tree structure.

        dvecs : bool, optional
            If True, show contents of du and dp vectors instead of
            u and p (the default).
        """
        klass = self.__class__.__name__
        if dvecs:
            ulabel, plabel, uvecname, pvecname = 'du', 'dp', 'dunknowns', 'dparams'
        else:
            ulabel, plabel, uvecname, pvecname = 'u', 'p', 'unknowns', 'params'

        uvec = getattr(self, uvecname)
        pvec = getattr(self, pvecname)

        lens = [len(n) for n in uvec.keys()]
        nwid = max(lens) if lens else 12

        commsz = self.comm.size if hasattr(self.comm, 'size') else 0

        out_stream.write("%s %s '%s'    req: %s  usize:%d  psize:%d  commsize:%d\n" %
                   (" "*nest,
                    klass,
                    self.name,
                    self.get_req_procs(),
                    uvec.vec.size,
                    pvec.vec.size,
                    commsz))

        for v, meta in uvec.items():
            if verbose:
                if v in uvec._slices:
                    uslice = '{0}[{1[0]}:{1[1]}]'.format(ulabel, uvec._slices[v])
                    out_stream.write("{0}{1:<{nwid}} {2:<21} {3:>10}\n".format(" "*(nest+8),
                                                                         v,
                                                                         uslice,
                                                                         repr(uvec[v]),
                                                                         nwid=nwid))
                elif not dvecs: # deriv vecs don't have passing by obj
                    out_stream.write("{0}{1:<{nwid}}  (by_obj) ({2})\n".format(" "*(nest+8),
                                                                         v,
                                                                         repr(uvec[v]),
                                                                         nwid=nwid))

        out_stream.flush()
