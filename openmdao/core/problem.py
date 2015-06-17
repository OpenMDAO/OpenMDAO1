""" OpenMDAO Problem class defintion."""

from __future__ import print_function

import warnings
from itertools import chain
from six import iteritems, string_types
import sys

# pylint: disable=E0611, F0401
import numpy as np

from openmdao.core.system import System
from openmdao.core.basicimpl import BasicImpl
from openmdao.core.checks import check_connections
from openmdao.core.component import Component
from openmdao.core.driver import Driver
from openmdao.core.mpiwrap import MPI, FakeComm
from openmdao.core.relevance import Relevance
from openmdao.units.units import get_conversion_tuple
from openmdao.util.strutil import get_common_ancestor
from openmdao.devtools.debug import debug


class Problem(System):
    """ The Problem is always the top object for running an OpenMDAO
    model.
    """

    def __init__(self, root=None, driver=None, impl=None):
        super(Problem, self).__init__()
        self.root = root
        if impl is None:
            self._impl = BasicImpl
        else:
            self._impl = impl
        if driver is None:
            self.driver = Driver()
        else:
            self.driver = driver

    def __getitem__(self, name):
        """Retrieve unflattened value of named variable from the root system.

        Parameters
        ----------
        name : str   OR   tuple : (name, vector)
             The name of the variable to retrieve from the unknowns vector OR
             a tuple of the name of the variable and the vector to get its
             value from.

        Returns
        -------
        The unflattened value of the given variable.
        """
        return self.root[name]

    def __setitem__(self, name, val):
        """Sets the given value into the appropriate `VecWrapper`.

        Parameters
        ----------
        name : str
             The name of the variable to set into the unknowns vector.
        """
        self.root[name] = val

    def subsystem(self, name):
        """
        Parameters
        ----------
        name : str
            Name of the subsystem to retrieve.

        Returns
        -------
        `System`
            A reference to the named subsystem.
        """
        return self.root.subsystem(name)

    def setup(self):
        """Performs all setup of vector storage, data transfer, etc.,
        necessary to perform calculations.
        """
        # Give every system an absolute pathname
        self.root._setup_paths(self.pathname)

        # divide MPI communicators among subsystems
        if MPI:
            self.root._setup_communicators(MPI.COMM_WORLD)
        else:
            self.root._setup_communicators(FakeComm())

        # Give every system a dictionary of parameters and of unknowns
        # that are visible to that system, keyed on absolute pathnames.
        # Metadata for each variable will contain the name of the
        # variable relative to that system as well as size and shape if
        # known.

        # Returns the parameters and unknowns dictionaries for the root.
        params_dict, unknowns_dict = self.root._setup_variables()

        # Get all explicit connections (stated with absolute pathnames)
        connections = self.root._get_explicit_connections()

        # go through relative names of all top level params/unknowns
        # if relative name in unknowns matches relative name in params
        # that indicates an implicit connection. All connections are returned
        # in absolute form.
        implicit_conns = _get_implicit_connections(params_dict, unknowns_dict)

        # check for conflicting explicit/implicit connections
        for tgt, src in connections.items():
            if tgt in implicit_conns:
                msg = "'%s' is explicitly connected to '%s' but implicitly connected to '%s'" % \
                      (tgt, connections[tgt], implicit_conns[tgt])
                raise RuntimeError(msg)

        # combine implicit and explicit connections
        connections.update(implicit_conns)

        # calculate unit conversions and store in param metadata
        _setup_units(connections, params_dict, unknowns_dict)

        # perform additional checks on connections (e.g. for compatible types and shapes)
        check_connections(connections, params_dict, unknowns_dict)

        # check for parameters that are not connected to a source/unknown
        hanging_params = []
        for p in params_dict:
            if p not in connections.keys():
                hanging_params.append(p)

        if hanging_params:
            msg = 'Parameters %s have no associated unknowns.' % hanging_params
            warnings.warn(msg)

        # propagate top level metadata, e.g. unit_conv, to subsystems
        self.root._update_sub_unit_conv()

        # Given connection information, create mapping from system pathname
        # to the parameters that system must transfer data to
        param_owners = assign_parameters(connections)

        # some mode determination requires connection information, so pass
        # it down the tree
        self.root.connections = connections
        for name, sub in self.root.subsystems(recurse=True):
            sub.connections = connections

        mode = self._check_for_matrix_matrix(self.driver._inputs_of_interest,
                                             self.driver._outputs_of_interest)

        relevance = Relevance(params_dict, unknowns_dict, connections,
                              self.driver._inputs_of_interest,
                              self.driver._outputs_of_interest,
                              mode)

        # create VarManagers and VecWrappers for all groups in the system tree.
        self.root._setup_vectors(param_owners, relevance=relevance,
                                 impl=self._impl)

        # Prep for case recording
        self._start_recorders()

    def run(self):
        """ Runs the Driver in self.driver. """

        if self.root.is_active():
            self.driver.run(self)

    def _mode(self, mode, param_list, unknown_list):
        """ Determine the mode based on precedence. The mode in `mode` is
        first. If that is 'auto', then the mode in root.ln_options takes
        precedence. If that is 'auto', then mode is determined by the width
        of the prameter and quantity space."""

        root = self.root

        if mode == 'auto':
            mode = root.ln_solver.options['mode']
            if mode == 'auto':
                p_dict = root._params_dict
                u_dict = root._unknowns_dict

                # Sum up param size
                p_length = 0
                for param in param_list:

                    if param in p_dict:
                        meta = p_dict[param]
                    else:
                        # The user sometimes specifies the parameter output
                        # name instead of its target because it is more
                        # convenient
                        for key, val in iteritems(root.connections):
                            if val == param:
                                meta = u_dict[param]
                                break

                        # We need the absolute name, but the fd Jacobian
                        # holds relative promoted inputs
                        else:
                            for key in p_dict:
                                metadata = root.params.metadata(key)
                                if metadata['relative_name'] == param:
                                    meta = p_dict[metadata['pathname']]
                                    break

                    p_length += meta['size']

                # Sum up unknowns size
                u_length = 0
                for unkn in unknown_list:

                    if unkn in u_dict:
                        meta = u_dict[unkn]
                    else:
                        # We need the absolute name, but the fd Jacobian
                        # holds relative promoted inputs
                        for key in root.unknowns:
                            metadata = root.unknowns.metadata(key)
                            if metadata['pathname'] == unkn:
                                meta = u_dict[metadata['relative_name']]
                                break

                    u_length += meta['size']

                # Choose mode based on size
                if p_length > u_length:
                    mode = 'rev'
                else:
                    mode = 'fwd'

        return mode

    def calc_gradient(self, param_list, unknown_list, mode='auto',
                      return_format='array'):
        """ Returns the gradient for the system that is slotted in
        self.root. This function is used by the optimizer but also can be
        used for testing derivatives on your model.

        Parameters
        ----------
        param_list : list of strings (optional)
            List of parameter name strings with respect to which derivatives
            are desired. All params must have a paramcomp.

        unknown_list : list of strings (optional)
            List of output or state name strings for derivatives to be
            calculated. All must be valid unknowns in OpenMDAO.

        mode : string (optional)
            Deriviative direction, can be 'fwd', 'rev', 'fd', or 'auto'.
            Default is 'auto', which uses mode specified on the linear solver
            in root.

        return_format : string (optional)
            Format for the derivatives, can be 'array' or 'dict'.

        Returns
        -------
        ndarray or dict
            Jacobian of unknowns with respect to params.
        """

        if mode not in ['auto', 'fwd', 'rev', 'fd']:
            msg = "mode must be 'auto', 'fwd', 'rev', or 'fd'"
            raise ValueError(msg)

        if return_format not in ['array', 'dict']:
            msg = "return_format must be 'array' or 'dict'"
            raise ValueError(msg)

        # Either analytic or finite difference
        if mode == 'fd' or self.root.fd_options['force_fd'] == True:
            return self._calc_gradient_fd(param_list, unknown_list,
                                          return_format)
        else:
            return self._calc_gradient_ln_solver(param_list, unknown_list,
                                                 return_format, mode)

    def _calc_gradient_fd(self, param_list, unknown_list, return_format):
        """ Returns the finite differenced gradient for the system that is slotted in
        self.root.

        Parameters
        ----------
        param_list : list of strings (optional)
            List of parameter name strings with respect to which derivatives
            are desired. All params must have a paramcomp.

        unknown_list : list of strings (optional)
            List of output or state name strings for derivatives to be
            calculated. All must be valid unknowns in OpenMDAO.

        return_format : string (optional)
            Format for the derivatives, can be 'array' or 'dict'.

        Returns
        -------
        ndarray or dict
            Jacobian of unknowns with respect to params.
        """

        root = self.root
        unknowns = root.unknowns
        params = root.params

        Jfd = root.fd_jacobian(params, unknowns, root.resids,
                               total_derivs=True)
        J = {}
        for okey in unknown_list:
            J[okey] = {}
            for ikey in param_list:
                if isinstance(ikey, tuple):
                    ikey = ikey[0]

                # User might request an output via the absolute pathname
                fd_okey = okey
                if fd_okey not in unknowns:
                    for key in unknowns:
                        meta = unknowns.metadata(key)
                        if meta['pathname'] == fd_okey:
                            fd_okey = meta['relative_name']
                            break

                # FD Input keys are a little funny...
                fd_ikey = ikey
                if fd_ikey not in params:

                    # The user sometimes specifies the parameter output
                    # name instead of its target because it is more
                    # convenient
                    for key, val in iteritems(root.connections):
                        if val == ikey:
                            fd_ikey = key
                            break

                    # We need the absolute name, but the fd Jacobian
                    # holds relative promoted inputs
                    if fd_ikey not in params:
                        for key in params:
                            meta = params.metadata(key)
                            if meta['relative_name'] == fd_ikey:
                                fd_ikey = meta['pathname']
                                break

                J[okey][ikey] = Jfd[fd_okey, fd_ikey]
        return J

    def _calc_gradient_ln_solver(self, param_list, unknown_list, return_format, mode):
        """ Returns the gradient for the system that is slotted in
        self.root. The gradient is calculated using root.ln_solver.

        Parameters
        ----------
        param_list : list of strings (optional)
            List of parameter name strings with respect to which derivatives
            are desired. All params must have a paramcomp.

        unknown_list : list of strings (optional)
            List of output or state name strings for derivatives to be
            calculated. All must be valid unknowns in OpenMDAO.

        return_format : string (optional)
            Format for the derivatives, can be 'array' or 'dict'.

        mode : string (optional)
            Deriviative direction, can be 'fwd', 'rev', 'fd', or 'auto'.
            Default is 'auto', which uses mode specified on the linear solver
            in root.

        Returns
        -------
        ndarray or dict
            Jacobian of unknowns with respect to params.
        """

        root = self.root
        unknowns = root.unknowns
        params = root.params

        # Prepare model for calculation
        root.clear_dparams()
        for names in root._relevance.vars_of_interest():
            for name in names:
                if name in root.dumat:
                    root.dumat[name].vec[:] = 0.0
                    root.drmat[name].vec[:] = 0.0
        root.dumat[None].vec[:] = 0.0
        root.drmat[None].vec[:] = 0.0

        root.jacobian(params, unknowns, root.resids)

        rhs = np.zeros((len(unknowns.vec), ))

        # Initialize Jacobian
        if return_format == 'dict':
            J = {}
            for okey in unknown_list:
                J[okey] = {}
                for ikey in param_list:
                    if isinstance(ikey, tuple):
                        ikey = ikey[0]
                    J[okey][ikey] = None
        else:
            # TODO: need these functions
            num_input = system.get_size(param_list)
            num_output = system.get_size(unknown_list)
            J = np.zeros((num_output, num_input))

        # Respect choice of mode based on precedence.
        # Call arg > ln_solver option > auto-detect
        mode = self._mode(mode, param_list, unknown_list)

        if mode == 'fwd':
            input_list, output_list = param_list, unknown_list
        else:
            input_list, output_list = unknown_list, param_list

        # If Forward mode, solve linear system for each param
        # If Adjoint mode, solve linear system for each unknown
        j = 0
        for param in input_list:

            if param in unknowns:
                in_size, in_idxs = unknowns.get_local_idxs(param)
            else:
                param_src = root.connections.get(param)
                param_src = unknowns.get_relative_varname(param_src)
                in_size, in_idxs = unknowns.get_local_idxs(param_src)

            jbase = j

            for irhs in in_idxs:

                rhs[irhs] = 1.0

                # Call GMRES to solve the linear system
                dx = root.ln_solver.solve(rhs, root, mode)

                rhs[irhs] = 0.0

                i = 0
                for item in output_list:

                    if item in unknowns:
                        out_size, out_idxs = unknowns.get_local_idxs(item)
                    else:
                        param_src = root.connections.get(item)
                        param_src = unknowns.get_relative_varname(param_src)
                        out_size, out_idxs = unknowns.get_local_idxs(param_src)

                    nk = len(out_idxs)

                    if return_format == 'dict':
                        if mode == 'fwd':
                            if J[item][param] is None:
                                J[item][param] = np.zeros((nk, len(in_idxs)))
                            J[item][param][:, j-jbase] = dx[out_idxs]
                        else:
                            if J[param][item] is None:
                                J[param][item] = np.zeros((len(in_idxs), nk))
                            J[param][item][j-jbase, :] = dx[out_idxs]

                    else:
                        if mode == 'fwd':
                            J[i:i+nk, j] = dx[out_idxs]
                        else:
                            J[j, i:i+nk] = dx[out_idxs]
                        i += nk

                j += 1

        return J

    def check_partial_derivatives(self, out_stream=sys.stdout):
        """ Checks partial derivatives comprehensively for all components in
        your model.

        Parameters
        ----------

        out_stream : file_like
            Where to send human readable output. Default is sys.stdout. Set to
            None to suppress.

        Returns
        -------
        Dict of Dicts of Dicts of Tuples of Floats.

        First key is the component name; 2nd key is the (output, input) tuple
        of strings; third key is one of ['rel error', 'abs error',
        'magnitude', 'fdstep']; Tuple contains norms for forward - fd,
        adjoint - fd, forward - adjoint using the best case fdstep.
        """

        root = self.root

        # Linearize the model
        root.jacobian(root.params, root.unknowns, root.resids)

        if out_stream is not None:
            out_stream.write('Partial Derivatives Check\n\n')

        data = {}
        skip_keys = []
        model_hierarchy = _find_all_comps(root)

        # FIXME:
        voi = None

        # Check derivative calculations for all comps at every level of the
        # system hierarchy.
        for group, comps in model_hierarchy.items():
            for comp in comps:

                # No need to check comps that don't have any derivs.
                if comp.fd_options['force_fd'] == True:
                    continue

                cname = comp.pathname
                data[cname] = {}
                jac_fwd = {}
                jac_rev = {}
                jac_fd = {}

                params = comp.params
                unknowns = comp.unknowns
                resids = comp.resids
                dparams = comp.dpmat[voi]
                dunknowns = comp.dumat[voi]
                dresids = comp.drmat[voi]

                if out_stream is not None:
                    out_stream.write('-'*(len(cname)+15) + '\n')
                    out_stream.write("Component: '%s'\n" % cname)
                    out_stream.write('-'*(len(cname)+15) + '\n')

                # Figure out implicit states for this comp
                states = []
                for u_name, meta in iteritems(comp._unknowns_dict):
                    if meta.get('state'):
                        states.append(meta['relative_name'])

                # Create all our keys and allocate Jacs
                for p_name in chain(dparams, states):

                    dinputs = dunknowns if p_name in states else dparams
                    p_size = np.size(dinputs[p_name])

                    # Check dimensions of user-supplied Jacobian
                    for u_name in unknowns:

                        u_size = np.size(dunknowns[u_name])
                        if comp._jacobian_cache is not None:

                            # Go no further if we aren't defined.
                            if (u_name, p_name) not in comp._jacobian_cache:
                                skip_keys.append((u_name, p_name))
                                continue

                            user = comp._jacobian_cache[(u_name, p_name)].shape

                            # User may use floats for scalar jacobians
                            if len(user) < 2:
                                user = (user[0], 1)

                            if user[0] != u_size or user[1] != p_size:
                                msg = "Jacobian in component '{}' between the" + \
                                " variables '{}' and '{}' is the wrong size. " + \
                                "It should be {} by {}"
                                msg = msg.format(cname, p_name, u_name, p_size,
                                                 u_size)
                                raise ValueError(msg)

                        jac_fwd[(u_name, p_name)] = np.zeros((u_size, p_size))
                        jac_rev[(u_name, p_name)] = np.zeros((u_size, p_size))

                # Reverse derivatives first
                for u_name in dresids:
                    u_size = np.size(dunknowns[u_name])

                    # Send columns of identity
                    for idx in range(u_size):
                        dresids.vec[:] = 0.0
                        root.clear_dparams()
                        dunknowns.vec[:] = 0.0

                        dresids.flat[u_name][idx] = 1.0
                        comp.apply_linear(params, unknowns, dparams,
                                          dunknowns, dresids, 'rev')

                        for p_name in chain(dparams, states):
                            if (u_name, p_name) in skip_keys:
                                continue

                            dinputs = dunknowns if p_name in states else dparams

                            jac_rev[(u_name, p_name)][idx, :] = dinputs.flat[p_name]

                # Forward derivatives second
                for p_name in chain(dparams, states):

                    dinputs = dunknowns if p_name in states else dparams
                    p_size = np.size(dinputs[p_name])

                    # Send columns of identity
                    for idx in range(p_size):
                        dresids.vec[:] = 0.0
                        root.clear_dparams()
                        dunknowns.vec[:] = 0.0

                        dinputs.flat[p_name][idx] = 1.0
                        comp.apply_linear(params, unknowns, dparams,
                                          dunknowns, dresids, 'fwd')

                        for u_name in dresids:
                            if (u_name, p_name) in skip_keys:
                                continue

                            jac_fwd[(u_name, p_name)][:, idx] = dresids.flat[u_name]

                # Finite Difference goes last
                dresids.vec[:] = 0.0
                root.clear_dparams()
                dunknowns.vec[:] = 0.0
                jac_fd = comp.fd_jacobian(params, unknowns, resids,
                                          step_size=1e-6)

                # Assemble and Return all metrics.
                _assemble_deriv_data(chain(dparams, states), resids, data[cname],
                                     jac_fwd, jac_rev, jac_fd, out_stream,
                                     skip_keys, c_name=cname)

        return data

    def check_total_derivatives(self, out_stream=sys.stdout):
        """ Checks total derivatives for problem defined at the top.

        Parameters
        ----------

        out_stream : file_like
            Where to send human readable output. Default is sys.stdout. Set to
            None to suppress.

        Returns
        -------
        Dict of Dicts of Tuples of Floats

        First key is the (output, input) tuple of strings; second key is one
        of ['rel error', 'abs error', 'magnitude', 'fdstep']; Tuple contains
        norms for forward - fd, adjoint - fd, forward - adjoint using the
        best case fdstep.
        """

        if out_stream is not None:
            out_stream.write('Total Derivatives Check\n\n')

        # Params and Unknowns that we provide at this level.
        abs_param_list = self.root._get_fd_params()
        unknown_list = self.root._get_fd_unknowns()

        # Convert absolute parameter names to relative ones because it is
        # easier for the user to read.
        param_list = []
        params = self.root.params
        for param in abs_param_list:
            if param not in self.root.unknowns:
                param_list.append(params.metadata(param)['relative_name'])
            else:
                param_list.append(param)

        # Calculate all our Total Derivatives
        Jfor = self.calc_gradient(param_list, unknown_list, mode='fwd',
                                  return_format='dict')
        Jrev = self.calc_gradient(param_list, unknown_list, mode='rev',
                                  return_format='dict')
        Jfd = self.calc_gradient(param_list, unknown_list, mode='fd',
                                 return_format='dict')

        Jfor = jac_to_flat_dict(Jfor)
        Jrev = jac_to_flat_dict(Jrev)
        Jfd = jac_to_flat_dict(Jfd)

        # Assemble and Return all metrics.
        data = {}
        _assemble_deriv_data(param_list, unknown_list, data,
                             Jfor, Jrev, Jfd, out_stream)

        return data

    def _start_recorders(self):
        for recorder in self.driver.recorders:
            recorder.startup(self.root)

        for group, solvers in _find_all_solvers(self.root):
            for solver in solvers:
                for recorder in solver.recorders:
                    recorder.startup(group)

    def _check_for_matrix_matrix(self, params, unknowns):
        """ Checks a system hiearchy to make sure that no settings violate the
        assumptions needed for matrix-matrix calculation. Returns the mode that
        the system needs to use.
        """

        mode = self._mode('auto', params, unknowns)

        # TODO : Only Linear GS is supported on system

        for _, sub in self.root.subgroups(recurse=True):
            sub_mode = sub.ln_solver.options['mode']

            # Modes must match root for all subs
            if sub_mode not in (mode, 'auto'):
                msg  = "Group '{name}' has mode '{submode}' but the root group has mode '{rootmode}'." \
                        " Modes must match to use Matrix Matrix."
                msg = msg.format(name=sub.name, submode=sub_mode, rootmode=mode)
                raise RuntimeError(msg)

            # TODO : Only Linear GS is supported on sub

        return mode

def _setup_units(connections, params_dict, unknowns_dict):
    """
    Calculate unit conversion factors for any connected
    variables having different units and store them in params_dict.

    Parameters
    ----------
    connections : dict
        A dict of target variables (absolute name) mapped
        to the absolute name of their source variable.

    params_dict : OrderedDict
        A dict of parameter metadata for the whole `Problem`.

    unknowns_dict : OrderedDict
        A dict of unknowns metadata for the whole `Problem`.
    """

    for target, source in connections.items():
        tmeta = params_dict[target]
        smeta = unknowns_dict[source]

        # units must be in both src and target to have a conversion
        if 'units' not in tmeta or 'units' not in smeta:
            continue

        src_unit = smeta['units']
        tgt_unit = tmeta['units']

        try:
            scale, offset = get_conversion_tuple(src_unit, tgt_unit)
        except TypeError as err:
            if str(err) == "Incompatible units":
                msg = "Unit '{s[units]}' in source '{s[relative_name]}' "\
                    "is incompatible with unit '{t[units]}' "\
                    "in target '{t[relative_name]}'.".format(s=smeta, t=tmeta)
                raise TypeError(msg)
            else:
                raise

        # If units are not equivalent, store unit conversion tuple
        # in the parameter metadata
        if scale != 1.0 or offset != 0.0:
            tmeta['unit_conv'] = (scale, offset)


def assign_parameters(connections):
    """Map absolute system names to the absolute names of the
    parameters they transfer data to.
    """
    param_owners = {}

    for par, unk in connections.items():
        param_owners.setdefault(get_common_ancestor(par, unk), []).append(par)

    return param_owners


def _find_all_solvers(group):
    """Recursively finds all solvers in the given group and sub-groups."""
    yield (group, (group.ln_solver, group.nl_solver))
    for _, sub in group.subgroups():
        for solvers in _find_all_solvers(sub):
            yield solvers

def _find_all_comps(group):
    """ Recursive function that assembles a dictionary whose keys are Group
    instances and whose values are lists of Component instances."""

    data = {group:[]}
    for c_name, c in group.components():
        data[group].append(c)
    for sg_name, sg in group.subgroups():
        data.update(_find_all_comps(sg))
    return data


def jac_to_flat_dict(jac):
    """ Converts a double `dict` jacobian to a flat `dict` Jacobian. Keys go
    from [out][in] to [out,in].

    Parameters
    ----------

    jac : dict of dicts of ndarrays
        Jacobian that comes from calc_gradient when the return_type is 'dict'.

    Returns
    -------

    dict of ndarrays"""

    new_jac = {}
    for key1, val1 in jac.items():
        for key2, val2 in val1.items():
            new_jac[(key1, key2)] = val2

    return new_jac

def _assemble_deriv_data(params, resids, cdata, jac_fwd, jac_rev, jac_fd,
                         out_stream, skip_keys=[None], c_name='root'):
    """ Assembles dictionaries and prints output for check derivatives
    functions. This is used by both the partial and total derivative
    checks."""
    started = False

    for p_name in params:
        for u_name in resids:

            ldata = cdata[(u_name, p_name)] = {}

            Jsub_fd = jac_fd[(u_name, p_name)]

            if (u_name, p_name) in skip_keys:
                Jsub_for = np.zeros(Jsub_fd.shape)
                Jsub_rev = np.zeros(Jsub_fd.shape)
            else:
                Jsub_for = jac_fwd[(u_name, p_name)]
                Jsub_rev = jac_rev[(u_name, p_name)]

            ldata['J_fd'] = Jsub_fd
            ldata['J_fwd'] = Jsub_for
            ldata['J_rev'] = Jsub_rev

            magfor = np.linalg.norm(Jsub_for)
            magrev = np.linalg.norm(Jsub_rev)
            magfd = np.linalg.norm(Jsub_fd)

            ldata['magnitude'] = (magfor, magrev, magfd)

            abs1 = np.linalg.norm(Jsub_for - Jsub_fd)
            abs2 = np.linalg.norm(Jsub_rev - Jsub_fd)
            abs3 = np.linalg.norm(Jsub_for - Jsub_rev)

            ldata['abs error'] = (abs1, abs2, abs3)

            rel1 = np.linalg.norm(Jsub_for - Jsub_fd)/magfd
            rel2 = np.linalg.norm(Jsub_rev - Jsub_fd)/magfd
            rel3 = np.linalg.norm(Jsub_for - Jsub_rev)/magfd

            ldata['rel error'] = (rel1, rel2, rel3)

            if out_stream is None:
                continue

            if started is True:
                out_stream.write(' -'*30 + '\n')
            else:
                started = True

            # Optional file_like output
            out_stream.write("  %s: '%s' wrt '%s'\n\n"% (c_name, u_name, p_name))

            out_stream.write('    Forward Magnitude : %.6e\n' % magfor)
            out_stream.write('    Reverse Magnitude : %.6e\n' % magrev)
            out_stream.write('         Fd Magnitude : %.6e\n\n' % magfd)

            out_stream.write('    Absolute Error (Jfor - Jfd) : %.6e\n' % abs1)
            out_stream.write('    Absolute Error (Jrev - Jfd) : %.6e\n' % abs2)
            out_stream.write('    Absolute Error (Jfor - Jrev): %.6e\n\n' % abs3)

            out_stream.write('    Relative Error (Jfor - Jfd) : %.6e\n' % rel1)
            out_stream.write('    Relative Error (Jrev - Jfd) : %.6e\n' % rel2)
            out_stream.write('    Relative Error (Jfor - Jrev): %.6e\n\n' % rel3)

            out_stream.write('    Raw Forward Derivative (Jfor)\n\n')
            out_stream.write(str(Jsub_for))
            out_stream.write('\n\n')
            out_stream.write('    Raw Reverse Derivative (Jrev)\n\n')
            out_stream.write(str(Jsub_rev))
            out_stream.write('\n\n')
            out_stream.write('    Raw FD Derivative (Jfor)\n\n')
            out_stream.write(str(Jsub_fd))
            out_stream.write('\n\n')

def _get_implicit_connections(params_dict, unknowns_dict):
    """
    Finds all matches between relative names of parameters and
    unknowns.  Any matches imply an implicit connection.  All
    connections are expressed using absolute pathnames.

    This should only be called using params and unknowns from the
    top level `Group` in the system tree.

    Parameters
    ----------
    params_dict : dict
        dictionary of metadata for all parameters in this `Group`

    unknowns_dict : dict
        dictionary of metadata for all unknowns in this `Group`

    Returns
    -------
    dict
        implicit connections in this `Group`, represented as a mapping
        from the pathname of the target to the pathname of the source

    Raises
    ------
    RuntimeError
        if a a promoted variable name matches multiple unknowns
    """

    # collect all absolute names that map to each relative name
    abs_unknowns = {}
    for abs_name, u in unknowns_dict.items():
        abs_unknowns.setdefault(u['relative_name'], []).append(abs_name)

    abs_params = {}
    for abs_name, p in params_dict.items():
        abs_params.setdefault(p['relative_name'], []).append(abs_name)

    # check if any relative names correspond to mutiple unknowns
    for name, lst in abs_unknowns.items():
        if len(lst) > 1:
            raise RuntimeError("Promoted name '%s' matches multiple unknowns: %s" %
                               (name, lst))

    connections = {}
    for uname, uabs in abs_unknowns.items():
        pabs = abs_params.get(uname, ())
        for p in pabs:
            connections[p] = uabs[0]

    return connections

