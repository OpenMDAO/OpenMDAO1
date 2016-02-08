""" OpenMDAO Problem class defintion."""

from __future__ import print_function

import os
import sys
import json
import warnings
from itertools import chain
from six import iteritems, itervalues
from six.moves import cStringIO
import networkx as nx

import numpy as np

from openmdao.core.system import System
from openmdao.core.group import Group
from openmdao.core.component import Component
from openmdao.core.parallel_group import ParallelGroup
from openmdao.core.parallel_fd_group import ParallelFDGroup
from openmdao.core.basic_impl import BasicImpl
from openmdao.core._checks import check_connections
from openmdao.core.driver import Driver
from openmdao.core.mpi_wrap import MPI, under_mpirun
from openmdao.core.relevance import Relevance

from openmdao.components.indep_var_comp import IndepVarComp
from openmdao.solvers.scipy_gmres import ScipyGMRES
from openmdao.solvers.ln_gauss_seidel import LinearGaussSeidel

from openmdao.units.units import get_conversion_tuple
from collections import OrderedDict
from openmdao.util.string_util import get_common_ancestor, nearest_child, name_relative_to

force_check = os.environ.get('OPENMDAO_FORCE_CHECK_SETUP')

trace = os.environ.get('OPENMDAO_TRACE')
if trace:
    from openmdao.core.mpi_wrap import debug

class _ProbData(object):
    """
    A container for Problem level data that is needed by subsystems
    and VecWrappers.
    """
    def __init__(self):
        self.top_lin_gs = False
        self.in_complex_step = False


class Problem(object):
    """ The Problem is always the top object for running an OpenMDAO
    model.

    Args
    ----
    root : `Group`, optional
        The top-level `Group` for the `Problem`.  If not specified, a default
        `Group` will be created

    driver : `Driver`, optional
        The top-level `Driver` for the `Problem`.  If not specified, a default
        "Run Once" `Driver` will be used

    impl : `BasicImpl` or `PetscImpl`, optional
        The vector and data transfer implementation for the `Problem`.
        For parallel processing support using MPI, `PetscImpl` is required.
        If not specified, the default `BasicImpl` will be used.

    comm : an MPI communicator (real or fake), optional
        A communicator that can be used for distributed operations when running
        under MPI. If not specified, the default "COMM_WORLD" will be used.

    Options
    -------
    fd_options['force_fd'] :  bool(False)
        Set to True to finite difference this system.
    fd_options['form'] :  str('forward')
        Finite difference mode. (forward, backward, central) You can also set to 'complex_step' to peform the complex step method if your components support it.
    fd_options['step_size'] :  float(1e-06)
        Default finite difference stepsize
    fd_options['step_type'] :  str('absolute')
        Set to absolute, relative

    """

    def __init__(self, root=None, driver=None, impl=None, comm=None):
        super(Problem, self).__init__()
        self.root = root
        self._probdata = _ProbData()

        if MPI:
            from openmdao.core.petsc_impl import PetscImpl
            if impl != PetscImpl:
                raise ValueError("To run under MPI, the impl for a Problem must be PetscImpl.")

        if impl is None:
            self._impl = BasicImpl
        else:
            self._impl = impl

        self.comm = comm

        if driver is None:
            self.driver = Driver()
        else:
            self.driver = driver

        self.pathname = ''

    def __getitem__(self, name):
        """Retrieve unflattened value of named unknown or unconnected
        param variable from the root system.

        Args
        ----
        name : str
             The name of the variable.

        Returns
        -------
        The unflattened value of the given variable.
        """
        if name in self.root.unknowns:
            return self.root.unknowns[name]
        elif name in self.root.params:
            return self.root.params[name]
        elif name in self.root._sysdata.to_abs_pnames:
            for p in self.root._sysdata.to_abs_pnames[name]:
                return self._rec_get_param(p)
        elif name in self._dangling:
            for p in self._dangling[name]:
                return self._rec_get_param(p)
        else:
            raise KeyError("Variable '%s' not found." % name)

    def _rec_get_param(self, absname):
        parts = absname.rsplit('.', 1)
        if len(parts) == 1:
            return self.root.params[absname]
        else:
            grp = self.root._subsystem(parts[0])
            return grp.params[parts[1]]

    def __setitem__(self, name, val):
        """Sets the given value into the appropriate `VecWrapper`.
        'name' is assumed to be a promoted name.

        Args
        ----
        name : str
             The promoted name of the variable to set into the
             unknowns vector, or into params vectors if the params are
             unconnected.
        """
        if name in self.root.unknowns:
            self.root.unknowns[name] = val
        elif name in self._dangling:
            for p in self._dangling[name]:
                parts = p.rsplit('.', 1)
                if len(parts) == 1:
                    self.root.params[p] = val
                else:
                    grp = self.root._subsystem(parts[0])
                    grp.params[parts[1]] = val
        else:
            raise KeyError("Variable '%s' not found." % name)

    def _setup_connections(self, params_dict, unknowns_dict, compute_indices=True):
        """Generate a mapping of absolute param pathname to the pathname
        of its unknown.

        Args
        ----

        params_dict : OrderedDict
            A dict of parameter metadata for the whole `Problem`.

        unknowns_dict : OrderedDict
            A dict of unknowns metadata for the whole `Problem`.

        compute_indices : bool, optional
            If True, perform mapping of src_indices for input to input
            connections.
        """

        # Get all explicit connections (stated with absolute pathnames)
        connections = self.root._get_explicit_connections()

        # get dictionary of implicit connections {param: [unknowns]}
        # and dictionary of params that are not implicitly connected
        # to anything {promoted_name: pathname}
        implicit_conns, prom_noconns = self._get_implicit_connections()


        # combine implicit and explicit connections
        for tgt, srcs in iteritems(implicit_conns):
            connections.setdefault(tgt, []).extend(srcs)

        input_graph = nx.Graph()

        # resolve any input to input connections
        for tgt, srcs in iteritems(connections):
            for src, idxs in srcs:
                if src in params_dict:
                    input_graph.add_edge(src, tgt, idxs=idxs)

        # find any promoted but not connected inputs
        for p, prom in iteritems(self.root._sysdata.to_prom_pname):
            if prom in prom_noconns:
                for n in prom_noconns[prom]:
                    if p != n:
                        input_graph.add_edge(p, n, idxs=None)

        # store all of the connected sets of inputs for later use
        self._input_inputs = {}

        for tgt in connections:
            if tgt in input_graph and tgt not in self._input_inputs:
                # force list here, since some versions of networkx return a
                # set here.
                connected = list(nx.node_connected_component(input_graph, tgt))
                for c in connected:
                    self._input_inputs[c] = connected

        # initialize this here since we may have unit diffs for input-input
        # connections that get filtered out of the connection dict by the
        # time setup_units is called.
        self._unit_diffs = {}

        # for all connections where the source is an input, we want to connect
        # the 'unknown' source for that target to all other inputs that are
        # connected to it
        to_add = []
        for tgt, srcs in iteritems(connections):
            if tgt in input_graph:
                connected_inputs = self._input_inputs[tgt]

                for src, idxs in srcs:
                    if src in unknowns_dict:
                        for new_tgt in connected_inputs:
                            new_idxs = idxs
                            if compute_indices:
                                # follow path to new target, apply src_idxs along the way
                                path = nx.shortest_path(input_graph, tgt, new_tgt)
                                for i, node in enumerate(path[:-1]):
                                    next_idxs = input_graph[node][path[i+1]]['idxs']
                                    if next_idxs is not None:
                                        if new_idxs is not None:
                                            new_idxs = np.array(new_idxs)[next_idxs]
                                        else:
                                            new_idxs = next_idxs
                            to_add.append((new_tgt, (src, new_idxs)))

        for tgt, (src, idxs) in to_add:
            if tgt in connections:
                srcs = connections[tgt]
                if (src, idxs) not in srcs:
                    srcs.append((src, idxs))
            else:
                connections[tgt] = [(src, idxs)]

        # remove all the input to input connections, leaving just one unknown
        # connection to each param
        newconns = OrderedDict()
        for tgt, srcs in iteritems(connections):
            unknown_srcs = [src for src in srcs if src[0] in unknowns_dict]
            if len(unknown_srcs) > 1:
                src_names = (name for name, idx in unknown_srcs)
                raise RuntimeError("Target '%s' is connected to multiple unknowns: %s" %
                                   (tgt, sorted(src_names)))

            if unknown_srcs:
                newconns[tgt] = unknown_srcs[0]

        connections = newconns

        self._dangling = OrderedDict()
        for p, prom in iteritems(self.root._sysdata.to_prom_pname):
            if p not in connections:
                if p in input_graph:
                    self._dangling[prom] = \
                        set(nx.node_connected_component(input_graph, p))
                else:
                    self._dangling[prom] = set([p])

        return connections

    def _check_input_diffs(self, connections, params_dict, unknowns_dict):
        """For all sets of connected inputs, find any differences in units
        or initial value.
        """
        input_diffs = {}

        for tgt, connected_inputs in iteritems(self._input_inputs):

            # figure out if any connected inputs have different initial
            # values or different units
            if tgt not in input_diffs:
                for inp in connected_inputs:
                    input_diffs[inp] = ([], [])

                tgt_idx = connected_inputs.index(tgt)
                units = [params_dict[n].get('units') for n in connected_inputs]
                vals = [params_dict[n]['val'] for n in connected_inputs]

                diff_units = []

                for i, u in enumerate(units):
                    if i != tgt_idx and u != units[tgt_idx]:
                        if units[tgt_idx] is None:
                            sname, s = connected_inputs[i], u
                            tname, t = connected_inputs[tgt_idx], units[tgt_idx]
                        else:
                            sname, s = connected_inputs[tgt_idx], units[tgt_idx]
                            tname, t = connected_inputs[i], u

                        # report these in check_setup later
                        self._unit_diffs[(sname, tname)] = (s, t)
                        diff_units.append((connected_inputs[i], u))

                if isinstance(vals[tgt_idx], np.ndarray):
                    diff_vals = [(connected_inputs[i],v) for i,v in
                                   enumerate(vals) if not
                                       (isinstance(v, np.ndarray) and
                                          v.shape==vals[tgt_idx].shape and
                                          (v==vals[tgt_idx]).all())]
                else:
                    diff_vals = [(connected_inputs[i],v) for i,v in
                                     enumerate(vals) if v!=vals[tgt_idx]]

                # if tgt has no unknown source, units MUST match, unless
                # one of them is None. At this point, connections contains
                # only unknown to input connections, so if the target is
                # in connections, it has an unknown source.
                if tgt not in connections:
                    if diff_units:
                        filt = set([u for n,u in diff_units])
                        if None in filt:
                            filt.remove(None)
                        if filt:
                            raise RuntimeError("The following sourceless "
                                "connected inputs have different units: %s" %
                                sorted([(tgt,params_dict[tgt].get('units'))]+
                                                                    diff_units))
                    if diff_vals:
                        msg = ("The following sourceless connected inputs have "
                               "different initial values: "
                               "%s.  Connect one of them to the output of "
                               "an IndepVarComp to ensure that they have the "
                               "same initial value." %
                               (sorted([(tgt,params_dict[tgt]['val'])]+
                                                 diff_vals)))
                        raise RuntimeError(msg)

        # now check for differences in step_size, step_type, or form for
        # promoted inputs
        for promname, absnames in iteritems(self.root._sysdata.to_abs_pnames):
            if len(absnames) > 1:
                step_sizes, step_types, forms = {}, {}, {}
                for name in absnames:
                    meta = self.root._params_dict[name]
                    ss = meta.get('step_size')
                    if ss is not None:
                        step_sizes[ss] = name
                    st = meta.get('step_type')
                    if st is not None:
                        step_types[st] = name
                    f = meta.get('form')
                    if f is not None:
                        forms[f] = name

                if len(step_sizes) > 1:
                    raise RuntimeError("The following parameters have the same "
                                  "promoted name, '%s', but different "
                                  "'step_size' values: %s" % (promname,
                                  sorted([(v,k) for k,v in step_sizes.items()])))

                if len(step_types) > 1:
                    raise RuntimeError("The following parameters have the same "
                                  "promoted name, '%s', but different "
                                  "'step_type' values: %s" % (promname,
                                 sorted([(v,k) for k,v in step_types.items()])))

                if len(forms) > 1:
                    raise RuntimeError("The following parameters have the same "
                                  "promoted name, '%s', but different 'form' "
                                  "values: %s" % (promname,
                                      sorted([(v,k) for k,v in forms.items()])))

        return input_diffs

    def _get_ubc_vars(self, connections):
        """Return a list of any connected inputs that are used before they
        are set by their connected unknowns.
        """
        # this is the order that each component would run in if executed
        # a single time from the root system.
        full_order = {s.pathname : i for i,s in
                     enumerate(self.root.subsystems(recurse=True))}

        ubcs = []
        for tgt, srcs in iteritems(connections):
            tsys = tgt.rsplit('.', 1)[0]
            ssys = srcs[0].rsplit('.', 1)[0]
            if full_order[ssys] > full_order[tsys]:
                ubcs.append(tgt)

        return ubcs

    def _check_layout(self, stream=sys.stdout):
        """
        Check the current system tree to see if it's optimal.
        """
        problem_groups = {}
        for group in self.root.subgroups(recurse=True, include_self=True):
            problem_groups[group.pathname] = {}
            uses_lings = isinstance(group.ln_solver, LinearGaussSeidel)
            maxiter = group.ln_solver.options['maxiter']

            subs = [s for s in group.subsystems()]
            graph = group._get_sys_graph()
            strong = [sorted(s) for s in nx.strongly_connected_components(graph)
                      if len(s) > 1]
            cycle_systems = set()
            for s in strong:
                cycle_systems.update(s)

            if strong and len(strong[0]) == len(subs):
                # all subsystems form a single cycle
                if uses_lings:
                    print("\nAll systems in group '%s' form a cycle, so the "
                          "linear solver should be ScipyGMRES or PetscKSP." %
                          group.pathname, file=stream)
                    problem_groups[group.pathname]['ln_solver'] = _get_gmres_name()
            else:
                if strong:
                    print("\nIn group '%s' the following cycles should be "
                          "grouped into subgroups with a ScipyGMRES or PetscKSP "
                          "linear solver: %s." % (group.pathname, strong),
                          file=stream)
                    problem_groups[group.pathname]['sub_cycles'] = strong

                if (not uses_lings and (len(subs) > 1 or
                                       (len(subs)==1 and
                                        not _needs_iteration(subs[0])))):
                    print("\nGroup '%s' should have a LinearGaussSeidel linear solver." %
                           group.pathname, file=stream)
                    problem_groups[group.pathname]['ln_solver'] = 'LinearGaussSeidel'

            if len(subs) > 1 or uses_lings:
                for s in subs:
                    if (s.is_active() and s.name not in cycle_systems and
                               _needs_iteration(s)):
                        print("\nSystem '%s' has implicit states and should be "
                              "in its own subgroup with a GMRES linear solver." %
                              s.pathname, file=stream)
                        problem_groups[group.pathname].setdefault(
                                                         'sub_implicit_comps',
                                                         []).append(s.name)

            if not problem_groups[group.pathname]:
                del problem_groups[group.pathname]

        return problem_groups

    def setup(self, check=True, out_stream=sys.stdout):
        """Performs all setup of vector storage, data transfer, etc.,
        necessary to perform calculations.

        Args
        ----
        check : bool, optional
            Check for potential issues after setup is complete (the default
            is True)

        out_stream : a file-like object, optional
            Stream where report will be written if check is performed.
        """
        # if we modify the system tree, we'll need to call _init_sys_data,
        # _setup_variables and _setup_connections again
        tree_changed = False

        # call _setup_variables again if we change metadata
        meta_changed = False

        self._probdata = _ProbData()
        if isinstance(self.root.ln_solver, LinearGaussSeidel):
            self._probdata.top_lin_gs = True

        self.driver.root = self.root

        # Give every system an absolute pathname
        self.root._init_sys_data(self.pathname, self._probdata)

        # divide MPI communicators among subsystems
        self._setup_communicators()

        # Returns the parameters and unknowns metadata dictionaries
        # for the root, which has an entry for each variable contained
        # in any child of root. Metadata for each variable will contain
        # the name of the variable relative to that system, the absolute
        # name of the variable, any user defined metadata, and the value,
        # size and/or shape if known. For example:
        #  unknowns_dict['G1.G2.foo.v1'] = {
        #     'pathname' : 'G1.G2.foo.v1', # absolute path from the top
        #     'size' : 1,
        #     'shape' : 1,
        #     'val': 2.5,   # the initial value of that variable (if known)
        #  }
        params_dict, unknowns_dict = self.root._setup_variables()
        self._probdata.params_dict = params_dict
        self._probdata.unknowns_dict = unknowns_dict

        self._probdata.to_prom_name = self.root._sysdata.to_prom_name

        # collect all connections, both implicit and explicit from
        # anywhere in the tree, and put them in a dict where each key
        # is an absolute param name that maps to the absolute name of
        # a single source.
        connections = self._setup_connections(params_dict, unknowns_dict)
        self._probdata.connections = connections

        # Allow the user to omit the size of a parameter and pull the size
        # and shape from the connection source.
        for tgt, src in iteritems(connections):
            tmeta = params_dict[tgt]
            if not tmeta.get('pass_by_obj') and tmeta['shape'] == ():

                src_name, src_idx = src
                smeta = unknowns_dict[src_name]

                # Connected with src_indices specified
                if src_idx is not None:
                    size = len(src_idx)
                    tmeta['shape'] = (size, )
                    tmeta['size'] = size
                    tmeta['val'] = smeta['val'][np.array(src_idx)]

                # Regular connection
                else:
                    tmeta['shape'] = smeta['shape']
                    tmeta['size'] = smeta['size']
                    tmeta['val'] = smeta['val']

        # push connection src_indices down into the metadata for all target
        # params in all component level systems, then flag meta_changed so
        # it will get percolated back up to all groups in next setup_vars()
        src_idx_conns = [(tgt, src, idxs) for tgt, (src, idxs) in
                         iteritems(connections) if idxs is not None]
        if src_idx_conns:
            meta_changed = True
            for comp in self.root.components(recurse=True):
                for tgt, src, idxs in src_idx_conns:
                    meta = comp._params_dict.get(tgt)
                    if meta and meta['pathname'] == tgt:
                        meta['src_indices'] = idxs

        # TODO: handle any automatic grouping of systems here...
        #       If we modify the system tree here, we'll have to call
        #       the full setup over again...

        # mark any variables in non-local Systems as 'remote'
        for comp in self.root.components(recurse=True):
            if not comp.is_active():
                meta_changed = True
                comp._set_vars_as_remote()

        if MPI:
            for s in self.root.components(recurse=True):
                # get rid of check for setup_distrib_idxs when we move to beta
                if hasattr(s, 'setup_distrib_idxs') or (
                         hasattr(s, 'setup_distrib') and (s.setup_distrib
                                                is not Component.setup_distrib)):
                    # component defines its own setup_distrib, so
                    # the metadata will change
                    meta_changed = True

        # All changes to the system tree or variable metadata
        # must be complete at this point.

        # if the system tree has changed, we need to recompute pathnames,
        # variable metadata, and connections
        if tree_changed:
            self.root._init_sys_data(self.pathname, self._probdata)
            params_dict, unknowns_dict = \
                self.root._setup_variables(compute_indices=True)
            connections = self._setup_connections(params_dict, unknowns_dict,
                                                  compute_indices=False)
        elif meta_changed:
            params_dict, unknowns_dict = \
                self.root._setup_variables(compute_indices=True)

        # perform additional checks on connections
        # (e.g. for compatible types and shapes)
        check_connections(connections, params_dict, unknowns_dict,
                          self.root._sysdata.to_prom_name)

        # calculate unit conversions and store in param metadata
        self._setup_units(connections, params_dict, unknowns_dict)

        # propagate top level promoted names, unit conversions,
        # and connections down to all subsystems
        to_prom_name = self.root._sysdata.to_prom_name
        self._probdata.to_prom_name = to_prom_name
        for sub in self.root.subsystems(recurse=True, include_self=True):
            sub.connections = connections

        # set top_promoted_name and unit_conv in top system (all metatdata
        # is shared, so not need to propagate down the tree)
        for path, meta in iteritems(self.root._params_dict):
            meta['top_promoted_name'] = to_prom_name[path]
            unit_conv = params_dict[path].get('unit_conv')
            if unit_conv:
                meta['unit_conv'] = unit_conv

        for path, meta in iteritems(self.root._unknowns_dict):
            meta['top_promoted_name'] = to_prom_name[path]

        # Given connection information, create mapping from system pathname
        # to the parameters that system must transfer data to
        param_owners = _assign_parameters(connections)

        pois = self.driver.desvars_of_interest()
        oois = self.driver.outputs_of_interest()

        self._driver_vois = set()
        for tup in chain(pois, oois):
            self._driver_vois.update(tup)

        # make sure pois and oois all refer to existing vars.
        # NOTE: all variables of interest (includeing POIs) must exist in
        #      the unknowns dict
        promoted_unknowns = self.root._sysdata.to_abs_uname

        parallel_p = False
        for vnames in pois:
            if len(vnames) > 1:
                parallel_p = True
            for v in vnames:
                if v not in promoted_unknowns:
                    raise NameError("Can't find param of interest '%s'." % v)

        parallel_u = False
        for vnames in oois:
            if len(vnames) > 1:
                parallel_u = True
            for v in vnames:
                if v not in promoted_unknowns:
                    raise NameError("Can't find quantity of interest '%s'." % v)

        mode = self._check_for_parallel_derivs(pois, oois, parallel_u, parallel_p)

        self._probdata.relevance = Relevance(self.root, params_dict,
                                             unknowns_dict, connections,
                                             pois, oois, mode)


        # perform auto ordering
        for s in self.root.subgroups(recurse=True, include_self=True):
            # set auto order if order not already set
            if not s._order_set:
                order = None
                broken_edges = None
                if self.comm.rank == 0:
                    order, broken_edges = s.list_auto_order()
                if MPI:
                    if trace:
                        debug("problem setup order bcast")
                    order, broken_edges = self.comm.bcast((order, broken_edges), root=0)
                    if trace:
                        debug("problem setup order bcast DONE")
                s.set_order(order)

                # Mark "head" of each broken edge
                for edge in broken_edges:
                    cname = edge[1]
                    head_sys = self.root
                    for name in cname.split('.'):
                        head_sys = getattr(head_sys, name)
                    head_sys._run_apply = True

        # report any differences in units or initial values for
        # sourceless connected inputs
        self._check_input_diffs(connections, params_dict, unknowns_dict)

        # Check for dangling params that have no size or shape
        dangling_params = set([p for p in self.root._params_dict
                               if p not in self.root.connections])
        for param in dangling_params:
            tmeta = self.root._params_dict[param]
            if not tmeta.get('pass_by_obj') and tmeta['shape'] == ():
                msg = "Unconnected param '{}' is missing a shape or default value."
                raise RuntimeError(msg.format(param))

        # create VecWrappers for all systems in the tree.
        self.root._setup_vectors(param_owners, impl=self._impl)

        # Prepare Driver
        self.driver._setup()

        # get map of vars to VOI indices
        self._poi_indices, self._qoi_indices = self.driver._map_voi_indices()

        # Prepare Solvers
        for sub in self.root.subgroups(recurse=True, include_self=True):
            sub.nl_solver.setup(sub)
            sub.ln_solver.setup(sub)

        self._check_solvers()

        # Prep for case recording
        self._start_recorders()

        # check for any potential issues
        if check or force_check:
            return self.check_setup(out_stream)

        return {}

    def cleanup(self):
        """ Clean up resources prior to exit. """
        self.driver.cleanup()
        self.root.cleanup()

    def _check_solvers(self):
        """ Search over all solvers and raise errors for unsupported
        configurations. These include:

        Raise an exception if we detect a LinearGaussSeidel
        solver and that group has either cycles or uniterated states.

        Raise an exception if a Newton solver is found under any system that
        is set to complex step.
        """

        # all states that have some maxiter>1 linear solver above them in the tree
        iterated_states = set()
        group_states = []

        for group in self.root.subgroups(recurse=True, include_self=True):

            # Look for nl solvers that require derivs under Complex Step.
            opt = group.fd_options
            if opt['force_fd'] == True and opt['form'] == 'complex_step':

                # TODO: Support using complex step on a subsystem
                if group.name != '':
                    msg = "Complex step is currently not supported for groups"
                    msg += " other than root."
                    raise RuntimeError(msg)

                # Complex Step, so check for deriv requirement in subsolvers
                for sub in self.root.subgroups(recurse=True, include_self=True):
                    if hasattr(sub.nl_solver, 'ln_solver'):
                        msg = "The solver in '{}' requires derivatives. We "
                        msg += "currently do not support complex step around it."
                        raise RuntimeError(msg.format(sub.name))

            if isinstance(group.ln_solver, LinearGaussSeidel) and \
                                     group.ln_solver.options['maxiter'] == 1:
                # If group has a cycle and lings can't iterate, that's
                # an error.
                graph = group._get_sys_graph()
                strong = [sorted(s) for s in nx.strongly_connected_components(graph)
                          if len(s) > 1]
                if strong:
                    raise RuntimeError("Group '%s' has a LinearGaussSeidel "
                                   "solver with maxiter==1 but it contains "
                                   "cycles %s. To fix this error, change to "
                                   "a different linear solver, e.g. ScipyGMRES "
                                   "or PetscKSP, or increase maxiter (not "
                                   "recommended)."
                                   % (group.pathname, strong))

            states = [n for n,m in iteritems(group._unknowns_dict) if m.get('state')]
            if states:
                group_states.append((group, states))

                # this group has an iterative lin solver, so all states in it are ok
                if group.ln_solver.options['maxiter'] > 1:
                    iterated_states.update(states)
                else:
                    # see if any states are in comps that have their own
                    # solve_linear method
                    for s in states:
                        if s not in iterated_states:
                            cname = s.rsplit('.', 1)[0]
                            comp = self.root
                            for name in cname.split('.'):
                                comp = getattr(comp, name)
                            if not _needs_iteration(comp):
                                iterated_states.add(s)

        for group, states in group_states:
            uniterated_states = [s for s in states if s not in iterated_states]

            # It's an error if we find states that don't have some
            # iterative linear solver as a parent somewhere in the tree, or they
            # don't live in a Component that defines its own solve_linear method.

            if uniterated_states:
                raise RuntimeError("Group '%s' has a LinearGaussSeidel "
                               "solver with maxiter==1 but it contains "
                               "implicit states %s. To fix this error, "
                               "change to a different linear solver, e.g. "
                               "ScipyGMRES or PetscKSP, or increase maxiter "
                               "(not recommended)." %
                               (group.pathname, uniterated_states))

    def _check_dangling_params(self, out_stream=sys.stdout):
        """ Check for parameters that are not connected to a source/unknown.
        this includes ALL dangling params, both promoted and unpromoted.
        """
        to_prom_name = self.root._sysdata.to_prom_name

        dangling_params = sorted(set([
            to_prom_name[p] for p, m in iteritems(self.root._params_dict)
            if p not in self.root.connections
        ]))
        if dangling_params:
            print("\nThe following parameters have no associated unknowns:",
                  file=out_stream)
            for d in dangling_params:
                print(d, file=out_stream)

        return dangling_params

    def _check_mode(self, out_stream=sys.stdout):
        """ Adjoint vs Forward mode appropriateness """
        if self._calculated_mode != self.root._probdata.relevance.mode:
            print("\nSpecified derivative mode is '%s', but calculated mode is '%s'\n(based "
                  "on param size of %d and unknown size of %d)" % (self.root._probdata.relevance.mode,
                                                                   self._calculated_mode,
                                                                   self._p_length,
                                                                   self._u_length),
                  file=out_stream)

        return (self.root._probdata.relevance.mode, self._calculated_mode)

    def _list_unit_conversions(self, out_stream=sys.stdout):
        """ List all unit conversions being made (including only units on one
        side)"""
        if self._unit_diffs:
            tuples = sorted(iteritems(self._unit_diffs))
            print("\nUnit Conversions", file=out_stream)

            vec = self.root.unknowns
            pbos = [var for var in vec if vec.metadata(var).get('pass_by_obj') is True]

            for (src, tgt), (sunit, tunit) in tuples:
                if src in pbos:
                    pbo_str = ' (pass_by_obj)'
                else:
                    pbo_str = ''
                print("%s -> %s : %s -> %s%s" % (src, tgt, sunit, tunit, pbo_str),
                      file=out_stream)

            return tuples
        return []

    def _check_no_unknown_comps(self, out_stream=sys.stdout):
        """ Check for components without unknowns. """
        nocomps = sorted([c.pathname for c in self.root.components(recurse=True,
                                                                   local=True)
                          if len(c.unknowns) == 0])
        if nocomps:
            print("\nThe following components have no unknowns:", file=out_stream)
            for n in nocomps:
                print(n, file=out_stream)

        return nocomps

    def _check_no_recorders(self, out_stream=sys.stdout):
        """ Check for no case recorder. """
        recorders = []
        recorders.extend(self.driver.recorders)
        for grp in self.root.subgroups(recurse=True, local=True,
                                       include_self=True):
            recorders.extend(grp.nl_solver.recorders)
            recorders.extend(grp.ln_solver.recorders)

        if not recorders:
            print("\nNo recorders have been specified, so no data will be saved.",
                  file=out_stream)

        return recorders

    def _check_no_connect_comps(self, out_stream=sys.stdout):
        """ Check for unconnected components. """
        conn_comps = set([t.rsplit('.', 1)[0]
                          for t in self.root.connections])
        conn_comps.update([s.rsplit('.', 1)[0]
                           for s, i in itervalues(self.root.connections)])
        noconn_comps = sorted([c.pathname
                               for c in self.root.components(recurse=True, local=True)
                               if c.pathname not in conn_comps])
        if noconn_comps:
            print("\nThe following components have no connections:", file=out_stream)
            for comp in noconn_comps:
                print(comp, file=out_stream)

        return noconn_comps

    def _check_mpi(self, out_stream=sys.stdout):
        """ Some simple MPI checks. """
        if under_mpirun():
            parr = True
            # Indicate that there are no parallel systems if user is running under MPI
            if self.comm.rank == 0:
                for grp in self.root.subgroups(recurse=True, include_self=True):
                    if (isinstance(grp, ParallelGroup) or
                        isinstance(grp, ParallelFDGroup)):
                        break
                else:
                    parr = False
                    print("\nRunning under MPI, but no ParallelGroups or ParallelFDGroups were found.",
                          file=out_stream)

                mincpu, maxcpu = self.root.get_req_procs()
                if maxcpu is not None and self.comm.size > maxcpu:
                    print("\nmpirun was given %d MPI processes, but the problem can only use %d" %
                          (self.comm.size, maxcpu))

                return (self.comm.size, maxcpu, parr)
        # or any ParalleGroups found when not running under MPI
        else:
            pargrps = []
            for grp in self.root.subgroups(recurse=True, include_self=True):
                if isinstance(grp, ParallelGroup):
                    print("\nFound ParallelGroup '%s', but not running under MPI." %
                          grp.pathname, file=out_stream)
                    pargrps.append(grp.pathname)
            return sorted(pargrps)

    def _check_graph(self, out_stream=sys.stdout):
        """ Check for cycles in group w/o solver. """
        cycles = []
        ooo = []

        for grp in self.root.subgroups(recurse=True, include_self=True):
            graph = grp._get_sys_graph()

            strong = [s for s in nx.strongly_connected_components(graph)
                      if len(s) > 1]

            if strong:
                relstrong = []
                for slist in strong:
                    relstrong.append([])
                    for s in slist:
                        relstrong[-1].append(nearest_child(grp.pathname, s))
                        # sort the cycle systems in execution order
                        subs = [s for s in grp._subsystems]
                        tups = sorted([(subs.index(s),s) for s in relstrong[-1]])
                        relstrong[-1] = [t[1] for t in tups]
                print("Group '%s' has the following cycles: %s" %
                          (grp.pathname, relstrong), file=out_stream)
                cycles.append(relstrong)

            # Components/Systems/Groups are not in the right execution order
            graph, _ = grp._break_cycles(grp.list_order(), graph)

            visited = set()
            out_of_order = {}
            for sub in itervalues(grp._subsystems):
                visited.add(sub.pathname)
                for u, v in nx.dfs_edges(graph, sub.pathname):
                    if v in visited:
                        out_of_order.setdefault(nearest_child(grp.pathname, v),
                                                set()).add(sub.pathname)

            if out_of_order:
                # scope ooo names to group
                for name in out_of_order:
                    out_of_order[name] = sorted([
                        nearest_child(grp.pathname, n) for n in out_of_order[name]
                    ])
                print("Group '%s' has the following out-of-order subsystems:" %
                      grp.pathname, file=out_stream)
                for n, subs in iteritems(out_of_order):
                    print("   %s should run after %s" % (n, subs), file=out_stream)
                ooo.append((grp.pathname, list(iteritems(out_of_order))))
                print("Auto ordering would be: %s" % grp.list_auto_order()[0],
                      file=out_stream)

        return (cycles, sorted(ooo))

    def _check_gmres_under_mpi(self, out_stream=sys.stdout):
        """ warn when using ScipyGMRES solver under MPI.
        """
        if under_mpirun():
            has_parallel = False
            for s in self.root.subgroups(recurse=True, include_self=True):
                if isinstance(s, ParallelGroup):
                    has_parallel = True
                    break

            if has_parallel and isinstance(self.root.ln_solver, ScipyGMRES):
                print("\nScipyGMRES is being used under MPI. Problems can arise "
                      "if a variable of interest (param/objective/constraint) "
                      "does not exist in all MPI processes.", file=out_stream)

    def _check_ubcs(self, out_stream=sys.stdout):
        ubcs = self._get_ubc_vars(self.root.connections)
        if ubcs:
            print("\nThe following params are connected to unknowns that are "
                  "updated out of order, so their initial values may contain "
                  "uninitialized unknown values: %s" % ubcs, file=out_stream)
        return ubcs

    def _check_unmarked_pbos(self, out_stream=sys.stdout):
        pbos = []
        for comp in self.root.components(recurse=True, include_self=True):
            if comp._pbo_warns:
                pbos.append((comp.pathname, comp._pbo_warns))

        if pbos:
            print("\nThe following variables are not differentiable but were "
                  "not labeled by the user as pass_by_obj:", file=out_stream)
            for cname, pbo_warns in sorted(pbos, key=lambda x: x[0]):
                for vname, val in pbo_warns:
                    print("%s: type %s" % ('.'.join((cname, vname)),
                          type(val).__name__), file=out_stream)

        return pbos

    def _check_relevant_pbos(self, out_stream=sys.stdout):
        """ Warn if any pass_by_object variables are in any relevant set if
        top driver requires derivatives."""

        # Only warn if we are taking gradients across model with a pbo
        # variable.
        if self.driver.__class__ is Driver or \
           self.driver.supports['gradients'] is False or \
           self.root.fd_options['force_fd'] is True:
            return []

        vec = self.root.unknowns
        pbos = [var for var in vec if vec.metadata(var).get('pass_by_obj') is True]

        rels = set()
        for key, rel in iteritems(self._probdata.relevance.relevant):
            rels.update(rel)

        rel_pbos = rels.intersection(pbos)
        if rel_pbos:
            print("\nThe following relevant connections are marked as pass_by_obj:",
                  file=out_stream)
            for src in rel_pbos:
                val = vec[src]

                # Find target(s) and print whole relevant connection
                for tgt, src_tuple in iteritems(self.root.connections):
                    if src_tuple[0] == src and tgt in rels:
                        print("%s -> %s: type %s" % (src, tgt, type(val).__name__),
                              file=out_stream)

            print("\nYour driver requires a gradient across a model with pass_by_obj "
                  "connections. We strongly recommend either setting the root "
                  "fd_options 'force_fd' to True, or isolating the pass_by_obj "
                  "connection into a Group and setting its fd_options 'force_fd' "
                  "to True.",
                  file=out_stream)

        return list(rel_pbos)

    def check_setup(self, out_stream=sys.stdout):
        """Write a report to the given stream indicating any potential problems
        found with the current configuration of this ``Problem``.

        Args
        ----
        out_stream : a file-like object, optional
            Stream where report will be written.
        """
        print("##############################################", file=out_stream)
        print("Setup: Checking for potential issues...", file=out_stream)

        results = {}  # dict of results for easier testing
        results['unit_diffs'] = self._list_unit_conversions(out_stream)
        results['recorders'] = self._check_no_recorders(out_stream)
        results['mpi'] = self._check_mpi(out_stream)
        results['dangling_params'] = self._check_dangling_params(out_stream)
        results['mode'] = self._check_mode(out_stream)
        results['no_unknown_comps'] = self._check_no_unknown_comps(out_stream)
        results['no_connect_comps'] = self._check_no_connect_comps(out_stream)
        results['cycles'], results['out_of_order'] = self._check_graph(out_stream)
        results['ubcs'] = self._check_ubcs(out_stream)
        results['solver_issues'] = self._check_gmres_under_mpi(out_stream)
        results['unmarked_pbos'] = self._check_unmarked_pbos(out_stream)
        results['relevant_pbos'] = self._check_relevant_pbos(out_stream)
        results['layout'] = self._check_layout(out_stream)

        # TODO: Incomplete optimization driver configuration
        # TODO: Parallelizability for users running serial models
        # TODO: io state of recorder-specific files?

        # loop over subsystems and let them add any specific checks to the stream
        for s in self.root.subsystems(recurse=True, local=True, include_self=True):
            stream = cStringIO()
            s.check_setup(out_stream=stream)
            content = stream.getvalue()
            if content:
                print("%s:\n%s\n" % (s.pathname, content), file=out_stream)
                results["@%s" % s.pathname] = content

        print("\nSetup: Check complete.", file=out_stream)
        print("##############################################\n", file=out_stream)

        return results

    def run(self):
        """ Runs the Driver in self.driver. """
        if self.root.is_active():
            self.driver.run(self)

        # if we're running under MPI, ensure that all of the processes
        # are finished in order to ensure that scripting code outside of
        # Problem doesn't attempt to access variables or files that have
        # not finished updating.  This can happen with FileRef vars and
        # potentially other pass_by_obj variables.
        if MPI:
            self.comm.barrier()

    def _mode(self, mode, indep_list, unknown_list):
        """ Determine the mode based on precedence. The mode in `mode` is
        first. If that is 'auto', then the mode in root.ln_options takes
        precedence. If that is 'auto', then mode is determined by the width
        of the independent variable and quantity space."""

        self._p_length = 0
        self._u_length = 0
        uset = set()
        for unames in unknown_list:
            if isinstance(unames, tuple):
                uset.update(unames)
            else:
                uset.add(unames)
        pset = set()
        for pnames in indep_list:
            if isinstance(pnames, tuple):
                pset.update(pnames)
            else:
                pset.add(pnames)

        to_prom_name = self.root._sysdata.to_prom_name

        for path, meta in chain(iteritems(self.root._unknowns_dict),
                                iteritems(self.root._params_dict)):
            prom_name = to_prom_name[path]
            if prom_name in uset:
                self._u_length += meta['size']
                uset.remove(prom_name)
            if prom_name in pset:
                self._p_length += meta['size']
                pset.remove(prom_name)

        if uset:
            raise RuntimeError("Can't determine size of unknowns %s." % list(uset))
        if pset:
            raise RuntimeError("Can't determine size of params %s." % list(pset))

        # Choose mode based on size
        if self._p_length > self._u_length:
            self._calculated_mode = 'rev'
        else:
            self._calculated_mode = 'fwd'

        if mode == 'auto':
            mode = self.root.ln_solver.options['mode']
            if mode == 'auto':
                mode = self._calculated_mode

        return mode

    def calc_gradient(self, indep_list, unknown_list, mode='auto',
                      return_format='array', dv_scale=None, cn_scale=None,
                      sparsity=None):
        """ Returns the gradient for the system that is slotted in
        self.root. This function is used by the optimizer but also can be
        used for testing derivatives on your model.

        Args
        ----
        indep_list : iter of strings
            Iterator of independent variable names that derivatives are to
            be calculated with respect to. All params must have a IndepVarComp.

        unknown_list : iter of strings
            Iterator of output or state names that derivatives are to
            be calculated for. All must be valid unknowns in OpenMDAO.

        mode : string, optional
            Deriviative direction, can be 'fwd', 'rev', 'fd', or 'auto'.
            Default is 'auto', which uses mode specified on the linear solver
            in root.

        return_format : string, optional
            Format for the derivatives, can be 'array' or 'dict'.

        dv_scale : dict, optional
            Dictionary of driver-defined scale factors on the design variables.

        cn_scale : dict, optional
            Dictionary of driver-defined scale factors on the constraints.

        sparsity : dict, optional
            Dictionary that gives the relevant design variables for each
            constraint. This option is only supported in the `dict` return
            format.

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
        if mode == 'fd' or self.root.fd_options['force_fd']:
            return self._calc_gradient_fd(indep_list, unknown_list,
                                          return_format, dv_scale=dv_scale,
                                          cn_scale=cn_scale, sparsity=sparsity)
        else:
            return self._calc_gradient_ln_solver(indep_list, unknown_list,
                                                 return_format, mode,
                                                 dv_scale=dv_scale,
                                                 cn_scale=cn_scale,
                                                 sparsity=sparsity)

    def _calc_gradient_fd(self, indep_list, unknown_list, return_format,
                          dv_scale=None, cn_scale=None, sparsity=None):
        """ Returns the finite differenced gradient for the system that is slotted in
        self.root.

        Args
        ----
        indep_list : iter of strings
            Iterator of independent variable names that derivatives are to
            be calculated with respect to. All params must have a IndepVarComp.

        unknown_list : iter of strings
            Iterator of output or state names that derivatives are to
            be calculated for. All must be valid unknowns in OpenMDAO.

        return_format : string
            Format for the derivatives, can be 'array' or 'dict'.

        dv_scale : dict, optional
            Dictionary of driver-defined scale factors on the design variables.

        cn_scale : dict, optional
            Dictionary of driver-defined scale factors on the constraints.

        sparsity : dict, optional
            Dictionary that gives the relevant design variables for each
            constraint. This option is only supported in the `dict` return
            format.

        Returns
        -------
        ndarray or dict
            Jacobian of unknowns with respect to params.
        """
        root = self.root
        unknowns = root.unknowns
        params = root.params

        to_prom_name = root._sysdata.to_prom_name
        to_abs_pnames = root._sysdata.to_abs_pnames
        to_abs_uname = root._sysdata.to_abs_uname

        if dv_scale is None:
            dv_scale = {} # Order not guaranteed in python 3.
        if cn_scale is None:
            cn_scale = {} # Order not guaranteed in python 3.

        abs_params = []
        fd_unknowns = [var for var in unknown_list if var not in indep_list]
        pass_unknowns = [var for var in unknown_list if var in indep_list]
        for name in indep_list:

            if name in unknowns:
                name = to_abs_uname[name]

            for tgt, (src, idxs) in iteritems(root.connections):
                if name == src:
                    name = tgt
                    break

            abs_params.append(name)

        Jfd = root.fd_jacobian(params, unknowns, root.resids, total_derivs=True,
                               fd_params=abs_params, fd_unknowns=fd_unknowns,
                               pass_unknowns=pass_unknowns,
                               poi_indices=self._poi_indices,
                               qoi_indices=self._qoi_indices)

        def get_fd_ikey(ikey):
            # FD Input keys are a little funny....
            if isinstance(ikey, tuple):
                ikey = ikey[0]

            fd_ikey = ikey

            if fd_ikey not in params:
                # The user sometimes specifies the parameter output
                # name instead of its target because it is more
                # convenient
                for tgt, (src, idxs) in iteritems(root.connections):
                    if src == ikey:
                        fd_ikey = tgt
                        break

                # We need the absolute name, but the fd Jacobian
                # holds relative promoted inputs
                if fd_ikey not in params:
                    for key, meta in iteritems(params):
                        if to_prom_name[key] == fd_ikey:
                            fd_ikey = meta['pathname']
                            break

            return fd_ikey

        if return_format == 'dict':
            J = OrderedDict()
            for okey in unknown_list:
                J[okey] = OrderedDict()
                for j, ikey in enumerate(indep_list):

                    # Support sparsity
                    if sparsity is not None:
                        if ikey not in sparsity[okey]:
                            continue

                    abs_ikey = abs_params[j]
                    fd_ikey = get_fd_ikey(abs_ikey)

                    # Support for IndepVarComps that are buried in sub-Groups
                    if (okey, fd_ikey) not in Jfd:
                        fd_ikey = to_abs_pnames[fd_ikey][0]

                    J[okey][ikey] = Jfd[(okey, fd_ikey)]

                    # Driver scaling
                    if ikey in dv_scale:
                        J[okey][ikey] *= dv_scale[ikey]
                    if okey in cn_scale:
                        J[okey][ikey] *= cn_scale[okey]

        else:
            usize = 0
            psize = 0
            for u in unknown_list:
                if u in self._qoi_indices:
                    idx = self._qoi_indices[u]
                    usize += len(idx)
                else:
                    usize += self.root.unknowns.metadata(u)['size']
            for p in indep_list:
                if p in self._poi_indices:
                    idx = self._poi_indices[p]
                    psize += len(idx)
                else:
                    psize += self.root.unknowns.metadata(p)['size']
            J = np.zeros((usize, psize))

            ui = 0
            for u in unknown_list:
                pi = 0
                for j, p in enumerate(indep_list):
                    abs_ikey = abs_params[j]
                    fd_ikey = get_fd_ikey(abs_ikey)

                    # Support for IndepVarComps that are buried in sub-Groups
                    if (u, fd_ikey) not in Jfd:
                        fd_ikey = to_abs_pnames[fd_ikey][0]

                    pd = Jfd[u, fd_ikey]
                    rows, cols = pd.shape

                    for row in range(0, rows):
                        for col in range(0, cols):
                            J[ui+row][pi+col] = pd[row][col]
                            # Driver scaling
                            if p in dv_scale:
                                J[ui+row][pi+col] *= dv_scale[p]
                            if u in cn_scale:
                                J[ui+row][pi+col] *= cn_scale[u]
                    pi += cols
                ui += rows
        return J

    def _calc_gradient_ln_solver(self, indep_list, unknown_list, return_format, mode,
                                 dv_scale=None, cn_scale=None, sparsity=None):
        """ Returns the gradient for the system that is slotted in
        self.root. The gradient is calculated using root.ln_solver.

        Args
        ----
        indep_list : list of strings
            List of independent variable names that derivatives are to
            be calculated with respect to. All params must have a IndepVarComp.

        unknown_list : list of strings
            List of output or state names that derivatives are to
            be calculated for. All must be valid unknowns in OpenMDAO.

        return_format : string
            Format for the derivatives, can be 'array' or 'dict'.

        mode : string
            Deriviative direction, can be 'fwd', 'rev', 'fd', or 'auto'.
            Default is 'auto', which uses mode specified on the linear solver
            in root.

        dv_scale : dict, optional
            Dictionary of driver-defined scale factors on the design variables.

        cn_scale : dict, optional
            Dictionary of driver-defined scale factors on the constraints.

        sparsity : dict, optional
            Dictionary that gives the relevant design variables for each
            constraint. This option is only supported in the `dict` return
            format.

        Returns
        -------
        ndarray or dict
            Jacobian of unknowns with respect to params.
        """

        root = self.root
        relevance = root._probdata.relevance
        unknowns = root.unknowns
        unknowns_dict = root._unknowns_dict
        to_abs_uname = root._sysdata.to_abs_uname
        comm = root.comm
        iproc = comm.rank
        nproc = comm.size
        owned = root._owning_ranks

        if dv_scale is None:
            dv_scale = {} # Order not guaranteed in python 3.
        if cn_scale is None:
            cn_scale = {} # Order not guaranteed in python 3.

        # Respect choice of mode based on precedence.
        # Call arg > ln_solver option > auto-detect
        mode = self._mode(mode, indep_list, unknown_list)

        fwd = mode == 'fwd'

        # Prepare model for calculation
        root.clear_dparams()
        for names in root._probdata.relevance.vars_of_interest(mode):
            for name in names:
                if name in root.dumat:
                    root.dumat[name].vec[:] = 0.0
                    root.drmat[name].vec[:] = 0.0
        root.dumat[None].vec[:] = 0.0
        root.drmat[None].vec[:] = 0.0

        # Linearize Model
        root._sys_linearize(root.params, unknowns, root.resids)

        # Initialize Jacobian
        if return_format == 'dict':
            J = OrderedDict()
            for okeys in unknown_list:
                if isinstance(okeys, str):
                    okeys = (okeys,)
                for okey in okeys:
                    J[okey] = OrderedDict()
                    for ikeys in indep_list:
                        if isinstance(ikeys, str):
                            ikeys = (ikeys,)
                        for ikey in ikeys:

                            # Support sparsity
                            if sparsity is not None:
                                if ikey not in sparsity[okey]:
                                    continue

                            J[okey][ikey] = None
        else:
            usize = 0
            psize = 0
            Jslices = OrderedDict()
            for u in unknown_list:
                start = usize
                if u in self._qoi_indices:
                    idx = self._qoi_indices[u]
                    usize += len(idx)
                else:
                    usize += self.root.unknowns.metadata(u)['size']
                Jslices[u] = slice(start, usize)

            for p in indep_list:
                start = psize
                if p in self._poi_indices:
                    idx = self._poi_indices[p]
                    psize += len(idx)
                else:
                    psize += unknowns.metadata(p)['size']
                Jslices[p] = slice(start, psize)
            J = np.zeros((usize, psize))

        if fwd:
            input_list, output_list = indep_list, unknown_list
            poi_indices, qoi_indices = self._poi_indices, self._qoi_indices
            in_scale, un_scale = dv_scale, cn_scale
        else:
            input_list, output_list = unknown_list, indep_list
            qoi_indices, poi_indices = self._poi_indices, self._qoi_indices
            in_scale, un_scale = cn_scale, dv_scale

        # Process our inputs/outputs of interest for parallel groups
        all_vois = self.root._probdata.relevance.vars_of_interest(mode)

        input_set = set()
        for inp in input_list:
            if isinstance(inp, str):
                input_set.add(inp)
            else:
                input_set.update(inp)

        # Our variables of interest include all sets for which at least
        # one variable is requested.
        voi_sets = []
        for voi_set in all_vois:
            for voi in voi_set:
                if voi in input_set:
                    voi_sets.append(voi_set)
                    break

        # Add any variables that the user "forgot". TODO: This won't be
        # necessary when we have an API to automatically generate the
        # IOI and OOI.
        flat_voi = [item for sublist in all_vois for item in sublist]
        for items in input_list:
            if isinstance(items, str):
                items = (items,)
            for item in items:
                if item not in flat_voi:
                    # Put them in serial groups
                    voi_sets.append((item,))

        voi_srcs = {}

        # If Forward mode, solve linear system for each param
        # If Adjoint mode, solve linear system for each unknown
        for params in voi_sets:
            rhs = OrderedDict()
            voi_idxs = {}

            old_size = None

            # Allocate all of our Right Hand Sides for this parallel set.
            for voi in params:
                vkey = self._get_voi_key(voi, params)

                duvec = self.root.dumat[vkey]
                rhs[vkey] = np.zeros((len(duvec.vec), ))

                voi_srcs[vkey] = voi
                if voi in duvec:
                    in_idxs = duvec._get_local_idxs(voi, poi_indices)
                else:
                    in_idxs = []

                if len(in_idxs) == 0:
                    in_idxs = np.arange(0, unknowns_dict[to_abs_uname[voi]]['size'], dtype=int)

                if old_size is None:
                    old_size = len(in_idxs)
                elif old_size != len(in_idxs):
                    raise RuntimeError("Indices within the same VOI group must be the same size, but"
                                       " in the group %s, %d != %d" % (params, old_size, len(in_idxs)))
                voi_idxs[vkey] = in_idxs

            # at this point, we know that for all vars in the current
            # group of interest, the number of indices is the same. We loop
            # over the *size* of the indices and use the loop index to look
            # up the actual indices for the current members of the group
            # of interest.
            for i in range(len(in_idxs)):
                for voi in params:
                    vkey = self._get_voi_key(voi, params)
                    rhs[vkey][:] = 0.0
                    # only set a 1.0 in the entry if that var is 'owned' by this rank
                    if self.root._owning_ranks[voi_srcs[vkey]] == iproc:
                        rhs[vkey][voi_idxs[vkey][i]] = 1.0

                # Solve the linear system
                dx_mat = root.ln_solver.solve(rhs, root, mode)

                for param, dx in iteritems(dx_mat):
                    vkey = self._get_voi_key(param, params)
                    if param is None:
                        param = params[0]

                    for item in output_list:

                        # Support sparsity
                        if sparsity is not None:
                            if fwd and param not in sparsity[item]:
                                continue
                            elif not fwd and item not in sparsity[param]:
                                continue

                        if relevance.is_relevant(vkey, item):
                            if fwd or owned[item] == iproc:
                                out_idxs = self.root.dumat[vkey]._get_local_idxs(item,
                                                                                 qoi_indices,
                                                                                 get_slice=True)
                                dxval = dx[out_idxs]
                                if dxval.size == 0:
                                    dxval = None
                            else:
                                dxval = None
                            if nproc > 1:
                                # TODO: make this use Bcast for efficiency
                                if trace:
                                    debug("calc_gradient_ln_solver dxval bcast. dxval=%s, root=%s"%
                                            (dxval, owned[item]))
                                    debug("input_list: %s, output_list: %s" % (input_list, output_list))
                                dxval = comm.bcast(dxval, root=owned[item])
                                if trace:
                                    debug("dxval bcast DONE")
                        else:  # irrelevant variable.  just give'em zeros
                            if item in qoi_indices:
                                zsize = len(qoi_indices[item])
                            else:
                                zsize = unknowns.metadata(item)['size']
                            dxval = np.zeros(zsize)

                        if dxval is not None:
                            nk = len(dxval)

                            if return_format == 'dict':
                                if fwd:
                                    if J[item][param] is None:
                                        J[item][param] = np.zeros((nk, len(in_idxs)))
                                    J[item][param][:, i] = dxval

                                    # Driver scaling
                                    if param in in_scale:
                                        J[item][param][:, i] *= in_scale[param]
                                    if item in un_scale:
                                        J[item][param][:, i] *= un_scale[item]
                                else:
                                    if J[param][item] is None:
                                        J[param][item] = np.zeros((len(in_idxs), nk))
                                    J[param][item][i, :] = dxval

                                    # Driver scaling
                                    if param in in_scale:
                                        J[param][item][i, :] *= in_scale[param]
                                    if item in un_scale:
                                        J[param][item][i, :] *= un_scale[item]

                            else:
                                if fwd:
                                    J[Jslices[item], Jslices[param].start+i] = dxval

                                    # Driver scaling
                                    if param in in_scale:
                                        J[Jslices[item], Jslices[param].start+i] *= in_scale[param]
                                    if item in un_scale:
                                        J[Jslices[item], Jslices[param].start+i] *= un_scale[item]

                                else:
                                    J[Jslices[param].start+i, Jslices[item]] = dxval

                                    # Driver scaling
                                    if param in in_scale:
                                        J[Jslices[param].start+i, Jslices[item]] *= in_scale[param]
                                    if item in un_scale:
                                        J[Jslices[param].start+i, Jslices[item]] *= un_scale[item]

        # Clean up after ourselves
        root.clear_dparams()

        return J

    def _get_voi_key(self, voi, grp):
        """Return the voi name, which allows for parallel derivative calculations
        (currently only works with LinearGaussSeidel), or None for those
        solvers that can only do a single linear solve at a time.
        """
        if (voi in self._driver_vois and
                isinstance(self.root.ln_solver, LinearGaussSeidel)):
            if (len(grp) > 1 or
                    self.root.ln_solver.options['single_voi_relevance_reduction']):
                return voi

        return None

    def check_partial_derivatives(self, out_stream=sys.stdout):
        """ Checks partial derivatives comprehensively for all components in
        your model.

        Args
        ----

        out_stream : file_like
            Where to send human readable output. Default is sys.stdout. Set to
            None to suppress.

        Returns
        -------
        Dict of Dicts of Dicts

        First key is the component name;
        2nd key is the (output, input) tuple of strings;
        third key is one of ['rel error', 'abs error', 'magnitude', 'J_fd', 'J_fwd', 'J_rev'];

        For 'rel error', 'abs error', 'magnitude' the value is:

            A tuple containing norms for forward - fd, adjoint - fd, forward - adjoint using the best case fdstep

        For 'J_fd', 'J_fwd', 'J_rev' the value is:

            A numpy array representing the computed Jacobian for the three different methods of computation

        """

        root = self.root

        # Linearize the model
        root._sys_linearize(root.params, root.unknowns, root.resids)

        if out_stream is not None:
            out_stream.write('Partial Derivatives Check\n\n')

        data = {}

        # Derivatives should just be checked without parallel adjoint for now.
        voi = None

        # Check derivative calculations for all comps at every level of the
        # system hierarchy.
        for comp in root.components(recurse=True):
            cname = comp.pathname

            # No need to check comps that don't have any derivs.
            if comp.fd_options['force_fd']:
                continue

            # IndepVarComps are just clutter too.
            if isinstance(comp, IndepVarComp):
                continue

            data[cname] = {}
            jac_fwd = OrderedDict()
            jac_rev = OrderedDict()
            jac_fd = OrderedDict()

            params = comp.params
            unknowns = comp.unknowns
            resids = comp.resids
            dparams = comp.dpmat[voi]
            dunknowns = comp.dumat[voi]
            dresids = comp.drmat[voi]
            states = comp.states

            # Skip if all of our inputs are unconnected.
            if len(dparams) == 0:
                continue

            # Work with all params that are not pbo.
            param_list = [item for item in dparams if not \
                          dparams.metadata(item).get('pass_by_obj')]
            param_list.extend(states)
            unkn_list = [item for item in dunknowns if not \
                         dunknowns.metadata(item).get('pass_by_obj')]

            if out_stream is not None:
                out_stream.write('-'*(len(cname)+15) + '\n')
                out_stream.write("Component: '%s'\n" % cname)
                out_stream.write('-'*(len(cname)+15) + '\n')

            # Create all our keys and allocate Jacs
            for p_name in param_list:

                dinputs = dunknowns if p_name in states else dparams
                p_size = np.size(dinputs[p_name])

                # Check dimensions of user-supplied Jacobian
                for u_name in unkn_list:

                    u_size = np.size(dunknowns[u_name])
                    if comp._jacobian_cache:

                        # We can perform some additional helpful checks.
                        if (u_name, p_name) in comp._jacobian_cache:

                            user = comp._jacobian_cache[(u_name, p_name)].shape

                            # User may use floats for scalar jacobians
                            if len(user) < 2:
                                user = (user[0], 1)

                            if user[0] != u_size or user[1] != p_size:
                                msg = "derivative in component '{}' of '{}' wrt '{}' is the wrong size. " + \
                                      "It should be {}, but got {}"
                                msg = msg.format(cname, u_name, p_name, (u_size, p_size), user)
                                raise ValueError(msg)

                    jac_fwd[(u_name, p_name)] = np.zeros((u_size, p_size))
                    jac_rev[(u_name, p_name)] = np.zeros((u_size, p_size))

            # Reverse derivatives first
            for u_name in unkn_list:
                u_size = np.size(dunknowns[u_name])

                # Send columns of identity
                for idx in range(u_size):
                    dresids.vec[:] = 0.0
                    root.clear_dparams()
                    dunknowns.vec[:] = 0.0

                    dresids._dat[u_name].val[idx] = 1.0
                    try:
                        comp.apply_linear(params, unknowns, dparams,
                                          dunknowns, dresids, 'rev')
                    finally:
                        dparams._apply_unit_derivatives()

                    for p_name in param_list:

                        dinputs = dunknowns if p_name in states else dparams
                        jac_rev[(u_name, p_name)][idx, :] = dinputs._dat[p_name].val

            # Forward derivatives second
            for p_name in param_list:

                dinputs = dunknowns if p_name in states else dparams
                p_size = np.size(dinputs[p_name])

                # Send columns of identity
                for idx in range(p_size):
                    dresids.vec[:] = 0.0
                    root.clear_dparams()
                    dunknowns.vec[:] = 0.0

                    dinputs._dat[p_name].val[idx] = 1.0
                    dparams._apply_unit_derivatives()
                    comp.apply_linear(params, unknowns, dparams,
                                      dunknowns, dresids, 'fwd')

                    for u_name, u_val in dresids.vec_val_iter():
                        jac_fwd[(u_name, p_name)][:, idx] = u_val

            # Finite Difference goes last
            dresids.vec[:] = 0.0
            root.clear_dparams()
            dunknowns.vec[:] = 0.0

            # Component can request to use complex step.
            if comp.fd_options['form'] == 'complex_step':
                fd_func = comp.complex_step_jacobian
            else:
                fd_func = comp.fd_jacobian

            jac_fd = fd_func(params, unknowns, resids)

            # Assemble and Return all metrics.
            _assemble_deriv_data(chain(dparams, states), resids, data[cname],
                                 jac_fwd, jac_rev, jac_fd, out_stream,
                                 c_name=cname)

        return data

    def check_total_derivatives(self, out_stream=sys.stdout):
        """ Checks total derivatives for problem defined at the top.

        Args
        ----

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
        root = self.root
        abs_indep_list = root._get_fd_params()
        param_srcs = [root.connections[p] for p in abs_indep_list \
                      if not root._params_dict[p].get('pass_by_obj')]
        unknown_list = root._get_fd_unknowns()
        unknown_list = [item for item in unknown_list \
                        if not root.unknowns.metadata(item).get('pass_by_obj')]

        # Convert absolute parameter names to promoted ones because it is
        # easier for the user to read.
        to_prom_name = self.root._sysdata.to_prom_name
        indep_list = [
            to_prom_name[p] for p, idxs in param_srcs
        ]

        # Calculate all our Total Derivatives
        Jfor = self.calc_gradient(indep_list, unknown_list, mode='fwd',
                                  return_format='dict')
        Jrev = self.calc_gradient(indep_list, unknown_list, mode='rev',
                                  return_format='dict')
        Jfd = self.calc_gradient(indep_list, unknown_list, mode='fd',
                                 return_format='dict')

        Jfor = _jac_to_flat_dict(Jfor)
        Jrev = _jac_to_flat_dict(Jrev)
        Jfd = _jac_to_flat_dict(Jfd)

        # Assemble and Return all metrics.
        data = {}
        _assemble_deriv_data(indep_list, unknown_list, data,
                             Jfor, Jrev, Jfd, out_stream)

        return data

    def _start_recorders(self):
        """ Prepare recorders for recording."""

        self.driver.recorders.startup(self.root)
        self.driver.recorders.record_metadata(self.root)

        for group in self.root.subgroups(recurse=True, include_self=True):
            for solver in (group.nl_solver, group.ln_solver):
                solver.recorders.startup(group)
                solver.recorders.record_metadata(self.root)

    def _check_for_parallel_derivs(self, params, unknowns, par_u, par_p):
        """ Checks a system hiearchy to make sure that no settings violate the
        assumptions needed for parallel dervivative calculation. Returns the
        mode that the system needs to use.
        """

        mode = self._mode('auto', params, unknowns)

        if mode == 'fwd':
            has_parallel_derivs = par_p
        else:
            has_parallel_derivs = par_u

        # the type of the root linear solver determines whether we solve
        # multiple RHS in parallel. Currently only LinearGaussSeidel can
        # support this.
        if (isinstance(self.root.ln_solver, LinearGaussSeidel) and
                self.root.ln_solver.options['single_voi_relevance_reduction']) \
                and has_parallel_derivs:

            for sub in self.root.subgroups(recurse=True):
                sub_mode = sub.ln_solver.options['mode']

                # Modes must match root for all subs
                if isinstance(sub.ln_solver, LinearGaussSeidel) and sub_mode not in (mode, 'auto'):
                    msg = "Group '{name}' has mode '{submode}' but the root group has mode '{rootmode}'." \
                          " Modes must match to use parallel derivative groups."
                    msg = msg.format(name=sub.name, submode=sub_mode, rootmode=mode)
                    raise RuntimeError(msg)

        return mode

    def _json_system_tree(self):
        """ Returns a json representation of the system hierarchy for the
        model in root.

        Returns
        -------
        json string
        """

        def _tree_dict(system):
            dct = OrderedDict()
            for s in system.subsystems(recurse=True):
                if isinstance(s, Group):
                    dct[s.name] = _tree_dict(s)
                else:
                    dct[s.name] = OrderedDict()
                    for vname, meta in iteritems(s.unknowns):
                        dct[s.name][vname] = m = meta.copy()
                        for mname in m:
                            if isinstance(m[mname], np.ndarray):
                                m[mname] = m[mname].tolist()
            return dct

        tree = OrderedDict()
        tree['root'] = _tree_dict(self.root)
        return json.dumps(tree)

    def _json_dependencies(self):
        """ Returns a json representation of the data dependency graph for
        the model in root..

        Returns
        -------
        A json string with a dependency matrix and a list of variable
        name labels.
        """
        return self.root._probdata.relevance.json_dependencies()

    def _setup_communicators(self):
        if self.comm is None:
            self.comm = self._impl.world_comm()

        # first determine how many procs that root can possibly use
        minproc, maxproc = self.driver.get_req_procs()
        if MPI:
            if not (maxproc is None or maxproc >= self.comm.size):
                # we have more procs than we can use, so just raise an
                # exception to encourage the user not to waste resources :)
                raise RuntimeError("This problem was given %d MPI processes, "
                                   "but it requires between %d and %d." %
                                   (self.comm.size, minproc, maxproc))
            elif self.comm.size < minproc:
                if maxproc is None:
                    maxproc = '(any)'
                raise RuntimeError("This problem was given %d MPI processes, "
                                   "but it requires between %s and %s." %
                                   (self.comm.size, minproc, maxproc))

        # TODO: once we have nested Problems, figure out proper Problem
        #       directory instead of just using getcwd().
        self.driver._setup_communicators(self.comm, os.getcwd())

    def _setup_units(self, connections, params_dict, unknowns_dict):
        """
        Calculate unit conversion factors for any connected
        variables having different units and store them in params_dict.

        Args
        ----
        connections : dict
            A dict of target variables (absolute name) mapped
            to the absolute name of their source variable and the
            relevant indices of that source if applicable.

        params_dict : OrderedDict
            A dict of parameter metadata for the whole `Problem`.

        unknowns_dict : OrderedDict
            A dict of unknowns metadata for the whole `Problem`.
        """

        to_prom_name = self.root._sysdata.to_prom_name

        for target, (source, idxs) in iteritems(connections):
            tmeta = params_dict[target]
            smeta = unknowns_dict[source]

            # units must be in both src and target to have a conversion
            if 'units' not in tmeta or 'units' not in smeta:
                # for later reporting in check_setup, keep track of any unit differences,
                # even for connections where one side has units and the other doesn't
                if 'units' in tmeta or 'units' in smeta:
                    self._unit_diffs[(source, target)] = (smeta.get('units'),
                                                          tmeta.get('units'))
                continue

            src_unit = smeta['units']
            tgt_unit = tmeta['units']

            try:
                scale, offset = get_conversion_tuple(src_unit, tgt_unit)
            except TypeError as err:
                if str(err) == "Incompatible units":
                    msg = "Unit '{s[units]}' in source '{sprom}' "\
                        "is incompatible with unit '{t[units]}' "\
                        "in target '{tprom}'.".format(s=smeta, t=tmeta,
                                                                 sprom=to_prom_name[source],
                                                                 tprom=to_prom_name[target])
                    raise TypeError(msg)
                else:
                    raise

            # If units are not equivalent, store unit conversion tuple
            # in the parameter metadata
            if scale != 1.0 or offset != 0.0:
                tmeta['unit_conv'] = (scale, offset)
                self._unit_diffs[(source, target)] = (smeta.get('units'),
                                                      tmeta.get('units'))

    def _get_implicit_connections(self):
        """
        Finds all matches between promoted names of parameters and unknowns
        in this `Problem`.  Any matches imply an implicit connection.
        All connections are expressed using absolute pathnames.

        Returns
        -------
        dict
            implicit connections in this `Problem`, represented as a mapping
            from the pathname of the target to the pathname of the source

        dict
            parameters in this `Problem` that are not implicitly connected,
            represented as a mapping from the promoted name of the parameter
            to it's pathname

        Raises
        ------
        RuntimeError
            if a a promoted variable name matches multiple unknowns
        """

        connections = OrderedDict()
        dangling = {} # Order not guaranteed in python 3.

        abs_unames = self.root._sysdata.to_abs_uname

        for prom_name, pabs_list in iteritems(self.root._sysdata.to_abs_pnames):
            if prom_name in abs_unames:  # param has a src in unknowns
                uprom = abs_unames[prom_name]
                for pabs in pabs_list:
                    connections[pabs] = ((uprom, None),)
            else:
                dangling.setdefault(prom_name, set()).update(pabs_list)

        return connections, dangling

    def print_all_convergence(self):
        """ Sets iprint to True for all solvers and subsolvers in the model."""

        root = self.root
        root.ln_solver.print_all_convergence()
        root.nl_solver.print_all_convergence()
        for grp in root.subgroups(recurse=True):
            grp.ln_solver.print_all_convergence()
            grp.nl_solver.print_all_convergence()

def _assign_parameters(connections):
    """Map absolute system names to the absolute names of the
    parameters they transfer data to.
    """
    param_owners = {} # Order not guaranteed in python 3.

    for par, (unk, idxs) in iteritems(connections):
        param_owners.setdefault(get_common_ancestor(par, unk), []).append(par)

    return param_owners


def _jac_to_flat_dict(jac):
    """ Converts a double `dict` jacobian to a flat `dict` Jacobian. Keys go
    from [out][in] to [out,in].

    Args
    ----

    jac : dict of dicts of ndarrays
        Jacobian that comes from calc_gradient when the return_type is 'dict'.

    Returns
    -------

    dict of ndarrays"""

    new_jac = OrderedDict()
    for key1, val1 in iteritems(jac):
        for key2, val2 in iteritems(val1):
            new_jac[(key1, key2)] = val2

    return new_jac


def _assemble_deriv_data(params, resids, cdata, jac_fwd, jac_rev, jac_fd,
                         out_stream, c_name='root'):
    """ Assembles dictionaries and prints output for check derivatives
    functions. This is used by both the partial and total derivative
    checks."""
    started = False

    for p_name in params:
        for u_name in resids:

            key = (u_name, p_name)

            # Ignore non-differentiables
            if key not in jac_fd:
                continue

            ldata = cdata[key] = {}

            Jsub_fd = jac_fd[key]

            Jsub_for = jac_fwd[key]
            Jsub_rev = jac_rev[key]

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

            if magfd == 0.0:
                rel1 = rel2 = rel3 = float('nan')
            else:
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
            out_stream.write("  %s: '%s' wrt '%s'\n\n" % (c_name, u_name, p_name))

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
            out_stream.write('\n')

def _needs_iteration(comp):
    """Return True if the given component needs an iterative
    solver to converge it.
    """
    if isinstance(comp, Component) and comp.is_active() and comp.states:
        for klass in comp.__class__.__mro__:
            if klass is Component:
                break
            if 'solve_linear' in klass.__dict__:
                # class has defined it's own solve_linear
                return  False
        return True
    return False

def _get_gmres_name():
    if MPI:
        return 'PetscKSP'
    else:
        return 'ScipyGMRES'
