""" OpenMDAO Problem class defintion."""

from __future__ import print_function

import sys
import json
from itertools import chain
from six import iteritems, iterkeys, itervalues
from six.moves import cStringIO
import networkx as nx

import numpy as np

from openmdao.core.system import System
from openmdao.core.group import Group
from openmdao.core.component import Component
from openmdao.core.parallel_group import ParallelGroup
from openmdao.core.basic_impl import BasicImpl
from openmdao.core.checks import check_connections
from openmdao.core.driver import Driver
from openmdao.core.mpi_wrap import MPI, under_mpirun
from openmdao.core.relevance import Relevance

from openmdao.components.indep_var_comp import IndepVarComp
from openmdao.solvers.scipy_gmres import ScipyGMRES
from openmdao.solvers.ln_gauss_seidel import LinearGaussSeidel

from openmdao.units.units import get_conversion_tuple
from collections import OrderedDict
from openmdao.util.string_util import get_common_ancestor, name_relative_to


class _ProbData(object):
    """
    A container for Problem level data that is needed by subsystems.
    """
    def __init__(self):
        self.top_lin_gs = False


class Problem(System):
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

        if MPI:  # pragma: no cover
            from openmdao.core.petsc_impl import PetscImpl
            if impl != PetscImpl:
                raise ValueError("To run under MPI, the impl for a Problem must be PetscImpl.")

        if impl is None:
            self._impl = BasicImpl
        else:
            self._impl = impl

        self._comm = comm

        if driver is None:
            self.driver = Driver()
        else:
            self.driver = driver

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
        elif name in self.root._to_abs_pnames:
            for p in self.root._to_abs_pnames[name]:
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
        for p, meta in iteritems(params_dict):
            prom = meta['promoted_name']
            if prom in prom_noconns:
                for n in prom_noconns[prom]:
                    if p != n:
                        input_graph.add_edge(p, n, idxs=None)

        # for all connections where the target is an input, we want to connect
        # the 'unknown' sources for that target to all other inputs that are
        # connected to it
        to_add = []
        for tgt, srcs in iteritems(connections):
            if tgt in input_graph:
                connected_inputs = nx.node_connected_component(input_graph, tgt)
                for src, idxs in srcs:
                    if src in unknowns_dict:
                        for new_tgt in connected_inputs:
                            new_idxs = idxs
                            if compute_indices:
                                # follow path to new target, apply src_idxs along the way
                                path = nx.shortest_path(input_graph, tgt, new_tgt)
                                x = 0
                                while x < len(path)-1:
                                    next_idxs = input_graph[path[x]][path[x+1]]['idxs']
                                    if next_idxs is not None:
                                        if new_idxs is not None:
                                            new_idxs = np.array(new_idxs)[next_idxs]
                                        else:
                                            new_idxs = next_idxs
                                    x = x + 1
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
        newconns = {}
        for tgt, srcs in iteritems(connections):
            unknown_srcs = list(src for src in srcs if src[0] in unknowns_dict)
            if len(unknown_srcs) > 1:
                src_names = (name for name, idx in unknown_srcs)
                raise RuntimeError("Target '%s' is connected to multiple unknowns: %s" %
                                   (tgt, sorted(src_names)))
            if unknown_srcs:
                newconns[tgt] = unknown_srcs.pop()

        connections = newconns

        self._dangling = {}
        prom_unknowns = self.root._to_abs_unames
        for p, meta in iteritems(params_dict):
            if p not in connections:
                if meta['promoted_name'] not in prom_unknowns and p in input_graph:
                    self._dangling[meta['promoted_name']] = \
                        set(nx.node_connected_component(input_graph, p))
                else:
                    self._dangling[meta['promoted_name']] = set([meta['pathname']])

        return connections

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

        # Give every system an absolute pathname
        self.root._init_sys_data(self.pathname, self._probdata)

        # Returns the parameters and unknowns metadata dictionaries
        # for the root, which has an entry for each variable contained
        # in any child of root. Metadata for each variable will contain
        # the name of the variable relative to that system, the absolute
        # name of the variable, any user defined metadata, and the value,
        # size and/or shape if known. For example:
        #  unknowns_dict['G1.G2.foo.v1'] = {
        #     'promoted_name' :  'v1',
        #     'pathname' : 'G1.G2.foo.v1', # absolute path from the top
        #     'size' : 1,
        #     'shape' : 1,
        #     'val': 2.5,   # the initial value of that variable (if known)
        #  }
        params_dict, unknowns_dict = self.root._setup_variables()

        # collect all connections, both implicit and explicit from
        # anywhere in the tree, and put them in a dict where each key
        # is an absolute param name that maps to the absolute name of
        # a single source.
        connections = self._setup_connections(params_dict, unknowns_dict)

        # push connection src_indices down into the metadata for all target
        # params in all component level systems, then flag meta_changed so
        # it will get percolated back up to all groups in next setup_vars()
        src_idx_conns = [(tgt, src, idxs) for tgt, (src, idxs) in
                         iteritems(connections) if idxs is not None]
        if src_idx_conns:
            meta_changed = True
            for comp in self.root.components(recurse=True):
                for tgt, src, idxs in src_idx_conns:
                    # component dicts are keyed on var name, not pathname
                    path, name = tgt.rsplit('.', 1)
                    meta = comp._params_dict.get(name)
                    if meta and meta['pathname'] == tgt:
                        meta['src_indices'] = idxs

        # TODO: handle any automatic grouping of systems here...

        # divide MPI communicators among subsystems
        self._setup_communicators()

        # mark any variables in non-local Systems as 'remote'
        for comp in self.root.components(recurse=True):
            if not comp.is_active():
                meta_changed = True
                comp._set_vars_as_remote()

        if MPI:  # pragma: no cover
            for s in self.root.components(recurse=True):
                if s.setup_distrib_idxs is not Component.setup_distrib_idxs:
                    # component defines its own setup_distrib_idxs, so
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
        check_connections(connections, params_dict, unknowns_dict)

        # calculate unit conversions and store in param metadata
        self._setup_units(connections, params_dict, unknowns_dict)

        # propagate top level promoted names, unit conversions,
        # and connections down to all subsystems
        for sub in self.root.subsystems(recurse=True, include_self=True):
            sub.connections = connections

            for meta in chain(itervalues(sub._params_dict),
                              itervalues(sub._unknowns_dict)):
                path = meta['pathname']
                if path in unknowns_dict:
                    meta['top_promoted_name'] = unknowns_dict[path]['promoted_name']
                else:
                    meta['top_promoted_name'] = params_dict[path]['promoted_name']
                    unit_conv = params_dict[path].get('unit_conv')
                    if unit_conv:
                        meta['unit_conv'] = unit_conv

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
        promoted_unknowns = self.root._to_abs_unames

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

        relevance = Relevance(self.root, params_dict, unknowns_dict, connections,
                              pois, oois, mode)

        # pass relevance object down to all systems and perform
        # auto ordering
        for s in self.root.subsystems(recurse=True, include_self=True):
            s._relevance = relevance
            if isinstance(s, Group):
                # set auto order if order not already set
                if not s._order_set:
                    s.set_order(s.list_auto_order())

        # create VecWrappers for all systems in the tree.
        self.root._setup_vectors(param_owners, impl=self._impl)


        # Prepare Driver
        self.driver._setup(self.root)
        

        # get map of vars to VOI indices
        self._poi_indices, self._qoi_indices = self.driver._map_voi_indices()

        # Prepare Solvers
        for sub in self.root.subgroups(recurse=True, include_self=True):
            sub.nl_solver.setup(sub)
            sub.ln_solver.setup(sub)
        
        # Prep for case recording
        self._start_recorders()

        # check for any potential issues
        if check:
            return self.check_setup(out_stream)
        return {}

    def _check_dangling_params(self, out_stream=sys.stdout):
        """ Check for parameters that are not connected to a source/unknown.
        this includes ALL dangling params, both promoted and unpromoted.
        """
        dangling_params = sorted(set([
            m['promoted_name'] for p, m in iteritems(self.root._params_dict)
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
        if self._calculated_mode != self.root._relevance.mode:
            print("\nSpecified derivative mode is '%s', but calculated mode is '%s'\n(based "
                  "on param size of %d and unknown size of %d)" % (self.root._relevance.mode,
                                                                   self._calculated_mode,
                                                                   self._p_length,
                                                                   self._u_length),
                  file=out_stream)

        return (self.root._relevance.mode, self._calculated_mode)

    def _list_unit_conversions(self, out_stream=sys.stdout):
        """ List all unit conversions being made (including only units on one
        side)"""
        if self._unit_diffs:
            tuples = sorted(iteritems(self._unit_diffs))
            print("\nUnit Conversions", file=out_stream)
            for (src, tgt), (sunit, tunit) in tuples:
                print("%s -> %s : %s -> %s" % (src, tgt, sunit, tunit),
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
                          for t in iterkeys(self.root.connections)])
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
        if under_mpirun():  # pragma: no cover
            parr = True
            # Indicate that there are no parallel systems if user is running under MPI
            if self._comm.rank == 0:
                for grp in self.root.subgroups(recurse=True, include_self=True):
                    if isinstance(grp, ParallelGroup):
                        break
                else:
                    parr = False
                    print("\nRunning under MPI, but no ParallelGroups were found.",
                          file=out_stream)

                mincpu, maxcpu = self.root.get_req_procs()
                if maxcpu is not None and self._comm.size > maxcpu:
                    print("\nmpirun was given %d MPI processes, but the problem can only use %d" %
                          (self._comm.size, maxcpu))

                return (self._comm.size, maxcpu, parr)
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
                        relstrong[-1].append(name_relative_to(grp.pathname, s))
                        relstrong[-1] = sorted(relstrong[-1])
                print("Group '%s' has the following cycles: %s" %
                      (grp.pathname, relstrong), file=out_stream)
                cycles.append(relstrong)

            # Components/Systems/Groups are not in the right execution order
            graph = grp._break_cycles(grp.list_order(), graph)

            visited = set()
            out_of_order = {}
            for sub in itervalues(grp._subsystems):
                visited.add(sub.pathname)
                for u, v in nx.dfs_edges(graph, sub.pathname):
                    if v in visited:
                        out_of_order.setdefault(name_relative_to(grp.pathname, v),
                                                set()).add(sub.pathname)

            if out_of_order:
                # scope ooo names to group
                for name in out_of_order:
                    out_of_order[name] = sorted([
                        name_relative_to(grp.pathname, n) for n in out_of_order[name]
                    ])
                print("Group '%s' has the following out-of-order subsystems:" %
                      grp.pathname, file=out_stream)
                for n, subs in iteritems(out_of_order):
                    print("   %s should run after %s" % (n, subs), file=out_stream)
                ooo.append((grp.pathname, list(iteritems(out_of_order))))
                print("Auto ordering would be: %s" % grp.list_auto_order(),
                      file=out_stream)

        return (cycles, sorted(ooo))

    def _check_gmres_under_mpi(self, out_stream=sys.stdout):
        """ warn when using ScipyGMRES solver under MPI.
        """
        if under_mpirun():  # pragma: no cover
            has_parallel = False
            for s in self.root.subgroups(recurse=True, include_self=True):
                if isinstance(s, ParallelGroup):
                    has_parallel = True
                    break

            if has_parallel and isinstance(self.root.ln_solver, ScipyGMRES):
                print("ScipyGMRES is being used under MPI. Problems can arise "
                      "if a variable of interest (param/objective/constraint) "
                      "does not exist in all MPI processes.", file=out_stream)

    def check_setup(self, out_stream=sys.stdout):
        """Write a report to the given stream indicating any potential problems found
        with the current configuration of this ``Problem``.

        Args
        ----
        out_stream : a file-like object, optional
            Stream where report will be written.
        """
        print("##############################################", file=out_stream)
        print("Setup: Checking for potential issues...", file=out_stream)

        results = {}  # dict of results for easier testing
        results['dangling_params'] = self._check_dangling_params(out_stream)
        results['mode'] = self._check_mode(out_stream)
        results['unit_diffs'] = self._list_unit_conversions(out_stream)
        results['no_unknown_comps'] = self._check_no_unknown_comps(out_stream)
        results['no_connect_comps'] = self._check_no_connect_comps(out_stream)
        results['recorders'] = self._check_no_recorders(out_stream)
        results['mpi'] = self._check_mpi(out_stream)
        results['cycles'], results['out_of_order'] = self._check_graph(out_stream)
        results['solver_issues'] = self._check_gmres_under_mpi(out_stream)

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

        for meta in chain(itervalues(self.root._unknowns_dict),
                          itervalues(self.root._params_dict)):
            prom_name = meta['promoted_name']
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
        indep_list : list of strings
            List of independent variable names that derivatives are to
            be calculated with respect to. All params must have a IndepVarComp.

        unknown_list : list of strings
            List of output or state names that derivatives are to
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
        indep_list : list of strings
            List of independent variable names that derivatives are to
            be calculated with respect to. All params must have a IndepVarComp.

        unknown_list : list of strings
            List of output or state names that derivatives are to
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

        if dv_scale is None:
            dv_scale = {}
        if cn_scale is None:
            cn_scale = {}

        abs_params = []
        for name in indep_list:

            if name in unknowns:
                name = unknowns.metadata(name)['pathname']

            for tgt, (src, idxs) in iteritems(root.connections):
                if name == src:
                    name = tgt
                    break

            abs_params.append(name)

        Jfd = root.fd_jacobian(params, unknowns, root.resids, total_derivs=True,
                               fd_params=abs_params, fd_unknowns=unknown_list,
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
                    for key in params:
                        meta = params.metadata(key)
                        if meta['promoted_name'] == fd_ikey:
                            fd_ikey = meta['pathname']
                            break

            return fd_ikey

        if return_format == 'dict':
            J = {}
            for okey in unknown_list:
                J[okey] = {}
                for j, ikey in enumerate(indep_list):

                    # Support sparsity
                    if sparsity is not None:
                        if ikey not in sparsity[okey]:
                            continue

                    abs_ikey = abs_params[j]
                    fd_ikey = get_fd_ikey(abs_ikey)

                    # Support for IndepVarComps that are buried in sub-Groups
                    if (okey, fd_ikey) not in Jfd:
                        fd_ikey = root._to_abs_pnames[fd_ikey][0]

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
                        fd_ikey = root._to_abs_pnames[fd_ikey][0]

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
        relevance = root._relevance
        unknowns = root.unknowns
        unknowns_dict = root._unknowns_dict
        to_abs_unames = root._to_abs_unames
        comm = root.comm
        iproc = comm.rank
        nproc = comm.size
        owned = root._owning_ranks

        if dv_scale is None:
            dv_scale = {}
        if cn_scale is None:
            cn_scale = {}

        # Respect choice of mode based on precedence.
        # Call arg > ln_solver option > auto-detect
        mode = self._mode(mode, indep_list, unknown_list)

        fwd = mode == 'fwd'

        # Prepare model for calculation
        root.clear_dparams()
        for names in root._relevance.vars_of_interest(mode):
            for name in names:
                if name in root.dumat:
                    root.dumat[name].vec[:] = 0.0
                    root.drmat[name].vec[:] = 0.0
        root.dumat[None].vec[:] = 0.0
        root.drmat[None].vec[:] = 0.0

        # Linearize Model
        root._sys_jacobian(root.params, unknowns, root.resids)

        # Initialize Jacobian
        if return_format == 'dict':
            J = {}
            for okeys in unknown_list:
                if isinstance(okeys, str):
                    okeys = (okeys,)
                for okey in okeys:
                    J[okey] = {}
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
            Jslices = {}
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
        all_vois = self.root._relevance.vars_of_interest(mode)

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
                    in_idxs = np.arange(0, unknowns_dict[to_abs_unames[voi][0]]['size'], dtype=int)

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
                                dxval = comm.bcast(dxval, root=owned[item])
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
        root._sys_jacobian(root.params, root.unknowns, root.resids)

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
            jac_fwd = {}
            jac_rev = {}
            jac_fd = {}

            params = comp.params
            unknowns = comp.unknowns
            resids = comp.resids
            dparams = comp.dpmat[voi]
            dunknowns = comp.dumat[voi]
            dresids = comp.drmat[voi]

            # Skip if all of our inputs are unconnected.
            if len(dparams) == 0:
                continue

            if out_stream is not None:
                out_stream.write('-'*(len(cname)+15) + '\n')
                out_stream.write("Component: '%s'\n" % cname)
                out_stream.write('-'*(len(cname)+15) + '\n')

            # Figure out implicit states for this comp
            states = [n for n, m in iteritems(comp.unknowns) if m.get('state')]

            # Create all our keys and allocate Jacs
            for p_name in chain(dparams, states):

                dinputs = dunknowns if p_name in states else dparams
                p_size = np.size(dinputs[p_name])

                # Check dimensions of user-supplied Jacobian
                for u_name in unknowns:

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
            for u_name in dresids:
                u_size = np.size(dunknowns[u_name])

                # Send columns of identity
                for idx in range(u_size):
                    dresids.vec[:] = 0.0
                    root.clear_dparams()
                    dunknowns.vec[:] = 0.0

                    dresids.flat[u_name][idx] = 1.0
                    try:
                        comp.apply_linear(params, unknowns, dparams,
                                          dunknowns, dresids, 'rev')
                    finally:
                        dparams._apply_unit_derivatives()

                    for p_name in chain(dparams, states):

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
                    dparams._apply_unit_derivatives()
                    comp.apply_linear(params, unknowns, dparams,
                                      dunknowns, dresids, 'fwd')

                    for u_name, u_val in iteritems(dresids.flat):
                        jac_fwd[(u_name, p_name)][:, idx] = u_val

            # Finite Difference goes last
            dresids.vec[:] = 0.0
            root.clear_dparams()
            dunknowns.vec[:] = 0.0
            jac_fd = comp.fd_jacobian(params, unknowns, resids)

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
        abs_indep_list = self.root._get_fd_params()
        param_srcs = [self.root.connections[p] for p in abs_indep_list]
        unknown_list = self.root._get_fd_unknowns()

        # Convert absolute parameter names to promoted ones because it is
        # easier for the user to read.
        indep_list = [
            self.root._unknowns_dict[p]['promoted_name'] for p, idxs in param_srcs
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
        exclude = set([])

        self.driver.recorders.startup(self.root)
        self.driver.recorders.record_metadata(self.root, exclude=exclude)
        
        for group in self.root.subgroups(recurse=True, include_self=True):
            for solver in (group.nl_solver, group.ln_solver):
                solver.recorders.startup(group)
                solver.recorders.record_metadata(self.root, exclude=exclude)

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
        return self.root._relevance.json_dependencies()

    def _setup_communicators(self):
        if self._comm is None:
            self._comm = self._impl.world_comm()

        # first determine how many procs that root can possibly use
        minproc, maxproc = self.root.get_req_procs()
        if MPI:  # pragma: no cover
            if not (maxproc is None or maxproc >= self._comm.size):
                # we have more procs than we can use, so just raise an
                # exception to encourage the user not to waste resources :)
                raise RuntimeError("This problem was given %d MPI processes, "
                                   "but it requires between %d and %d." %
                                   (self._comm.size, minproc, maxproc))
            elif self._comm.size < minproc:
                if maxproc is None:
                    maxproc = '(any)'
                raise RuntimeError("This problem was given %d MPI processes, "
                                   "but it requires between %s and %s." %
                                   (self._comm.size, minproc, maxproc))

        self.root._setup_communicators(self._comm)

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

        self._unit_diffs = {}
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
                    msg = "Unit '{s[units]}' in source '{s[promoted_name]}' "\
                        "is incompatible with unit '{t[units]}' "\
                        "in target '{t[promoted_name]}'.".format(s=smeta, t=tmeta)
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

        # check if any promoted names correspond to mutiple unknowns
        for name, lst in iteritems(self.root._to_abs_unames):
            if len(lst) > 1:
                raise RuntimeError("Promoted name '%s' matches multiple unknowns: %s" %
                                   (name, lst))

        connections = {}
        dangling = {}

        for prom_name, pabs_list in iteritems(self.root._to_abs_pnames):
            if prom_name in self.root._to_abs_unames:  # param has a src in unknowns
                uprom = self.root._to_abs_unames[prom_name]
                for pabs in pabs_list:
                    uprom = set(uprom)
                    connections[pabs] = ((u, None) for u in uprom)
            else:
                dangling.setdefault(prom_name, set()).update(pabs_list)

        return connections, dangling


def _assign_parameters(connections):
    """Map absolute system names to the absolute names of the
    parameters they transfer data to.
    """
    param_owners = {}

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

    new_jac = {}
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
