""" OpenMDAO Problem class defintion."""

from __future__ import print_function

import sys
import json
from collections import OrderedDict
from itertools import chain
from six import iteritems
from six.moves import cStringIO
import networkx as nx

import numpy as np

from openmdao.components.paramcomp import ParamComp
from openmdao.core.system import System
from openmdao.core.group import Group
from openmdao.core.parallelgroup import ParallelGroup
from openmdao.core.basicimpl import BasicImpl
from openmdao.core.checks import check_connections
from openmdao.core.driver import Driver
from openmdao.core.mpiwrap import MPI, FakeComm, under_mpirun
from openmdao.core.relevance import Relevance
from openmdao.solvers.run_once import RunOnce
from openmdao.units.units import get_conversion_tuple
from openmdao.util.strutil import get_common_ancestor, name_relative_to


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
        elif name in self._dangling:
            for p in self._dangling[name]:
                parts = p.rsplit('.', 1)
                if len(parts) == 1:
                    return self.root.params[p]
                else:
                    grp = self.root._subsystem(parts[0])
                    return grp.params[parts[1]]
        else:
            raise KeyError("Variable '%s' not found." % name)

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

    def _setup_connections(self, params_dict, unknowns_dict):
        """Generate a mapping of absolute param pathname to the pathname
        of its unknown.
        """
        # Get all explicit connections (stated with absolute pathnames)
        connections = self.root._get_explicit_connections()

        # go through promoted names of all top level params/unknowns
        # if promoted name in unknowns matches promoted name in params
        # that indicates an implicit connection. All connections are returned
        # in absolute form.
        implicit_conns, prom_noconns = _get_implicit_connections(params_dict, unknowns_dict)

        # combine implicit and explicit connections
        for tgt, srcs in implicit_conns.items():
            connections.setdefault(tgt, []).extend(srcs)

        # resolve any input to input explicit connections
        input_sets = {}
        for tgt, srcs in connections.items():
            for src in srcs:
                if src in params_dict:
                    input_sets.setdefault(src, set()).update((tgt, src))
                    input_sets.setdefault(tgt, set()).update((tgt, src))

        # find any promoted but not connected inputs
        for p, meta in params_dict.items():
            prom = meta['promoted_name']
            if prom in prom_noconns:
                input_sets.setdefault(meta['pathname'], set()).update(prom_noconns[prom])

        for tgt, srcs in list(connections.items()):
            if tgt in input_sets:
                for s in srcs:
                    if s in unknowns_dict:
                        for t in input_sets[tgt]:
                            if s not in connections.get(t, ()):
                                connections.setdefault(t, []).append(s)

        newconns = {}
        for tgt, srcs in connections.items():
            unknown_srcs = [s for s in srcs if s in unknowns_dict]
            if len(unknown_srcs) > 1:
                raise RuntimeError("Target '%s' is connected to multiple unknowns: %s" %
                                   (tgt, unknown_srcs))

            if unknown_srcs:
                newconns[tgt] = unknown_srcs[0]

        connections = newconns

        self._dangling = {}
        prom_unknowns = {m['promoted_name'] for m in unknowns_dict.values()}
        for p, meta in params_dict.items():
            if meta['pathname'] not in connections:
                if meta['promoted_name'] not in prom_unknowns and meta['pathname'] in input_sets:
                    self._dangling[meta['promoted_name']] = input_sets[meta['pathname']]
                else:
                    self._dangling[meta['promoted_name']] = set([meta['pathname']])


        # perform additional checks on connections (e.g. for compatible types and shapes)
        check_connections(connections, params_dict, unknowns_dict)

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
        # if we modify the system tree, we'll need to call _setup_variables
        # and _setup_connections again
        tree_changed = False

        # call _setup_variables again if we change metadata
        meta_changed = False

        # Give every system an absolute pathname
        self.root._setup_paths(self.pathname)

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

        # TODO: handle any automatic grouping of systems here...

        # divide MPI communicators among subsystems
        if MPI:
            self.root._setup_communicators(MPI.COMM_WORLD)
        else:
            self.root._setup_communicators(FakeComm())

        # mark any variables in non-local Systems as 'remote'
        for comp in self.root.components(recurse=True):
            if not comp.is_active():
                meta_changed = True
                comp._set_vars_as_remote()

        # All changes to the system tree or variable metadata
        # must be complete at this point.

        if tree_changed:
            self.root._setup_paths(self.pathname)
            params_dict, unknowns_dict = self.root._setup_variables()
            connections = self._setup_connections(params_dict, unknowns_dict)
        elif meta_changed:
            params_dict, unknowns_dict = self.root._setup_variables()

        # calculate unit conversions and store in param metadata
        self._setup_units(connections, params_dict, unknowns_dict)

        # propagate top level promoted names, unit conversions,
        # and connections down to all subsystems
        for sub in self.root.subsystems(recurse=True, include_self=True):
            sub.connections = connections

            for meta in chain(sub._params_dict.values(),
                              sub._unknowns_dict.values()):
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

        # get map of vars to VOI indices
        self._poi_indices, self._qoi_indices = self.driver._map_voi_indices()

        pois = self.driver.params_of_interest()
        oois = self.driver.outputs_of_interest()

        # make sure pois and oois all refer to existing vars.
        # NOTE: all variables of interest (includeing POIs) must exist in the unknowns dict
        promoted_unknowns = [m['promoted_name'] for m in unknowns_dict.values()]

        for vnames in pois:
            for v in vnames:
                if v not in promoted_unknowns:
                    raise NameError("Can't find param of interest '%s'." % v)

        for vnames in oois:
            for v in vnames:
                if v not in promoted_unknowns:
                    raise NameError("Can't find quantity of interest '%s'." % v)

        mode = self._check_for_matrix_matrix(pois, oois)

        relevance = Relevance(params_dict, unknowns_dict, connections,
                              pois, oois, mode)

        # create VecWrappers for all systems in the tree.
        self.root._setup_vectors(param_owners, relevance=relevance,
                                 impl=self._impl)

        # Prep for case recording
        self._start_recorders()

        # Prepare Driver
        self.driver._setup(self.root)

        # check for any potential issues
        if check:
            return self.check_setup(out_stream)
        return {}

    def _check_dangling_params(self, out_stream=sys.stdout):
        """ Check for parameters that are not connected to a source/unknown.
        this includes ALL dangling params, both promoted and unpromoted.
        """
        dangling_params = sorted(set([m['promoted_name']
                             for p, m in self.root._params_dict.items()
                               if p not in self.root.connections]))
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
            tuples = sorted(self._unit_diffs.items())
            print("\nUnit Conversions")
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
        conn_comps = set([t.rsplit('.', 1)[0] for t in self.root.connections.keys()])
        conn_comps.update([s.rsplit('.', 1)[0] for s in self.root.connections.values()])
        noconn_comps = sorted([c.pathname for c in self.root.components(recurse=True, local=True)
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
            if MPI.COMM_WORLD.rank == 0:
                for grp in self.root.subgroups(recurse=True, include_self=True):
                    if isinstance(grp, ParallelGroup):
                        break
                else:
                    parr = False
                    print("\nRunning under MPI, but no ParallelGroups were found.",
                          file=out_stream)

                mincpu, maxcpu = self.root.get_req_procs()
                if maxcpu is not None and MPI.COMM_WORLD.size > maxcpu:
                    print("\nmpirun was given %d MPI processes, but the problem can only use %d" %
                          (MPI.COMM_WORLD.size, maxcpu))

                return (MPI.COMM_WORLD.size, maxcpu, parr)
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
        cgraph = self.root._relevance._cgraph
        for grp in self.root.subgroups(recurse=True, include_self=True):
            path = [] if not grp.pathname else grp.pathname.split('.')
            graph = cgraph.subgraph([n for n in cgraph if n.startswith(grp.pathname)])
            renames = {}
            for node in graph.nodes_iter():
                renames[node] = '.'.join(node.split('.')[:len(path)+1])
                if renames[node] == node:
                    del renames[node]

            # get the graph of direct children of current group
            nx.relabel_nodes(graph, renames, copy=False)

            # remove self loops created by renaming
            graph.remove_edges_from([(u, v) for u, v in graph.edges()
                                     if u == v])

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
            subnames = [s.pathname for s in grp.subsystems()]
            while strong:
                # break cycles to check order
                lsys = [s for s in subnames if s in strong[0]]
                for p in graph.predecessors(lsys[0]):
                    if p in lsys:
                        graph.remove_edge(p, lsys[0])
                strong = [s for s in nx.strongly_connected_components(graph)
                          if len(s) > 1]

            visited = set()
            out_of_order = {}
            for sub in grp.subsystems():
                visited.add(sub.pathname)
                for u, v in nx.dfs_edges(graph, sub.pathname):
                    if v in visited:
                        out_of_order.setdefault(name_relative_to(grp.pathname, v),
                                                set()).add(sub.pathname)

            if out_of_order:
                # scope ooo names to group
                for name in out_of_order:
                    out_of_order[name] = sorted([name_relative_to(grp.pathname, n)
                                                   for n in out_of_order[name]])
                print("Group '%s' has the following out-of-order subsystems:" %
                        grp.pathname, file=out_stream)
                for n, subs in out_of_order.items():
                    print("   %s should run after %s" % (n, subs), file=out_stream)
                ooo.append((grp.pathname, list(out_of_order.items())))

        return (cycles, sorted(ooo))

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

        results = {} # dict of results for easier testing
        results['dangling_params'] = self._check_dangling_params(out_stream)
        results['mode'] = self._check_mode(out_stream)
        results['unit_diffs'] = self._list_unit_conversions(out_stream)
        results['no_unknown_comps'] = self._check_no_unknown_comps(out_stream)
        results['no_connect_comps'] = self._check_no_connect_comps(out_stream)
        results['recorders'] = self._check_no_recorders(out_stream)
        results['mpi'] = self._check_mpi(out_stream)
        results['cycles'], results['out_of_order'] = self._check_graph(out_stream)

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

    def _mode(self, mode, param_list, unknown_list):
        """ Determine the mode based on precedence. The mode in `mode` is
        first. If that is 'auto', then the mode in root.ln_options takes
        precedence. If that is 'auto', then mode is determined by the width
        of the parameter and quantity space."""

        self._p_length = 0
        self._u_length = 0
        uset = set()
        for unames in unknown_list:
            if isinstance(unames, tuple):
                uset.update(unames)
            else:
                uset.add(unames)
        pset = set()
        for pnames in param_list:
            if isinstance(pnames, tuple):
                pset.update(pnames)
            else:
                pset.add(pnames)

        for meta in chain(self.root._unknowns_dict.values(),
                          self.root._params_dict.values()):
            prom_name = meta['promoted_name']
            if prom_name in uset:
                self._u_length += meta['size']
                uset.remove(prom_name)
            elif prom_name in pset:
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

    def calc_gradient(self, param_list, unknown_list, mode='auto',
                      return_format='array'):
        """ Returns the gradient for the system that is slotted in
        self.root. This function is used by the optimizer but also can be
        used for testing derivatives on your model.

        Args
        ----
        param_list : list of strings
            List of parameter name strings with respect to which derivatives
            are desired. All params must have a paramcomp.

        unknown_list : list of strings
            List of output or state name strings for derivatives to be
            calculated. All must be valid unknowns in OpenMDAO.

        mode : string, optional
            Deriviative direction, can be 'fwd', 'rev', 'fd', or 'auto'.
            Default is 'auto', which uses mode specified on the linear solver
            in root.

        return_format : string, optional
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

        Args
        ----
        param_list : list of strings
            List of parameter name strings with respect to which derivatives
            are desired. All params must have a paramcomp.

        unknown_list : list of strings
            List of output or state name strings for derivatives to be
            calculated. All must be valid unknowns in OpenMDAO.

        return_format : string
            Format for the derivatives, can be 'array' or 'dict'.

        Returns
        -------
        ndarray or dict
            Jacobian of unknowns with respect to params.
        """
        root = self.root
        unknowns = root.unknowns
        params = root.params

        Jfd = root.fd_jacobian(params, unknowns, root.resids, total_derivs=True)

        def get_fd_ikey(ikey):
            # FD Input keys are a little funny....
            if isinstance(ikey, tuple):
                ikey = ikey[0]

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
                        if meta['promoted_name'] == fd_ikey:
                            fd_ikey = meta['pathname']
                            break

            return fd_ikey

        if return_format == 'dict':
            J = {}
            for okey in unknown_list:
                J[okey] = {}
                for ikey in param_list:
                    fd_ikey = get_fd_ikey(ikey)
                    J[okey][ikey] = Jfd[(okey, fd_ikey)]
        else:
            usize = 0
            psize = 0
            for u in unknown_list:
                usize += self.root.unknowns.metadata(u)['size']
            for p in param_list:
                psize += self.root.unknowns.metadata(p)['size']
            J = np.zeros((usize, psize))

            ui = 0
            for u in unknown_list:
                pi = 0
                for p in param_list:
                    pd = Jfd[u, get_fd_ikey(p)]
                    rows, cols = pd.shape
                    for row in range(0, rows):
                        for col in range(0, cols):
                            J[ui+row][pi+col] = pd[row][col]
                    pi += cols
                ui += rows
        return J

    def _calc_gradient_ln_solver(self, param_list, unknown_list, return_format, mode):
        """ Returns the gradient for the system that is slotted in
        self.root. The gradient is calculated using root.ln_solver.

        Args
        ----
        param_list : list of strings
            List of parameter name strings with respect to which derivatives
            are desired. All params must have a paramcomp.

        unknown_list : list of strings
            List of output or state name strings for derivatives to be
            calculated. All must be valid unknowns in OpenMDAO.

        return_format : string
            Format for the derivatives, can be 'array' or 'dict'.

        mode : string
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
        iproc = root.comm.rank

        # Respect choice of mode based on precedence.
        # Call arg > ln_solver option > auto-detect
        mode = self._mode(mode, param_list, unknown_list)

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
        root.jacobian(params, unknowns, root.resids)

        # Initialize Jacobian
        if return_format == 'dict':
            J = {}
            for okeys in unknown_list:
                if isinstance(okeys, str):
                    okeys = (okeys,)
                for okey in okeys:
                    J[okey] = {}
                    for ikeys in param_list:
                        if isinstance(ikeys, str):
                            ikeys = (ikeys,)
                        for ikey in ikeys:
                            J[okey][ikey] = None
        else:
            usize = 0
            psize = 0
            for u in unknown_list:
                usize += self.root.unknowns.metadata(u)['size']
            for p in param_list:
                psize += self.root.unknowns.metadata(p)['size']
            J = np.zeros((usize, psize))

        if mode == 'fwd':
            input_list, output_list = param_list, unknown_list
            poi_indices, qoi_indices = self._poi_indices, self._qoi_indices
        else:
            input_list, output_list = unknown_list, param_list
            qoi_indices, poi_indices = self._poi_indices, self._qoi_indices

        # Process our inputs/outputs of interest for parallel groups
        all_vois = self.root._relevance.vars_of_interest(mode)

        input_set = set()
        for inp in input_list:
            if isinstance(inp, str):
                input_set.add(inp)
            else:
                input_set.update(inp)

        # Our variables of interest inlude all sets for which at least
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
        j = 0
        for params in voi_sets:
            rhs = {}
            voi_idxs = {}

            # Allocate all of our Right Hand Sides for this parallel set.
            for voi in params:
                vkey = voi if len(params) > 1 else None

                duvec = self.root.dumat[vkey]
                rhs[vkey] = np.zeros((len(duvec.vec), ))

                voi_srcs[vkey] = voi
                _, in_idxs = duvec.get_local_idxs(voi, poi_indices)
                voi_idxs[vkey] = in_idxs

            # TODO: check that all vois are the same size!!!

            jbase = j

            for i in range(len(in_idxs)):
                for voi in params:
                    vkey = voi if len(params) > 1 else None
                    # only set a 1.0 in the entry if that var is 'owned' by this rank
                    if self.root._owning_ranks[voi_srcs[vkey]] == iproc:
                        #print("setting %s to 1.0 in rank %d" % (voi, iproc))
                        rhs[vkey][voi_idxs[vkey][i]] = 1.0

                # Solve the linear system
                dx_mat = root.ln_solver.solve(rhs, root, mode)

                for voi in rhs:
                    rhs[voi][voi_idxs[voi][i]] = 0.0

                for param, dx in dx_mat.items():
                    if len(params) == 1:
                        vkey = None
                        param = params[0] # if voi is None, params has only one serial entry
                    else:
                        vkey = param

                    i = 0
                    for item in output_list:

                        _, out_idxs = self.root.dumat[vkey].get_local_idxs(item,
                                                                           qoi_indices)
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

        Args
        ----

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

        # Derivatives should just be checked without parallel adjoint for now.
        voi = None

        # Check derivative calculations for all comps at every level of the
        # system hierarchy.
        for comp in root.components(recurse=True):
            cname = comp.pathname

            # No need to check comps that don't have any derivs.
            if comp.fd_options['force_fd'] == True:
                continue

            # Paramcomps are just clutter too.
            if isinstance(comp, ParamComp):
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

            if out_stream is not None:
                out_stream.write('-'*(len(cname)+15) + '\n')
                out_stream.write("Component: '%s'\n" % cname)
                out_stream.write('-'*(len(cname)+15) + '\n')

            # Figure out implicit states for this comp
            states = [n for n, m in comp.unknowns.items() if m.get('state')]

            # Create all our keys and allocate Jacs
            for p_name in chain(dparams, states):

                dinputs = dunknowns if p_name in states else dparams
                p_size = np.size(dinputs[p_name])

                # Check dimensions of user-supplied Jacobian
                for u_name in unknowns:

                    u_size = np.size(dunknowns[u_name])
                    if comp._jacobian_cache:

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
                    try:
                        dparams._set_adjoint_mode(True)
                        comp.apply_linear(params, unknowns, dparams,
                                          dunknowns, dresids, 'rev')
                    finally:
                        dparams._set_adjoint_mode(False)

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
        abs_param_list = self.root._get_fd_params()
        param_srcs = [self.root.connections[p] for p in abs_param_list]
        unknown_list = self.root._get_fd_unknowns()

        # Convert absolute parameter names to promoted ones because it is
        # easier for the user to read.
        param_list = [self.root._unknowns_dict[p]['promoted_name'] for p in param_srcs]

        # Calculate all our Total Derivatives
        Jfor = self.calc_gradient(param_list, unknown_list, mode='fwd',
                                  return_format='dict')
        Jrev = self.calc_gradient(param_list, unknown_list, mode='rev',
                                  return_format='dict')
        Jfd = self.calc_gradient(param_list, unknown_list, mode='fd',
                                 return_format='dict')

        Jfor = _jac_to_flat_dict(Jfor)
        Jrev = _jac_to_flat_dict(Jrev)
        Jfd = _jac_to_flat_dict(Jfd)

        # Assemble and Return all metrics.
        data = {}
        _assemble_deriv_data(param_list, unknown_list, data,
                             Jfor, Jrev, Jfd, out_stream)

        return data

    def _start_recorders(self):
        """ Prepare recorders for recording."""
        for recorder in self.driver.recorders:
            recorder.startup(self.root)

        for group in self.root.subgroups(recurse=True, include_self=True):
            for solver in (group.nl_solver, group.ln_solver):
                for recorder in solver.recorders:
                    recorder.startup(group)

    def _check_for_matrix_matrix(self, params, unknowns):
        """ Checks a system hiearchy to make sure that no settings violate the
        assumptions needed for matrix-matrix calculation. Returns the mode that
        the system needs to use.
        """

        mode = self._mode('auto', params, unknowns)

        # TODO : Only Linear GS is supported on system

        for sub in self.root.subgroups(recurse=True):
            sub_mode = sub.ln_solver.options['mode']

            # Modes must match root for all subs
            if sub_mode not in (mode, 'auto'):
                msg = "Group '{name}' has mode '{submode}' but the root group has mode '{rootmode}'." \
                        " Modes must match to use Matrix Matrix."
                msg = msg.format(name=sub.name, submode=sub_mode, rootmode=mode)
                raise RuntimeError(msg)

            # TODO : Only Linear GS is supported on sub

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
                    for vname, meta in s.unknowns.items():
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


    def _setup_units(self, connections, params_dict, unknowns_dict):
        """
        Calculate unit conversion factors for any connected
        variables having different units and store them in params_dict.

        Args
        ----
        connections : dict
            A dict of target variables (absolute name) mapped
            to the absolute name of their source variable.

        params_dict : OrderedDict
            A dict of parameter metadata for the whole `Problem`.

        unknowns_dict : OrderedDict
            A dict of unknowns metadata for the whole `Problem`.
        """

        self._unit_diffs = {}
        for target, source in connections.items():
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


def _assign_parameters(connections):
    """Map absolute system names to the absolute names of the
    parameters they transfer data to.
    """
    param_owners = {}

    for par, unk in connections.items():
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
    for key1, val1 in jac.items():
        for key2, val2 in val1.items():
            new_jac[(key1, key2)] = val2

    return new_jac

def _assemble_deriv_data(params, resids, cdata, jac_fwd, jac_rev, jac_fd,
                         out_stream, skip_keys=(None, ), c_name='root'):
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
    Finds all matches between promoted names of parameters and
    unknowns.  Any matches imply an implicit connection.  All
    connections are expressed using absolute pathnames.

    This should only be called using params and unknowns from the
    top level `Group` in the system tree.

    Args
    ----
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

    # collect all absolute names that map to each promoted name
    abs_unknowns = {}
    for u in unknowns_dict.values():
        abs_unknowns.setdefault(u['promoted_name'], []).append(u['pathname'])

    abs_params = {}
    for p in params_dict.values():
        abs_params.setdefault(p['promoted_name'], []).append(p['pathname'])

    # check if any promoted names correspond to mutiple unknowns
    for name, lst in abs_unknowns.items():
        if len(lst) > 1:
            raise RuntimeError("Promoted name '%s' matches multiple unknowns: %s" %
                               (name, lst))

    prom_unknowns = [m['promoted_name'] for m in unknowns_dict.values()]

    connections = {}
    dangling = {}

    for prom_name, pabs_list in abs_params.items():
        uabs = abs_unknowns.get(prom_name, ())
        if uabs:  # param has a src in unknowns
            for pabs in pabs_list:
                connections[pabs] = uabs
        elif prom_name not in prom_unknowns:
            dangling.setdefault(prom_name, set()).update(pabs_list)

    return connections, dangling
