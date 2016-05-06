""" OpenMDAO Problem class defintion."""

from __future__ import print_function

import os
import sys
import json
import warnings
import traceback
from collections import OrderedDict
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
from openmdao.core._checks import check_connections, _both_names
from openmdao.core.driver import Driver
from openmdao.core.mpi_wrap import MPI, under_mpirun, debug
from openmdao.core.relevance import Relevance

from openmdao.components.indep_var_comp import IndepVarComp
from openmdao.solvers.scipy_gmres import ScipyGMRES
from openmdao.solvers.ln_direct import DirectSolver
from openmdao.solvers.ln_gauss_seidel import LinearGaussSeidel

from openmdao.units.units import get_conversion_tuple
from openmdao.util.string_util import get_common_ancestor, nearest_child, name_relative_to
from openmdao.util.graph import plain_bfs
from openmdao.util.options import OptionsDictionary

force_check = os.environ.get('OPENMDAO_FORCE_CHECK_SETUP')
trace = os.environ.get('OPENMDAO_TRACE')


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

    def _setup_connections(self, params_dict, unknowns_dict):
        """Generate a mapping of absolute param pathname to the pathname
        of its unknown.

        Args
        ----

        params_dict : OrderedDict
            A dict of parameter metadata for the whole `Problem`.

        unknowns_dict : OrderedDict
            A dict of unknowns metadata for the whole `Problem`.

        """
        to_prom_name = self._probdata.to_prom_name

        # Get all explicit connections (stated with absolute pathnames)
        connections = self.root._get_explicit_connections()

        # get set of promoted params that are not *implicitly* connected
        # to anything, and add all implicit connections to the connections dict.
        prom_noconns = self._add_implicit_connections(connections)

        input_graph = nx.DiGraph()
        self._dangling = {}

        to_abs_pnames = self.root._sysdata.to_abs_pnames

        usrcs = set()

        # resolve any input to input connections
        for tgt, srcs in iteritems(connections):
            for src, idxs in srcs:
                input_graph.add_edge(src, tgt, idxs=idxs)
                if src in unknowns_dict:
                    usrcs.add(src)

        for prom, plist in iteritems(to_abs_pnames):
            input_graph.add_nodes_from(plist)
            if prom in prom_noconns:
                # include connections in the graph due to multiple params that
                # are promoted to the same name
                start = plist[0]
                input_graph.add_edges_from(((start,p) for p in plist[1:]),
                                           idxs=None)

        newconns = {}
        # loop over srcs that are unknowns
        for src in usrcs:
            newconns[src] = None
            src_idxs = {src:None}
            # walk depth first from each unknown src to each connected input,
            # updating src_indices if necessary
            for s, t in nx.dfs_edges(input_graph, src):
                tidxs = input_graph[s][t]['idxs']
                sidxs = src_idxs[s]

                if tidxs is None:
                    tidxs = sidxs
                elif sidxs is not None:
                    tidxs = np.array(sidxs)[tidxs]

                src_idxs[t] = tidxs

                if t in newconns:
                    newconns[t].append((src, tidxs))
                else:
                    newconns[t] = [(src, tidxs)]

        self._input_inputs = {}

        # now all nodes that are downstream of an unknown source have been
        # marked.  Anything left must be an input that is either dangling or
        # upstream of an input that does have an unknown source.
        for node in input_graph.nodes_iter():
            # only look at unmarked nodes that have 0 in_degree
            if node not in newconns and len(input_graph.pred[node]) == 0:
                nosrc = [node]
                # walk dfs from this input 'src' node until we hit a param
                # that has an unknown src
                for s, t in nx.dfs_edges(input_graph, node):
                    if t in newconns:  # found param with unknown src
                        src = newconns[t][0][0]
                        # connect the unknown src to all of the inputs connected
                        # to the current node that have no unknown src
                        for n in nosrc:
                            newconns[n] = [(src, None)]
                        break
                    else:
                        nosrc.append(t)
                else: # didn't find an unknown src, so must be dangling
                    set_nosrc = set(nosrc)
                    for n in nosrc:
                        self._dangling[to_prom_name[n]] = set_nosrc
                        self._input_inputs[n] = nosrc

        # connections must be in order across processes, so use an OrderedDict
        # and sort targets before adding them
        connections = OrderedDict()
        for tgt, srcs in sorted(newconns.items()):
            if srcs is not None:
                if len(srcs) > 1:
                    src_names = (n for n, idx in srcs)
                    self._setup_errors.append("Target '%s' is connected to "
                                              "multiple unknowns: %s" %
                                               (tgt, sorted(src_names)))
                connections[tgt] = srcs[0]

        return connections

    def _check_input_diffs(self, connections, params_dict, unknowns_dict):
        """For all sets of connected inputs, find any differences in units
        or initial value.
        """
        # loop over all dangling inputs
        for tgt, connected_inputs in iteritems(self._input_inputs):

            # figure out if any connected inputs have different initial
            # values or different units
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

                    diff_units.append((connected_inputs[i], u))

            if isinstance(vals[tgt_idx], np.ndarray):
                diff_vals = [(connected_inputs[i],v) for i,v in
                               enumerate(vals) if not
                                   (isinstance(v, np.ndarray) and
                                      v.shape==vals[tgt_idx].shape and
                                      (v==vals[tgt_idx]).all())]
            else:
                vtype = type(vals[tgt_idx])
                diff_vals = [(connected_inputs[i],v) for i,v in
                                 enumerate(vals) if vtype!=type(v) or
                                                      v!=vals[tgt_idx]]

            # if tgt has no unknown source, units MUST match, unless
            # one of them is None. At this point, connections contains
            # only unknown to input connections, so if the target is
            # in connections, it has an unknown source.

            if diff_units:
                filt = set([u for n,u in diff_units])
                if None in filt:
                    filt.remove(None)
                if filt:
                    proms = set([params_dict[item]['top_promoted_name'] \
                                 for item in connected_inputs])

                    # All params are promoted, so extra message for clarity.
                    if len(proms) == 1:
                        msg = "The following connected inputs are promoted to " + \
                            "'%s', but have different units" % proms.pop()
                    else:
                        msg = "The following connected inputs have no source and different " + \
                              "units"

                    msg += ": %s." % sorted([(tgt, params_dict[tgt].get('units'))] + \
                                            diff_units)
                    correct_src = params_dict[connected_inputs[0]]['top_promoted_name']
                    msg += " Connect '%s' to a source (such as an IndepVarComp)" % correct_src + \
                           " with defined units."

                    self._setup_errors.append(msg)
            if diff_vals:
                msg = ("The following sourceless connected inputs have "
                       "different initial values: "
                       "%s.  Connect one of them to the output of "
                       "an IndepVarComp to ensure that they have the "
                       "same initial value." %
                       (sorted([(tgt,params_dict[tgt]['val'])]+
                                         diff_vals)))
                self._setup_errors.append(msg)

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
                    self._setup_errors.append("The following parameters have the same "
                                  "promoted name, '%s', but different "
                                  "'step_size' values: %s" % (promname,
                                  sorted([(v,k) for k,v in step_sizes.items()])))

                if len(step_types) > 1:
                    self._setup_errors.append("The following parameters have the same "
                                  "promoted name, '%s', but different "
                                  "'step_type' values: %s" % (promname,
                                 sorted([(v,k) for k,v in step_types.items()])))

                if len(forms) > 1:
                    self._setup_errors.append("The following parameters have the same "
                                  "promoted name, '%s', but different 'form' "
                                  "values: %s" % (promname,
                                      sorted([(v,k) for k,v in forms.items()])))

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
        self._setup_errors = []

        # if we modify the system tree, we'll need to call _init_sys_data,
        # _setup_variables and _setup_connections again
        tree_changed = False

        # call _setup_variables again if we change metadata
        meta_changed = False

        self._probdata = _ProbData()
        if isinstance(self.root.ln_solver, LinearGaussSeidel):
            self._probdata.top_lin_gs = True

        self.driver.root = self.root
        self.driver.pathname = self.pathname + "." + self.driver.__class__.__name__
        self.driver.recorders.pathname = self.driver.pathname + ".recorders"

        # Give every system and solver an absolute pathname
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

        for tgt, (src, idxs) in iteritems(connections):
            tmeta = params_dict[tgt]
            if 'pass_by_obj' not in tmeta or not tmeta['pass_by_obj']:

                # Allow the user to omit the size of a parameter and pull
                # the size and shape from the connection source.
                if tmeta['shape'] == ():

                    smeta = unknowns_dict[src]

                    # Connected with src_indices specified
                    if idxs is not None:
                        size = len(idxs)
                        tmeta['shape'] = (size, )
                        tmeta['size'] = size
                        tmeta['val'] = smeta['val'][np.array(idxs)]

                    # Regular connection
                    else:
                        tmeta['shape'] = smeta['shape']
                        tmeta['size'] = smeta['size']
                        tmeta['val'] = smeta['val']

                # set src_indices into variable metadata
                if idxs is not None:
                    if isinstance(idxs, np.ndarray):
                        tmeta['src_indices'] = idxs
                    else:
                        tmeta['src_indices'] = np.array(idxs,
                                                  dtype=self._impl.idx_arr_type)

        # TODO: handle any automatic grouping of systems here...
        #       If we modify the system tree here, we'll have to call
        #       the full setup over again...

        if MPI:
            for s in self.root.components(recurse=True):
                # TODO: get rid of check for setup_distrib_idxs when we move to beta
                if hasattr(s, 'setup_distrib_idxs') or (
                         hasattr(s, 'setup_distrib') and (s.setup_distrib
                                                is not Component.setup_distrib)):
                    # component defines its own setup_distrib, so
                    # the metadata will change
                    meta_changed = True

        # All changes to the system tree or variable metadata
        # must be complete at this point.

        # if the system tree has changed, we have to redo the entire setup
        if tree_changed:
            return self.setup(check=check, out_stream=out_stream)
        elif meta_changed:
            params_dict, unknowns_dict = \
                self.root._setup_variables(compute_indices=True)

        # perform additional checks on connections
        # (e.g. for compatible types and shapes)
        self._setup_errors.extend(check_connections(connections, params_dict,
                                               unknowns_dict,
                                               self.root._sysdata.to_prom_name))

        # calculate unit conversions and store in param metadata
        self._setup_units(connections, params_dict, unknowns_dict)

        # propagate top level promoted names, unit conversions,
        # and connections down to all subsystems
        to_prom_name = self.root._sysdata.to_prom_name
        self._probdata.to_prom_name = to_prom_name

        for path, meta in iteritems(params_dict):
            # set top promoted name into var metadata
            meta['top_promoted_name'] = to_prom_name[path]

            # Check for dangling params that have no size or shape
            if path not in connections:
                if 'pass_by_obj' not in meta or not meta['pass_by_obj']:
                    if meta['shape'] == ():
                        self._setup_errors.append("Unconnected param '{}' is missing "
                                           "a shape or default value.".format(path))

        for path, meta in iteritems(unknowns_dict):
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

        # If we force_fd root and don't need derivatives in solvers, then we
        # don't have to allocate any deriv vectors.
        alloc_derivs = not self.root.fd_options['force_fd']
        for sub in self.root.subgroups(recurse=True, include_self=True):
            alloc_derivs = alloc_derivs or sub.nl_solver.supports['uses_derivatives']

        # create VecWrappers for all systems in the tree.
        self.root._setup_vectors(param_owners, impl=self._impl, alloc_derivs=alloc_derivs)

        # Prepare Driver
        self.driver._setup()

        # get map of vars to VOI indices
        self._poi_indices, self._qoi_indices = self.driver._map_voi_indices()

        # Prepare Solvers
        for sub in self.root.subgroups(recurse=True, include_self=True):
            sub.nl_solver.setup(sub)
            sub.ln_solver.setup(sub)

        self._check_solvers()

        # Prep for case recording and record metadata
        self._start_recorders()

        if self._setup_errors:
            stream = cStringIO()
            stream.write("\nThe following errors occurred during setup:\n")
            for err in self._setup_errors:
                stream.write("%s\n" % err)
            raise RuntimeError(stream.getvalue())

        # Lock any restricted options in the options dictionaries.
        OptionsDictionary.locked = True

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

        has_iter_solver = {}
        for group in self.root.subgroups(recurse=True, include_self=True):
            try:
                has_iter_solver[group.pathname] = (group.ln_solver.options['maxiter'] > 1)
            except KeyError:

                # DirectSolver handles coupled derivatives without iteration
                if isinstance(group.ln_solver, DirectSolver):
                    has_iter_solver[group.pathname] = (True)

            # Look for nl solvers that require derivs under Complex Step.
            opt = group.fd_options
            if opt['force_fd'] == True and opt['form'] == 'complex_step':

                # TODO: Support using complex step on a subsystem
                if group.name != '':
                    msg = "Complex step is currently not supported for groups"
                    msg += " other than root."
                    self._setup_errors.append(msg)

                # Complex Step, so check for deriv requirement in subsolvers
                for sub in self.root.subgroups(recurse=True, include_self=True):
                    if hasattr(sub.nl_solver, 'ln_solver'):
                        msg = "The solver in '{}' requires derivatives. We "
                        msg += "currently do not support complex step around it."
                        self._setup_errors.append(msg.format(sub.name))

            parts = group.pathname.split('.')
            for i in range(len(parts)):
                # if an ancestor solver iterates, we're ok
                if has_iter_solver['.'.join(parts[:i])]:
                    is_iterated_somewhere = True
                    break
            else:
                is_iterated_somewhere = False

            # if we're iterated at this level or somewhere above, then it's
            # ok if we have cycles or states.
            if is_iterated_somewhere:
                continue

            if isinstance(group.ln_solver, LinearGaussSeidel) and \
                                     group.ln_solver.options['maxiter'] == 1:
                # If group has a cycle and lings can't iterate, that's
                # an error if current lin solver or ancestor lin solver doesn't
                # iterate.
                graph = group._get_sys_graph()
                strong = [sorted(s) for s in nx.strongly_connected_components(graph)
                          if len(s) > 1]
                if strong:
                    self._setup_errors.append("Group '%s' has a LinearGaussSeidel "
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
                if isinstance(group.ln_solver, DirectSolver) or \
                   group.ln_solver.options['maxiter'] > 1:
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
                self._setup_errors.append("Group '%s' has a LinearGaussSeidel "
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
           self.root.fd_options['force_fd']:
            return []

        vec = self.root.unknowns
        pbos = [var for var in vec if vec.metadata(var).get('pass_by_obj')]

        rels = set()
        for key, rel in iteritems(self._probdata.relevance.relevant):
            rels.update(rel)

        rel_pbos = rels.intersection(pbos)
        if rel_pbos:
            rel_conns = []

            for src in rel_pbos:
                # Find target(s) and print whole relevant connection
                for tgt, src_tuple in iteritems(self.root.connections):
                    if src_tuple[0] == src and tgt in rels:
                        rel_conns.append((src, tgt))

            if rel_conns:
                print("\nThe following relevant connections are marked as pass_by_obj:",
                      file=out_stream)
                for src, tgt in rel_conns:
                    val = vec[src]
                    print("%s -> %s: type %s" % (src, tgt, type(val).__name__),
                          file=out_stream)
            else:
                print("\nThe following pass_by_obj variables are relevant to "
                      "a derivative calculation:", sorted(rel_pbos))

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

    def pre_run_check(self):
        """ Last chance for some checks. The checks that should be performed
        here are those that would generate a cryptic error message. We can
        raise a readable error for the user."""

        # New message if you forget to run setup first, or if you assign a
        # new ln or nl solver and forget to run setup.
        if not self.root.fd_options.locked:
            msg = "Before running the model, setup() must be called. If " + \
                 "the configuration has changed since it was called, then " + \
                 "setup must be called again before running the model."
            raise RuntimeError(msg)

    def run(self):
        """ Runs the Driver in self.driver. """
        self.pre_run_check()
        if self.root.is_active():
            self.driver.run(self)

            # if we're running under MPI, ensure that all of the processes
            # are finished in order to ensure that scripting code outside of
            # Problem doesn't attempt to access variables or files that have
            # not finished updating.  This can happen with FileRef vars and
            # potentially other pass_by_obj variables.
            if MPI:
                if trace: debug("waiting on problem run() comm.barrier")
                self.root.comm.barrier()
                if trace: debug("problem run() comm.barrier DONE")

    def run_once(self):
        """ Execute run_once in the driver, executing the model at the
        the current design point. """
        self.pre_run_check()
        root = self.root
        driver = self.driver
        if root.is_active():
            driver.run_once(self)

            # Make sure our residuals are up-to-date
            with root._dircontext:
                root.apply_nonlinear(root.params, root.unknowns, root.resids,
                                     metadata=driver.metadata)

            # if we're running under MPI, ensure that all of the processes
            # are finished in order to ensure that scripting code outside of
            # Problem doesn't attempt to access variables or files that have
            # not finished updating.  This can happen with FileRef vars and
            # potentially other pass_by_obj variables.
            if MPI:
                if trace: debug("waiting on problem run() comm.barrier")
                root.comm.barrier()
                if trace: debug("problem run() comm.barrier DONE")

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
        """ Returns the gradient for the system that is specified in
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

        with self.root._dircontext:
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
        """ Returns the finite differenced gradient for the system that is
        specified in self.root.

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
            dv_scale = {}
        if cn_scale is None:
            cn_scale = {}

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
        """ Returns the gradient for the system that is specified in
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
            dv_scale = {}
        if cn_scale is None:
            cn_scale = {}

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
                rhs[vkey] = np.empty((len(duvec.vec), ))

                voi_srcs[vkey] = voi
                if voi in duvec:
                    in_idxs = duvec._get_local_idxs(voi, poi_indices)
                else:
                    in_idxs = []

                if len(in_idxs) == 0:
                    if voi in poi_indices:
                        # offset doesn't matter since we only care about the size
                        in_idxs = duvec.to_idx_array(poi_indices[voi])
                    else:
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
                    # only set a -1.0 in the entry if that var is 'owned' by this rank
                    # Note, we solve a slightly modified version of the unified
                    # derivatives equations in OpenMDAO.
                    # (dR/du) * (du/dr) = -I
                    if self.root._owning_ranks[voi_srcs[vkey]] == iproc:
                        rhs[vkey][voi_idxs[vkey][i]] = -1.0

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
                                    debug("calc_gradient_ln_solver dxval bcast. dxval=%s, root=%s, param=%s, item=%s" %
                                            (dxval, owned[item], param, item))
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

    def check_partial_derivatives(self, out_stream=sys.stdout, comps=None,
                                  compact_print=False, abs_err_tol=1.0E-6,
                                  rel_err_tol=1.0E-6):
        """ Checks partial derivatives comprehensively for all components in
        your model.

        Args
        ----

        out_stream : file_like
            Where to send human readable output. Default is sys.stdout. Set to
            None to suppress.

        comps : None or list_like
            List of component names to check the partials of (all others will be skipped).
            Set to None (default) to run all components

        compact_print : bool
            Set to True to just print the essentials, one line per unknown-param
            pair.

        abs_err_tol : float
            Threshold value for absolute error.  Errors about this value will
            have a '*' displayed next to them in output, making them easy
            to search for. Default is 1.0E-6.

        rel_err_tol : float
            Threshold value for relative error.  Errors about this value will
            have a '*' displayed next to them in output, making them easy
            to search for. Note at times there may be a significant relative
            error due to a minor absolute error.  Default is 1.0E-6.


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

        if self.driver.iter_count < 1:
            out_stream.write('Executing model to populate unknowns...\n\n')
            self.run_once()

        # Linearize the model
        root._sys_linearize(root.params, root.unknowns, root.resids)

        if out_stream is not None:
            out_stream.write('Partial Derivatives Check\n\n')

        data = {}

        # Derivatives should just be checked without parallel adjoint for now.
        voi = None

        # Check derivative calculations for all comps at every level of the
        # system hierarchy.
        allcomps = root.components(recurse=True)
        if comps is None:
            comps = allcomps
        else:
            allcompnames = set([c.pathname for c in allcomps])
            requested = set(comps)
            diff = requested.difference(allcompnames)

            if diff:
                sorted_diff = list(diff)
                sorted_diff.sort()
                msg = "The following are not valid comp names: "
                msg += str(sorted_diff)
                raise RuntimeError(msg)

            comps = [root._subsystem(c_name) for c_name in comps]

        for comp in comps:
            cname = comp.pathname
            opt = comp.fd_options

            fwd_rev = True
            if opt['extra_check_partials_form']:
                f_d_2 = True
                fd_desc = opt['form']
                fd_desc2 = opt['extra_check_partials_form']
            else:
                f_d_2 = False
                fd_desc = None
                fd_desc2 = None

            # If we don't have analytic, then only continue if we are
            # comparing 2 different fds.
            if opt['force_fd']:
                if not f_d_2:
                    continue
                fwd_rev = False

            # IndepVarComps are just clutter too.
            if isinstance(comp, IndepVarComp):
                continue

            data[cname] = {}
            jac_fwd = OrderedDict()
            jac_rev = OrderedDict()
            jac_fd = OrderedDict()
            jac_fd2 = OrderedDict()

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

                # No need to pre-allocate if we are not calculating them
                if not fwd_rev:
                    break

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
            if fwd_rev:
                for u_name in unkn_list:
                    u_size = np.size(dunknowns[u_name])

                    # Send columns of identity
                    for idx in range(u_size):
                        dresids.vec[:] = 0.0
                        root.clear_dparams()
                        dunknowns.vec[:] = 0.0

                        dresids._dat[u_name].val[idx] = 1.0
                        dresids._scale_derivatives()
                        try:
                            comp.apply_linear(params, unknowns, dparams,
                                              dunknowns, dresids, 'rev')
                        finally:
                            dparams._apply_unit_derivatives()
                            dunknowns._scale_derivatives()

                        for p_name in param_list:

                            dinputs = dunknowns if p_name in states else dparams
                            jac_rev[(u_name, p_name)][idx, :] = dinputs._dat[p_name].val

            # Forward derivatives second
            if fwd_rev:
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
                        dunknowns._scale_derivatives()
                        comp.apply_linear(params, unknowns, dparams,
                                          dunknowns, dresids, 'fwd')
                        dresids._scale_derivatives()

                        for u_name, u_val in dresids.vec_val_iter():
                            jac_fwd[(u_name, p_name)][:, idx] = u_val

            # Finite Difference goes last
            dresids.vec[:] = 0.0
            root.clear_dparams()
            dunknowns.vec[:] = 0.0

            # Component can request to use complex step.
            if opt['form'] == 'complex_step':
                fd_func = comp.complex_step_jacobian
            else:
                fd_func = comp.fd_jacobian

            jac_fd = fd_func(params, unknowns, resids)

            # Extra Finite Difference if requested
            if f_d_2:
                dresids.vec[:] = 0.0
                root.clear_dparams()
                dunknowns.vec[:] = 0.0

                # Component can request to use complex step.
                if opt['extra_check_partials_form'] == 'complex_step':
                    fd_func = comp.complex_step_jacobian
                else:
                    fd_func = comp.fd_jacobian

                # Cache old form so we can overide temporarily
                save_form = opt['form']
                OptionsDictionary.locked = False
                opt['form'] = opt['extra_check_partials_form']

                jac_fd2 = fd_func(params, unknowns, resids)

                opt['form'] = save_form
                OptionsDictionary.locked = True

            # Assemble and Return all metrics.
            _assemble_deriv_data(chain(dparams, states), resids, data[cname],
                                 jac_fwd, jac_rev, jac_fd, out_stream,
                                 c_name=cname, jac_fd2=jac_fd2, fd_desc=fd_desc,
                                 fd_desc2=fd_desc2, compact_print=compact_print,
                                 abs_err_tol=abs_err_tol,
                                 rel_err_tol=rel_err_tol)

        return data

    def check_total_derivatives(self, out_stream=sys.stdout, abs_err_tol=1.0E-6,
                                rel_err_tol=1.0E-6):
        """ Checks total derivatives for problem defined at the top.

        Args
        ----

        out_stream : file_like
            Where to send human readable output. Default is sys.stdout. Set to
            None to suppress.

        abs_err_tol : float
            Threshold value for absolute error.  Errors about this value will
            have a '*' displayed next to them in output, making them easy
            to search for. Default is 1.0E-6.

        rel_err_tol : float
            Threshold value for relative error.  Errors about this value will
            have a '*' displayed next to them in output, making them easy
            to search for. Note at times there may be a significant relative
            error due to a minor absolute error.  Default is 1.0E-6.

        Returns
        -------
        Dict of Dicts of Tuples of Floats

        First key is the (output, input) tuple of strings; second key is one
        of ['rel error', 'abs error', 'magnitude', 'fdstep']; Tuple contains
        norms for forward - fd, adjoint - fd, forward - adjoint using the
        best case fdstep.
        """
        root = self.root
        driver = self.driver

        if driver.iter_count < 1:
            out_stream.write('Executing model to populate unknowns...\n\n')
            self.run_once()

        if out_stream is not None:
            out_stream.write('Total Derivatives Check\n\n')

        # Check derivatives with respect to design variables, if they have
        # been defined..
        if len(driver._desvars) > 0:
            param_srcs = list(driver._desvars.keys())
            to_abs_name = root._sysdata.to_abs_uname
            indep_list = [p for p in param_srcs if not \
                          root._unknowns_dict[to_abs_name[p]].get('pass_by_obj')]

        # Otherwise, use all available params.
        else:
            abs_indep_list = root._get_fd_params()
            param_srcs = [root.connections[p] for p in abs_indep_list \
                          if not root._params_dict[p].get('pass_by_obj')]

            # Convert absolute parameter names to promoted ones because it is
            # easier for the user to read.
            to_prom_name = self.root._sysdata.to_prom_name
            indep_list = [
                to_prom_name[p] for p, idxs in param_srcs
            ]

        # Check derivatives of objectives and constraints, if they have
        # been defined..
        if len(driver._objs) > 0 or len(driver._cons) > 0:
            unknown_list = list(driver._objs.keys())
            unknown_list.extend(list(driver._cons.keys()))
            unknown_list = [item for item in unknown_list \
                            if not root.unknowns.metadata(item).get('pass_by_obj')]

        # Otherwise, use all available unknowns.
        else:
            unknown_list = root._get_fd_unknowns()
            unknown_list = [item for item in unknown_list \
                            if not root.unknowns.metadata(item).get('pass_by_obj')]

        # If we are using relevance reducton, then we are hard-wired for only
        # one mode
        if root.ln_solver.options.get('single_voi_relevance_reduction'):
            mode = self._mode('auto', indep_list, unknown_list)
            if mode == 'fwd':
                fwd, rev = True, False
                Jrev = None
                if out_stream is not None:
                    out_stream.write('Relevance Checking is enabled\n')
                    out_stream.write('Only calculating fwd derviatives.\n\n')
            else:
                fwd, rev = False, True
                Jfor = None
                if out_stream is not None:
                    out_stream.write('Relevance Checking is enabled\n')
                    out_stream.write('Only calculating rev derviatives.\n\n')
        else:
            fwd = rev = True

        # Calculate all our Total Derivatives
        if fwd:
            Jfor = self.calc_gradient(indep_list, unknown_list, mode='fwd',
                                      return_format='dict')
            Jfor = _jac_to_flat_dict(Jfor)

        if rev:
            Jrev = self.calc_gradient(indep_list, unknown_list, mode='rev',
                                      return_format='dict')
            Jrev = _jac_to_flat_dict(Jrev)

        Jfd = self.calc_gradient(indep_list, unknown_list, mode='fd',
                                 return_format='dict')
        Jfd = _jac_to_flat_dict(Jfd)

        # Assemble and Return all metrics.
        data = {}
        _assemble_deriv_data(indep_list, unknown_list, data,
                             Jfor, Jrev, Jfd, out_stream,
                             abs_err_tol=abs_err_tol, rel_err_tol=rel_err_tol)

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
                    self._setup_errors.append(msg)

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

                # We treat a scaler in the source as a type of unit
                # conversion.
                if 'scaler' in smeta:
                    tmeta['unit_conv'] = (smeta['scaler'], 0.0)

                continue

            src_unit = smeta['units']
            tgt_unit = tmeta['units']

            try:
                scale, offset = get_conversion_tuple(src_unit, tgt_unit)
            except TypeError as err:
                if str(err) == "Incompatible units":
                    msg = "Unit '{0}' in source {1} "\
                        "is incompatible with unit '{2}' "\
                        "in target {3}.".format(src_unit,
                                                _both_names(smeta, to_prom_name),
                                                tgt_unit,
                                                _both_names(tmeta, to_prom_name))
                    self._setup_errors.append(msg)
                    continue
                else:
                    raise

            # We treat a scaler in the source as a type of unit
            # conversion.
            if 'scaler' in smeta:
                scale *= smeta['scaler']
                offset /= smeta['scaler']

            # If units are not equivalent, store unit conversion tuple
            # in the parameter metadata
            if scale != 1.0 or offset != 0.0:
                tmeta['unit_conv'] = (scale, offset)

    def _add_implicit_connections(self, connections):
        """
        Finds all matches between promoted names of parameters and unknowns
        in this `Problem`.  Any matches imply an implicit connection.
        All connections are expressed using absolute pathnames and are
        added to the dict of explicit connections passed in.

        Args
        ----
        connections : dict
            A dict containing all explicit connections.

        Returns
        -------
        set
            promoted parameters in this `Problem` that are not implicitly
            connected

        """

        dangling = set()

        abs_unames = self.root._sysdata.to_abs_uname

        for prom_name, pabs_list in iteritems(self.root._sysdata.to_abs_pnames):
            if prom_name in abs_unames:  # param has a src in unknowns
                for pabs in pabs_list:
                    connections.setdefault(pabs, []).append((abs_unames[prom_name], None))
            else:
                dangling.add(prom_name)

        return dangling

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
    param_owners = {}

    for par, (unk, idxs) in iteritems(connections):
        param_owners.setdefault(get_common_ancestor(par, unk), set()).add(par)

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


def _pad_name(name, pad_num=13, quotes=True):
    """ Pads a string so that they all line up when stacked."""
    l_name = len(name)
    if l_name < pad_num:
        pad = pad_num - l_name
        if quotes:
            pad_str = "'{name}'{sep:<{pad}}"
        else:
            pad_str = "{name}{sep:<{pad}}"
        pad_name = pad_str.format(name=name, sep='', pad=pad)
        return pad_name
    else:
        return '{0}'.format(name)


def _assemble_deriv_data(params, resids, cdata, jac_fwd, jac_rev, jac_fd,
                         out_stream, c_name='root', jac_fd2=None, fd_desc=None,
                         fd_desc2=None, compact_print=False, rel_err_tol=1.0E-6,
                         abs_err_tol=1.0E-6):
    """ Assembles dictionaries and prints output for check derivatives
    functions. This is used by both the partial and total derivative
    checks.
    """
    started = False

    for p_name in params:
        for u_name in resids:

            key = (u_name, p_name)

            # Ignore non-differentiables
            if key not in jac_fd:
                continue

            ldata = cdata[key] = {}

            Jsub_fd = jac_fd[key]
            ldata['J_fd'] = Jsub_fd
            magfd = np.linalg.norm(Jsub_fd)

            if jac_fwd:
                Jsub_for = jac_fwd[key]
                ldata['J_fwd'] = Jsub_for
                magfor = np.linalg.norm(Jsub_for)
            else:
                magfor = None

            if jac_rev:
                Jsub_rev = jac_rev[key]
                ldata['J_rev'] = Jsub_rev
                magrev = np.linalg.norm(Jsub_rev)
            else:
                magrev = None

            if jac_fd2:
                Jsub_fd2 = jac_fd2[key]
                ldata['J_fd2'] = Jsub_fd2
                magfd2 = np.linalg.norm(Jsub_fd2)
            else:
                magfd2 = None

            ldata['magnitude'] = (magfor, magrev, magfd)

            if jac_fwd:
                abs1 = np.linalg.norm(Jsub_for - Jsub_fd)
            else:
                abs1 = None
            if jac_rev:
                abs2 = np.linalg.norm(Jsub_rev - Jsub_fd)
            else:
                abs2 = None

            if jac_fwd and jac_rev:
                abs3 = np.linalg.norm(Jsub_for - Jsub_rev)
            else:
                abs3 = None

            if jac_fd2:
                abs4 = np.linalg.norm(Jsub_fd2 - Jsub_fd)
            else:
                abs4 = None

            ldata['abs error'] = (abs1, abs2, abs3)

            if magfd == 0.0:
                rel1 = rel2 = rel3 = rel4 = float('nan')
            else:
                if jac_fwd:
                    rel1 = np.linalg.norm(Jsub_for - Jsub_fd)/magfd
                else:
                    rel1 = None

                if jac_rev:
                    rel2 = np.linalg.norm(Jsub_rev - Jsub_fd)/magfd
                else:
                    rel2 = None

                if jac_fwd and jac_rev:
                    rel3 = np.linalg.norm(Jsub_for - Jsub_rev)/magfd
                else:
                    rel3 = None

                if jac_fd2:
                    rel4 = np.linalg.norm(Jsub_fd2 - Jsub_fd)/magfd
                else:
                    rel4 = None

            ldata['rel error'] = (rel1, rel2, rel3)

            if out_stream is None:
                continue

            if compact_print:
                if jac_fwd and jac_rev:
                    if not started:
                        tmp1 = "{0} wrt {1} | {2} | {3} |  {4} | {5} | {6} | {7} | {8}\n"
                        out_str = tmp1.format(_pad_name('<unknown>'), _pad_name('<param>'),
                                              _pad_name('fwd mag.', 10, quotes=False),
                                              _pad_name('rev mag.', 10, quotes=False),
                                              _pad_name('fd mag.', 10, quotes=False),
                                              _pad_name('a(fwd-fd)', 10, quotes=False),
                                              _pad_name('a(rev-fd)', 10, quotes=False),
                                              _pad_name('r(fwd-rev)', 10, quotes=False),
                                              _pad_name('r(rev-fd)', 10, quotes=False)
                        )
                        out_stream.write(out_str)
                        out_stream.write('-'*len(out_str)+'\n')
                        started=True

                    tmp1 = "{0} wrt {1} | {2:.4e} | {3:.4e} |  {4:.4e} | {5:.4e} | {6:.4e} | {7:.4e} | {8:.4e}\n"
                    out_stream.write(tmp1.format(_pad_name(u_name), _pad_name(p_name),
                                                 magfor, magrev, magfd, abs1, abs2,
                                                 rel1, rel2))

                elif jac_fd and jac_fd2:
                    if not started:
                        tmp1 = "{0} wrt {1} | {2} | {3} | {4} | {5}\n"
                        out_str = tmp1.format(_pad_name('<unknown>'), _pad_name('<param>'),
                                              _pad_name('fd1 mag.', 13, quotes=False),
                                              _pad_name('fd2 mag.', 12, quotes=False),
                                              _pad_name('ab(fd2 - fd1)', 12, quotes=False),
                                              _pad_name('rel(fd2 - fd1)', 12, quotes=False)
                        )
                        out_stream.write(out_str)
                        out_stream.write('-'*len(out_str)+'\n')
                        started=True

                    tmp1 = "{0} wrt {1} | {2: .6e} | {3:.6e} | {4: .6e} | {5: .6e}\n"
                    out_stream.write(tmp1.format(_pad_name(u_name), _pad_name(p_name),
                                                 magfd, magfd2, abs4, rel4))
            else:

                if started:
                    out_stream.write(' -'*30 + '\n')
                else:
                    started = True

                # Optional file_like output
                out_stream.write("  %s: '%s' wrt '%s'\n\n" % (c_name, u_name, p_name))

                if jac_fwd:
                    out_stream.write('    Forward Magnitude : %.6e\n' % magfor)
                if jac_rev:
                    out_stream.write('    Reverse Magnitude : %.6e\n' % magrev)
                if not jac_fwd and not jac_rev:
                    out_stream.write('    Fwd/Rev Magnitude : Component supplies no analytic derivatives.\n')
                if jac_fd:
                    out_stream.write('         Fd Magnitude : %.6e' % magfd)
                    if fd_desc:
                        out_stream.write(' (%s)' % fd_desc)
                    out_stream.write('\n')
                if jac_fd2:
                    out_stream.write('        Fd2 Magnitude : %.6e' % magfd2)
                    if fd_desc2:
                        out_stream.write(' (%s)' % fd_desc2)
                    out_stream.write('\n')
                out_stream.write('\n')

                if jac_fwd:
                    flag = '' if abs1 < abs_err_tol else ' *'
                    out_stream.write('    Absolute Error (Jfor - Jfd) : %.6e%s\n' % (abs1, flag))
                if jac_rev:
                    flag = '' if abs2 < abs_err_tol else ' *'
                    out_stream.write('    Absolute Error (Jrev - Jfd) : %.6e%s\n' % (abs2, flag))
                if jac_fwd and jac_rev:
                    flag = '' if abs3 < abs_err_tol else ' *'
                    out_stream.write('    Absolute Error (Jfor - Jrev): %.6e%s\n' % (abs3, flag))
                if jac_fd2:
                    flag = '' if abs4 < abs_err_tol else ' *'
                    out_stream.write('    Absolute Error (Jfd2 - Jfd): %.6e%s\n' % (abs4, flag))
                out_stream.write('\n')

                if jac_fwd:
                    flag = '' if np.isnan(rel1) or rel1 < rel_err_tol else ' *'
                    out_stream.write('    Relative Error (Jfor - Jfd) : %.6e%s\n' % (rel1, flag))
                if jac_rev:
                    flag = '' if np.isnan(rel2) or rel2 < rel_err_tol else ' *'
                    out_stream.write('    Relative Error (Jrev - Jfd) : %.6e%s\n' % (rel2, flag))
                if jac_fwd and jac_rev:
                    flag = '' if np.isnan(rel3) or rel3 < rel_err_tol else ' *'
                    out_stream.write('    Relative Error (Jfor - Jrev): %.6e%s\n' % (rel3, flag))
                if jac_fd2:
                    flag = '' if np.isnan(rel4) or rel4 < rel_err_tol else ' *'
                    out_stream.write('    Relative Error (Jfd2 - Jfd) : %.6e%s\n' % (rel4, flag))
                out_stream.write('\n')

                if jac_fwd:
                    out_stream.write('    Raw Forward Derivative (Jfor)\n\n')
                    out_stream.write(str(Jsub_for))
                    out_stream.write('\n\n')
                if jac_rev:
                    out_stream.write('    Raw Reverse Derivative (Jrev)\n\n')
                    out_stream.write(str(Jsub_rev))
                    out_stream.write('\n\n')
                out_stream.write('    Raw FD Derivative (Jfd)\n\n')
                out_stream.write(str(Jsub_fd))
                out_stream.write('\n\n')
                if jac_fd2:
                    out_stream.write('    Raw FD Check Derivative (Jfd2)\n\n')
                    out_stream.write(str(Jsub_fd2))
                    out_stream.write('\n\n')

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
