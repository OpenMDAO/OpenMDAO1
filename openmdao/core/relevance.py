""" Relevance object for systems. Manages the data connectivity graph."""

from __future__ import print_function

from collections import OrderedDict
import json
from six import string_types, itervalues, iteritems

import networkx as nx

from openmdao.util.graph import collapse_nodes


class Relevance(object):
    """ Object that manages the data connectivity graph for systems."""

    def __init__(self, group, params_dict, unknowns_dict, connections,
                 inputs, outputs, mode):

        self.params_dict = params_dict
        self.unknowns_dict = unknowns_dict
        self.mode = mode
        self._sysdata = group._sysdata

        param_groups = []
        output_groups = []

        # turn all inputs and outputs, even singletons, into tuples
        self.inputs = []
        for inp in inputs:
            if isinstance(inp, string_types):
                inp = (inp,)
            param_groups.append(tuple(inp))
            self.inputs.append(tuple(inp))

        self.outputs = []
        for out in outputs:
            if isinstance(out, string_types):
                out = (out,)
            output_groups.append(tuple(out))
            self.outputs.append(tuple(out))

        self._sgraph = self._setup_sys_graph(group, connections)
        self._compute_relevant_vars(group, connections)

        # when voi is None, everything is relevant
        self.relevant[None] = set(m['top_promoted_name']
                                    for m in itervalues(unknowns_dict))
        self.relevant[None].update(m['top_promoted_name']
                                    for m in itervalues(params_dict))

        if mode == 'fwd':
            self.groups = param_groups
        else:
            self.groups = output_groups

    def __getitem__(self, name):
        try:
            return self.relevant[name]
        except KeyError:
            return ()

    def is_relevant(self, var_of_interest, varname):
        """ Returns True if a variable is relevant to a particular variable
        of interest.

        Args
        ----
        var_of_interest : str
            Name of a variable of interest (either a parameter or a constraint
            or objective output, depending on mode.)

        varname : str
            Name of some other variable in the model.

        Returns
        -------
        bool: True if varname is in the relevant path of var_of_interest
        """
        try:
            return varname in self.relevant[var_of_interest]
        except KeyError:
            return True

    def vars_of_interest(self, mode=None):
        """ Determines our list of var_of_interest depending on mode.

        Args
        ----
        mode : str
            Derivative mode, can be 'fwd' or 'rev'.

        Returns
        -------
        list : Our inputs, or output, or both, depending on mode.
        """
        if mode is None:
            mode = self.mode
        if mode == 'fwd':
            return self.inputs
        elif mode == 'rev':
            return self.outputs
        else:
            return self.inputs+self.outputs

    def is_relevant_system(self, var_of_interest, system):
        """
        Args
        ----
        var_of_interest : str
            Name of a variable of interest (either a parameter or a constraint
            or objective output, depending on mode.)

        system : `System`
            The system being checked for relevant w.r.t. the variable of
            interest.

        Returns
        -------
        bool
            True if the given system is relevant for the given variable of
            interest.
        """
        return var_of_interest is None or system.pathname in self._relevant_systems[var_of_interest]

    def _setup_sys_graph(self, group, connections):
        """
        Set up dependency graph for systems in the Problem.

        Returns
        -------
        nx.DiGraph
            The system graph.

        """
        sgraph = nx.DiGraph()  # subsystem graph

        # ensure we have system graph nodes even for unconnected subsystems
        sgraph.add_nodes_from(s.pathname for s in group.subsystems(recurse=True))

        for target, (source, idxs) in iteritems(connections):
            sgraph.add_edge(source.rsplit('.', 1)[0], target.rsplit('.', 1)[0])

        return sgraph

    def _compute_relevant_vars(self, group, connections):
        """
        Calculate the relevant variables and relevant systems for the
        current variables of interest.

        Args
        ----
        group : Group
            The top level group.

        connections : OrderedDict
            Dict of targets mapped to (src, idxs)

        """

        relevant = {}
        succs = {}

        sgraph = self._sgraph      # system graph

        to_prom_name = group._sysdata.to_prom_name
        to_abs_uname = group._sysdata.to_abs_uname

        for nodes in self.inputs:
            for node in nodes:
                relevant[node] = set()
                pnode = to_abs_uname[node]
                comp = pnode.rsplit('.', 1)[0]
                succs[node] = []
                if comp in sgraph:
                    succs[node].append(comp)
                    succs[node].extend(v for u,v in nx.dfs_edges(sgraph, comp))

        grev = sgraph.reverse()
        self._outset = set()
        for nodes in self.outputs:
            self._outset.update(nodes)
            for node in nodes:
                unode = to_abs_uname[node]
                comp = unode.rsplit('.', 1)[0]
                relevant[node] = set()
                if comp in sgraph:
                    preds = set((comp,))
                    preds.update(v for u, v in nx.dfs_edges(grev, comp))
                    for inps in self.inputs:
                        for inp in inps:
                            common = preds.intersection(succs[inp])
                            relevant[node].update(common)
                            relevant[inp].update(common)

        # at this point, relevant contains the relevent *systems*, so now
        # we have to determine the relevant variables based on those systems
        # and our connections
        relvars = {}
        rcomps = [to_abs_uname[n].rsplit('.', 1)[0] for n in relevant]

        # make sure we don't miss any other VOIs that are relevant but are not
        # part of a connection
        for name, relcomps in iteritems(relevant):
            relvars[name] = set()
            for i, n in enumerate(relevant):
                if rcomps[i] in relcomps:
                    relvars[name].add(n)

        for tgt, (src, idxs) in iteritems(connections):
            prom_tgt = to_prom_name[tgt]
            prom_src = to_prom_name[src]
            tcomp = tgt.rsplit('.', 1)[0]
            scomp = src.rsplit('.', 1)[0]
            for n, relcomps in iteritems(relevant):
                if tcomp in relcomps and scomp in relcomps:
                    relvars[n].update((prom_tgt, prom_src))

        # finally, add ancestors of relevant systems to the relevant set
        for voi, relsystems in iteritems(relevant):
            to_add = set()
            for system in relsystems:
                parts = system.split('.')[:-1]
                for i in range(0, len(parts)):
                    to_add.add('.'.join(parts[:i+1]))
            relsystems.update(to_add)

        self._relevant_systems = relevant
        self.relevant = relvars
