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

        self._vgraph, self._sgraph = self._setup_graphs(group, connections)
        self.relevant = self._get_relevant_vars(group, self._vgraph)
        # when voi is None, everything is relevant
        self.relevant[None] = set(m['top_promoted_name']
                                    for m in itervalues(unknowns_dict))
        self.relevant[None].update(m['top_promoted_name']
                                    for m in itervalues(params_dict))
        self._relevant_systems = self._get_relevant_systems()

        if mode == 'fwd':
            self.groups = param_groups
        else:
            self.groups = output_groups

    def __getitem__(self, name):
        # if name is None, everything is relevant
        if name in self.relevant:
            return self.relevant[name]
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
        if var_of_interest is None:
            return True
        return system.pathname in self._relevant_systems[var_of_interest]

    def _setup_graphs(self, group, connections):
        """
        Set up dependency graphs for variables and components in the Problem.

        Returns
        -------
        tuple of (nx.DiGraph, nx.DiGraph)
            The variable graph and the component graph.

        """
        params_dict = self.params_dict
        unknowns_dict = self.unknowns_dict

        vgraph = nx.DiGraph()  # var graph
        sgraph = nx.DiGraph()  # subsystem graph

        self._compins = compins = {}  # maps input vars to components
        self._compouts = compouts = {} # maps output vars to components

        # ensure we have system graph nodes even for unconnected subsystems
        sgraph.add_nodes_from(s.pathname for s in group.subsystems(recurse=True))

        for param, meta in iteritems(params_dict):
            tcomp = param.rsplit('.', 1)[0]
            compins.setdefault(tcomp, set()).add(param)

        for unknown, meta in iteritems(unknowns_dict):
            scomp = unknown.rsplit('.', 1)[0]
            compouts.setdefault(scomp, set()).add(unknown)

        for target, (source, idxs) in iteritems(connections):
            vgraph.add_edge(source, target)
            sgraph.add_edge(source.rsplit('.', 1)[0], target.rsplit('.', 1)[0])

        # connect inputs to outputs on same component in order to fully
        # connect the variable graph.
        for comp, inputs in iteritems(compins):
            outs = compouts.get(comp, ())
            for inp in inputs:
                for out in outs:
                    vgraph.add_edge(inp, out)

        return vgraph, sgraph

    def _get_relevant_vars(self, group, g):
        """
        Args
        ----
        group : Group
            The top level group.

        g : nx.DiGraph
            A graph of variable dependencies.

        Returns
        -------
        dict
            Dictionary that maps a variable name to all other variables in the
            graph that are relevant to it.
        """

        relevant = {}
        succs = {}

        compins = self._compins
        compouts = self._compouts

        to_prom_name = group._sysdata.to_prom_name
        to_abs_uname = group._sysdata.to_abs_uname

        for nodes in self.inputs:
            for node in nodes:
                relevant[node] = set()
                succs[node] = set((node,))
                pnode = to_abs_uname[node]
                if pnode in g:
                    comp = pnode.rsplit('.', 1)[0]
                    succs[node].update(to_prom_name[v]
                                        for u, v in nx.dfs_edges(g, pnode))

        grev = g.reverse(copy=False)
        self._outset = set()
        for nodes in self.outputs:
            self._outset.update(nodes)
            for node in nodes:
                unode = to_abs_uname[node]
                relevant[node] = set()
                if unode in g:
                    preds = set(to_prom_name[v] for u, v in nx.dfs_edges(grev, unode))
                    preds.add(node)
                    for inps in self.inputs:
                        for inp in inps:
                            if to_abs_uname[inp] in g:
                                common = preds.intersection(succs[inp])
                                relevant[node].update(common)
                                relevant[inp].update(common)

        return relevant

    def _get_relevant_systems(self):
        """
        Given the dict that maps relevant vars to each VOI, find the mapping
        of each VOI to the set of systems that need to run.
        """
        relevant_systems = {}
        grev = self._sgraph.reverse()

        to_abs_uname = self._sysdata.to_abs_uname
        to_abs_pnames = self._sysdata.to_abs_pnames

        for voi, relvars in iteritems(self.relevant):
            rev = True if voi in self._outset else False
            if rev:
                voicomp = to_abs_uname[voi].rsplit('.', 1)[0]
                gpath = set([voicomp])
                gpath.update([v for u,v in nx.dfs_edges(grev, voicomp)])
            comps = set()
            for relvar in relvars:
                if relvar in to_abs_uname:
                    absvars = (to_abs_uname[relvar],)
                else:
                    absvars = iter(to_abs_pnames[relvar])
                for absvar in absvars:
                    parts = absvar.split('.')
                    for i in range(len(parts)-1):
                        cname = '.'.join(parts[:i+1])
                        # in rev mode, need to eliminate irrelevant systems that
                        # have shared promoted vars
                        if rev:
                            if cname in gpath:
                                comps.add(cname)
                        else:
                            comps.add(cname)
            relevant_systems[voi] = tuple(comps)

        return relevant_systems
