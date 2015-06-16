""" OpenMDAO Problem class defintion."""

from itertools import chain
from six import string_types

import networkx as nx


class Relevance(object):
    def __init__(self, params_dict, unknowns_dict, connections,
                 inputs, outputs, mode):

        self.params_dict = params_dict
        self.unknowns_dict = unknowns_dict
        self.mode = mode

        # turn all inputs and outputs, even singletons, into tuples
        self.inputs = []
        for inp in inputs:
            if isinstance(inp, string_types):
                self.inputs.append((inp,))
            else:
                self.inputs.append(tuple(inp))

        self.outputs = []
        for out in outputs:
            if isinstance(out, string_types):
                self.outputs.append((out,))
            else:
                self.outputs.append(tuple(out))

        vgraph = self._setup_graph(connections)
        self.relevant = self._get_relevant_vars(vgraph)

    def is_relevant(self, var_of_interest, varname):
        if not var_of_interest:
            return True
        return varname in self.relevant[var_of_interest]

    def vars_of_interest(self):
        if self.mode == 'fwd':
            return iter(self.inputs)
        elif self.mode == 'rev':
            return iter(self.outputs)
        else:
            return iter(self.inputs+self.outputs)

    def _setup_graph(self, connections):
        """
        Set up a dependency graph for all variables in the Problem.

        Returns
        -------
        nx.DiGraph
            A graph containing all variables and their connections

        """
        params_dict = self.params_dict
        unknowns_dict = self.unknowns_dict

        vgraph = nx.DiGraph()  # var graph

        compins = {}  # maps input vars to components
        compouts = {} # maps output vars to components

        for param in params_dict:
            tcomp = param.rsplit(':',1)[0]
            compins.setdefault(tcomp, []).append(param)

        for unknown in unknowns_dict:
            scomp = unknown.rsplit(':',1)[0]
            compouts.setdefault(scomp, []).append(unknown)

        for target, source in connections.items():
            vgraph.add_edge(source, target)

        # connect inputs to outputs on same component in order to fully
        # connect the variable graph.
        for comp, inputs in compins.items():
            for inp in inputs:
                for out in compouts.get(comp, ()):
                    vgraph.add_edge(inp, out)

        return vgraph

    def _get_relevant_vars(self, g):
        """
        Parameters
        ----------
        g : nx.DiGraph
            A graph of variable dependencies.

        Returns
        -------
        dict
            Dictionary that maps a variable name to all other variables in the graph that
            are relevant to it.
        """
        succs = {}
        for nodes in self.inputs:
            for node in nodes:
                succs[node] = set([v for u,v in nx.dfs_edges(g, node)])
                succs[node].add(node)

        relevant = {}
        grev = g.reverse()
        for nodes in self.outputs:
            for node in nodes:
                relevant[node] = set()
                preds = set([v for u,v in nx.dfs_edges(grev, node)])
                preds.add(node)
                for inps in self.inputs:
                    for inp in inps:
                        common = preds.intersection(succs[inp])
                        relevant[node].update(common)
                        relevant.setdefault(inp, set()).update(common)

        return relevant
