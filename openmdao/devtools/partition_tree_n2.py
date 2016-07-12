
import os
import sys
import json
from six import iteritems
import networkx as nx

import webbrowser

from openmdao.core.component import Component



def _system_tree_dict(system, component_execution_orders, component_execution_index):
    """
    Returns a dict representation of the system hierarchy with
    the given System as root.
    """

    def _tree_dict(ss, component_execution_orders, component_execution_index):
        subsystem_type = 'group'
        if isinstance(ss, Component):
            subsystem_type = 'component'
            component_execution_orders[ss.pathname] = component_execution_index[0]
            component_execution_index[0] += 1
        dct = { 'name': ss.name, 'type': 'subsystem', 'subsystem_type': subsystem_type }
        children = [_tree_dict(s, component_execution_orders, component_execution_index) for s in ss.subsystems()]

        if isinstance(ss, Component):
            for vname, meta in ss.unknowns.items():
                dtype=type(meta['val']).__name__
                implicit = False
                if meta.get('state'):
                    implicit = True
                children.append({'name': vname, 'type': 'unknown', 'implicit': implicit, 'dtype': dtype})

            for vname, meta in ss.params.items():
                dtype=type(meta['val']).__name__
                children.append({'name': vname, 'type': 'param', 'dtype': dtype})

        dct['children'] = children

        return dct

    tree = _tree_dict(system, component_execution_orders, component_execution_index)
    if not tree['name']:
        tree['name'] = 'root'
        tree['type'] = 'root'

    return tree

def view_tree(problem, outfile='partition_tree_n2.html', show_browser=True):
    """
    Generates a self-contained html file containing a tree viewer
    of the specified type.  Optionally pops up a web browser to
    view the file.

    Args
    ----
    problem : Problem()
        The Problem (after problem.setup()) for the desired tree.

    outfile : str, optional
        The name of the output html file.  Defaults to 'partition_tree_n2.html'.

    show_browser : bool, optional
        If True, pop up a browser to view the generated html file.
        Defaults to True.
    """
    component_execution_orders = {}
    component_execution_index = [0] #list so pass by ref
    tree = _system_tree_dict(problem.root, component_execution_orders, component_execution_index)
    viewer = 'partition_tree_n2.template'

    code_dir = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(code_dir, viewer), "r") as f:
        template = f.read()

    treejson = json.dumps(tree)

    connections_list = []
    G = problem._probdata.relevance._sgraph
    scc = nx.strongly_connected_components(G)
    scc_list = [s for s in scc if len(s)>1] #list(scc)

    for tgt, (src, idxs) in iteritems(problem._probdata.connections):
        src_subsystem = src.rsplit('.', 1)[0]
        tgt_subsystem = tgt.rsplit('.', 1)[0]


        count = 0
        edges_set = set()
        for li in scc_list:
            if src_subsystem in li and tgt_subsystem in li:
                count = count+1
                if(count > 1):
                    raise ValueError('Count greater than 1')

                exe_tgt = component_execution_orders[tgt_subsystem]
                exe_src = component_execution_orders[src_subsystem]
                exe_low = min(exe_tgt,exe_src)
                exe_high = max(exe_tgt,exe_src)
                subg = G.subgraph(li)
                for n in subg.nodes():
                    exe_order = component_execution_orders[n]
                    if(exe_order < exe_low or exe_order > exe_high):
                        subg.remove_node(n)


                list_sim = list(nx.all_simple_paths(subg,source=tgt_subsystem,target=src_subsystem))





                for this_list in list_sim:
                    if(len(this_list) >= 2):
                        for i in range(len(this_list)-1):
                            edge_str = this_list[i] + ' ' + this_list[i+1]
                            edges_set.add(edge_str)


        edges_set_list = list(edges_set)
        if(len(edges_set_list) > 0):
            connections_list.append({'src':src, 'tgt':tgt, 'cycle_arrows': edges_set_list})
        else:
            connections_list.append({'src':src, 'tgt':tgt})

    connsjson = json.dumps(connections_list)

    with open(outfile, 'w') as f:
        f.write(template % (treejson, connsjson))

    if show_browser:
        from openmdao.devtools.d3graph import webview
        webview(outfile)
