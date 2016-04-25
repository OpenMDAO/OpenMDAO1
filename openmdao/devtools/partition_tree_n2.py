
import os
import sys
import json
from six import iteritems

import webbrowser

from openmdao.core.component import Component


def _system_tree_dict(system):
    """
    Returns a dict representation of the system hierarchy with
    the given System as root.
    """

    def _tree_dict(ss):
        subsystem_type = 'group'
        if isinstance(ss, Component):
            subsystem_type = 'component'
        dct = { 'name': ss.name, 'type': 'subsystem', 'subsystem_type': subsystem_type }
        children = [_tree_dict(s) for s in ss.subsystems()]

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

    tree = _tree_dict(system)
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
    tree = _system_tree_dict(problem.root)
    viewer = 'partition_tree_n2.template'

    code_dir = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(code_dir, viewer), "r") as f:
        template = f.read()

    treejson = json.dumps(tree)

    myList = []
    for target, (src, idxs) in iteritems(problem._probdata.connections):
        myList.append({'src':src, 'tgt':target})
    connsjson = json.dumps(myList)

    with open(outfile, 'w') as f:
        f.write(template % (treejson, connsjson))

    if show_browser:
        from openmdao.devtools.d3graph import webview
        webview(outfile)
