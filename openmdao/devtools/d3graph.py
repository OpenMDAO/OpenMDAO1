from __future__ import print_function

import os
import sys
import shutil
import json
import tempfile
import threading
import time
import pprint

import webbrowser
from six.moves import SimpleHTTPServer, socketserver 

import networkx as nx
from networkx.readwrite.json_graph import node_link_data

from openmdao.core.component import Component

def _launch_browser(port, fname):
    time.sleep(1)
    webbrowser.get().open('http://localhost:%s/%s' % (port,fname))

def _startThread(fn):
    thread = threading.Thread(target=fn)
    thread.setDaemon(True)
    thread.start()
    return thread

def _system_tree_dict(system):
    """
    Returns a dict representation of the system hierarchy with
    this System as root.
    """

    def _tree_dict(ss):
        dct = { 'name': ss.name }
        children = [_tree_dict(s) for s in ss.subsystems()]

        if isinstance(ss, Component):
            for vname, meta in ss.unknowns.items():
                size = meta['size'] if meta['size'] else 1
                children.append({'name': vname, 'size': size })

            for vname, meta in ss.params.items():
                size = meta['size'] if meta['size'] else 1
                children.append({'name': vname, 'size': size })

            dct['_children'] = children # start with child var nodes toggled off
            dct['children'] = None
        else:
            dct['children'] = children
            dct['_children'] = None

        return dct

    tree = _tree_dict(system)
    if not tree['name']:
        tree['name'] = 'root'

    return tree

def view_tree(system, viewer='collapse_tree', port=8001):
    """
    Args
    ----
    system : system
        The root system for the desired tree.

    viewer : str, optional
        The name of web viewer used to view the tree. Options are:
        collapse_tree, circlepack, circletree, indenttree, and treemap.

    port : int, optional
        The port number for the web server that serves the tree viewing page.

    """
    if not viewer.endswith('.html'):
        viewer += '.html'

    tree = _system_tree_dict(system)
    try:
        startdir = os.getcwd()
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

        with open('__graph.json', 'w') as f:
            json.dump(tree, f)

        httpd = socketserver.TCPServer(("localhost", port),
                           SimpleHTTPServer.SimpleHTTPRequestHandler)

        print("starting server on port %d" % port)

        serve_thread  = _startThread(httpd.serve_forever)
        _launch_browser(port, viewer)

        while serve_thread.isAlive():
            serve_thread.join(timeout=1)

    finally:
        try:
            os.remove('__graph.json')
        except:
            pass
        os.chdir(startdir)
