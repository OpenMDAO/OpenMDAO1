import os
import sys
import time
import inspect
import fnmatch
import argparse
import json
from functools import wraps

import types

from six import iteritems

from openmdao.core.mpi_wrap import MPI
from openmdao.core.group import Group
from openmdao.devtools.d3graph import webview

def get_method_class(meth):
    """Return the class that actually defined the given method."""
    for cls in inspect.getmro(meth.__self__.__class__):
        if meth.__name__ in cls.__dict__:
            return cls

# list of default methods to profile
_profile_methods = [
    "calc_gradient",
    "solve_nonlinear",
    "apply_nonlinear",
    "solve_linear",
    "apply_linear",
    "solve",
    "fd_jacobian",
    "linearize",
    "complex_step_jacobian",
]

_profile_prefix = None
_profile_out = None
_profile_by_class = None

def activate_profiling(prefix='prof_raw', methods=None, by_class=False):
    """Turns on profiling of certain important openmdao methods.

    Args
    ----

    prefix : str ('prof_raw')
        Prefix used for the raw profile data. Process rank will be appended
        to it to get the actual filename.  When not using MPI, rank=0.

    methods : list of str, optional
        A list of profiled methods to override the default set.  Method names
        should be simple names like "solve", or "fd_jacobian" an should not
        include a class name.  The default set of methods is:
        ["solve_nonlinear", "apply_nonlinear",
         "solve_linear", "apply_linear", "solve",
         "fd_jacobian", "linearize", "complex_step_jacobian"]

    by_class : bool (False)
        If True, use classes to group call information rather than instances.
    """
    global _profile_prefix, _profile_methods, _profile_by_class
    _profile_prefix = prefix
    _profile_by_class = by_class

    if methods:
        _profile_methods = methods

def _setup_profiling(problem):
    """
    Create the profile data output file and instrument the methods to be
    profiled.  Does nothing unless activate_profiling() has been called.
    """
    global _profile_out, _profile_prefix, _profile_by_class

    if _profile_prefix:
        if MPI:
            rank = MPI.COMM_WORLD.rank
        else:
            rank = 0
        _profile_out = open("%s.%d" % (_profile_prefix, rank), 'w')
        _profile_out.write(','.join(['class','pathname','funcname',
                                     'elapsed_time', 'timestamp',
                                     'caller_class', 'caller_pathname',
                                     'caller_funcname']))
        _profile_out.write('\n')

    if _profile_out is not None:
        rootsys = problem.root

        for meth in _profile_methods:
            if hasattr(problem, meth):
                setattr(problem, meth,
                        profile()(getattr(problem, meth)).__get__(problem,
                                                                  problem.__class__))

        # wrap a bunch of methods for profiling
        for s in rootsys.subsystems(recurse=True, include_self=True):
            if isinstance(s, Group):
                objs = (s, s.ln_solver, s.nl_solver)
            else:
                objs = (s,)
            for obj in objs:
                for meth in _profile_methods:
                    if hasattr(obj, meth):
                        setattr(obj, meth,
                                profile()(getattr(obj, meth)).__get__(obj,
                                                                      obj.__class__))

class profile(object):
    """ Use as a decorator on functions that should be profiled.
    The data collected will include time elapsed, number of calls, ...
    """
    _call_stack = []

    def __init__(self):
        pass

    def __call__(self, fn):
        global _profile_out, _profile_by_class
        # don't actually wrap the function unless OPENMDAO_PROFILE is set
        if _profile_out is not None:
            @wraps(fn)
            def wrapper(*args, **kwargs):

                if _profile_by_class:
                    try:
                        name = get_method_class(fn).__name__
                    except AttributeError:
                        name = ''
                else:  # profile by instance
                    try:
                        name = fn.__self__.pathname
                    except AttributeError:
                        name = "<%s>" % args[0].__class__.__name__

                name = '.'.join((name, fn.__name__))

                stack = profile._call_stack

                if stack:
                    caller = profile._call_stack[-1]
                else:
                    caller = ''

                stack.append(name)

                start = time.time()

                ret = fn(*args[1:], **kwargs)

                end = time.time()

                profile._call_stack.pop()

                data = [
                    name,
                    str(end-start),
                    str(start),
                ]

                data.extend(profile._call_stack)

                _profile_out.write(','.join(data))
                _profile_out.write('\n')
                _profile_out.flush()

                return ret

            return wrapper
        return fn

def _update_counts(dct, name, elapsed):
    if name not in dct:
        dct[name] = {
                'count': 1,
                'elapsed_time': elapsed,
            }
    else:
        tot = dct[name]
        tot['count'] += 1
        tot['elapsed_time'] += elapsed

def process_profile(profs):
    """Take the generated raw profile data, potentially from multiple files,
    and combine it to get hierarchy structure and total execution counts and
    timing data.

    Args
    ----

    prof : str or None
        Name of profile data file.  Can contain wildcards to process multiple
        profiles together, e.g., when MPI is used.

    """

    if profs is None:
        prof = _profile_prefix
        prof += '.*'
        flist = fnmatch.filter(os.listdir('.'), prof)
    else:
        flist = profs

    funcs = {}
    totals = {}

    for fname in flist:
        with open(fname, 'r') as f:
            for i, line in enumerate(f):
                if i==0:
                    continue # skip header

                line = line.strip()

                parts = line.split(',')

                name, elapsed, tstamp = parts[:3]
                stack = tuple(parts[3:])

                elapsed = float(elapsed)

                _update_counts(totals, name, elapsed)

                if stack not in funcs:
                    funcs[stack] = {}

                _update_counts(funcs[stack], name, elapsed)

    info = {} # mapping of full stack path to callee
    tree = { 'name': '', 'children': [] }

    info[()] = tree

    for stack, fdicts in iteritems(funcs):

        if stack:
            caller = stack[-1]
            if stack not in info:
                info[stack] = {
                    'name': caller,
                    'children': [],
                }
        else:
            caller = tree['name']

        caller_dct = info[stack]

        for callee, fdict in iteritems(fdicts):
            stk = list(stack)
            stk.append(callee)
            stk = tuple(stk)
            if stk not in info:
                info[stk] = {
                    'name': callee,
                    'totals': totals[callee],
                    'elapsed_time': fdict['elapsed_time'],
                    'count': fdict['count'],
                    'children': [],
                }
            dct = info[stk]
            caller_dct['children'].append(dct)

    return tree, totals

def viewprof():
    """Called from a command line to process profile data files."""

    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--outfile', action='store', dest='outfile',
                        metavar='OUTFILE', default='profile.out',
                        help='Name of file containing function total counts and elapsed times.')
    parser.add_argument("--show_browser", dest='show', help="If true pop up a browser",
                        action="store_true", default=True)
    parser.add_argument('rawfiles', metavar='rawfile', nargs='*',
                        help='File(s) containing raw profile data to be processed. Wildcards are allowed.')

    options = parser.parse_args()

    if options.outfile == 'sys.stdout':
        out_stream = sys.stdout
    else:
        out_stream = open(options.outfile, 'w')

    call_graph, totals = process_profile(options.rawfiles)

    out_stream.write("Function Name, Total Time, Max Time, Min Time, Calls\n")
    for func, data in sorted(((k,v) for k,v in iteritems(totals)),
                                key=lambda x:x[1]['elapsed_time'],
                                reverse=True):
        out_stream.write("%s, %s, %s\n" %
                           (func, data['elapsed_time'], data['count']))

    viewer = 'sunburst.html'
    code_dir = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(code_dir, viewer), "r") as f:
        template = f.read()

    graphjson = json.dumps(call_graph)

    outfile = 'call_graph.html'
    with open(outfile, 'w') as f:
        s = template.replace("<call_graph_data>", graphjson)
        f.write(s)

    if options.show:
        webview(outfile)
