import os
import sys
import time
import inspect
import fnmatch
import argparse
import json
import atexit
import types
from collections import OrderedDict
from functools import wraps
from struct import Struct
from ctypes import Structure, c_uint, c_double

from six import iteritems

from openmdao.core.mpi_wrap import MPI
from openmdao.core.system import System
from openmdao.core.group import Group
from openmdao.core.component import Component
from openmdao.core.driver import Driver
from openmdao.solvers.solver_base import SolverBase
from openmdao.recorders.recording_manager import RecordingManager
from openmdao.devtools.d3graph import webview

def get_method_class(meth):
    """Return the class that actually defined the given method."""
    for cls in inspect.getmro(meth.__self__.__class__):
        if meth.__name__ in cls.__dict__:
            return cls


class _ProfData(Structure):
    _fields_ = [ ('t',c_double), ('tstamp',c_double), ('id',c_uint) ]

_profile_methods = None
_profile_prefix = None
_profile_out = None
_profile_by_class = None
_profile_start = None
_profile_on = False
_profile_struct = _ProfData()
_profile_funcs_dict = OrderedDict()

def activate_profiling(prefix='prof_raw', methods=None, by_class=False):
    """Turns on profiling of certain important openmdao methods.

    Args
    ----

    prefix : str ('prof_raw')
        Prefix used for the raw profile data. Process rank will be appended
        to it to get the actual filename.  When not using MPI, rank=0.

    methods : dict, optional
        A dict of profiled methods to override the default set.  The key
        is the method name and the value is a tuple of class objects used
        for isinstance checking.  The default set of methods is:
        {
            "run": (Problem,),
            "calc_gradient": (Problem,),
            "solve_nonlinear": (System,),
            "apply_nonlinear": (System,),
            "solve_linear": (System,),
            "apply_linear": (System,),
            "solve": (SolverBase,),
            "fd_jacobian": (System,),
            "linearize": (System,),
            "complex_step_jacobian": (Component,),
            "record_iteration": (RecordingManager,),
            "record_derivatives": (RecordingManager,),
            "_transfer_data": (Group,),
        }

    by_class : bool (False)
        If True, use classes to group call information rather than instances.
    """
    from openmdao.core.problem import Problem # avoid circular import

    global _profile_prefix, _profile_methods, _profile_by_class, _profile_on

    if _profile_on:
        raise RuntimeError("profiling is already active.")

    _profile_prefix = prefix
    _profile_by_class = by_class
    _profile_on = True

    if methods:
        _profile_methods = methods
    else:
        _profile_methods = {
            "run": (Problem,),
            "calc_gradient": (Problem,),
            "solve_nonlinear": (System,),
            "apply_nonlinear": (System,),
            "solve_linear": (System,),
            "apply_linear": (System,),
            "solve": (SolverBase,),
            "fd_jacobian": (System,),
            "linearize": (System,),
            "complex_step_jacobian": (Component,),
            "record_iteration": (RecordingManager,),
            "record_derivatives": (RecordingManager,),
            "_transfer_data": (Group,),
        }

def deactivate_profiling():
    """Turn off profiling.  Note that currently you can't turn it back on
    after you've turned it off.
    """
    global _profile_on
    _profile_on = False

def _iter_raw_prof_file(rawname, fdict):
    """Returns an iterator of (elapsed_time, timestamp, funcpath)
    from a raw profile data file.
    """
    global _profile_struct

    fn, ext = os.path.splitext(rawname)
    funcs_fname = "funcs_" + fn + ext

    with open(funcs_fname, 'r') as f:
        for line in f:
            line = line.strip()
            path, ident = line.split(' ')
            fdict[ident] = path

    with open(rawname, 'rb') as f:
        while f.readinto(_profile_struct):
            path = fdict[str(_profile_struct.id)]
            yield _profile_struct.t, _profile_struct.tstamp, path

def _write_funcs_dict():
    """called at exit to write out the file mapping function call paths
    to identifiers.
    """
    global _profile_prefix, _profile_funcs_dict, _profile_start
    rank = MPI.COMM_WORLD.rank if MPI else 0
    with open("funcs_%s.%d" % (_profile_prefix, rank), 'w') as f:
        for name, ident in iteritems(_profile_funcs_dict):
            f.write("%s %s\n" % (name, ident))
        # also write out the total time so that we can report how much of
        # the runtime is invisible to our profile.
        f.write("%s %s\n" % (time.time()-_profile_start, "@total"))

def _prof_iter(prob):
    """Iterator over objects to be checked for functions to wrap for profiling."""
    yield prob
    yield prob.driver
    if prob.driver.recorders._recorders:
        yield prob.driver.recorders
    for s in prob.root.subsystems(recurse=True, include_self=True):
        yield s
        if isinstance(s, Group):
            yield s.ln_solver
            yield s.nl_solver
            if s.ln_solver.recorders._recorders:
                yield s.ln_solver.recorders
            if s.nl_solver.recorders._recorders:
                yield s.nl_solver.recorders

def _setup_profiling(problem):
    """
    Create the profile data output file and instrument the methods to be
    profiled.  Does nothing unless activate_profiling() has been called.
    """
    global _profile_out, _profile_prefix, _profile_by_class, _profile_on
    global _profile_funcs_dict, _profile_start

    if _profile_on:
        rank = MPI.COMM_WORLD.rank if MPI else 0
        _profile_out = open("%s.%d" % (_profile_prefix, rank), 'wb')

        atexit.register(_write_funcs_dict)

        rootsys = problem.root

        # wrap a bunch of methods for profiling
        for obj in _prof_iter(problem):
            for meth, classes in iteritems(_profile_methods):
                if isinstance(obj, classes):
                    match = getattr(obj, meth, None)
                    if match is not None:
                        setattr(obj, meth, profile()(match).__get__(obj,
                                                                obj.__class__))
        _profile_start = time.time()

class profile(object):
    """ Use as a decorator on functions that should be profiled.
    The data collected will include time elapsed, number of calls, ...
    """
    _call_stack = []

    def __init__(self):
        pass

    def __call__(self, fn):
        global _profile_out, _profile_by_class, _profile_buff, _profile_struct
        # don't actually wrap the function unless OPENMDAO_PROFILE is set
        if _profile_on is not None:
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
                    caller = stack[-1]
                else:
                    caller = ''

                stack.append(name)

                path = ','.join(stack)

                if path not in _profile_funcs_dict:
                    # save the id for this path
                    _profile_funcs_dict[path] = len(_profile_funcs_dict)

                start = time.time()
                ret = fn(*args[1:], **kwargs)
                end = time.time()

                stack.pop()

                _profile_struct.t = end - start
                _profile_struct.tstamp = start
                _profile_struct.id = _profile_funcs_dict[path]
                _profile_out.write(_profile_struct)

                return ret

            return wrapper
        return fn

def _update_counts(dct, name, elapsed):
    if name not in dct:
        dct[name] = {
                'count': 1,
                'time': elapsed,
            }
    else:
        d = dct[name]
        d['count'] += 1
        d['time'] += elapsed

def _get_dict(path, funcs, totals):
    parts = path.split(',')
    name = parts[-1]
    fdict = funcs[path]
    tdict = totals[name]

    return {
        'name': name,
        'children': [],
        'time': fdict['time'],
        'tot_time': tdict['time'],
        'count': fdict['count'],
        'tot_count': tdict['count'],
    }


def process_profile(flist):
    """Take the generated raw profile data, potentially from multiple files,
    and combine it to get hierarchy structure and total execution counts and
    timing data.

    Args
    ----

    flist : list of str
        Names of raw profiling data files.

    """

    funcs = {}
    totals = {}
    tops = set()
    fdict = {}

    nfiles = len(flist)

    for fname in flist:
        ext = os.path.splitext(fname)[1]
        try:
            extval = int(ext.lstrip('.'))
            dec = ext
        except:
            dec = False

        for t, tstamp, funcpath in _iter_raw_prof_file(fname, fdict):
            parts = funcpath.split(',')

            # for multi-file MPI profiles, decorate names with the rank
            if nfiles > 1 and dec:
                parts = ["%s%s" % (p,dec) for p in parts]
                funcpath = ','.join(parts)

            name = parts[-1]

            elapsed = float(t)

            _update_counts(totals, name, elapsed)
            _update_counts(funcs, funcpath, elapsed)

            stack = parts[:-1]
            if not stack:
                tops.add(funcpath)

    tree = {
        'name': '',
        'time': sum([totals[n]['time'] for n in tops]),
        'tot_time': fdict['@total'],
        'count': 1,
        'tot_count': 1,
        'children': [],
    }

    tmp = {} # just for temporary lookup of objects

    for path, fdict in sorted(iteritems(funcs)):

        dct = _get_dict(path, funcs, totals)
        tmp[path] = dct

        if path in tops:
            tree['children'].append(dct)
        else:
            parts = path.split(',')
            caller = ','.join(parts[:-1])
            tmp[caller]['children'].append(dct)

    return tree, totals

def prof_totals():
    """Called from the command line to create a file containing total elapsed
    times and number of calls for all profiled functions.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--outfile', action='store', dest='outfile',
                        metavar='OUTFILE', default='sys.stdout',
                        help='Name of file containing function total counts and elapsed times.')
    parser.add_argument('rawfiles', metavar='rawfile', nargs='*',
                        help='File(s) containing raw profile data to be processed. Wildcards are allowed.')

    options = parser.parse_args()

    if not options.rawfiles:
        print("No files to process.")
        sys.exit(0)

    if options.outfile == 'sys.stdout':
        out_stream = sys.stdout
    else:
        out_stream = open(options.outfile, 'w')

    _, totals = process_profile(options.rawfiles)

    try:
        out_stream.write("Function Name, Total Time, Calls\n")
        for func, data in sorted(((k,v) for k,v in iteritems(totals)),
                                    key=lambda x:x[1]['time'],
                                    reverse=True):
            out_stream.write("%s, %s, %s\n" %
                               (func, data['time'], data['count']))
    finally:
        if out_stream is not sys.stdout:
            out_stream.close()

def viewprof():
    """Called from a command line to generate an html viewer for profile data."""

    parser = argparse.ArgumentParser()
    parser.add_argument('--show', action='store_true', dest='show',
                        help="Pop up a browser to view the data.")
    parser.add_argument('rawfiles', metavar='rawfile', nargs='*',
                        help='File(s) containing raw profile data to be processed. Wildcards are allowed.')

    options = parser.parse_args()

    if not options.rawfiles:
        print("No files to process.")
        sys.exit(0)

    call_graph, totals = process_profile(options.rawfiles)

    viewer = 'sunburst.html'
    code_dir = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(code_dir, viewer), "r") as f:
        template = f.read()

    graphjson = json.dumps(call_graph)

    outfile = 'profile_sunburst.html'
    with open(outfile, 'w') as f:
        s = template.replace("<call_graph_data>", graphjson)
        f.write(s)

    if options.show:
        webview(outfile)
