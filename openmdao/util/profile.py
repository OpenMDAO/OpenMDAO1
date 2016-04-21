import os
import sys
import time
import inspect
import fnmatch
from functools import wraps

import types

from six import iteritems

from openmdao.core.mpi_wrap import MPI
from openmdao.core.group import Group

def get_method_class(meth):
    """Return the class that actually defined the given method."""
    for cls in inspect.getmro(meth.__self__.__class__):
        if meth.__name__ in cls.__dict__:
            return cls

# list of default methods to profile
_profile_methods = [
    "solve_nonlinear",
    "solve_linear",
    "apply_linear",
    "solve",
    "fd_jacobian",
    "linearize",
]

_profile_prefix = None
_profile_out = None

def activate_profiling(prefix='prof_out', methods=None):
    """Turns on profiling of certain important openmdao methods.

    Args
    ----

    prefix : str ('prof_out')
        Prefix used for the raw profile data. Process rank will be appended
        to it to get the actual filename.  When not using MPI, rank=0.

    methods : list of str, optional
        A list of profiled methods to override the default set.  Method names
        should be simple names like "solve", or "fd_jacobian" an should not
        include a class name.  The default set of methods is:
        ["solve_nonlinear", "solve_linear", "apply_linear", "solve",
         "fd_jacobian", "linearize"]
    """
    global _profile_prefix, _profile_methods
    _profile_prefix = prefix

    if methods:
        _profile_methods = methods

def _setup_profiling(rootsys):
    """
    Create the profile data output file and instrument the methods to be
    profiled.  Does nothing unless activate_profiling() has been called.
    """
    global _profile_out, _profile_prefix

    if _profile_prefix:
        if MPI:
            rank = MPI.COMM_WORLD.rank
        else:
            rank = 0
        _profile_out = open("%s.%d" % (_profile_prefix, rank), 'w')
        _profile_out.write(','.join(['class','pathname','funcname','elapsed_time']))
        _profile_out.write('\n')

    if _profile_out is not None:
        # wrap a bunch of methods for profiling
        for s in rootsys.subsystems(recurse=True, include_self=True):
            if isinstance(s, Group):
                objs = (s, s.ln_solver, s.nl_solver)
            else:
                objs = (s,)
            for obj in objs:
                for meth in _profile_methods:
                    if hasattr(obj, meth):
                        setattr(obj, meth, profile()(getattr(obj, meth)).__get__(obj, obj.__class__))

class profile(object):
    """ Use as a decorator on functions that should be profiled.
    The data collected will include time elapsed, number of calls, ...
    """
    def __init__(self):
        pass

    def __call__(self, fn):
        global _profile_out
        # don't actually wrap the function unless OPENMDAO_PROFILE is set
        if _profile_out is not None:
            @wraps(fn)
            def wrapper(*args, **kwargs):
                start = time.time()

                ret = fn(*args[1:], **kwargs)

                end = time.time()

                try:
                    klass = get_method_class(fn).__name__
                except AttributeError:
                    klass = ''

                try:
                    path = fn.__self__.pathname
                except AttributeError:
                    path = "<%s>" % args[0].__class__.__name__

                data = [
                    klass,
                    path,
                    fn.__name__,
                    str(end-start),
                ]

                _profile_out.write(','.join(data))
                _profile_out.write('\n')
                _profile_out.flush()

                return ret

            return wrapper
        return fn

def process_profile(prof=None, by_instance=True):
    """Take the generated profile data, potentially from multiple files,
    and combine it to get execution counts.

    Args
    ----

    prof : str or None
        Name of profile data file.  Can contain wildcards to process multiple
        profiles together, e.g., when MPI is used.

    by_instance : bool (True)
        If True, group result data by instance.  Otherwise group by class.
    """

    if prof is None:
        prof = _profile_prefix
        prof += '.*'

    results = {}
    for fname in fnmatch.filter(os.listdir('.'), prof):
        with open(fname, 'r') as f:
            for i, line in enumerate(f):
                if i==0:
                    continue

                klass, path, func, elapsed = line.split(',')
                elapsed = float(elapsed)

                if i==0:
                    continue

                if by_instance:
                    name = '.'.join((path, func))
                else:
                    name = '.'.join((klass, func))
                if name not in results:
                    results[name] = {
                        'count': 0,
                        'tot_time': 0.,
                        'max_time': 0.,
                        'min_time': 9999999.
                    }
                res = results[name]
                res['count'] += 1
                res['tot_time'] += elapsed
                res['max_time'] = max(res['max_time'], elapsed)
                res['min_time'] = min(res['min_time'], elapsed)

    return results

def viewprof():
    """Called from a command line to process profile data files."""
    args = sys.argv[1:]
    prof = args[0]

    by_instance = not ('byclass' in args)
    out_stream = sys.stdout

    results = process_profile(prof, by_instance)

    out_stream.write("Function Name, Total Time, Max Time, Min Time, Calls\n")
    for func, data in sorted(((k,v) for k,v in iteritems(results)),
                                key=lambda x:x[1]['tot_time'],
                                reverse=True):

        out_stream.write("%s, %s, %s, %s, %s\n" %
                           (func, data['tot_time'], data['max_time'],
                            data['min_time'], data['count']))
