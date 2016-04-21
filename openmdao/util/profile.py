import os
import time
import inspect
import fnmatch
from functools import wraps

import types

from openmdao.core.mpi_wrap import MPI

def get_method_class(meth):
    """Return the class that actually defined the given method."""
    for cls in inspect.getmro(meth.__self__.__class__):
        if meth.__name__ in cls.__dict__:
            return cls

# list of default methods to profile
_methods = [
    "solve_nonlinear",
    "solve_linear",
    "apply_linear",
]

_profile_prefix = os.environ.get("OPENMDAO_PROFILE")
_profile_prefix = 'prof_out'

# to override the default method list, define this env var
_profile_methods = os.environ.get("OPENMDAO_PROFILE_METHODS")
if _profile_methods is None:
    _profile_methods = _methods
else:
    _profile_methods = [m.strip() for m in _profile_methods.split(',') if m.strip()]

if _profile_prefix is not None:
    if MPI:
        rank = MPI.COMM_WORLD.rank
    else:
        rank = 0
    _profile_out = open("%s.%d" % (_profile_prefix, rank), 'w')
    _profile_out.write(','.join(['class','pathname','funcname','elapsed_time']))
    _profile_out.write('\n')

def setup_profiling(rootsys):
    if _profile_out is not None:
        # wrap a bunch of methods for profiling
        for s in rootsys.subsystems(recurse=True, include_self=True):
            for meth in _profile_methods:
                if hasattr(s, meth):
                    setattr(s, meth, profile()(getattr(s, meth)).__get__(s, s.__class__))

class profile(object):
    """ Use as a decorator on functions that should be profiled.
    The data collected will include time elapsed, number of calls,
    rank, ...
    """
    def __init__(self):
        pass

    def __call__(self, fn):
        # don't actually wrap the function unless OPENMDAO_PROFILE is set
        if _profile_out is not None:
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
                    path = '<None>'

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

# def process_profile(prof_prefix=None, out_stream=sys.stdout):
#     """Take the generated profile data, potentially from multiple files,
#     and combine it to get execution counts.
#     """
#     if prof_prefix is None:
#         prof_prefix = _profile_prefix
#     prof_prefix += '.*'
#
#     results = {}
#     for fname in fnmatch.filter(os.listdir('.'), prof_prefix):
#         with open(fname, 'r') as f:
#             for i,line in f:
#                 if i==0:
#
