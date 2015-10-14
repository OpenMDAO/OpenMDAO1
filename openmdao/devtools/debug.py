"""functions useful for debugging openmdao"""
from __future__ import print_function

import sys
from pprint import pformat
from functools import wraps
from resource import getrusage, RUSAGE_SELF, RUSAGE_CHILDREN


def dump_meta(system, nest=0, out_stream=sys.stdout):
    """
    Dumps the system tree with associated metadata for the params and unknowns
    `VecWrappers`.

    Args
    ----
    system : `System`
        The node in the `System` tree where dumping begins.

    nest : int, optional
        Starting nesting level.  Defaults to 0.

    out_stream : file-like, optional
        Where output is written.  Defaults to sys.stdout.

    """
    klass = system.__class__.__name__

    commsz = system.comm.size if hasattr(system.comm, 'size') else 0

    margin = ' '*nest
    if system.is_active():
        out_stream.write("%s %s '%s'    req: %s  usize:%d  psize:%d  commsize:%d\n" %
                         (margin,
                          klass,
                          system.name,
                          system.get_req_procs(),
                          system.unknowns.vec.size,
                          system.params.vec.size,
                          commsz))

        margin = ' '*(nest+6)
        out_stream.write("%sunknowns:\n" % margin)
        for v, meta in system.unknowns.items():
            out_stream.write("%s%s: " % (margin, v))
            out_stream.write(pformat(meta, indent=nest+9).replace("{","{\n",1))
            out_stream.write('\n')

        out_stream.write("%sparams:\n" % margin)
        for v, meta in system.params.items():
            out_stream.write("%s%s: " % (margin, v))
            out_stream.write(pformat(meta, indent=nest+9).replace("{","{\n",1))
            out_stream.write('\n')
    else:
        out_stream.write("%s %s '%s'   (inactive)\n" %
                         (margin, klass, system.name))

    nest += 3
    for sub in itervalues(system._subsystems):
        sub.dump_meta(nest, out_stream=out_stream)

    out_stream.flush()

def max_mem_usage():
    """
    Returns
    -------
    The max memory used by this process and its children, in MB.
    """
    denom = 1024.
    if sys.platform == 'darwin':
        denom *= denom
    total = getrusage(RUSAGE_SELF).ru_maxrss / denom
    total += getrusage(RUSAGE_CHILDREN).ru_maxrss / denom
    return total


def diff_max_mem(fn):
    """
    This gives the difference in max memory before and after the
    decorated function is called.  Results can sometimes be
    deceptive since it only deals with max memory, i.e., the
    value coming back from getrusage never goes down, even if
    memory is freed up.
    """
    @wraps(fn)
    def wrapper(*args, **kwargs):
        startmem = max_mem_usage()
        ret = fn(*args, **kwargs)
        diff = max_mem_usage()-startmem
        if diff > 0.0:
            print("%s added %s MB" % (fn.__name__, diff))
        return ret
    return wrapper
