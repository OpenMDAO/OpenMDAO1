"""functions useful for debugging openmdao"""

import sys
from pprint import pformat

from openmdao.core.mpiwrap import MPI, under_mpirun


if under_mpirun():
    def debug(*msg):
        newmsg = ["%d: " % MPI.COMM_WORLD.rank] + list(msg)
        for m in newmsg:
            sys.stdout.write("%s " % m)
        sys.stdout.write('\n')
        sys.stdout.flush()
else:
    def debug(*msg):
        for m in msg:
            sys.stdout.write("%s " % str(m))
        sys.stdout.write('\n')


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
    for name, sub in system.subsystems():
        sub.dump_meta(nest, out_stream=out_stream)

    out_stream.flush()
