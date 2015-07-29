""" Class definition for DumpRecorder, a recorder that prints
human-readable text output to a stream."""

import sys

from six import string_types

from openmdao.recorders.base_recorder import BaseRecorder
from openmdao.util.record_util import format_iteration_coordinate


class DumpRecorder(BaseRecorder):
    """Dumps cases in a "pretty" form to `out`, which may be a string or a
    file-like object (defaults to ``stdout``). If `out` is ``stdout`` or
    ``stderr``, then that standard stream is used. Otherwise, if `out` is a
    string, then a file with that name will be opened in the current directory.
    If `out` is None, cases will be ignored.
    """

    def __init__(self, out='stdout'):
        super(DumpRecorder, self).__init__()
        if isinstance(out, string_types):
            if out == 'stdout':
                out = sys.stdout
            elif out == 'stderr':
                out = sys.stderr
            else:
                out = open(out, 'w')
        self.out = out

    def startup(self, group):
        """ Write out info that applies to the entire run.

        Args
        ----
        group : `Group`
            Group that owns this recorder.
        """
        super(DumpRecorder, self).startup(group)

    def record(self, params, unknowns, resids, metadata):
        """Dump the given run data in a "pretty" form.

        Args
        ----
        params : `VecWrapper`
            `VecWrapper` containing parameters. (p)

        unknowns : `VecWrapper`
            `VecWrapper` containing outputs and states. (u)

        resids : `VecWrapper`
            `VecWrapper` containing residuals. (r)

        metadata : dict
            Dictionary containing execution metadata (e.g. iteration coordinate).
        """

        if not self.out:  # if self.out is None, just do nothing
            return

        write = self.out.write
        fmat = "Iteration Coordinate: {0:s}\n"
        write(fmat.format(format_iteration_coordinate(metadata['coord'])))

        write("Params:\n")
        for param, val in sorted(params.items()):
            write("  {0}: {1}\n".format(param, str(val)))

        write("Unknowns:\n")
        for unknown, val in sorted(unknowns.items()):
            write("  {0}: {1}\n".format(unknown, str(val)))

        write("Resids:\n")
        for resid, val in sorted(resids.items()):
            write("  {0}: {1}\n".format(resid, str(val)))
