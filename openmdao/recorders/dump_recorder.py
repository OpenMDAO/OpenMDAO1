""" Class definition for DumpRecorder, a recorder that prints
human-readable text output to a stream."""

import sys

from six import string_types, iteritems

from openmdao.core.mpi_wrap import MPI
from openmdao.recorders.base_recorder import BaseRecorder
from openmdao.util.record_util import format_iteration_coordinate

class DumpRecorder(BaseRecorder):
    """Dumps cases in a "pretty" form to `out`, which may be a string or a
    file-like object (defaults to ``stdout``). If `out` is ``stdout`` or
    ``stderr``, then that standard stream is used. Otherwise, if `out` is a
    string, then a file with that name will be opened in the current
    directory. If `out` is None, cases will be ignored. When called under
    MPI, the dumprecorder writes to a separate file for each rank, with the
    rank number appended to each filename. In this case, only variables that
    exist on all processes can be printed.

    Args
    ----
    out : stream
        Output stream or file name to write the data.

    Options
    -------
    options['record_metadata'] :  bool(True)
        Tells recorder whether to record variable attribute metadata.
    options['record_unknowns'] :  bool(True)
        Tells recorder whether to record the unknowns vector.
    options['record_params'] :  bool(False)
        Tells recorder whether to record the params vector.
    options['record_resids'] :  bool(False)
        Tells recorder whether to record the ressiduals vector.
    options['record_derivs'] :  bool(True)
        Tells recorder whether to record derivatives that are requested by a `Driver`.
    options['includes'] :  list of strings
        Patterns for variables to include in recording.
    options['excludes'] :  list of strings
        Patterns for variables to exclude in recording (processed after includes).
    """

    def __init__(self, out='stdout'):
        super(DumpRecorder, self).__init__()
        self._parallel = True

        if isinstance(out, string_types):

            if out == 'stdout':
                out = sys.stdout

            elif out == 'stderr':
                out = sys.stderr

            else:
                # Dump to separate file for each process if we are under MPI
                if MPI:
                    if '.' in out:
                        parts = out.split('.')
                        parts[-2] += '_' + str(MPI.COMM_WORLD.rank)
                        out = '.'.join(parts)
                    else:
                        out += str(MPI.COMM_WORLD.rank)

                out = open(out, 'w')

        self.out = out

    def record_metadata(self, group):
        """Dump the metadata of the given group in a "pretty" form.

        Args
        ----
        group : `System`
            `System` containing vectors
        """
        params = list(iteritems(group.params))
        unknowns = list(iteritems(group.unknowns))

        self.out.write("Metadata:\n")
        self.out.write("Params:\n")

        for name, metadata in params:
            fmat = "  {0}: {1}\n"
            self.out.write(fmat.format(name, metadata))

        self.out.write("Unknowns:\n")

        for name, metadata in unknowns:
            fmat = "  {0}: {1}\n"
            self.out.write(fmat.format(name, metadata))

    def record_iteration(self, params, unknowns, resids, metadata):
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

        iteration_coordinate = metadata['coord']
        timestamp = metadata['timestamp']

        write = self.out.write
        fmat = "Timestamp: {0!r}\n"
        write(fmat.format(timestamp))

        fmat = "Iteration Coordinate: {0:s}\n"
        write(fmat.format(format_iteration_coordinate(iteration_coordinate)))

        self._write_success_info(metadata)

        if self.options['record_params']:
            write("Params:\n")
            for param, val in sorted(iteritems(self._filter_vector(params,
                                                   'p',iteration_coordinate))):
                write("  {0}: {1}\n".format(param, str(val)))

        if self.options['record_unknowns']:
            write("Unknowns:\n")
            for unknown, val in sorted(iteritems(self._filter_vector(unknowns,
                                                   'u',iteration_coordinate))):
                write("  {0}: {1}\n".format(unknown, str(val)))

        if self.options['record_resids']:
            write("Resids:\n")
            for resid, val in sorted(iteritems(self._filter_vector(resids,
                                                   'r',iteration_coordinate))):
                write("  {0}: {1}\n".format(resid, str(val)))

        # Flush once per iteration to allow external scripts to process the data.
        self.out.flush()

    def _write_success_info(self, metadata):
        """Writes 'Iteration succeeded: yes/no', followed by the error msg
        if latest iteration didn't succeed.
        """
        self.out.write("Iteration succeeded: %s\n" %
                           'yes' if metadata['success'] else 'no')
        if not metadata['success']:
            self.out.write("Error msg: %s\n" % metadata['msg'])

    def record_derivatives(self, derivs, metadata):
        """Writes the derivatives that were calculated for the driver.

        Args
        ----
        derivs : dict
            Dictionary containing derivatives

        metadata : dict, optional
            Dictionary containing execution metadata (e.g. iteration coordinate).
        """

        iteration_coordinate = metadata['coord']
        timestamp = metadata['timestamp']

        write = self.out.write
        fmat = "Timestamp: {0!r}\n"
        write(fmat.format(timestamp))

        fmat = "Iteration Coordinate: {0:s}/Derivs\n"
        write(fmat.format(format_iteration_coordinate(iteration_coordinate)))

        self._write_success_info(metadata)

        write("Derivatives:\n")
        if isinstance(derivs, dict):
            for okey, sub in sorted(iteritems(derivs)):
                for ikey, deriv in sorted(iteritems(sub)):
                    write("  {0} wrt {1}: {2}\n".format(okey, ikey, str(deriv)))
        else:
            write("  {0} \n".format(str(derivs)))

        # Flush once per iteration to allow external scripts to process the data.
        self.out.flush()
