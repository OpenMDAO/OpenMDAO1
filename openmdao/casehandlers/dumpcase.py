import sys

from six import StringIO, string_types
from six.moves import cStringIO

from openmdao.casehandlers.baserecorder import _BaseRecorder

class DumpCaseRecorder(_BaseRecorder):
    """Dumps cases in a "pretty" form to `out`, which may be a string or a
    file-like object (defaults to ``stdout``). If `out` is ``stdout`` or
    ``stderr``, then that standard stream is used. Otherwise, if `out` is a
    string, then a file with that name will be opened in the current directory.
    If `out` is None, cases will be ignored.
    """

    def __init__(self, driver, out='stdout'):
        super(DumpCaseRecorder, self).__init__(driver)
        if isinstance(out, string_types):
            if out == 'stdout':
                out = sys.stdout
            elif out == 'stderr':
                out = sys.stderr
            else:
                out = open(out, 'w')
        self.out = out

    def startup(self):
        """ Write out info that applies to the entire run"""
        write = self.out.write
        sim_info = self.get_simulation_info()
        write("Simulation Info:\n")
        write("  OpenMDAO Version: %s\n" % sim_info['OpenMDAO_Version'])
        driver_info = self.get_driver_info()
        write("Driver Info:\n")
        write("  Driver Class: %s\n" % driver_info['class_name'])

    def register(self, driver, inputs, outputs):
        """Register names for later record call from `driver`."""
        pass

    def record(self, params, unknowns, resids):
        """Dump the given run data in a "pretty" form."""
        if not self.out:  # if self.out is None, just do nothing
            return

        write = self.out.write

        write("Case:\n")

        #TODO: Need to look at Group.dump to see how it handles this
        write("  Params:\n")
        for param, meta in params.items():
            if self._check_path(param):
                write("%s: %s\n" % ( param, str(meta['val'])))

        write("  Unknowns:\n")
        for unknown, meta in unknowns.items():
            if self._check_path(unknown):
                write("%s: %s\n" % ( unknown, str(meta['val'])))

        write("  Resids:\n")
        for resid, meta in resids.items():
            if self._check_path(resid):
                write("%s: %s\n" % ( resid, str(meta['val'])))


    def close(self):
        """Closes `out` unless it's ``sys.stdout`` or ``sys.stderr``.
        Note that a closed recorder will do nothing in :meth:`record`."""
        if self.out not in (None, sys.stdout, sys.stderr):
            if not isinstance(self.out,
                              (StringIO, cStringIO.OutputType)):
                # Closing a StringIO deletes its contents.
                self.out.close()
            self.out = None

