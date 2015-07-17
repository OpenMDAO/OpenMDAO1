""" Class definition for BaseRecorder, the base class for all recorders."""

from fnmatch import fnmatch
import sys

from six.moves import filter
from six import StringIO

from openmdao.core.options import OptionsDictionary


class BaseRecorder(object):
    """ Base class for all case recorders. """

    def __init__(self):
        self.options = OptionsDictionary()
        self.options.add_option('includes', ['*'],
                                desc='Patterns for variables to include in recording')
        self.options.add_option('excludes', [],
                                desc='Patterns for variables to exclude from recording '
                                '(processed after includes)')

        self.out = None

        self._filtered = {}
        # TODO: System specific includes/excludes

    def startup(self, group):
        """ Prepare for a new run.

        Args
        ----
        group : `Group`
            Group that owns this recorder.
        """

        # Compute the inclusion lists for recording
        params = list(filter(self._check_path, group.params))
        unknowns = list(filter(self._check_path, group.unknowns))
        resids = list(filter(self._check_path, group.resids))

        self._filtered[group.pathname] = (params, unknowns, resids)

    def _check_path(self, path):
        """ Return True if `path` should be recorded. """

        includes = self.options['includes']
        excludes = self.options['excludes']

        # First see if it's included
        for pattern in includes:
            if fnmatch(path, pattern):
                # We found a match. Check to see if it is excluded.
                for ex_pattern in excludes:
                    if fnmatch(path, ex_pattern):
                        return False
                return True

        # Did not match anything in includes.
        return False

    def raw_record(self, params, unknowns, resids, metadata):
        """
        This is the method that drivers and solvers will call during their
        execution to record their current state. This method is responsible
        for filtering the provided data to reflect the includes/excludes
        provided by the user and then calling `self.record`.

        Recorder subclasses should override `record`, altering this function
        should not be necessary.

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

        # Coord will look like ['Driver', (1,), 'root', (1,), 'G1', (1,1), ...]
        # So the pathname is every other entry, starting with the fifth.
        pathname = '.'.join(metadata['coord'][4::2])
        pnames, unames, rnames = self._filtered[pathname]
        filtered_params = {key: params[key] for key in pnames}
        filtered_unknowns = {key: unknowns[key] for key in unames}
        filtered_resids = {key: resids[key] for key in rnames}
        self.record(filtered_params, filtered_unknowns, filtered_resids, metadata)

    def record(self, params, unknowns, resids, metadata):
        """ Records the requested variables. This method must be defined in
        all recorders.

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
        raise NotImplementedError("record")

    def close(self):
        """Closes `out` unless it's ``sys.stdout``, ``sys.stderr``, or StringIO.
        Note that a closed recorder will do nothing in :meth:`record`."""
        # Closing a StringIO deletes its contents.
        if self.out not in (None, sys.stdout, sys.stderr):
            if not isinstance(self.out, StringIO):
                self.out.close()
            self.out = None
