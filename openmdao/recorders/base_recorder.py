""" Class definition for BaseRecorder, the base class for all recorders."""

from types import MethodType
from fnmatch import fnmatch
import sys

from six.moves import filter
from six import StringIO

from openmdao.util.options import OptionsDictionary

class BaseRecorder(object):
    """ This is a base class for all case recorders and is not a functioning
    case recorder on its own.

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

    def __init__(self):
        self.options = OptionsDictionary()
        self.options.add_option('record_metadata', True)
        self.options.add_option('record_unknowns', True)
        self.options.add_option('record_params', False)
        self.options.add_option('record_resids', False)
        self.options.add_option('record_derivs', True,
                                desc='Set to True to record derivatives at the driver level')
        self.options.add_option('includes', ['*'],
                                desc='Patterns for variables to include in recording')
        self.options.add_option('excludes', [],
                                desc='Patterns for variables to exclude from recording '
                                '(processed after includes)')
        self.out = None

        # This is for drivers to determine if a recorder supports
        # real parallel recording (recording on each process), because
        # if it doesn't, the driver figures out what variables must
        # be gathered to rank 0 if running under MPI.
        #
        # By default, this is False, but it should be set to True
        # if the recorder will record data on each process to avoid
        # unnecessary gathering.
        self._parallel = False

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

    def _get_pathname(self, iteration_coordinate):
        '''
        Converts an iteration coordinate to key to index
        `_filtered` to retrieve names of variables to be recorder
        '''
        return '.'.join(iteration_coordinate[4::2])

    def _filter_vectors(self, params, unknowns, resids, iteration_coordinate):
        '''
        Returns subset of `params`, `unknowns` and `resids` to be recoder
        '''
        pathname = self._get_pathname(iteration_coordinate)
        pnames, unames, rnames = self._filtered[pathname]

        params = {key: params[key] for key in pnames}
        unknowns = {key: unknowns[key] for key in unames}
        resids = {key: resids[key] for key in rnames}

        return params, unknowns, resids

    def record_metadata(self, group):
        """Writes the metadata of the given group

        Args
        ----
        group : `System`
            `System` containing vectors
        """
        raise NotImplementedError()

    def record_iteration(self, params, unknowns, resids, metadata):
        """
        Writes the provided data.

        Args
        ----
        params : dict
            Dictionary containing parameters. (p)

        unknowns : dict
            Dictionary containing outputs and states. (u)

        resids : dict
            Dictionary containing residuals. (r)

        metadata : dict, optional
            Dictionary containing execution metadata (e.g. iteration coordinate).
        """
        raise NotImplementedError()

    def record_derivatives(self, derivs, metadata):
        """Writes the metadata of the given group

        Args
        ----
        derivs : dict
            Dictionary containing derivatives

        metadata : dict, optional
            Dictionary containing execution metadata (e.g. iteration coordinate).
        """
        raise NotImplementedError()

    def close(self):
        """Closes `out` unless it's ``sys.stdout``, ``sys.stderr``, or StringIO.
        Note that a closed recorder will do nothing in :meth:`record`, and
        closing a closed recorder also does nothing.
        """
        # Closing a StringIO deletes its contents.
        if self.out not in (None, sys.stdout, sys.stderr):
            if not isinstance(self.out, StringIO):
                self.out.close()
            self.out = None
