""" Class definition for BaseRecorder, the base class for all recorders."""

from types import MethodType
from fnmatch import fnmatch
import sys

from six.moves import filter
from six import StringIO

from openmdao.core.options import OptionsDictionary

class BaseRecorder(object):
    """ Base class for all case recorders. """

    def __init__(self):
        self.options = OptionsDictionary()
        self.options.add_option('record_metadata', False)
        self.options.add_option('record_unknowns', True)
        self.options.add_option('record_params', False)
        self.options.add_option('record_resids', False)
        self.options.add_option('includes', ['*'])
        self.options.add_option('excludes', [])

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

    def record_metadata(self, group):
        """Writes the metadata of the given group

        Args
        ----
        group : `System`
            `System` containing vectors 
        """
        raise NotImplementedError()

    def close(self):
        """Closes `out` unless it's ``sys.stdout``, ``sys.stderr``, or StringIO.
        Note that a closed recorder will do nothing in :meth:`record`."""
        # Closing a StringIO deletes its contents.
        if self.out not in (None, sys.stdout, sys.stderr):
            if not isinstance(self.out, StringIO):
                self.out.close()
            self.out = None
