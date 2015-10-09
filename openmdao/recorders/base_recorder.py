""" Class definition for BaseRecorder, the base class for all recorders."""

from types import MethodType
from fnmatch import fnmatch
import sys

from six.moves import filter
from six import StringIO

from openmdao.core.options import OptionsDictionary
#from openmdao.core.mpi_wrap import MPI

class BaseRecorder(object):
    """ Base class for all case recorders. """
    supported_recorders = []

    def __init__(self):
        self.options = OptionsDictionary()
        
        for recorder in self.supported_recorders:
            self.support(recorder)

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

    def support(self, datatype):
        self.options.update(datatype.options)

        for method in datatype.methods:
            name = method.func_name
            if not hasattr(self, name):
                setattr(self, name, MethodType(method, self))

    @classmethod
    def supported_options(self):
        options = []
        
        for recorder in self.supported_recorders:
            for key, _ in recorder.options.items():
                options.append(key)
            
        return options

    @classmethod
    def supported_methods(self):
        methods = []
        
        for recorder in self.supported_recorders:
            for method in recorder.methods:
                methods.append(method.func_name)
            
        return methods


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

#    def record(self, params, unknowns, resids, metadata):
#        """ Records the requested variables. This method must be defined in
#        all recorders.
#
#        Args
#        ----
#        params : `VecWrapper`
#            `VecWrapper` containing parameters. (p)
#
#        unknowns : `VecWrapper`
#            `VecWrapper` containing outputs and states. (u)
#
#        resids : `VecWrapper`
#            `VecWrapper` containing residuals. (r)
#
#        metadata : dict
#            Dictionary containing execution metadata (e.g. iteration coordinate).
#        """
#        raise NotImplementedError("record")

    def close(self):
        """Closes `out` unless it's ``sys.stdout``, ``sys.stderr``, or StringIO.
        Note that a closed recorder will do nothing in :meth:`record`."""
        # Closing a StringIO deletes its contents.
        if self.out not in (None, sys.stdout, sys.stderr):
            if not isinstance(self.out, StringIO):
                self.out.close()
            self.out = None
