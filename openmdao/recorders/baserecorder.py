from fnmatch import fnmatch

from openmdao.core.options import OptionsDictionary
from six.moves import filter

class BaseRecorder(object):
    """ Base class for all case recorders. """

    def __init__(self):
        self.options = OptionsDictionary()
        self.options.add_option('includes', ['*'], desc='Patterns for variables to include in recording')
        self.options.add_option('excludes', [], desc='Patterns for variables to exclude from recording '
                                '(processed after includes)')

        self._filtered = {}

    def startup(self, group):
        """ Prepare for new run. """

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

        """
        # Coord will look like ['Driver', (1,), '', (1,), 'G1', (1,1), ...]
        # So the pathname is composed of every other entry, starting with the third
        pathname = '.'.join(metadata['coord'][4::2])
        pnames, unames, rnames = self._filtered[pathname]
        filtered_params = {key: params[key] for key in pnames}
        filtered_unknowns = {key: unknowns[key] for key in unames}
        filtered_resids = {key: resids[key] for key in rnames}
        self.record(filtered_params, filtered_unknowns, filtered_resids, metadata)

    def record(self, params, unknowns, resids, metadata):
        raise NotImplementedError("record")
