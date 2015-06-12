from fnmatch import fnmatch

from openmdao.core.options import OptionsDictionary
from six.moves import filter

class BaseRecorder(object):
    """ Base class for all case recorders. """

    def __init__(self, driver):
        self.driver = driver
        self.options = OptionsDictionary()
        self.options.add_option('includes', ['*'], desc='Patterns for variables to include in recording')
        self.options.add_option('excludes', [], desc='Patterns for variables to exclude from recording '
                                '(processed after includes)')
        self.options.add_option('save_problem_formulation', True, desc='Save problem formulation '
                                               '(parameters, constraints, etc.)')

    def startup(self, group):
        """ Prepare for new run. """

        # Compute the inclusion lists for recording
        self._params = list(filter(self._check_path, group.params))
        self._unknowns = list(filter(self._check_path, group.unknowns))
        self._resids = list(filter(self._check_path, group.resids))


    def get_simulation_info(self):
        """ Return simulation info dictionary. """

        # TODO: depgraphs
        openmdao_version = '1.0' # TODO: how do I get this info?
        return dict(OpenMDAO_Version=openmdao_version)

    def get_driver_info(self):
        """ Return list of driver info dictionaries. """

        # TODO: add more info
        class_name = type(self.driver).__name__
        return dict(class_name=class_name)

    def _check_path(self,path):
        """ Return True if `path` should be recorded. """

        includes = self.options['includes']
        excludes = self.options['excludes']

        # first see if it's included
        for pattern in includes:
            if fnmatch(path, pattern):
                # We found a match. Check to see if it is excluded.
                for ex_pattern in excludes:
                    if fnmatch(path, ex_pattern):
                        return False
                return True

        # Did not match anything in includes.
        return False

    def _record(self, params, unknowns, resids):
        filtered_params = {key:params[key] for key in self._params}
        filtered_unknowns = {key:unknowns[key] for key in self._unknowns}
        filtered_resids = {key:resids[key] for key in self._resids}
        self.record(filtered_params, filtered_unknowns, filtered_resids)

    def record(self, params, unknowns, resids):
        raise NotImplementedError("record")
