from fnmatch import fnmatch

from openmdao.core.options import OptionsDictionary

class _BaseRecorder(object):
    """ Base class for all case recorders. """

    def __init__(self, driver):
        self.driver = driver
        self.options = OptionsDictionary()
        self.options.add_option('includes', ['*'], desc='Patterns for variables to include in recording')
        self.options.add_option('excludes', [], desc='Patterns for variables to exclude from recording '
                                '(processed after includes)')
        self.options.add_option('save_problem_formulation', True, desc='Save problem formulation '
                                               '(parameters, constraints, etc.)')

    def startup(self):
        """ Prepare for new run. """

        # In Classic, only CSV recorder did anything in this method.
        # All the others did opening of files, etc... in their __init__
        raise NotImplementedError("startup")

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
        record = False

        includes = self.options['includes']
        excludes = self.options['excludes']
        
        # first see if it's included
        for pattern in includes:
            if fnmatch(path, pattern):
                record = True

        # if it passes include filter, check exclude filter
        if record:
            for pattern in excludes:
                if fnmatch(path, pattern):
                    record = False

        return record

    def record(self, params, unknowns, resids):
        raise NotImplementedError("record")

