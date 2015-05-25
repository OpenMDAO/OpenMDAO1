from fnmatch import fnmatch

from openmdao.core.options import OptionsDictionary

class _BaseRecorder(object):
    """ Base class for all case recorders. """

    def __init__(self):
        # self._cfg_map = {}
        self._cases = None
        # self.recorders = [DumpCaseRecorder(sout1), DumpCaseRecorder(sout2)]
        self.options = OptionsDictionary()
        self.options.add_option('includes', ['*'], doc='Patterns for variables to include in recording')
        self.options.add_option('excludes', [], doc='Patterns for variables to exclude from recording '
                                '(processed after includes)')
        self.options.add_option('save_problem_formulation', True, doc='Save problem formulation '
                                               '(parameters, constraints, etc.)')

    def startup(self):
        """ Prepare for new run. """

        # In Classic, only CSV recorder did anything in this method.
        # All the others did opening of files, etc... in their __init__
        # TODO: Do we need this method?
        raise NotImplementedError("startup")

    # def register(self, driver, inputs, outputs):
    #     """ Register names for later record call from `driver`. """
    #     self._cfg_map[driver] = (inputs, outputs)

    def get_simulation_info(self, constants):
        """ Return simulation info dictionary. """

        # dep_graph = top.get_graph(format='json')
        # comp_graph = top.get_graph(components_only=True, format='json')

        return dict(OpenMDAO_Version=__version__)

    def get_driver_info(self):
        """ Return list of driver info dictionaries. """

        # Locate top level assembly from first driver registered.
        top = self._cfg_map.keys()[0].parent
        while top.parent:
            top = top.parent
        #prefix_drop = len(top.name) + 1 if top.name else 0
        prefix_drop = 0

        driver_info = []
        # for driver, (ins, outs) in sorted(self._cfg_map.items(),
        #                                   key=lambda item: item[0].get_pathname()):
        #     name = driver.get_pathname()[prefix_drop:]
        #     info = dict(name=name, _id=id(driver), recording=ins+outs)
        #     if hasattr(driver, 'get_parameters'):
        #         info['parameters'] = \
        #             [str(param) for param in driver.get_parameters().values()]
        #     if hasattr(driver, 'eval_objectives'):
        #         info['objectives'] = \
        #             [key for key in driver.get_objectives()]
        #     if hasattr(driver, 'eval_responses'):
        #         info['responses'] = \
        #             [key for key in driver.get_responses()]
        #     if hasattr(driver, 'get_eq_constraints'):
        #         info['eq_constraints'] = \
        #             [str(con) for con in driver.get_eq_constraints().values()]
        #     if hasattr(driver, 'get_ineq_constraints'):
        #         info['ineq_constraints'] = \
        #             [str(con) for con in driver.get_ineq_constraints().values()]
        #     driver_info.append(info)
        return driver_info

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

    # def get_case_info(self, driver, inputs, outputs, exc):
    #     """ Return case info dictionary. """
    #     in_names, out_names = self._cfg_map[driver]

    #     scope = driver.parent
    #     prefix = scope.get_pathname()
    #     if prefix:
    #         prefix += '.'
    #     in_names = [prefix+name for name in in_names]
    #     out_names = [prefix+name for name in out_names]

    #     data = dict(zip(in_names, inputs))
    #     data.update(zip(out_names, outputs))

    #     #subdriver_last_case_uuids = {}
    #     #for subdriver in driver.subdrivers():
    #         #subdriver_last_case_uuids[ id(subdriver) ] = self._last_child_case_uuids[ id(subdriver) ]
    #     #self._last_child_case_uuids[ id(driver) ] = case_uuid


    #     return dict(_id=case_uuid,
    #                 _parent_id=parent_uuid or self._uuid,
    #                 _driver_id=id(driver),
    #                 #subdriver_last_case_uuids = subdriver_last_case_uuids,
    #                 error_status=None,
    #                 error_message=str(exc) if exc else '', timestamp=time.time(),
    #                 data=data)


