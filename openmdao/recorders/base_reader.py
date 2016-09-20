from abc import ABCMeta, abstractmethod, abstractproperty


class Case(object):
    ''' Case wraps the data from a single iteration/case of a recording
    to make it more easily accessible to the user.
    '''

    def __init__(self, case_id, case_dict, derivs_dict):
        self.case_id = case_id

        self.timestamp = case_dict.get('timestamp', None)
        self.success = case_dict.get('success', None)
        self.msg = case_dict.get('msg', None)

        self._parameters = case_dict.get('Parameters', {})
        self._unknowns = case_dict.get('Unknowns', {})
        self._residuals = case_dict.get('Residuals', {})
        self._derivs = derivs_dict.get('Derivatives', {})

    def __getitem__(self, item):
        return self._unknowns[item]

    def _to_json(self, filename):
        pass


class CaseReaderBase(object):

    __metaclass__ = ABCMeta

    def __init__(self):
        self._filename = None

    # @abstractmethod
    # def __getitem__(self, item):
    #     pass

    @abstractmethod
    def get_case(self, case_id):
        pass

    @abstractmethod
    def list_cases(self):
        pass

    @property
    def num_cases(self):
        return len(self.list_cases())
