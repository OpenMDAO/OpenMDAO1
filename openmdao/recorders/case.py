class Case(object):
    """ Case wraps the data from a single iteration/case of a recording
    to make it more easily accessible to the user.
    """

    def __init__(self, filename, case_id, case_dict):
        self.filename = filename
        self.case_id = case_id

        self.timestamp = case_dict.get('timestamp', None)
        self.success = case_dict.get('success', None)
        self.msg = case_dict.get('msg', None)

        self.parameters = case_dict.get('Parameters', None)
        self.unknowns = case_dict.get('Unknowns', None)
        self.derivs = case_dict.get('Derivatives', None)
        self.resids = case_dict.get('Residuals', None)

    def __getitem__(self, item):
        if self.unknowns is None:
            raise ValueError('No unknowns are available'
                             ' in file {0}'.format(self.filename))
        return self.unknowns[item]

    def _to_json(self, filename):
        pass
