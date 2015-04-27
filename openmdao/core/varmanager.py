
class VarManager(object):
    """A manager of the data transfer of a possibly distributed
    collection of variables.
    """
    def __init__(self):
        self.params = None
        self.dparams = None
        self.unknowns = None
        self.dunknowns = None
        self.resids = None
        self.dresids = None


class ProblemVarManager(VarManager):
    def __init__(self, params, unknowns, states):
        pass


class GroupVarManager(VarManager):
    def __init__(self, vm, path):
        pass
