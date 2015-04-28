
from openmdao.core.vecwrapper import VecWrapper

class VarManagerBase(object):
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

    def setup_scatters(self):
        pass


class VarManager(VarManagerBase):
    def __init__(self, params, outputs, states):
        self.unknowns = VecWrapper.create_source_vector(outputs, states, store_noflats=True)
        self.dunknowns = VecWrapper.create_source_vector(outputs, states)
        self.resids = VecWrapper.create_source_vector(outputs, states)
        self.dresids = VecWrapper.create_source_vector(outputs, states)
        self.params = VecWrapper.create_target_vector(params, self.unknowns, store_noflats=True)
        self.dparams = VecWrapper.create_target_vector(params, self.unknowns)


class VarViewManager(VarManagerBase):
    def __init__(self, parent_vm, name,
                 promotes, params, outputs, states):

        ulist = []
        for u in parent_vm.unknowns.keys():
            parts = u.split(':',1)
            if parts[0] == name and parts[1] in outputs:
                ulist.append((u,parts[1]))
            elif u in promotes and u in outputs:
                ulist.append((u,u))

        for u in parent_vm.unknowns.keys():
            parts = u.split(':',1)
            if parts[0] == name and parts[1] in states:
                ulist.append((u,parts[1]))
            elif u in promotes and u in states:
                ulist.append((u,u))

        self.unknowns  = parent_vm.unknowns.get_view(ulist)
        self.dunknowns = parent_vm.dunknowns.get_view(ulist)
        self.resids    = parent_vm.resids.get_view(ulist)
        self.dresids   = parent_vm.dresids.get_view(ulist)
        self.params    = VecWrapper.create_target_vector(params, self.unknowns)
        self.dparams   = VecWrapper.create_target_vector(params, self.unknowns)
