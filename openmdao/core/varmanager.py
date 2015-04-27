
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
    def __init__(self, params, unknowns, states):
        self.unknowns = SourceVecWrapper(unknowns, states, store_noflats=True)
        self.dunknowns = SourceVecWrapper(unknowns, states)
        self.resids = SourceVecWrapper(unknowns, states)
        self.dresids = SourceVecWrapper(unknowns, states)
        self.params = TargetVecWrapper(params, self.unknowns, store_noflats=True)
        self.dparams = TargetVecWrapper(params, self.unknowns)


class VarViewManager(VarManagerBase):
    def __init__(self, parent_vm, name,
                 promotes, params, unknowns, states):
        ulist = []
        for u in parent_vm.unknowns.keys():
            parts = u.split(':',1)
            if parts[0] == name and parts[1] in unknowns:
                ulist.append((u,parts[1]))
            elif u in promotes and u in unknowns:
                ulist.append((u,u))

        for u in parent_vm.states.keys():
            parts = u.split(':',1)
            if parts[0] == name and parts[1] in states:
                ulist.append((u,parts[1]))
            elif u in promotes and u in states:
                ulist.append((u,u))

        self.unknowns = parent_vm.unknowns.get_view(ulist)
        self.dunknowns = parent_vm.dunknowns.get_view(ulist)
        self.resids = parent_vm.resids.get_view(ulist)
        self.dresids = parent_vm.dresids.get_view(ulist)
        self.params = TargetVecWrapper(params, self.unknowns, store_noflats=True)
        self.dparams = TargetVecWrapper(params, self.unknowns)
        
