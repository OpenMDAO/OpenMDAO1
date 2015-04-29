
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

        umap = {}
        for u in parent_vm.unknowns.keys():
            parts = u.split(':',1)
            if parts[0] == name and parts[1] in outputs:
                umap[u] = parts[1]
            elif u in promotes and u in outputs:
                umap[u] = u

        for u in parent_vm.unknowns.keys():
            parts = u.split(':',1)
            if parts[0] == name and parts[1] in states:
                umap[u] = parts[1]
            elif u in promotes and u in states:
                umap[u] = u

        pmap = {}
        for p in parent_vm.params.keys():
            parts = p.split(':',1)
            if parts[0] == name and parts[1] in params:
                pmap[p] = parts[1]
            elif p in promotes and p in params:
                pmap[p] = p

        self.unknowns  = parent_vm.unknowns.get_view(umap)
        self.dunknowns = parent_vm.dunknowns.get_view(umap)
        self.resids    = parent_vm.resids.get_view(umap)
        self.dresids   = parent_vm.dresids.get_view(umap)
        self.params    = parent_vm.params.get_view(pmap, copy=True)
        self.dparams   = parent_vm.dparams.get_view(pmap, copy=True)
