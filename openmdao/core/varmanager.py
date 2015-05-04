
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

    def _setup_scatters(self):
        pass


class VarManager(VarManagerBase):
    def __init__(self, params, unknowns, my_params, connections):
        self.unknowns  = VecWrapper.create_source_vector(unknowns, store_noflats=True)
        self.dunknowns = VecWrapper.create_source_vector(unknowns)
        self.resids    = VecWrapper.create_source_vector(unknowns)
        self.dresids   = VecWrapper.create_source_vector(unknowns)
        self.params    = VecWrapper.create_target_vector(params, self.unknowns, my_params, connections, store_noflats=True)
        self.dparams   = VecWrapper.create_target_vector(params, self.unknowns, my_params, connections)


class VarViewManager(VarManagerBase):
    def __init__(self, parent_vm, sys_pathname, params, unknowns,
                 param_owners, connections):

        # parent_vm.unknowns is keyed on name relative to the parent system/varmanager
        # unknowns is keyed on absolute pathname

        umap = {}
        for rel, meta in parent_vm.unknowns.items():
            abspath = meta['pathname']
            if abspath.startswith(sys_pathname+':'):
                newmeta = unknowns.get(abspath)
                if newmeta is not None:
                    newrel = newmeta['relative_name']
                else:
                    newrel = rel
                umap[rel] = newrel

        self.unknowns  = parent_vm.unknowns.get_view(umap)
        self.dunknowns = parent_vm.dunknowns.get_view(umap)
        self.resids    = parent_vm.resids.get_view(umap)
        self.dresids   = parent_vm.dresids.get_view(umap)
        self.params    = VecWrapper.create_target_vector(params, self.unknowns, param_owners, connections, store_noflats=True)
        self.dparams   = VecWrapper.create_target_vector(params, self.unknowns, param_owners, connections)
