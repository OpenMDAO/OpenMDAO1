
from openmdao.core.vecwrapper import VecWrapper, get_relative_varname

class VarManagerBase(object):
    """A manager of the data transfer of a possibly distributed
    collection of variables.
    """
    def __init__(self, connections):
        self.params = None
        self.dparams = None
        self.unknowns = None
        self.dunknowns = None
        self.resids = None
        self.dresids = None
        self.connections = connections

    def _setup_scatters(self):
        # nothing to do for non-MPI case???????
        pass

    def _scatter(self, target_system):
        # simple non-MPI version:
        # propagate unknown values to the connected parameters
        for p, pmeta in self.params.items():
            if pmeta['pathname'].startswith(target_system+':'):
                psource = self.connections[p]
                unknown_name = get_relative_varname(psource, self.unknowns)
                self.params[p] = self.unknowns[unknown_name]


class VarManager(VarManagerBase):
    def __init__(self, params_dict, unknowns_dict, my_params, connections):
        super(VarManager, self).__init__(connections)

        self.unknowns  = VecWrapper.create_source_vector(unknowns_dict, store_noflats=True)
        self.dunknowns = VecWrapper.create_source_vector(unknowns_dict)
        self.resids    = VecWrapper.create_source_vector(unknowns_dict)
        self.dresids   = VecWrapper.create_source_vector(unknowns_dict)
        self.params    = VecWrapper.create_target_vector(params_dict, self.unknowns, my_params, connections, store_noflats=True)
        self.dparams   = VecWrapper.create_target_vector(params_dict, self.unknowns, my_params, connections)


class VarViewManager(VarManagerBase):
    def __init__(self, parent_vm, sys_pathname, params_dict, unknowns_dict, param_owners, connections):
        super(VarViewManager, self).__init__(connections)

        # parent_vm.unknowns is keyed on name relative to the parent system/varmanager
        # unknowns_dict is keyed on absolute pathname
        umap = {}
        for rel, meta in parent_vm.unknowns.items():
            abspath = meta['pathname']
            if abspath.startswith(sys_pathname+':'):
                newmeta = unknowns_dict.get(abspath)
                if newmeta is not None:
                    newrel = newmeta['relative_name']
                else:
                    newrel = rel
                umap[rel] = newrel

        self.unknowns  = parent_vm.unknowns.get_view(umap)
        self.dunknowns = parent_vm.dunknowns.get_view(umap)
        self.resids    = parent_vm.resids.get_view(umap)
        self.dresids   = parent_vm.dresids.get_view(umap)
        self.params    = VecWrapper.create_target_vector(params_dict, self.unknowns, param_owners, connections, store_noflats=True)
        self.dparams   = VecWrapper.create_target_vector(params_dict, self.unknowns, param_owners, connections)
