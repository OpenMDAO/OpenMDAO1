
import numpy
from openmdao.core.vecwrapper import VecWrapper, get_relative_varname
from openmdao.core.dataxfer import DataXfer

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
        self.data_xfer = {}

    def _setup_data_transfer(self, my_params):
        # collect all flattenable var sizes from self.unknowns
        flats = [m['size'] for m in self.unknowns._vardict.values()
                     if not m.get('noflat')]
        # create a 1x<num_flat_vars> numpy array with the sizes of each var
        self._local_sizes = numpy.array([[flats]])

        # we would do an Allgather of the local_sizes in the distributed case so all
        # processes would know the sizes of all variables (needed to determine distributed
        # indices)

        #TODO: invesigate providing enough system info here to detrmine what types of scatters
        # are necessary (for example, full scatter isn't needed except when solving using jacobi,
        # so why allocate space for the index arrays?)

        xfer_dict = {}
        for param, unknown in self.connections.items():
            if param in my_params:
                dest_comp = param.split(':',1)[0]
                src_idx_list, dest_idx_list, noflat_conns = xfer_dict.setdefault(dest_comp, ([],[],[]))
                urelname = get_relative_varname(unknown, self.unknowns)
                prelname = get_relative_varname(param, self.params)
                noflat = self.unknowns.metadata(urelname).get('noflat')
                if noflat:
                    noflat_conns.append(prelname, urelname)
                else:
                    src_idx_list.append(self.unknowns.get_idxs(urelname))
                    dest_idx_list.append(self.params.get_idxs(prelname))

        for tgt_comp, (srcs, tgts, noflat_conns) in xfer_dict.items():
            src_idxs, tgt_idxs = self.unknowns.merge_idxs(srcs, tgts)
            self.data_xfer[tgt_comp] = DataXfer(src_idxs, tgt_idxs, noflat_conns)

        #TODO: create a jacobi DataXfer object (if necessary) that combines all of the
        #      individual subsystem src_idxs, tgt_idxs, and noflat_conns

    def _transfer_data(self, target_system, mode='fwd', deriv=False):
        """Transfer data to/from target_system depending on mode.

        Parameters
        ----------
        target_system : str
            Name of the target `System`.

        mode : { 'fwd', 'rev' }, optional
            Specifies forward or reverse data transfer.

        deriv : bool, optional
            If True, perform a data transfer between derivative `VecWrapper`s
        """
        x = self.data_xfer.get(target_system)
        if x is not None:
            if deriv:
                x.transfer(self.dunknowns, self.dparams, mode)
            else:
                x.transfer(self.unknowns, self.params, mode)


class VarManager(VarManagerBase):
    def __init__(self, params_dict, unknowns_dict, my_params, connections):
        super(VarManager, self).__init__(connections)

        self.unknowns  = VecWrapper.create_source_vector(unknowns_dict, store_noflats=True)
        self.dunknowns = VecWrapper.create_source_vector(unknowns_dict)
        self.resids    = VecWrapper.create_source_vector(unknowns_dict)
        self.dresids   = VecWrapper.create_source_vector(unknowns_dict)
        self.params    = VecWrapper.create_target_vector(params_dict, self.unknowns, my_params, connections, store_noflats=True)
        self.dparams   = VecWrapper.create_target_vector(params_dict, self.unknowns, my_params, connections)

        self._setup_data_transfer(my_params)

class VarViewManager(VarManagerBase):
    def __init__(self, parent_vm, sys_pathname, params_dict, unknowns_dict, my_params, connections):
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
        self.params    = VecWrapper.create_target_vector(params_dict, self.unknowns,
                                                         my_params, connections, store_noflats=True)
        self.dparams   = VecWrapper.create_target_vector(params_dict, self.unknowns,
                                                         my_params, connections)

        self._setup_data_transfer(my_params)

