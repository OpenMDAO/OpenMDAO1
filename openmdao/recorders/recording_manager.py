import itertools
from openmdao.core.mpi_wrap import MPI

class RecordingManager(list):
    def __init__(self, *args, **kargs):
        super(Recorders, self).__init__(*args, **kargs)
        self._root = root
        self._vars_to_record = {
                'pnames' : set(),
                'unames' : set(),
                'rnames' : set(),
            }

        self._recorders = []
        self.__has_serial_recorders = False

    def _gather_vars(self, vec, varnames):
        '''
        Gathers and returns only variables listed in 
        `varnames` from the vector `vec`
        '''
        local_vars = []

        for name in varnames:
            if self._root.comm.rank == self._root._owning_ranks[name]:
                local_vars.append((name, vec[name]))

        all_vars = self._root.comm.gather(local_vars, root=0)

        if self._root.comm.rank == 0:
            return dict(itertools.chain(*all_vars))

    def startup(self, root):
        self._root = root

        for recorder in self._recorders:
            recorder.startup(self._root)

            if not recorder._parallel:
                self.__has_serial_recorders = True
                
                self._vars_to_record['pnames'].update(pnames)
                self._vars_to_record['unames'].update(unames)
                self._vars_to_record['rnames'].update(rnames)

    def record(self, metadata):
        '''
        Gathers variables for non-parallel case recorders and
        calls record for all recorders

        Args
        ----
        metadata: `dict`
        Metadata for iteration coordinate
        '''
        params = self._root.params
        unknowns = self._root.unknowns
        resids = self._root.resids

        if MPI and self.__has_serial_recorders:
            pnames = self._vars_to_record['pnames']
            unames = self._vars_to_record['unames']
            rnames = self._vars_to_record['rnames']

            params = self._gather_vars(params, pnames)
            unknowns = self._gather_vars(unknowns, unames)
            resids = self._gather_vars(resids, rnames)

        # If the recorder does not support parallel recording
        # we need to make sure we only record on rank 0.
        for recorder in self._recorders:
            if self._supports_parallel(recorder) or self._root.comm.rank == 0:
                recorder.raw_record(params, unknowns, resids, metadata)gg
