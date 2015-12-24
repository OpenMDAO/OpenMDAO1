""" RecordingManager class definition. """

import itertools
import time

from openmdao.core.mpi_wrap import MPI


class RecordingManager(object):
    """ Object that routes function calls to all attached recorders. """

    def __init__(self):
        self._vars_to_record = {
            'pnames': set(),
            'unames': set(),
            'rnames': set(),
            }

        self._recorders = []
        self.__has_serial_recorders = False
        if MPI:
            self.rank = MPI.COMM_WORLD.rank
        else:
            self.rank = 0

    def append(self, recorder):
        """ Add a recorder for recording.

        Args
        ----
        recorder : `BaseRecorder`
           Recorder instance.
        """
        self._recorders.append(recorder)

    def __getitem__(self, index):
        return self._recorders[index]

    def __iter__(self):
        return iter(self._recorders)

    def _local_vars(self, root, vec, varnames):
        local_vars = []

        for name in varnames:
            if root.comm.rank == root._owning_ranks[name]:
                local_vars.append((name, vec[name]))

        return local_vars

    def _gather_vars(self, root, local_vars):
        """Gathers and returns only variables listed in
        `varnames` from the vector `vec`"""

        all_vars = root.comm.gather(local_vars, root=0)

        if root.comm.rank == 0:
            return dict(itertools.chain(*all_vars))

    def startup(self, root):
        """ Initial startup for this recorder.

        Args
        ----
        root : `System`
           System containing variables.
        """
        for recorder in self._recorders:
            recorder.startup(root)

            if not recorder._parallel:
                self.__has_serial_recorders = True
                pnames, unames, rnames = recorder._filtered[root.pathname]

                self._vars_to_record['pnames'].update(pnames)
                self._vars_to_record['unames'].update(unames)
                self._vars_to_record['rnames'].update(rnames)

    def close(self):
        """ Close all recorders. """
        for recorder in self._recorders:
            recorder.close()

    def record_metadata(self, root):
        """ Record metadata for all variables of interest.

        Args
        ----
        root : `System`
           System containing variables.
        """

        for recorder in self._recorders:
            # If the recorder does not support parallel recording
            # we need to make sure we only record on rank 0.
            if recorder._parallel or self.rank == 0:
                metadata_option = recorder.options['record_metadata']

                if metadata_option is False:
                    continue

                recorder.record_metadata(root)

    def record_iteration(self, root, metadata):
        """ Gathers variables for non-parallel case recorders and calls
        record for all recorders.

        Args
        ----
        root : `System`
           System containing variables.
        metadata : dict
            Metadata for iteration coordinate
        """

        metadata['timestamp'] = time.time()
        params = root.params
        unknowns = root.unknowns
        resids = root.resids

        if MPI and self.__has_serial_recorders:
            pnames = self._vars_to_record['pnames']
            unames = self._vars_to_record['unames']
            rnames = self._vars_to_record['rnames']

            params = self._gather_vars(root, self._local_vars(root, params, pnames))
            unknowns = self._gather_vars(root, self._local_vars(root, unknowns, unames))
            resids = self._gather_vars(root, self._local_vars(root, resids, rnames))

        # If the recorder does not support parallel recording
        # we need to make sure we only record on rank 0.
        for recorder in self._recorders:
            if recorder._parallel or self.rank == 0:
                recorder.record_iteration(params, unknowns, resids, metadata)

    def record_derivatives(self, derivs, metadata):
        """" Records derivatives if requested.

        Args
        ----
        derivs : dict
            Dictionary containing derivatives
        metadata : `dict`
            Metadata for iteration coordinate
        """

        metadata['timestamp'] = time.time()

        # If the recorder does not support parallel recording
        # we need to make sure we only record on rank 0.
        for recorder in self._recorders:

            if recorder.options['record_derivs'] is False:
                continue

            if recorder._parallel or self.rank == 0:
                recorder.record_derivatives(derivs, metadata)
