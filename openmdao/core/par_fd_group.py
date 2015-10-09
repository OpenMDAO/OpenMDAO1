""" Defines the base class for a Group in OpenMDAO."""

from __future__ import print_function

from openmdoa.core.group import Group
from openmdao.util.array_util import evenly_distrib_idxs

class ParallelFDGroup(Group):
    """A Group that can do finite difference in parallel.

    Options
    -------
    fd_options['force_fd'] :  bool(False)
        Set to True to finite difference this system.
    fd_options['form'] :  str('forward')
        Finite difference mode. (forward, backward, central)
        You can also set to 'complex_step' to perform the complex step method
        if your components support it.
    fd_options['step_size'] :  float(1e-06)
        Default finite difference stepsize
    fd_options['step_type'] :  str('absolute')
        Set to absolute, relative

    """
    def __init__(self, num_par_fds):
        super(ParallelFDGroup, self).__init__()

        self._num_par_fds = num_par_fds
        self._par_fd_id = 0

    def _setup_communicators(self, comm):
        """
        Assign communicator to this `Group` and all of its subsystems.

        Args
        ----
        comm : an MPI communicator (real or fake)
            The communicator being offered by the parent system.
        """
        self._full_comm = comm

        minprocs, maxprocs = super(ParallelFDGroup, self).get_req_procs()
        sizes, offsets = evenly_distrib_idxs(self._num_par_fds, comm.size)

        # figure out which parallel FD we are associated with
        if self._num_par_fds >= 2:
            # a 'color' is assigned to each subsystem, with
            # an entry for each processor it will be given
            # e.g. [0, 0, 0, 1, 1, 1, 2, 2, 3, 3]
            color = []
            for i in range(self._num_par_fds):
                color.extend([i]*sizes[i])

            self._par_fd_id = color[comm.rank]

            # create a sub-communicator for each color and
            # get the one assigned to our color/process
            comm = comm.Split(self._par_fd_id)

        self._local_subsystems = []

        self.comm = comm

        for sub in itervalues(self._subsystems):
            sub._setup_communicators(comm)
            if self.is_active() and sub.is_active():
                self._local_subsystems.append(sub)

    def get_req_procs(self):
        """
        Returns
        -------
        tuple
            A tuple of the form (min_procs, max_procs), indicating the
            min and max processors usable by this `Group`.
        """
        minprocs, maxprocs = super(ParallelFDGroup, self).get_req_procs()

        minprocs *= self._num_par_fds
        if maxprocs is not None:
            maxprocs *= self._num_par_fds

        return (minprocs, maxprocs)
