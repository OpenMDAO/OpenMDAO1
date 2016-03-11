""" Defines the base class for a Group in OpenMDAO."""

from __future__ import print_function

import os
from six import itervalues

from openmdao.core.group import Group
from openmdao.util.array_util import evenly_distrib_idxs
from openmdao.core.mpi_wrap import MPI

trace = os.environ.get('OPENMDAO_TRACE')
if trace: # pragma: no cover
    from openmdao.core.mpi_wrap import debug


class ParallelFDGroup(Group):
    """A Group that can do finite difference in parallel.

    Args
    ----
    num_par_fds : int(1)
        Number of FD's to perform in parallel.  If num_par_fds is 1,
        this just behaves like a normal Group.

    Options
    -------
    fd_options['force_fd'] :  bool(True)
        Set to True to finite difference this system.
    fd_options['form'] :  str('forward')
        Finite difference mode. (forward, backward, central)
        You can also set to 'complex_step' to perform the complex step method
        if your components support it.
    fd_options['step_size'] :  float(1e-06)
        Default finite difference stepsize
    fd_options['step_type'] :  str('absolute')
        Set to absolute, relative
    fd_options['extra_check_partials_form'] :  None or str
        Finite difference mode: ("forward", "backward", "central", "complex_step")
        During check_partial_derivatives, you can optionally do a
        second finite difference with a different mode.
    fd_options['linearize'] : bool(False)
        Set to True if you want linearize to be called even though you are using FD.

    """
    def __init__(self, num_par_fds):
        super(ParallelFDGroup, self).__init__()

        self.fd_options['force_fd'] = True # change default
        self._num_par_fds = num_par_fds
        self._par_fd_id = 0

    def _setup_communicators(self, comm, parent_dir):
        """
        Assign communicator to this `Group` and all of its subsystems.

        Args
        ----
        comm : an MPI communicator (real or fake)
            The communicator being offered by the parent system.

        parent_dir : str
            Absolute dir of parent `System`.
        """
        if self._num_par_fds < 1:
            raise ValueError("'%s': num_par_fds must be >= 1 but value is %s." %
                              (self.pathname, self._num_par_fds))
        if not MPI:
            self._num_par_fds = 1

        self._full_comm = comm

        # figure out which parallel FD we are associated with
        if self._num_par_fds > 1:
            minprocs, maxprocs = super(ParallelFDGroup, self).get_req_procs()
            sizes, offsets = evenly_distrib_idxs(self._num_par_fds, comm.size)

            # a 'color' is assigned to each subsystem, with
            # an entry for each processor it will be given
            # e.g. [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
            color = []
            for i in range(self._num_par_fds):
                color.extend([i]*sizes[i])

            self._par_fd_id = color[comm.rank]

            # create a sub-communicator for each color and
            # get the one assigned to our color/process
            if trace:
                debug('%s: splitting comm, fd_id=%s' % (self.pathname,
                                                        self._par_fd_id))
            comm = comm.Split(self._par_fd_id)

        self._local_subsystems = []

        self.comm = comm

        self._setup_dir(parent_dir)

        for sub in itervalues(self._subsystems):
            sub._setup_communicators(comm, self._sysdata.absdir)
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
