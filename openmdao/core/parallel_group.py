""" Defines the base class for a ParallelGroup in OpenMDAO. ParallelGroup is
used for systems of `Components` or `Groups` that can be run in parallel."""

import warnings
from collections import OrderedDict
from six import itervalues

from openmdao.core.component import Component
from openmdao.core.group import Group
from openmdao.core.mpi_wrap import MPI


class ParallelGroup(Group):
    """ParallelGroup is used for systems of `Components` or `Groups` that can
    be run in parallel.

    Options
    -------
    fd_options['force_fd'] :  bool(False)
        Set to True to finite difference this system.
    fd_options['form'] :  str('forward')
        Finite difference mode. (forward, backward, central) You can also set to 'complex_step' to peform the complex step method if your components support it.
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

    def apply_nonlinear(self, params, unknowns, resids, metadata=None):
        """ Evaluates the residuals of our children systems.

        Args
        ----
        params : `VecWrapper`
            `VecWrapper` containing parameters. (p)

        unknowns : `VecWrapper`
            `VecWrapper`  containing outputs and states. (u)

        resids : `VecWrapper`
            `VecWrapper` containing residuals. (r)

        metadata : dict, optional
            Dictionary containing execution metadata (e.g. iteration
            coordinate).
        """

        # full scatter
        self._transfer_data()

        for sub in self._local_subsystems:
            if isinstance(sub, Component):
                sub.apply_nonlinear(sub.params, sub.unknowns, sub.resids)
            else:
                sub.apply_nonlinear(sub.params, sub.unknowns, sub.resids,
                                    metadata)

    def children_solve_nonlinear(self, metadata):
        """Loops over our children systems and asks them to solve."""

        # full scatter
        self._transfer_data()

        for sub in self._local_subsystems:
            with sub._dircontext:
                if isinstance(sub, Component):
                    sub.solve_nonlinear(sub.params, sub.unknowns, sub.resids)
                else:
                    sub.solve_nonlinear(sub.params, sub.unknowns, sub.resids,
                                        metadata)

    def get_req_procs(self):
        """
        Returns
        -------
        tuple
            A tuple of the form (min_procs, max_procs), indicating the min and
            max processors usable by this `ParallelGroup`.
        """
        min_procs = 0
        max_procs = 0

        for sub in itervalues(self._subsystems):
            sub_min, sub_max = sub.get_req_procs()
            if sub_min > min_procs:
                min_procs = sub_min
            if max_procs is not None:
                if sub_max is None:
                    max_procs = None
                else:
                    max_procs += sub_max

        if min_procs == 0:
            min_procs = 1

        if max_procs == 0:
            max_procs = 1

        return (min_procs, max_procs)

    def _setup_communicators(self, comm, parent_dir):
        """
        Assign communicator to this `ParallelGroup` and all of its subsystems.

        Args
        ----
        comm : an MPI communicator (real or fake)
            The communicator being offered by the parent system.

        parent_dir : str
            Absolute dir of parent `System`.
        """
        self.comm = comm
        self._local_subsystems = []

        # If we're not runnin in MPI, make this just a serial Group
        if not MPI or not self.is_active():
            super(ParallelGroup, self)._setup_communicators(comm, parent_dir)
            return

        self._setup_dir(parent_dir)

        size = comm.size
        rank = comm.rank

        subsystems = []
        requested_procs = []
        max_req_procs = []
        for system in itervalues(self._subsystems):
            subsystems.append(system)
            minproc, maxproc = system.get_req_procs()
            assert(minproc > 0)
            requested_procs.append(minproc)
            max_req_procs.append(maxproc)

        assigned_procs = [0]*len(subsystems)

        assigned = 0

        requested = sum(requested_procs)

        _, mx = self.get_req_procs()
        if mx is None:
            limit = size
            max_requested = size
        else:
            max_requested = sum(max_req_procs)
            limit = min(size, max_requested)

        # first, just use simple round robin assignment of requested procs
        # until everybody has what they asked for or we run out
        if requested:
            if size >= requested: # we have enough for all subsystems
                while assigned < limit:
                    for i, system in enumerate(subsystems):
                        if max_req_procs[i] is None or \
                           assigned_procs[i] < max_req_procs[i]:
                            assigned_procs[i] += 1
                            assigned += 1
                            if assigned == limit:
                                break

                # create buckets (one sub per bucket) to be consistent in how
                # we split procs below
                buckets = [(n,[i]) for i,n in enumerate(assigned_procs)]

            else: # we don't have enough, so have to group subsystems
                remaining = size
                # sort req procs in descending order
                tups = sorted([(n,i) for i,n in enumerate(requested_procs)],
                              reverse=True)
                buckets = []
                for i, (req, sub_idx) in enumerate(tups):
                    if remaining >= req:
                        buckets.append((req, [sub_idx]))
                        remaining -= req
                    elif i == 0:
                        # since we sorted in descending order by number of
                        # requested procs, only in the first iteration is there
                        # a chance that we've requested more procs than we have
                        raise RuntimeError("subsystem %s requested %d processes "
                                           "but got %d" %
                                                (subsystems[sub_idx].pathname,
                                                 req, remaining))
                    else: # we already have one in the bucket list that's big enough.
                        # go through buckets, find all that are big enough, and
                        # add the current sub to the one with the fewest number
                        # of subs already in it. In the event of a tie, take
                        # the bucket with the lowest number of requested procs.
                        lenlist = sorted([b for b in buckets if b[0]>=req],
                                         key=lambda t: len(t[1]))
                        shortest = len(lenlist[0][1])
                        final = sorted(b for b in lenlist
                                                if len(b[1]) == shortest)
                        final[0][1].append(sub_idx)

                warnings.warn("Group '%s' requested %d processes to run fully "
                              "in parallel, but it only got %d" % (self.pathname,
                                                             requested, size))

                # if we have any procs left over, apply them to any sub that
                # can use them
                while remaining > 0:
                    for i,b in enumerate(buckets):
                        procs, subs = b
                        for sub_idx in subs:
                            if (max_req_procs[sub_idx] is None or
                                       max_req_procs[sub_idx] > procs):
                                buckets[i] = (procs+1, subs)
                                remaining -= 1
                                break
                        if remaining == 0:
                            break

                assigned = size - remaining

        # a 'color' is assigned to each subsystem, with
        # an entry for each processor it will be given
        # e.g. [0, 1, 1, 1, 1, 2, 2, 3, 3, 3, UND, UND]
        color = []
        for i, b in enumerate(buckets):
            procs, _ = b
            color.extend([i]*procs)

        if size > assigned:
            color.extend([MPI.UNDEFINED]*(size-assigned))

        # create a sub-communicator for each color and
        # get the one assigned to our color/process
        rank_color = color[rank]
        sub_comm = comm.Split(rank_color)

        if sub_comm == MPI.COMM_NULL:
            return

        for i, b in enumerate(buckets):
            procs, subs = b
            for sub_idx in subs:
                sub = subsystems[sub_idx]
                if i == rank_color:
                    self._local_subsystems.append(sub)
                    sub._setup_communicators(sub_comm, self._sysdata.absdir)
                else:
                    sub._setup_communicators(MPI.COMM_NULL, self._sysdata.absdir)

    def list_auto_order(self):
        """
        Returns
        -------
        list of str
            Names of subsystems listed in their current order, since
            order is irrelevant in a ParallelGroup.

        list of str
            This will always be an empty list.
        """
        return [s.name for s in self.subsystems()], []
