"""
Class definition for InMemoryRecorder, a recorder that records values
in a memory resident object.  Note that this is primarily for testing,
and using it for real problems could use up large amounts of memory.
"""

import sys

from six import string_types, iteritems

from openmdao.core.mpi_wrap import MPI
from openmdao.recorders.base_recorder import BaseRecorder
from openmdao.util.record_util import format_iteration_coordinate

class InMemoryRecorder(BaseRecorder):
    def __init__(self):
        super(InMemoryRecorder, self).__init__()
        self._parallel = True

        self.iters = []
        self.meta = {}

    def record_iteration(self, params, unknowns, resids, metadata):
        """Record the given run data in memory.

        Args
        ----
        params : `VecWrapper`
            `VecWrapper` containing parameters. (p)

        unknowns : `VecWrapper`
            `VecWrapper` containing outputs and states. (u)

        resids : `VecWrapper`
            `VecWrapper` containing residuals. (r)

        metadata : dict
            Dictionary containing execution metadata (e.g. iteration coordinate).
        """

        data = {}
        iteration_coordinate = metadata['coord']
        params, unknowns, resids = self._filter_vectors(params, unknowns, resids,
                                                        iteration_coordinate)

        data['timestamp'] = metadata['timestamp']
        data['iter'] = format_iteration_coordinate(iteration_coordinate)

        if self.options['record_params']:
            data['params'] = {p:v for p,v in iteritems(params)}

        if self.options['record_unknowns']:
            data['unknowns'] = {u:v for u,v in iteritems(unknowns)}

        if self.options['record_resids']:
            data['resids'] = {r:v for r,v in iteritems(resids)}

        self.iters.append(data)

    def record_metadata(self, group):
        """Dump the metadata of the given group in a "pretty" form.

        Args
        ----
        group : `System`
            `System` containing vectors
        """
        self.meta['unknowns'] = {n:m.copy() for n,m in iteritems(group.unknowns)}
        self.meta['params'] = {n:m.copy() for n,m in iteritems(group.params)}
