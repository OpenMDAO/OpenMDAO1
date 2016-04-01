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
    """ Recorder that saves cases in memory. Note, this may take up large
    amounts of memory, so it is not recommended for large models or models
    with lots of iterations.

    Options
    -------
    options['record_metadata'] :  bool(True)
        Tells recorder whether to record variable attribute metadata.
    options['record_unknowns'] :  bool(True)
        Tells recorder whether to record the unknowns vector.
    options['record_params'] :  bool(False)
        Tells recorder whether to record the params vector.
    options['record_resids'] :  bool(False)
        Tells recorder whether to record the ressiduals vector.
    options['record_derivs'] :  bool(True)
        Tells recorder whether to record derivatives that are requested by a `Driver`.
    options['includes'] :  list of strings
        Patterns for variables to include in recording.
    options['excludes'] :  list of strings
        Patterns for variables to exclude in recording (processed after includes).
    """

    def __init__(self):
        super(InMemoryRecorder, self).__init__()
        self._parallel = True

        self.iters = []
        self.deriv_iters = []
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

        data['timestamp'] = metadata['timestamp']
        data['iter'] = format_iteration_coordinate(iteration_coordinate)
        data['success'] = metadata['success']
        data['msg'] = metadata['msg']

        if self.options['record_params']:
            data['params'] = {p:v for p,v in
                                 iteritems(self._filter_vector(params,'p',
                                                        iteration_coordinate))}

        if self.options['record_unknowns']:
            data['unknowns'] = {u:v for u,v in
                                  iteritems(self._filter_vector(unknowns,'u',
                                                        iteration_coordinate))}

        if self.options['record_resids']:
            data['resids'] = {r:v for r,v in
                                  iteritems(self._filter_vector(resids,'r',
                                                         iteration_coordinate))}

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

    def record_derivatives(self, derivs, metadata):
        """Writes the derivatives that were calculated for the driver.

        Args
        ----
        derivs : dict
            Dictionary containing derivatives

        metadata : dict, optional
            Dictionary containing execution metadata (e.g. iteration coordinate).
        """

        data = {}
        iteration_coordinate = metadata['coord']
        timestamp = metadata['timestamp']

        data['timestamp'] = timestamp
        data['Derivatives'] = derivs

        self.deriv_iters.append(data)
