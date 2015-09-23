""" Class definition for HDF5Recorder, which uses the shelve format."""

import shelve
import warnings

from collections import OrderedDict

from openmdao.recorders.base_recorder import BaseRecorder
from openmdao.util.record_util import format_iteration_coordinate


class ShelveRecorder(BaseRecorder):
    """
    A recorder that stores data using Python's shelve.

    ShelveRecorder is deprecated. Please consider using the SqliteRecorder instead.

    Args
    ----
    out : str
        String containing the filename for the shelve file.

    **shelve_args
        Additional keyword args to be passed to shelve.open().
    """

    def __init__(self, out, **shelve_args):
        super(ShelveRecorder, self).__init__()
        
        warnings.simplefilter('always', DeprecationWarning)
        warnings.warn("ShelveRecorder is deprecated, use SqliteRecorder instead",
                       DeprecationWarning,stacklevel=2)
        warnings.simplefilter('ignore', DeprecationWarning)

        self.out = shelve.open(out, **shelve_args)
        self.order = []

    def record(self, params, unknowns, resids, metadata):
        """
        Stores the provided data in the shelve file using the iteration
        coordinate for the key.

        Args
        ----
        params : dict
            Dictionary containing parameters. (p)

        unknowns : dict
            Dictionary containing outputs and states. (u)

        resids : dict
            Dictionary containing residuals. (r)

        metadata : dict, optional
            Dictionary containing execution metadata (e.g. iteration coordinate).
        """

        iteration_coordinate = metadata['coord']
        timestamp = metadata['timestamp']
        params, unknowns, resids = self._filter_vectors(params, unknowns, resids, iteration_coordinate)
        group_name = format_iteration_coordinate(iteration_coordinate)

        self.order.append(group_name)

        f = self.out

        data = OrderedDict([('timestamp', timestamp),
                            ('Parameters', params),
                            ('Unknowns', unknowns),
                            ('Residuals', resids)])

        f[group_name] = data
        f['order'] = self.order
