""" Class definition for HDF5Recorder, which uses the HDF5 format."""

from numbers import Number

from six import iteritems

import numpy as np
from h5py import File

from openmdao.recorders.base_recorder import BaseRecorder
from openmdao.util.record_util import format_iteration_coordinate


class HDF5Recorder(BaseRecorder):
    """
    A recorder that stores data using HDF5. This format naturally handles
    hierarchical data and is a standard for handling large datasets.

    Args
    ----
    out : str
        String containing the filename for the HDF5 file.

    **driver_kwargs
        Additional keyword args to be passed to the HDF5 driver.
    """

    def __init__(self, out, **driver_kwargs):

        super(HDF5Recorder, self).__init__()
        self.out = File(out, 'w', **driver_kwargs)

    def record(self, params, unknowns, resids, metadata):
        """
        Stores the provided data in the HDF5 file using the iteration
        coordinate for the Group name.

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

        f = self.out

        group = f.require_group(group_name)
        group.create_dataset('timestamp', data=timestamp)
        p_group = group.create_group("Parameters")
        u_group = group.create_group("Unknowns")
        r_group = group.create_group("Residuals")

        pairings = ((p_group, params),
                    (u_group, unknowns),
                    (r_group, resids))

        for grp, data in pairings:
            for key, val in iteritems(data):
                if isinstance(val, (np.ndarray, Number)):
                    grp.create_dataset(key, data=val)
                    # TODO: Compression/Checksum?
                else:
                    # TODO: Handling non-numeric data
                    msg = "HDF5 Recorder does not support data of type '{0}'".format(type(val))
                    raise NotImplementedError(msg)
