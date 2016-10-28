""" Class definition for HDF5Recorder, which uses the HDF5 format."""

from collections import OrderedDict
from numbers import Number

from six import iteritems

import numpy as np
import pickle

from h5py import File

from openmdao.recorders.base_recorder import BaseRecorder
from openmdao.util.record_util import format_iteration_coordinate

from openmdao.devtools.partition_tree_n2 import get_model_viewer_data

format_version = 4

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
    options['includes'] :  list of strings
        Patterns for variables to include in recording.
    options['excludes'] :  list of strings
        Patterns for variables to exclude in recording (processed after includes).
    """

    def __init__(self, out, **driver_kwargs):

        super(HDF5Recorder, self).__init__()
        self.out = File(out, 'w', **driver_kwargs)

        metadata_group = self.out.require_group('metadata')

        metadata_group.create_dataset('format_version', data = format_version)

    def record_metadata(self, group):
        """Stores the metadata of the given group in a HDF5 file using
        the variable name for the key.

        Args
        ----
        group : `System`
            `System` containing vectors
        """
        params = group.params.iteritems()
        resids = group.resids.iteritems()
        unknowns = group.unknowns.iteritems()

        metadata_group = self.out['metadata']

        # The group metadata could be anything so need to pickle it
        # There are other ways of storing any kind of Python object in HDF5 but this is the simplest
        system_metadata_val = np.array(pickle.dumps(group.metadata, pickle.HIGHEST_PROTOCOL))
        metadata_group.create_dataset('system_metadata', data=system_metadata_val)

        # Also store the model_viewer_data
        model_viewer_data = get_model_viewer_data(group)
        model_viewer_data_val = np.array(pickle.dumps(model_viewer_data, pickle.HIGHEST_PROTOCOL))
        metadata_group.create_dataset('model_viewer_data', data=model_viewer_data_val)

        pairings = (
            (metadata_group.create_group("Parameters"), params),
            (metadata_group.create_group("Unknowns"), unknowns),
        )

        for grp, data in pairings:
            for key, val in data:
                meta_group = grp.create_group(key)

                for mkey, mval in iteritems(val):
                    meta_group.create_dataset(mkey, data=mval)
                    # if isinstance(val, (np.ndarray, Number)):
                    #    grp.create_dataset(key, data=val)
                    #    # TODO: Compression/Checksum?
                    # else:
                    #    # TODO: Handling non-numeric data
                    #    msg = "HDF5 Recorder does not support data of type '{0}'".format(type(val))
                    #    raise NotImplementedError(msg)

    def record_iteration(self, params, unknowns, resids, metadata):
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
        group_name = format_iteration_coordinate(iteration_coordinate)

        f = self.out

        group = f.require_group(group_name)
        group.attrs['timestamp'] = metadata['timestamp']
        group.attrs['success'] = metadata['success']
        group.attrs['msg'] = metadata['msg']

        pairings = []

        if self.options['record_params']:
            p_group = group.create_group("Parameters")
            pairings.append((p_group, self._filter_vector(params, 'p',
                                                          iteration_coordinate)))

        if self.options['record_unknowns']:
            u_group = group.create_group("Unknowns")
            pairings.append((u_group, self._filter_vector(unknowns, 'u',
                                                          iteration_coordinate)))

        if self.options['record_resids']:
            r_group = group.create_group("Residuals")
            pairings.append((r_group, self._filter_vector(resids, 'r',
                                                          iteration_coordinate)))

        for grp, data in pairings:
            for key, val in iteritems(data):
                if isinstance(val, (np.ndarray, Number)):
                    grp.create_dataset(key, data=val)
                    # TODO: Compression/Checksum?
                else:
                    # TODO: Handling non-numeric data
                    msg = "HDF5 Recorder does not support data of type '{0}'".format(type(val))
                    raise NotImplementedError(msg)

    def record_derivatives(self, derivs, metadata):
        """Writes the derivatives that were calculated for the driver.

        Args
        ----
        derivs : dict
            Dictionary containing derivatives

        metadata : dict, optional
            Dictionary containing execution metadata (e.g. iteration coordinate).
        """

        iteration_coordinate = metadata['coord']
        group_name = format_iteration_coordinate(iteration_coordinate)

        # get the group for the iteration
        iteration_group = self.out[group_name]

        # Create a group under that called 'deriv'
        deriv_group = iteration_group.require_group('Derivs')

        # Then add timestamp, success, msg as attributes
        deriv_group.attrs['timestamp'] = metadata['timestamp']
        deriv_group.attrs['success'] = metadata['success']
        deriv_group.attrs['msg'] = metadata['msg']

        #  And actual deriv data. derivs could either be a dict or an ndarray
        #    depending on the optimizer
        if isinstance(derivs, np.ndarray):
            deriv_group.create_dataset('Derivatives', data=derivs)
        elif isinstance(derivs, OrderedDict):
            deriv_data_group = deriv_group.require_group('Derivatives')
            k = derivs.keys()
            for k,v in derivs.items():
                g = deriv_data_group.require_group(k)
                for k2,v2 in v.items():
                    g.create_dataset(k2,data=v2)
        else:
            raise ValueError("Currently can only record derivatives that are ndarrays or OrderedDicts")
