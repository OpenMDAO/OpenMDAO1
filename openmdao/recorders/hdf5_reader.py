from __future__ import print_function, absolute_import

import numpy as np

try:
    import h5py
except ImportError:
    h5py = None

from openmdao.recorders.case_reader_base import CaseReaderBase
from openmdao.recorders.case import Case


def _group_to_dict(grp):
    """ Given an HDF5 group, converted it to a nested dictionary.

    Parameters
    ----------
    grp : h5py.Group or h5py.Dataset
        The group to be converted to a nested python dictionary.

    Returns
    -------
    dict
        The given group converted to a nested python dictionary.
    """
    d = {}
    if isinstance(grp, h5py.Group):
        for k, v in grp.items():
            d[k] = _group_to_dict(v)
    elif isinstance(grp, h5py.Dataset):
        if not grp.shape:
            # Value is scalar
            val = grp[()]
        else:
            val = np.zeros(grp.shape, dtype=grp.dtype)
            grp.read_direct(val)
        return val
    elif isinstance(grp, dict):
        return grp
    else:
        raise(ValueError('Encountered unhandled type '
                         'in group {0}'.format(type(grp))))
    return d


class HDF5CaseReader(CaseReaderBase):
    """ A case reader intended to read data from files recorded using the
    HDF5Recorder.

    Args
    ----
    filename : str
        The name of the file to be read by the case reader.
    """

    def __init__(self, filename):
        super(HDF5CaseReader, self).__init__(filename)
        if h5py is None:
            raise RuntimeError('h5py not available. '
                               'Cannot instantiate HDF5CaseReader.')
        with h5py.File(self.filename, 'r') as f:
            self.format_version = f['metadata']['format_version'][()]
        self._load()
        self.num_cases = len(self._case_keys)

    def _load(self):
        """ Load the metadata and iteration keys from the given file. """
        if self.format_version in (3, 4):
            with h5py.File(self.filename, 'r') as f:
                self._parameters = f['metadata'].get('Parameters', None)
                self._unknowns = f['metadata'].get('Unknowns', None)

                if isinstance(self._parameters, h5py.Group):
                    self._parameters = _group_to_dict(self._parameters)

                if isinstance(self._unknowns, h5py.Group):
                    self._unknowns = _group_to_dict(self._unknowns)

                self._case_keys = tuple([key for key in f.keys()
                                         if key != 'metadata'])
        else:
            raise ValueError('HDF5CaseReader encountered an unhandled '
                             'format version: {0}'.format(self.format_version))

    def get_case(self, case_id):
        """
        Parameters
        ----------
        case_id : int or str
            The integer index or string-identifier of the case to be retrieved.

        Returns
        -------
            An instance of Case populated with data from the
            specified case/iteration.
        """
        if isinstance(case_id, int):
            # If case_id is an integer, assume the user
            # wants a case as an index
            _case_id = self._case_keys[case_id]
        else:
            # Otherwise assume we were given the case string identifier
            _case_id = case_id

        with h5py.File(self.filename, 'r') as f:
            case_dict = _group_to_dict(f[_case_id])
            return Case(self.filename, _case_id, case_dict)
