"""Class definition for SqliteRecorder, which provides dictionary backed by SQLite"""

from collections import OrderedDict
from sqlitedict import SqliteDict
from openmdao.recorders.base_recorder import BaseRecorder
from openmdao.util.record_util import format_iteration_coordinate

from openmdao.core.mpi_wrap import MPI

class SqliteRecorder(BaseRecorder):
    """ Recorder that saves cases in an SQLite dictionary.

    Args
    ----
    sqlite_dict_args : dict
        Dictionary lf any additional arguments for the SQL db.

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

    def __init__(self, out, **sqlite_dict_args):
        super(SqliteRecorder, self).__init__()

        if MPI and MPI.COMM_WORLD.rank > 0 :
            self._open_close_sqlitedict = False
        else:
            self._open_close_sqlitedict = True

        if self._open_close_sqlitedict:
            sqlite_dict_args.setdefault('autocommit', True)
            sqlite_dict_args.setdefault('tablename', 'openmdao')
            self.out = SqliteDict(filename=out, flag='n', **sqlite_dict_args)
        else:
            self.out = None

    def record_metadata(self, group):
        """Stores the metadata of the given group in a sqlite file using
        the variable name for the key.

        Args
        ----
        group : `System`
            `System` containing vectors
        """

        params = group.params.iteritems()
        resids = group.resids.iteritems()
        unknowns = group.unknowns.iteritems()

        data = OrderedDict([('Parameters', dict(params)),
                            ('Unknowns', dict(unknowns)),
                            ])

        self.out['metadata'] = data

    def record_iteration(self, params, unknowns, resids, metadata):
        """
        Stores the provided data in the sqlite file using the iteration
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

        data = OrderedDict()
        iteration_coordinate = metadata['coord']
        timestamp = metadata['timestamp']

        group_name = format_iteration_coordinate(iteration_coordinate)

        data['timestamp'] = timestamp
        data['success'] = metadata['success']
        data['msg'] = metadata['msg']

        if self.options['record_params']:
            data['Parameters'] = self._filter_vector(params, 'p', iteration_coordinate)

        if self.options['record_unknowns']:
            data['Unknowns'] = self._filter_vector(unknowns, 'u', iteration_coordinate)

        if self.options['record_resids']:
            data['Residuals'] = self._filter_vector(resids, 'r', iteration_coordinate)

        self.out[group_name] = data

    def record_derivatives(self, derivs, metadata):
        """Writes the derivatives that were calculated for the driver.

        Args
        ----
        derivs : dict
            Dictionary containing derivatives

        metadata : dict, optional
            Dictionary containing execution metadata (e.g. iteration coordinate).
        """

        data = OrderedDict()
        iteration_coordinate = metadata['coord']
        timestamp = metadata['timestamp']

        group_name = format_iteration_coordinate(iteration_coordinate)
        group_name = '%s/derivs' % group_name

        data['timestamp'] = timestamp
        data['success'] = metadata['success']
        data['msg'] = metadata['msg']
        data['Derivatives'] = derivs

        self.out[group_name] = data

    def close(self):
        """Closes `out`"""

        if self._open_close_sqlitedict:
            if self.out is not None:
                self.out.close()
                self.out = None
