"""Class definition for SqliteRecorder, which provides dictionary backed by SQLite"""

from collections import OrderedDict
from sqlitedict import SqliteDict
from openmdao.recorders.base_recorder import BaseRecorder
from openmdao.util.record_util import format_iteration_coordinate

class SqliteRecorder(BaseRecorder):

    def __init__(self, out, **sqlite_dict_args):
        super(SqliteRecorder, self).__init__()
        sqlite_dict_args.setdefault('autocommit', True)
        sqlite_dict_args.setdefault('tablename', 'openmdao')
        self.out = SqliteDict(filename=out, flag='n', **sqlite_dict_args)

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
        params, unknowns, resids = self._filter_vectors(params, unknowns, resids, iteration_coordinate)

        group_name = format_iteration_coordinate(iteration_coordinate)
        
        data['timestamp'] = timestamp

        if self.options['record_params']:
            data['Parameters'] = params

        if self.options['record_unknowns']:
            data['Unknowns'] = unknowns

        if self.options['record_resids']:
            data['Residuals'] = resids

        self.out[group_name] = data
