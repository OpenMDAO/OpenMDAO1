from __future__ import print_function, absolute_import

import os.path
import re

from sqlitedict import SqliteDict

from openmdao.recorders.base_reader import CaseReaderBase, Case


def _is_valid_sqlite3_db(filename):
    """ Returns true if the given filename
    contains a valid SQLite3 database file.

    Parameters
    ----------
    filename : str
        The path to the file to be tested

    Returns
    -------
        True if the filename specifies a valid SQlite3 database.

    """
    if not os.path.isfile(filename):
        return False
    if os.path.getsize(filename) < 100:
        # SQLite database file header is 100 bytes
        return False

    with open(filename, 'rb') as fd:
        header = fd.read(100)

    return header[:16] == b'SQLite format 3\x00'


class SqliteCaseReader(CaseReaderBase):

    def __init__(self, filename):
        super(SqliteCaseReader, self).__init__()

        self.format_version = None
        self.parameters = {}
        self.unknowns = {}

        if filename is not None:
            if not _is_valid_sqlite3_db(filename):
                raise ValueError('File does not contain a valid '
                                 'sqlite database ({0})'.format(filename))
            self._filename = filename

        # Read the metadata and save it in the reader
        with SqliteDict(self._filename, 'metadata', flag='r') as db:
            self.format_version = db.get('format_version', None)
            self.parameters = db.get('Parameters', None)
            self.unknowns = db.get('Unknowns', None)

        # Store the identifier for each iteration in _case_keys
        with SqliteDict(self._filename, 'iterations', flag='r') as db:
            self._case_keys = db.keys()

    def get_case(self, case_id):
        # Initialize the Case object from the iterations data
        with SqliteDict(self._filename, 'iterations', flag='r') as iter_db:
            with SqliteDict(self._filename, 'derivs', flag='r') as derivs_db:
                if isinstance(case_id, int):
                    # If case_id is an integer, assume the user
                    # wants a case as an index
                    case = Case(self._case_keys[case_id],
                                iter_db[self._case_keys[case_id]],
                                derivs_db[self._case_keys[case_id]])
                else:
                    # Otherwise assume we were given the case string identifier
                    case = Case(case_id, iter_db[case_id], derivs_db[case_id])
        return case

    def list_cases(self):
        return self._case_keys
