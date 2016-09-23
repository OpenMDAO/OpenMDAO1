from __future__ import print_function, absolute_import

import os.path

from sqlitedict import SqliteDict

from openmdao.recorders.case_reader_base import CaseReaderBase
from openmdao.recorders.case import Case


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
    """ A CaseReader specific to files created with SqliteRecorder.

    Parameters
    ----------
    filename : str
        The path to the filename containing the recorded data.

    """

    def __init__(self, filename):
        super(SqliteCaseReader, self).__init__(filename)

        if filename is not None:
            if not _is_valid_sqlite3_db(filename):
                raise IOError('File does not contain a valid '
                              'sqlite database ({0})'.format(filename))
            self._filename = filename
        self._load()

    def _load(self):
        """ The initial load of data from the sqlite database file.

        Load the metadata from the sqlite file, populating the
        `format_version`, `parameters`, and `unknowns` attributes of this
        CaseReader.

        The `iterations` table is read to load the keys which identify
        the individual cases/iterations from the recorded file.
        """

        # Read the metadata and save it in the reader
        with SqliteDict(self.filename, 'metadata', flag='r') as db:
            self._format_version = db.get('format_version', None)
            self._parameters = db.get('Parameters', None)
            self._unknowns = db.get('Unknowns', None)

        # Store the identifier for each iteration in _case_keys
        with SqliteDict(self.filename, 'iterations', flag='r') as db:
            self._case_keys = tuple(db.keys())

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

        # Initialize the Case object from the iterations data
        with SqliteDict(self.filename, 'iterations', flag='r') as iter_db:
            case = Case(self.filename, _case_id, iter_db[_case_id])

        # Set the derivs data for the case if available
        with SqliteDict(self.filename, 'derivs', flag='r') as derivs_db:
            # If derivs weren't recorded then don't bother sending them
            # to the Case.
            if len(derivs_db) == 0:
                pass
            else:
                case._derivs = derivs_db[_case_id].get('Derivatives', None)

        return case

    def list_cases(self):
        """ Return a tuple of the case string identifiers available in this
        instance of the CaseReader.
        """
        return self._case_keys
