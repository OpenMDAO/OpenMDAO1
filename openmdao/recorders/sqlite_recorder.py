"""Class definition for SqliteRecorder, which provides dictionary backed by SQLite"""

from collections import OrderedDict
from openmdao.recorders.base_recorder import BaseRecorder
from openmdao.util.record_util import format_iteration_coordinate

# code originally implemented at https://github.com/piskvorky/sqlitedict/
# -*- coding: utf-8 -*-
#
# This code is distributed under the terms and conditions
# from the Apache License, Version 2.0
#
# http://opensource.org/licenses/apache2.0.php
#
# This code was inspired by:
#  * http://code.activestate.com/recipes/576638-draft-for-an-sqlite3-based-dbm/
#  * http://code.activestate.com/recipes/526618/

"""
A lightweight wrapper around Python's sqlite3 database, with a dict-like interface
and multi-thread access support::

>>> mydict = SqliteDict('some.db', autocommit=True) # the mapping will be persisted to file `some.db`
>>> mydict['some_key'] = any_picklable_object
>>> print mydict['some_key']
>>> print len(mydict) # etc... all dict functions work

Pickle is used internally to serialize the values. Keys are strings.

If you don't use autocommit (default is no autocommit for performance), then
don't forget to call `mydict.commit()` when done with a transaction.

"""

import sqlite3
import os
import sys
import tempfile
import random
import logging
import traceback

major_version = sys.version_info[0]
if major_version < 3:  # py <= 2.x
    if sys.version_info[1] < 5:  # py <= 2.4
        raise ImportError("sqlitedict requires python 2.5 or higher (python 3.3 or higher supported)")

    # necessary to use exec()_ as this would be a SyntaxError in python3.
    # this is an exact port of six.reraise():
    def exec_(_code_, _globs_=None, _locs_=None):
        """Execute code in a namespace."""
        if _globs_ is None:
            frame = sys._getframe(1)
            _globs_ = frame.f_globals
            if _locs_ is None:
                _locs_ = frame.f_locals
            del frame
        elif _locs_ is None:
            _locs_ = _globs_
        exec("""exec _code_ in _globs_, _locs_""")

    exec_("def reraise(tp, value, tb=None):\n"
          "    raise tp, value, tb\n")
else:
    def reraise(tp, value, tb=None):
        if value is None:
            value = tp()
        if value.__traceback__ is not tb:
            raise value.with_traceback(tb)
        raise value

try:
    from cPickle import dumps, loads, HIGHEST_PROTOCOL as PICKLE_PROTOCOL
except ImportError:
    from pickle import dumps, loads, HIGHEST_PROTOCOL as PICKLE_PROTOCOL

# some Python 3 vs 2 imports
try:
    from collections import UserDict as DictClass
except ImportError:
    from UserDict import DictMixin as DictClass


logger = logging.getLogger(__name__)


def open(*args, **kwargs):
    """See documentation of the SqliteDict class."""
    return SqliteDict(*args, **kwargs)


def encode(obj):
    """Serialize an object using pickle to a binary format accepted by SQLite."""
    return sqlite3.Binary(dumps(obj, protocol=PICKLE_PROTOCOL))


def decode(obj):
    """Deserialize objects retrieved from SQLite."""
    return loads(bytes(obj))


class SqliteDict(DictClass):
    def __init__(self, filename=None, tablename='unnamed', flag='c',
                 autocommit=False, journal_mode="DELETE"):
        """
        Initialize a thread-safe sqlite-backed dictionary. The dictionary will
        be a table `tablename` in database file `filename`. A single file (=database)
        may contain multiple tables.

        If no `filename` is given, a random file in temp will be used (and deleted
        from temp once the dict is closed/deleted).

        If you enable `autocommit`, changes will be committed after each operation
        (more inefficient but safer). Otherwise, changes are committed on `self.commit()`,
        `self.clear()` and `self.close()`.

        Set `journal_mode` to 'OFF' if you're experiencing sqlite I/O problems
        or if you need performance and don't care about crash-consistency.

        The `flag` parameter:
          'c': default mode, open for read/write, creating the db/table if necessary.
          'w': open for r/w, but drop `tablename` contents first (start with empty table)
          'n': create a new database (erasing any existing tables, not just `tablename`!).

        """
        self.in_temp = filename is None
        if self.in_temp:
            randpart = hex(random.randint(0, 0xffffff))[2:]
            filename = os.path.join(tempfile.gettempdir(), 'sqldict' + randpart)
        if flag == 'n':
            if os.path.exists(filename):
                os.remove(filename)

        dirname = os.path.dirname(filename)
        if dirname:
            if not os.path.exists(dirname):
                raise RuntimeError('Error! The directory does not exist, %s' % dirname)

        self.filename = filename
        self.tablename = tablename

        logger.info("opening Sqlite table %r in %s" % (tablename, filename))
        MAKE_TABLE = 'CREATE TABLE IF NOT EXISTS %s (key TEXT PRIMARY KEY, value BLOB)' % self.tablename
        self.conn = SqliteMultithread(filename, autocommit=autocommit, journal_mode=journal_mode)
        self.conn.execute(MAKE_TABLE)
        self.conn.commit()
        if flag == 'w':
            self.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

    def __str__(self):
        return "SqliteDict(%s)" % (self.conn.filename)

    def __repr__(self):
        return str(self)  # no need of something complex

    def __len__(self):
        # `select count (*)` is super slow in sqlite (does a linear scan!!)
        # As a result, len() is very slow too once the table size grows beyond trivial.
        # We could keep the total count of rows ourselves, by means of triggers,
        # but that seems too complicated and would slow down normal operation
        # (insert/delete etc).
        GET_LEN = 'SELECT COUNT(*) FROM %s' % self.tablename
        rows = self.conn.select_one(GET_LEN)[0]
        return rows if rows is not None else 0

    def __bool__(self):
        # No elements is False, otherwise True
        GET_MAX = 'SELECT MAX(ROWID) FROM %s' % self.tablename
        m = self.conn.select_one(GET_MAX)[0]
        # Explicit better than implicit and bla bla
        return True if m is not None else False

    def keys(self):
        GET_KEYS = 'SELECT key FROM %s ORDER BY rowid' % self.tablename
        return [key[0] for key in self.conn.select(GET_KEYS)]

    def values(self):
        GET_VALUES = 'SELECT value FROM %s ORDER BY rowid' % self.tablename
        return [decode(value[0]) for value in self.conn.select(GET_VALUES)]

    def items(self):
        GET_ITEMS = 'SELECT key, value FROM %s ORDER BY rowid' % self.tablename
        return [(key, decode(value)) for key, value in self.conn.select(GET_ITEMS)]

    def __contains__(self, key):
        HAS_ITEM = 'SELECT 1 FROM %s WHERE key = ?' % self.tablename
        return self.conn.select_one(HAS_ITEM, (key,)) is not None

    def __getitem__(self, key):
        GET_ITEM = 'SELECT value FROM %s WHERE key = ?' % self.tablename
        item = self.conn.select_one(GET_ITEM, (key,))
        if item is None:
            raise KeyError(key)
        return decode(item[0])

    def __setitem__(self, key, value):
        ADD_ITEM = 'REPLACE INTO %s (key, value) VALUES (?,?)' % self.tablename
        self.conn.execute(ADD_ITEM, (key, encode(value)))

    def __delitem__(self, key):
        if key not in self:
            raise KeyError(key)
        DEL_ITEM = 'DELETE FROM %s WHERE key = ?' % self.tablename
        self.conn.execute(DEL_ITEM, (key,))

    def update(self, items=(), **kwds):
        try:
            items = [(k, encode(v)) for k, v in items.items()]
        except AttributeError:
            pass

        UPDATE_ITEMS = 'REPLACE INTO %s (key, value) VALUES (?, ?)' % self.tablename
        self.conn.executemany(UPDATE_ITEMS, items)
        if kwds:
            self.update(kwds)

    def __iter__(self):
        return iter(self.keys())

    def clear(self):
        CLEAR_ALL = 'DELETE FROM %s;' % self.tablename  # avoid VACUUM, as it gives "OperationalError: database schema has changed"
        self.conn.commit()
        self.conn.execute(CLEAR_ALL)
        self.conn.commit()

    def commit(self, blocking=True):
        """
        Persist all data to disk.

        When `blocking` is False, the commit command is queued, but the data is
        not guaranteed persisted (default implication when autocommit=True).
        """
        if self.conn is not None:
            self.conn.commit(blocking)
    sync = commit

    def close(self, do_log=True):
        if do_log:
            logger.debug("closing %s" % self)
        if hasattr(self, 'conn') and self.conn is not None:
            if self.conn.autocommit:
                # typically calls to commit are non-blocking when autocommit is
                # used.  However, we need to block on close() to ensure any
                # awaiting exceptions are handled and that all data is
                # persisted to disk before returning.
                self.conn.commit(blocking=True)
            self.conn.close()
            self.conn = None
        if self.in_temp:
            try:
                os.remove(self.filename)
            except:
                pass

    def terminate(self):
        """Delete the underlying database file. Use with care."""
        self.close()

        if self.filename == ':memory:':
            return

        logger.info("deleting %s" % self.filename)
        try:
            os.remove(self.filename)
        except (OSError, IOError):
            logger.exception("failed to delete %s" % (self.filename))

    def __del__(self):
        # like close(), but assume globals are gone by now (do not log!)
        self.close(do_log=False)

# Adding extra methods for python 2 compatibility (at import time)
if major_version == 2:
    setattr(SqliteDict, "iterkeys", lambda self: self.keys())
    setattr(SqliteDict, "itervalues", lambda self: self.values())
    setattr(SqliteDict, "iteritems", lambda self: self.items())
    SqliteDict.__nonzero__ = SqliteDict.__bool__
    del SqliteDict.__bool__  # not needed and confusing
#endclass SqliteDict

class SqliteRecorder(BaseRecorder):
    def __init__(self, out, **sqlite_dict_args):
        super(SqliteRecorder, self).__init__()
        self.out = SqliteDict(filename=out, **sqlite_dict_args)
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
        group_name = format_iteration_coordinate(iteration_coordinate)

        self.order.append(group_name)

        f = self.out

        data = OrderedDict([('Parameters', params),
                            ('Unknowns', unknowns),
                            ('Residuals', resids)])

        f[group_name] = data
        f['order'] = self.order
