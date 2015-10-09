""" Unit test for the SqliteRecorder. """
import errno
import os
import shelve
import unittest
from pickle import HIGHEST_PROTOCOL
from shutil import rmtree
from tempfile import mkdtemp

import numpy as np
from sqlitedict import SqliteDict

from openmdao.core.vec_wrapper import _ByObjWrapper
from openmdao.recorders import SqliteRecorder
from openmdao.recorders.test.test_sqlite import _assertMetadataRecorded, _assertIterationDataRecorded
from openmdao.test.record_util import create_testcase
from openmdao.test.mpi_util import MPITestCase

import iteration_data_tests
import metadata_tests

class TestSqliteRecorder(MPITestCase):
    filename = ""
    dir = ""
    N_PROCS = 2

    def setUp(self):
        self.dir = mkdtemp()
        self.filename = os.path.join(self.dir, "sqlite_test")
        self.tablename = 'openmdao'
        self.recorder = SqliteRecorder(self.filename)
        self.recorder.options['record_metadata'] = False
        self.eps = 1e-5

    def tearDown(self):
        try:
            rmtree(self.dir)
        except OSError as e:
            # If directory already deleted, keep going
            if e.errno != errno.ENOENT:
                raise e

    def assertMetadataRecorded(self, expected):
        db = SqliteDict(self.filename, self.tablename)
        _assertMetadataRecorded(self, db, expected)
        db.close()

    def assertIterationDataRecorded(self, expected, tolerance, root):
        if self.comm.rank != 0:
            return

        db = SqliteDict(self.filename, self.tablename)
        _assertIterationDataRecorded(self, db, expected, tolerance)
        db.close()

TestSqliteRecorder = create_testcase(TestSqliteRecorder, [iteration_data_tests, metadata_tests])

if __name__ == "__main__":
    from openmdao.test.mpi_util import mpirun_tests
    mpirun_tests()
