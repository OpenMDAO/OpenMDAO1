""" Unit test for the SqliteRecorder. """
import numpy as np
import errno
import os
import shelve
import unittest

from openmdao.core.vec_wrapper import _ByObjWrapper
from recorder_tests import RecorderTests
from sqlitedict import SqliteDict
from openmdao.core.mpi_wrap import MPI
from openmdao.api import SqliteRecorder
from openmdao.test.util import assert_rel_error
from openmdao.util.record_util import format_iteration_coordinate

from pickle import HIGHEST_PROTOCOL
from shutil import rmtree
from tempfile import mkdtemp

class TestSqliteRecorder(RecorderTests.Tests):
    filename = ""
    dir = ""

    def setUp(self):
        if not MPI or MPI.COMM_WORLD.rank == 0:
            self.dir = mkdtemp()
            self.filename = os.path.join(self.dir, "sqlite_test")
            self.tablename = 'openmdao'
            self.recorder = SqliteRecorder(self.filename)

    def tearDown(self):
        if not MPI or MPI.COMM_WORLD.rank == 0:
            super(TestSqliteRecorder, self).tearDown()
            try:
                rmtree(self.dir)
            except OSError as e:
                # If directory already deleted, keep going
                if e.errno != errno.ENOENT:
                    raise e

    def assertDatasetEquals(self, expected, tolerance):
        # Close the file to ensure it is written to disk.
        self.recorder.close()
        # self.recorder.out = None

        sentinel = object()

        db = SqliteDict( self.filename, self.tablename )


        for coord, expect in expected:
            iter_coord = format_iteration_coordinate(coord)

            groupings = (
                ("Parameters", expect[0]),
                ("Unknowns", expect[1]),
                ("Residuals", expect[2])
            )

            #### Need to get the record with the key of 'iter_coord'
            actual_group = db[iter_coord]
            timestamp = actual_group['timestamp']

            self.assertTrue(self.t0 <= timestamp and timestamp <= self.t1 )

            for label, values in groupings:
                actual = actual_group[label]
                # If len(actual) == len(expected) and actual <= expected, then
                # actual == expected.
                self.assertEqual(len(actual), len(values))
                for key, val in values:
                    found_val = actual.get(key, sentinel)
                    if found_val is sentinel:
                        self.fail("Did not find key '{0}'".format(key))

                    if isinstance(found_val, _ByObjWrapper):
                        found_val = found_val.val

                    try:
                        assert_rel_error(self, found_val, val, tolerance)
                    except TypeError as error:
                        self.assertEqual(found_val, val)

            del db[iter_coord]
            ######## delete the record with the key 'iter_coord'

        # Having deleted all found values, the file should now be empty.
        ###### Need a way to get the number of records in the main table
        self.assertEqual(len(db), 0)

        db.close()


if __name__ == "__main__":
    from openmdao.test.mpi_util import mpirun_tests
    mpirun_tests()
