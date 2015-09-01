""" Unit test for the SqliteRecorder. """

import errno
import os
import shelve
import unittest

from sqlitedict import SqliteDict
from openmdao.recorders.sqlite_recorder import SqliteRecorder
from openmdao.recorders.test.recorder_tests import RecorderTests
from openmdao.test.util import assert_rel_error
from openmdao.util.record_util import format_iteration_coordinate

from pickle import HIGHEST_PROTOCOL
from shutil import rmtree
from tempfile import mkdtemp


class TestSqliteRecorder(RecorderTests.Tests):
    filename = ""
    dir = ""

    def setUp(self):
        self.dir = mkdtemp()
        self.filename = os.path.join(self.dir, "sqlite_test")
        self.tablename = 'openmdao'
        self.recorder = SqliteRecorder(self.filename)

    def tearDown(self):
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

        ###### Need a way to get a list of the group_names in the order in which they were written and put it in  a variable named order
        order = db['order']
        del db['order']

        for coord, expect in expected:
            iter_coord = format_iteration_coordinate(coord)

            self.assertEqual(order.pop(0), iter_coord)

            groupings = (
                ("Parameters", expect[0]),
                ("Unknowns", expect[1]),
                ("Residuals", expect[2])
            )

            #### Need to get the record with the key of 'iter_coord'
            actual_group = db[iter_coord]

            for label, values in groupings:
                actual = actual_group[label]
                # If len(actual) == len(expected) and actual <= expected, then
                # actual == expected.
                self.assertEqual(len(actual), len(values))
                for key, val in values:
                    found_val = actual.get(key, sentinel)
                    if found_val is sentinel:
                        self.fail("Did not find key '{0}'".format(key))
                    assert_rel_error(self, found_val, val, tolerance)
            del db[iter_coord]
            ######## delete the record with the key 'iter_coord'

        # Having deleted all found values, the file should now be empty.
        ###### Need a way to get the number of records in the main table
        self.assertEqual(len(db), 0)

        # As should the ordering.
        self.assertEqual(len(order), 0)

        db.close()


if __name__ == "__main__":
    unittest.main()
