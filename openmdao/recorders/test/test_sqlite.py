""" Unit test for the SqliteRecorder. """

import errno
import os
import unittest

from sqlitedict import SqliteDict
from openmdao.recorders import SqliteRecorder
from openmdao.recorders.test import iteration_data_tests
from openmdao.recorders.test import metadata_tests
import openmdao.recorders.test.metadata_tests as MetadataTestCase
from openmdao.core.mpi_wrap import MPI
from openmdao.core.vec_wrapper import _ByObjWrapper
from openmdao.util.record_util import format_iteration_coordinate
from openmdao.test.record_util import create_testcase
from openmdao.test.util import assert_rel_error
from shutil import rmtree
from tempfile import mkdtemp
from six import iteritems
from numpy.testing import assert_allclose

def _assertIterationDataRecorded(test, db, expected, tolerance):
    sentinel = object()
    test.assertEquals(len(db.keys()), len(expected))

    for coord, (t0, t1), params, unknowns, resids in expected:
        iter_coord = format_iteration_coordinate(coord)
        actual_group = db[iter_coord]
        groupings = {
                "timestamp" : None,
                "Parameters" :  params,
                "Unknowns" :  unknowns,
                "Residuals" :  resids,
        }


        if params is None:
            test.assertIsNone(actual_group.get('Parameters', None))
            del groupings['Parameters']

        if unknowns is None:
            test.assertIsNone(actual_group.get('Unknowns', None))
            del groupings['Unknowns']
        
        if resids is None:
            test.assertIsNone(actual_group.get('Residuals', None))
            del groupings['Residuals']

        test.assertEquals(set(actual_group.keys()), set(groupings.keys()))

        timestamp = actual_group['timestamp']
        test.assertTrue( t0 <= timestamp and timestamp <= t1)
        del groupings["timestamp"]

        for label, values in iteritems(groupings):
            actual = actual_group.get(label, None)

            # If len(actual) == len(expected) and actual <= expected, then
            # actual == expected.
            test.assertEqual(len(actual), len(values))

            for key, val in values:
                found_val = actual.get(key, sentinel)

                if found_val is sentinel:
                    test.fail("Did not find key '{0}'".format(key))
             
                try:
                    assert_rel_error(test, found_val, val, tolerance)
                except TypeError:
                    test.assertEqual(val, found_val)

def _assertMetadataRecorded(test, db, expected):
    sentinel = object()
    metadata = db.get('metadata', None)

    if expected is None:
        test.assertIsNone(metadata)
        return

    test.assertEquals(len(metadata), 3)
    pairings = zip(expected, (metadata[x] for x in ('Parameters', 'Unknowns', 'Residuals')))

    for expected, actual in pairings:
        # If len(actual) == len(expected) and actual <= expected, then
        # actual == expected.
        test.assertEqual(len(expected), len(actual))

        for key, val in expected:
            found_val = actual.get(key, sentinel)

            if found_val is sentinel:
                test.fail("Did not find key '{0}'".format(key))
                
            for mkey, mval in iteritems(val):
                mfound_val = found_val[mkey]

                if isinstance(mfound_val, _ByObjWrapper):
                    mfound_val = mfound_val.val

                if isinstance(mval, _ByObjWrapper):
                    mval = mval.val

                try:
                    assert_allclose(mval, mfound_val)
                except TypeError:
                    test.assertEqual(mval, mfound_val)

class TestSqliteRecorder(unittest.TestCase):
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
        db = SqliteDict( self.filename, self.tablename )
        _assertMetadataRecorded( self, db, expected )
        db.close()

    def assertIterationDataRecorded(self, expected, tolerance):
        db = SqliteDict( self.filename, self.tablename )
        _assertIterationDataRecorded(self, db, expected, tolerance)
        db.close()

TestSqliteRecorder = create_testcase(TestSqliteRecorder, [iteration_data_tests, metadata_tests])

if __name__ == "__main__":
    unittest.main()
