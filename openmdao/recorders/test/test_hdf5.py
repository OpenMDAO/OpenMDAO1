""" Unit test for the HDF5Recorder. """

import errno
import os
from shutil import rmtree
from tempfile import mkdtemp
import unittest
import h5py
from openmdao.test.util import assert_rel_error
import openmdao.recorders.test.recorder_tests as iteration_testcase
import openmdao.recorders.test.metadata_tests as metadata_testcase
from openmdao.util.record_util import format_iteration_coordinate
from openmdao.test.record_util import create_testcase
from six.moves import zip
from six import iteritems

SKIP = False

try:
    from openmdao.recorders.hdf5_recorder import HDF5Recorder
except ImportError:
    # Necessary for the file to parse
    from openmdao.recorders.base_recorder import BaseRecorder
    HDF5Recorder = BaseRecorder
    SKIP = True

class TestHDF5Recorder(unittest.TestCase):
    def setUp(self):
        if SKIP:
            raise unittest.SkipTest("Could not import HDF5Recorder. Is h5py installed?")

        self.dir = mkdtemp()
        self.filename = os.path.join(self.dir, "tmp.hdf5")
        self.recorder = HDF5Recorder(self.filename)
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
        sentinel = object()
        hdf = h5py.File(self.filename, 'r')

        metadata = hdf.get('metadata', None)

        if expected is None:
            self.assertIsNone(metadata)
            return

        self.assertEquals(len(metadata), 3)

        pairings = zip(expected, (metadata[x] for x in ('Parameters', 'Unknowns', 'Residuals')))

        for expected, actual in pairings:
            self.assertEqual(len(expected), len(actual))

            for key, val in expected:
                found_val = actual.get(key, sentinel)

                if found_val is sentinel:
                    self.fail("Did not find key '{0}'".format(key))

                for mkey, mval in iteritems(val):
                    found_val = actual[key].get(mkey, sentinel)

                    if found_val is sentinel:
                        self.fail("Did not find metadata key '{0}'".format(mkey))

                    self.assertEqual(found_val.value, mval)

    def assertIterationDataRecorded(self, expected, tolerance):
        sentinel = object()
        hdf = h5py.File(self.filename, 'r')

        for coord, (t0, t1), params, unknowns, resids in expected:
            icoord = format_iteration_coordinate(coord)
            actual_group = hdf[icoord]

            groupings = {
                    "Parameters" :  params,
                    "Unknowns" :  unknowns,
                    "Residuals" :  resids,
            }
            
            if params is None:
                self.assertIsNone(actual_group.get('Parameters', None))
                del groupings['Parameters']

            if unknowns is None:
                self.assertIsNone(actual_group.get('Unknowns', None))
                del groupings['Unknowns']
            
            if resids is None:
                self.assertIsNone(actual_group.get('Residuals', None))
                del groupings['Residuals']
   
            timestamp = actual_group.attrs['timestamp']
            self.assertTrue(t0 <= timestamp and timestamp <= t1)

            for label, values in iteritems(groupings):
                actual = actual_group[label]

                # If len(actual) == len(expected) and actual <= expected, then
                # actual == expected.
                self.assertEqual(len(actual), len(values))

                for key, val in values:
                    found_val = actual.get(key, sentinel)

                    if found_val is sentinel:
                        self.fail("Did not find key '{0}'.".format(key))

                    assert_rel_error(self, found_val.value, val, tolerance)

        hdf.close()

TestHDF5Recorder = create_testcase(TestHDF5Recorder, [iteration_testcase, metadata_testcase])

if __name__ == "__main__":
    unittest.main()
