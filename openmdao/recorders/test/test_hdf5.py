""" Unit test for the HDF5Recorder. """

import unittest
from openmdao.test.testutil import assert_rel_error
from openmdao.recorders.test.recordertests import RecorderTests
from openmdao.util.recordutil import format_iteration_coordinate
from six.moves import zip

SKIP = False

try:
    from openmdao.recorders.hdf5recorder import HDF5Recorder
except ImportError:
    # Necessary for the file to parse
    from openmdao.recorders.baserecorder import BaseRecorder
    HDF5Recorder = BaseRecorder
    SKIP = True


class TestHDF5Recorder(RecorderTests.Tests):
    def setUp(self):
        if SKIP:
            raise unittest.SkipTest("Could not import HDF5Recorder. Is h5py installed?")
        self.recorder = HDF5Recorder('tmp.hdf5', driver='core', backing_store=False)

    def assertDatasetEquals(self, expected, tolerance):
        for coord, expect in expected:
            icoord = format_iteration_coordinate(coord)

            f = self.recorder.out[icoord]
            params = f['Parameters']
            unknowns = f['Unknowns']
            resids = f['Residuals']

            sentinel = object()

            # If len(actual) == len(expected) and actual <= expected, then
            # actual == expected.
            for actual, exp in zip((params, unknowns, resids), expect):
                self.assertEqual(len(actual), len(exp))
                for key, val in exp:
                    found_val = actual.get(key, sentinel)
                    if found_val is sentinel:
                        self.fail("Did not find key '{0}'.".format(key))
                    assert_rel_error(self, found_val[()], val, tolerance)

if __name__ == "__main__":
    unittest.main()
