""" Unit test for the ShelveRecorder. """

import errno
import os
import shelve
import unittest
from openmdao.recorders.shelverecorder import ShelveRecorder
from openmdao.recorders.test.recordertests import RecorderTests
from openmdao.test.testutil import assert_rel_error
from openmdao.util.recordutil import format_iteration_coordinate
from pickle import HIGHEST_PROTOCOL
from shutil import rmtree
from tempfile import mkdtemp


class TestShelveRecorder(RecorderTests.Tests):
    filename = ""
    dir = ""

    def setUp(self):
        self.dir = mkdtemp()
        self.filename = os.path.join(self.dir, "shelve_test")

        self.recorder = ShelveRecorder(self.filename, flag="n", protocol=HIGHEST_PROTOCOL)

    def tearDown(self):
        super(TestShelveRecorder, self).tearDown()
        try:
            rmtree(self.dir)
        except OSError as e:
            # If directory already deleted, keep going
            if e.errno != errno.ENOENT:
                raise e

    def assertDatasetEquals(self, expected, tolerance):
        # Close the file to ensure it is written to disk.
        self.recorder.out.close()
        self.recorder.out = None

        sentinel = object()

        f = shelve.open(self.filename)

        order = f['order']
        del f['order']

        for coord, expect in expected:
            iter_coord = format_iteration_coordinate(coord)

            self.assertEqual(order.pop(0), iter_coord)

            groupings = (
                ("Parameters", expect[0]),
                ("Unknowns", expect[1]),
                ("Residuals", expect[2])
            )

            actual_group = f[iter_coord]

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
            del f[iter_coord]

        # Having deleted all found values, the file should now be empty.
        self.assertEqual(len(f), 0)

        # As should the ordering.
        self.assertEqual(len(order), 0)

        f.close()


if __name__ == "__main__":
    unittest.main()
