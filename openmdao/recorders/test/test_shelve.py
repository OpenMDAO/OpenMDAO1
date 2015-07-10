""" Unit test for the DumpCaseRecorder. """

import unittest
import shelve
from pickle import HIGHEST_PROTOCOL
from openmdao.recorders.shelverecorder import ShelveRecorder
from openmdao.recorders.test.recordertests import RecorderTests
from openmdao.util.recordutil import format_iteration_coordinate
from openmdao.test.testutil import assert_rel_error


class TestShelveRecorder(RecorderTests.Tests):
    filename = "shelve_test"

    def setUp(self):
        self.recorder = ShelveRecorder(self.filename, flag="n", protocol=HIGHEST_PROTOCOL)

    def assertDatasetEquals(self, expected, tolerance):
        # Close the file to ensure it is written to disk.
        self.recorder.out.close()

        with shelve.open(self.filename) as f:
            for coord, expect in expected:
                icoord = format_iteration_coordinate(coord)
                groupings = (
                    ("/Parameters/", expect[0]),
                    ("/Unknowns/", expect[1]),
                    ("/Residuals/", expect[2])
                )

                for label, values in groupings:
                    local_name = icoord + label
                    for key, val in values:
                        assert_rel_error(self, f[local_name + key], val, self.eps)
                        del f[local_name + key]

            # Having deleted all found values, the file should now be empty.
            self.assertEqual(len(f), 0)


if __name__ == "__main__":
    unittest.main()
