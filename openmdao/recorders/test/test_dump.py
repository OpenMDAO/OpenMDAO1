""" Unit test for the DumpRecorder. """

import unittest

from six import StringIO
from openmdao.api import DumpRecorder
from openmdao.recorders.test.recorder_tests import RecorderTests
from openmdao.util.record_util import format_iteration_coordinate

class TestDumpRecorder(RecorderTests.Tests):
    def setUp(self):
        self.recorder = DumpRecorder(StringIO())

    def assertDatasetEquals(self, expected, tolerance):
        sout = self.recorder.out
        sout.seek(0)

        for coord, expect in expected:
            icoord = format_iteration_coordinate(coord)

            line = sout.readline()
            self.assertTrue('Timestamp: ' in line)
            timestamp = float(line[11:-1])
            self.assertTrue(self.t0 <= timestamp and timestamp <= self.t1)

            line = sout.readline()
            self.assertEqual("Iteration Coordinate: {0}\n".format(icoord), line)

            groupings = (
                ("Params:\n", expect[0]),
                ("Unknowns:\n", expect[1]),
                ("Resids:\n", expect[2])
            )

            for header, exp in groupings:
                line = sout.readline()
                self.assertEqual(header, line)
                for key, val in exp:
                    line = sout.readline()
                    self.assertEqual("  {0}: {1}\n".format(key, str(val)), line)

if __name__ == "__main__":
    unittest.main()
