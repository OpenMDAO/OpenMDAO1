""" Unit test for the DumpCaseRecorder. """

import unittest

from six import StringIO
from openmdao.recorders.dumpcase import DumpCaseRecorder
from openmdao.recorders.test.recordertests import RecorderTests
from openmdao.util.recordutil import format_iteration_coordinate

class TestDumpCaseRecorder(RecorderTests.Tests):
    def setUp(self):
        self.recorder = DumpCaseRecorder(StringIO())

    def assertDatasetEquals(self, expected, tolerance):
        sout = self.recorder.out
        sout.seek(0)

        for coord, expect in expected:
            icoord = format_iteration_coordinate(coord)

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
