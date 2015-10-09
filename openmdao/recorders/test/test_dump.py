""" Unit test for the DumpRecorder. """

from shutil import rmtree
from tempfile import mkdtemp
import errno
import os
import unittest

from six import StringIO

from openmdao.recorders import DumpRecorder
from openmdao.recorders.test import iteration_data_tests 
from openmdao.util.record_util import format_iteration_coordinate
from openmdao.test.record_util import create_testcase

class TestDumpRecorder(unittest.TestCase):
    def setUp(self):
        self.dir = mkdtemp()
        self.filename = os.path.join(self.dir, "sqlite_test")
        self.recorder = DumpRecorder(self.filename)
        self.eps = 1e-5

    def tearDown(self):
        try:
            rmtree(self.dir)
        except OSError as e:
            # If directory already deleted, keep going
            if e.errno != errno.ENOENT:
                raise e

    def assertIterationDataRecorded(self, expected, tolerance):
        sout = open(self.filename)

        for coord, (t0, t1), params, unknowns, resids in expected:
            icoord = format_iteration_coordinate(coord)

            line = sout.readline()
            self.assertTrue('Timestamp: ' in line)
            timestamp = float(line[11:-1])
            self.assertTrue(t0 <= timestamp and timestamp <= t1)

            line = sout.readline()
            self.assertEqual("Iteration Coordinate: {0}\n".format(icoord), line)

            groupings = []

            if params is not None:
                groupings.append(("Params:\n", params))

            if unknowns is not None:
                groupings.append(("Unknowns:\n", unknowns))

            if resids is not None:
                groupings.append(("Resids:\n", resids))

            for header, exp in groupings:
                line = sout.readline()
                self.assertEqual(header, line)
                for key, val in exp:
                    line = sout.readline()
                    self.assertEqual("  {0}: {1}\n".format(key, str(val)), line)

        sout.close()

TestDumpRecorder = create_testcase(TestDumpRecorder, [iteration_data_tests])

if __name__ == "__main__":
    unittest.main()
