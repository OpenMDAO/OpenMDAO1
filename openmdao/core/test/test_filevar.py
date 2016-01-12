
import os
import unittest
import struct
from tempfile import mkdtemp
from shutil import rmtree

from openmdao.api import Problem, Component, Group, ExecComp, FileRef

class FileSrc(Component):
    def __init__(self):
        super(FileSrc, self).__init__()
        self.add_output("ascii_dat", FileRef("ascii.dat"))
        self.add_output("bin_dat", FileRef("bin.dat"))

    def solve_nonlinear(self, params, unknowns, resids):
        # generate the output files
        ascii_fref = unknowns['ascii_dat']
        with ascii_fref.open(self, 'w') as f:
            f.write("this is line 1\nthis is line 2")

        bin_fref = unknowns['bin_dat']
        with bin_fref.open(self, 'wb') as f:
            f.write(struct.pack('ddd', 3.14, 10.6, 123.456))

class FilePass(Component):
    def __init__(self):
        super(FilePass, self).__init__()
        self.add_param("ascii_in", FileRef("ascii.dat"))
        self.add_param("bin_in", FileRef("bin.dat"))
        self.add_output("ascii_out", FileRef("ascii.out"))
        self.add_output("bin_out", FileRef("bin.out"))

    def solve_nonlinear(self, params, unknowns, resids):
        ascii_in_ref = params['ascii_in']
        bin_in_ref = params['bin_in']

        # read from input FileRefs
        with ascii_in_ref.open(self, 'r') as f:
            ascii_dat = f.read()

        with bin_in_ref.open(self, 'rb') as f:
            bin_dat = struct.unpack('ddd', f.read())

        # modify data
        ascii_dat += "\nthis is line 3"
        bin_dat = list(bin_dat) + [-98.76]

        ascii_out_ref = unknowns['ascii_out']
        bin_out_ref = unknowns['bin_out']

        # write to output FileRefs
        with ascii_out_ref.open(self, 'w') as f:
            f.write(ascii_dat)

        with bin_out_ref.open(self, 'wb') as f:
            f.write(struct.pack('dddd', *bin_dat))

class FileSink(Component):
    def __init__(self):
        super(FileSink, self).__init__()
        self.add_param("ascii_in", FileRef("ascii_final.dat"))
        self.add_param("bin_in", FileRef("bin_final.dat"))

    def solve_nonlinear(self, params, unknowns, resids):
        pass # nothing to do


class TestFileVar(unittest.TestCase):

    def setUp(self):
        self.startdir = os.getcwd()
        self.tmpdir = mkdtemp()
        os.chdir(self.tmpdir)

    def tearDown(self):
        os.chdir(self.startdir)
        try:
            rmtree(self.tmpdir)
        except OSError as e:
            # If directory already deleted, keep going
            if e.errno != errno.ENOENT:
                raise e

    def test_same_dir(self):
        p = Problem(root=Group())
        root = p.root
        src = root.add("src", FileSrc())
        middle = root.add("middle", FilePass())
        sink = root.add("sink", FileSink())

        root.connect("src.ascii_dat", "middle.ascii_in")
        root.connect("src.bin_dat", "middle.bin_in")
        root.connect("middle.ascii_out", "sink.ascii_in")
        root.connect("middle.bin_out", "sink.bin_in")

        p.setup(check=False)
        p.run()

        self.assertEqual(sink.params['ascii_in'].open(sink,'r').read(),
                         "this is line 1\nthis is line 2\nthis is line 3")
