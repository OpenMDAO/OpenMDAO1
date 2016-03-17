
import os
import unittest
import struct
from tempfile import mkdtemp
from shutil import rmtree
import errno

from openmdao.api import Problem, Component, Group, ExecComp, FileRef
from openmdao.util.file_util import build_directory

class FileSrc(Component):
    def __init__(self, path=''):
        super(FileSrc, self).__init__()
        self.add_output("ascii_dat", FileRef(os.path.join(path,"ascii.dat")))
        self.add_output("bin_dat", FileRef(os.path.join(path,"bin.dat")),
                                           binary=True)

    def solve_nonlinear(self, params, unknowns, resids):
        # generate the output files
        ascii_fref = unknowns['ascii_dat']
        with ascii_fref.open('w') as f:
            f.write("this is line 1\nthis is line 2")

        bin_fref = unknowns['bin_dat']
        with bin_fref.open('wb') as f:
            f.write(struct.pack('ddd', 3.14, 10.6, 123.456))

class FilePass(Component):
    def __init__(self, path=''):
        super(FilePass, self).__init__()
        self.add_param("ascii_in", FileRef(os.path.join(path,"ascii.dat")))
        self.add_param("bin_in", FileRef(os.path.join(path,"bin.dat")),
                                         binary=True)
        self.add_output("ascii_out", FileRef(os.path.join(path,"ascii.out")))
        self.add_output("bin_out", FileRef(os.path.join(path,"bin.out")),
                                           binary=True)

    def solve_nonlinear(self, params, unknowns, resids):
        ascii_in_ref = params['ascii_in']
        bin_in_ref = params['bin_in']

        # read from input FileRefs
        with ascii_in_ref.open('r') as f:
            ascii_dat = f.read()

        with bin_in_ref.open('rb') as f:
            bin_dat = struct.unpack('ddd', f.read())

        # modify data
        ascii_dat += "\nthis is line 3"
        bin_dat = list(bin_dat) + [-98.76]

        ascii_out_ref = unknowns['ascii_out']
        bin_out_ref = unknowns['bin_out']

        # write to output FileRefs
        with ascii_out_ref.open('w') as f:
            f.write(ascii_dat)

        with bin_out_ref.open('wb') as f:
            f.write(struct.pack('dddd', *bin_dat))

class FileSink(Component):
    def __init__(self, path=''):
        super(FileSink, self).__init__()
        self.add_param("ascii_in", FileRef(os.path.join(path,"ascii_final.dat")))
        self.add_param("bin_in", FileRef(os.path.join(path,"bin_final.dat")),
                                         binary=True)

    def solve_nonlinear(self, params, unknowns, resids):
        pass # nothing to do

class FileBin(Component):
    def __init__(self):
        super(FileBin, self).__init__()
        self.add_output("fout", FileRef("file.dat"), binary=True)

class FileNoBin(Component):
    def __init__(self):
        super(FileNoBin, self).__init__()
        self.add_param("fin", FileRef("file.dat"))



class TestFileRef(unittest.TestCase):

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
            if e.errno not in (errno.ENOENT, errno.EACCES, errno.EPERM):
                raise e

    def _compare_files(self, src, middle, sink):
        with src.unknowns['ascii_dat'].open('r') as f:
            src_ascii_dat = f.read()

        with src.unknowns['bin_dat'].open('r') as f:
            src_bin_dat = struct.unpack('ddd', f.read())

        with middle.params['ascii_in'].open('r') as f:
            middle_ascii_in = f.read()

        with middle.params['bin_in'].open('r') as f:
            middle_bin_dat = struct.unpack('ddd', f.read())

        with middle.unknowns['ascii_out'].open('r') as f:
            middle_ascii_out = f.read()

        with middle.unknowns['bin_out'].open('r') as f:
            middle_bin_out = struct.unpack('dddd', f.read())

        with sink.params['ascii_in'].open('r') as f:
            sink_ascii_in = f.read()

        with sink.params['bin_in'].open('rb') as f:
            sink_bin_in = struct.unpack('dddd', f.read())

        self.assertEqual(src_ascii_dat, "this is line 1\nthis is line 2")
        self.assertEqual(src_bin_dat, (3.14, 10.6, 123.456))
        self.assertEqual(middle_ascii_in, src_ascii_dat)
        self.assertEqual(middle_bin_dat, src_bin_dat)
        self.assertEqual(middle_ascii_out, "this is line 1\nthis is line 2\nthis is line 3")
        self.assertEqual(middle_bin_out, (3.14, 10.6, 123.456, -98.76))
        self.assertEqual(sink_ascii_in, middle_ascii_out)
        self.assertEqual(sink_bin_in, middle_bin_out)

    def _build_model(self, path=''):
        p = Problem(root=Group())
        root = p.root
        src = root.add("src", FileSrc(path=path))
        middle = root.add("middle", FilePass(path=path))
        sink = root.add("sink", FileSink(path=path))

        root.connect("src.ascii_dat", "middle.ascii_in")
        root.connect("src.bin_dat", "middle.bin_in")
        root.connect("middle.ascii_out", "sink.ascii_in")
        root.connect("middle.bin_out", "sink.bin_in")

        return p, src, middle, sink

    def test_same_dir(self):
        p, src, middle, sink = self._build_model()

        p.setup(check=False)
        p.run()

        self._compare_files(src, middle, sink)

        # check presence of files
        files = set(os.listdir('.'))
        self.assertEqual(files, set(['ascii.dat', 'ascii.out',
                                     'bin.dat', 'bin.out',
                                     'ascii_final.dat', 'bin_final.dat']))

    def test_diff_dirs1(self):
        os.mkdir('src')
        os.mkdir('middle')
        os.mkdir('sink')

        p, src, middle, sink = self._build_model()

        src.directory = 'src'
        middle.directory = "middle"
        sink.directory = 'sink'

        p.setup(check=False)
        p.run()

        self._compare_files(src, middle, sink)

        # check presence of files/directories
        files = set(os.listdir('.'))
        self.assertEqual(files, set(['src', 'middle', 'sink']))
        files = set(os.listdir('src'))
        self.assertEqual(files, set(['ascii.dat', 'bin.dat']))
        files = set(os.listdir('middle'))
        self.assertEqual(files, set(['ascii.dat', 'bin.dat', 'ascii.out', 'bin.out']))
        files = set(os.listdir('sink'))
        self.assertEqual(files, set(['ascii_final.dat', 'bin_final.dat']))

    def test_diff_dirs2(self):
        # dirs introduced via system.directory and in FileRef path attrs

        p, src, middle, sink = self._build_model(path='nest')

        src.directory = 'src'
        src.create_dirs = True
        middle.directory = "middle"
        middle.create_dirs = True
        sink.directory = 'sink'
        sink.create_dirs = True

        p.setup(check=False)
        p.run()

        self._compare_files(src, middle, sink)

        # check presence of files/directories
        files = set(os.listdir('.'))
        self.assertEqual(files, set(['src', 'middle', 'sink']))

        files = set(os.listdir('src'))
        self.assertEqual(files, set(['nest']))
        files = set(os.listdir('middle'))
        self.assertEqual(files, set(['nest']))
        files = set(os.listdir('sink'))
        self.assertEqual(files, set(['nest']))

        files = set(os.listdir(os.path.join('src', 'nest')))
        self.assertEqual(files, set(['ascii.dat', 'bin.dat']))
        files = set(os.listdir(os.path.join('middle', 'nest')))
        self.assertEqual(files, set(['ascii.dat', 'bin.dat', 'ascii.out', 'bin.out']))
        files = set(os.listdir(os.path.join('sink', 'nest')))
        self.assertEqual(files, set(['ascii_final.dat', 'bin_final.dat']))

    def test_diff_dirs3(self):
        # dirs introduced via system.directory and in FileRef path attrs but
        # create_dirs is not set

        p, src, middle, sink = self._build_model(path='nest')

        src.directory = 'src'
        middle.directory = "middle"
        sink.directory = 'sink'

        try:
            p.setup(check=False)
        except Exception as err:
            self.assertEqual(str(err).replace('/private',''), "directory '%s' doesn't "
                                       "exist for FileRef('%s'). Set "
                                       "create_dirs=True in system 'src' to create the "
                                       "directory automatically." %
                                       (os.path.join(self.tmpdir,'src','nest'),
                                       os.path.join('nest','ascii.dat')))
        else:
            self.fail("Exception expected")

    def test_mismatch(self):
        p = Problem(root=Group())
        root = p.root
        binsys = root.add("binsys", FileBin())
        nobinsys = root.add("nobinsys", FileNoBin())
        root.connect('binsys.fout', 'nobinsys.fin')
        try:
            p.setup(check=False)
        except Exception as err:
            self.assertEqual(str(err), "Source FileRef has (binary=True) and dest has (binary=False).")
        else:
            self.fail("Exception expected")

    def test_ref_unconnected(self):
        p = Problem(root=Group())
        root = p.root
        src = root.add("src", FileSrc())
        sink = root.add("sink", FileSink())
        sink2 = root.add("sink2", FileSink())

        root.connect("src.ascii_dat", "sink.ascii_in")

        try:
            p.setup(check=False)
        except Exception as err:
            self.assertTrue("FileRef param 'sink2.ascii_in' is unconnected but will "
                             "be overwritten by the following FileRef unknown(s): "
                             "['src.ascii_dat']. Files referred to by the FileRef unknowns are: "
                             "['%s']. To remove this error, make a "
                             "connection between sink2.ascii_in and a FileRef unknown." %
                             os.path.join(self.tmpdir, 'ascii.dat') in str(err).replace('\\\\','\\').replace('/private',''), )
        else:
            self.fail("Exception expected")

    def test_ref_multi_connections(self):
        p = Problem(root=Group())
        root = p.root
        src = root.add("src", FileSrc())
        src2 = root.add("src2", FileSrc())
        sink = root.add("sink", FileSink())
        sink2 = root.add("sink2", FileSink())

        root.connect("src.ascii_dat", "sink.ascii_in")
        root.connect("src2.ascii_dat", "sink2.ascii_in")

        try:
            p.setup(check=False)
        except Exception as err:
            # osx tacks a /private to the beginning of the tmp pathname, resulting
            # in test diffs, so just get rid of it
            msg = "Input file '%s' is referenced from FileRef param(s) ['sink.ascii_in', " \
                "'sink2.ascii_in'], which are connected to multiple output " \
                "FileRefs: ['src.ascii_dat', 'src2.ascii_dat']. Those FileRefs "  \
                "reference the following files: %s." % (
                 os.path.join(self.tmpdir, 'ascii_final.dat'),
                 [os.path.join(self.tmpdir, 'ascii.dat'),
                 os.path.join(self.tmpdir, 'ascii.dat')])
            self.assertTrue(msg in str(err).replace('/private',''))
        else:
            self.fail("Exception expected")

class FileComp(Component):
    def __init__(self, *args, **kwargs):
        super(FileComp, self).__init__(*args, **kwargs)
        self.add_output("out", 0.0)

    def solve_nonlinear(self, params, unknowns, resids):
        with open(self.name+".in", 'r') as f:
            unknowns['out'] = float(f.read())


class TestDirectory(unittest.TestCase):

    def setUp(self):
        self.startdir = os.getcwd()
        self.tmpdir = mkdtemp()
        os.chdir(self.tmpdir)
        build_directory({
            'top': {
                'nest1': {
                    'c1.in': '3.14'
                },
                'c2.in': '5.0'
            }
        })

    def tearDown(self):
        os.chdir(self.startdir)
        try:
            rmtree(self.tmpdir)
        except OSError as e:
            # If directory already deleted, keep going
            if e.errno not in (errno.ENOENT, errno.EACCES, errno.EPERM):
                raise e

    def test_sysdirs(self):
        p = Problem(root=Group())
        p.root.directory = 'top'
        nest1 = p.root.add('nest1', Group())
        nest1.directory = 'nest1'
        nest1.add('c1', FileComp())
        nest2 = p.root.add('nest2', Group())
        nest2.add('c2', FileComp())
        p.setup(check=False)
        p.run()
        self.assertEqual(p['nest1.c1.out'], 3.14)
        self.assertEqual(p['nest2.c2.out'], 5.0)


if __name__ == '__main__':
    unittest.main()
