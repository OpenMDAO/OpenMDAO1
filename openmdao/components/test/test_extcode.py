from __future__ import print_function

import unittest
import os
import tempfile
import shutil
import pkg_resources


from six import text_type

import numpy as np

from openmdao.components.external_code import ExternalCode

from openmdao.core.problem import Problem, Group

from openmdao.util.fileutil import FileMetadata

DIRECTORY = pkg_resources.resource_filename('openmdao.components', 'test')


class ExternalCodeForTesting(ExternalCode):
    def __init__(self):
        super(ExternalCodeForTesting, self).__init__()


class TestExternalCode(unittest.TestCase):

    def setUp(self):
        self.startdir = os.getcwd()
        self.tempdir = tempfile.mkdtemp(prefix='test_extcode-')
        print (self.tempdir)
        os.chdir(self.tempdir)
        shutil.copy(os.path.join(DIRECTORY, 'external_code_for_testing.py'), 
                    os.path.join(self.tempdir, 'external_code_for_testing.py'))

        self.extcode = ExternalCodeForTesting()

    def tearDown(self):
        os.chdir(self.startdir)
        if not os.environ.get('OPENMDAO_KEEPDIRS', False):
            try:
                shutil.rmtree(self.tempdir)
            except OSError:
                pass

    def test_normal(self):
        self.extcode.options['command'] = ['python', 'external_code_for_testing.py']

        self.extcode.options['external_files'] = [
            FileMetadata(path='external_code_for_testing.py', input=True, constant=True),
            FileMetadata(path='external_code_output.txt', output=True),
            ]

        top = Problem()
        root = top.root = Group()

        root.add('extcode', self.extcode)

        top.setup()
        top.run()

    def test_timeout(self):

        self.extcode.options['command'] = ['python', 'external_code_for_testing.py', '5']
        self.extcode.options['timeout'] = 1.0

        self.extcode.options['external_files'] = [
            FileMetadata(path='external_code_for_testing.py', input=True, constant=True),
            ]

        top = Problem()
        root = top.root = Group()

        root.add('extcode', self.extcode)

        top.setup()
        try:
            top.run()
        except RuntimeError as exc:
            self.assertEqual(str(exc), 'Timed out')
            self.assertEqual(self.extcode.timed_out, True)
        else:
            self.fail('Expected RunInterrupted')

    def test_badcmd(self):

        # Set command to nonexistant path.
        self.extcode.options['command'] = ['no-such-command', ]

        top = Problem()
        root = top.root = Group()

        root.add('extcode', self.extcode)

        top.setup()
        try:
            top.run()
        except ValueError as exc:
            msg = "The command to be executed, 'no-such-command', cannot be found"
            self.assertEqual(str(exc), msg)
            self.assertEqual(self.extcode.return_code, -999999)
        else:
            self.fail('Expected ValueError')

    def test_nullcmd(self):

        self.extcode.stdout = 'nullcmd.out'
        self.extcode.stderr = ExternalCode.STDOUT

        top = Problem()
        root = top.root = Group()

        root.add('extcode', self.extcode)

        top.setup()
        try:
            top.run()
        except ValueError as exc:
            self.assertEqual(str(exc), 'Empty command list')
        else:
            self.fail('Expected ValueError')
        finally:
            if os.path.exists(self.extcode.stdout):
                os.remove(self.extcode.stdout)


if __name__ == "__main__":
    unittest.main()
