from __future__ import print_function

import unittest
import os
import tempfile
import shutil
import pkg_resources

from openmdao.components.external_code import ExternalCode

from openmdao.core.problem import Problem, Group

DIRECTORY = pkg_resources.resource_filename('openmdao.components', 'test')


class ExternalCodeForTesting(ExternalCode):
    def __init__(self):
        super(ExternalCodeForTesting, self).__init__()


class TestExternalCode(unittest.TestCase):

    def setUp(self):
        self.startdir = os.getcwd()
        self.tempdir = tempfile.mkdtemp(prefix='test_extcode-')
        os.chdir(self.tempdir)
        shutil.copy(os.path.join(DIRECTORY, 'external_code_for_testing.py'), 
                    os.path.join(self.tempdir, 'external_code_for_testing.py'))

        self.extcode = ExternalCodeForTesting()
        self.top = Problem()
        self.top.root = Group()

        self.top.root.add('extcode', self.extcode)

    def tearDown(self):
        os.chdir(self.startdir)
        if not os.environ.get('OPENMDAO_KEEPDIRS', False):
            try:
                shutil.rmtree(self.tempdir)
            except OSError:
                pass

    def test_normal(self):
        self.extcode.options['command'] = ['python', 'external_code_for_testing.py', 'external_code_output.txt']

        self.extcode.options['external_files'] = [
            { 'path': 'external_code_for_testing.py', 'input': True},
            { 'path': 'external_code_output.txt', 'output': True},
            # FileMetadata(path='external_code_for_testing.py', input=True, constant=True),
            # FileMetadata(path='external_code_output.txt', output=True),
            ]

        self.top.setup()
        self.top.run()

    def test_timeout(self):

        self.extcode.options['command'] = ['python', 'external_code_for_testing.py',
             'external_code_output.txt', '--delay', '5']
        self.extcode.options['timeout'] = 1.0

        self.extcode.options['external_files'] = [
            # FileMetadata(path='external_code_for_testing.py', input=True, constant=True),
            {'path':'external_code_for_testing.py', 'input': True},
            ]

        self.top.setup()
        try:
            self.top.run()
        except RuntimeError as exc:
            self.assertEqual(str(exc), 'Timed out')
            self.assertEqual(self.extcode.timed_out, True)
        else:
            self.fail('Expected RunInterrupted')

    def test_badcmd(self):

        # Set command to nonexistant path.
        self.extcode.options['command'] = ['no-such-command', ]

        self.top.setup()
        try:
            self.top.run()
        except ValueError as exc:
            msg = "The command to be executed, 'no-such-command', cannot be found"
            self.assertEqual(str(exc), msg)
            self.assertEqual(self.extcode.return_code, -999999)
        else:
            self.fail('Expected ValueError')

    def test_nullcmd(self):

        self.extcode.stdout = 'nullcmd.out'
        self.extcode.stderr = ExternalCode.STDOUT

        self.top.setup()
        try:
            self.top.run()
        except ValueError as exc:
            self.assertEqual(str(exc), 'Empty command list')
        else:
            self.fail('Expected ValueError')
        finally:
            if os.path.exists(self.extcode.stdout):
                os.remove(self.extcode.stdout)

    def test_env_vars(self):

        self.extcode.options['env_vars'] = {'TEST_ENV_VAR': 'SOME_ENV_VAR_VALUE'}
        self.extcode.options['command'] = ['python', 'external_code_for_testing.py', 'external_code_output.txt', '--write_test_env_var']

        self.top.setup()
        self.top.run()

        # Check to see if output file contains the env var value
        with open(os.path.join(self.tempdir, 'external_code_output.txt'), 'r') as out:
            file_contents = out.read()
        self.assertTrue('SOME_ENV_VAR_VALUE' in file_contents)

    def test_check_external_outputs(self):

        # In the external_files list give it a file that will not be created
        # If check_external_outputs is True, there will be an exception, but since we set it 
        #   to False, no exception should be thrown
        self.extcode.options['check_external_outputs'] = False
        self.extcode.options['external_files'] = [
            # FileMetadata(path='external_code_for_testing.py', input=True, constant=True),
            # FileMetadata(path='does_not_exist.txt', output=True),
            { 'path': 'external_code_for_testing.py', 'input': True },
            { 'path': 'does_not_exist.txt', 'output': True},
            ]
        self.extcode.options['command'] = ['python', 'external_code_for_testing.py', 'external_code_output.txt']

        self.top.setup()
        self.top.run()



if __name__ == "__main__":
    unittest.main()
