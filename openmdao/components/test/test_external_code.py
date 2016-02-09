from __future__ import print_function

import unittest
import os
import sys
import tempfile
import shutil
import pkg_resources

from openmdao.api import Problem, Group, ExternalCode
from openmdao.components.external_code import STDOUT

DIRECTORY = os.path.dirname((os.path.abspath(__file__)))

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

        self.extcode.options['external_input_files'] = ['external_code_for_testing.py',]
        self.extcode.options['external_output_files'] = ['external_code_output.txt',]

        dev_null = open(os.devnull, 'w')
        self.top.setup(check=True, out_stream=dev_null)
        self.top.run()
        self.assertEqual(self.extcode.timed_out, False)
        self.assertEqual(self.extcode.errored_out, False)

    # def test_ls_command(self):
    #     output_filename = 'ls_output.txt'
    #     if sys.platform == 'win32':
    #         self.extcode.options['command'] = ['dir', ]
    #     else:
    #         self.extcode.options['command'] = ['ls', ]

    #     self.extcode.stdout = output_filename

    #     self.extcode.options['external_output_files'] = [output_filename,]

    #     self.top.setup()
    #     self.top.run()

    #     # check the contents of the output file for 'external_code_for_testing.py'
    #     with open(os.path.join(self.tempdir, output_filename), 'r') as out:
    #         file_contents = out.read()
    #     self.assertTrue('external_code_for_testing.py' in file_contents)

    def test_timeout_raise(self):

        self.extcode.options['command'] = ['python', 'external_code_for_testing.py',
             'external_code_output.txt', '--delay', '3']
        self.extcode.options['timeout'] = 1.0

        self.extcode.options['external_input_files'] = ['external_code_for_testing.py', ]

        dev_null = open(os.devnull, 'w')
        self.top.setup(check=True, out_stream=dev_null)
        try:
            self.top.run()
        except RuntimeError as exc:
            self.assertEqual(str(exc), 'Timed out')
            self.assertEqual(self.extcode.timed_out, True)
        else:
            self.fail('Expected RunInterrupted')

    def test_timeout_continue(self):

        self.extcode.options['command'] = ['python', 'external_code_for_testing.py',
             'external_code_output.txt', '--delay', '3']
        self.extcode.options['timeout'] = 1.0

        self.extcode.options['external_input_files'] = ['external_code_for_testing.py', ]
        self.extcode.options['on_timeout'] = 'continue'

        dev_null = open(os.devnull, 'w')
        self.top.setup(check=True, out_stream=dev_null)

        self.top.run()
        self.assertEqual(self.extcode.timed_out, True)

    def test_error_code_raise(self):

        self.extcode.options['command'] = ['python', 'external_code_for_testing.py',
             'external_code_output.txt', '--delay', '-3']
        self.extcode.options['timeout'] = 1.0

        self.extcode.options['external_input_files'] = ['external_code_for_testing.py', ]

        dev_null = open(os.devnull, 'w')
        self.top.setup(check=True, out_stream=dev_null)
        try:
            self.top.run()
        except RuntimeError as exc:
            self.assertTrue('Traceback' in str(exc))
            self.assertEqual(self.extcode.return_code, 1)
            self.assertEqual(self.extcode.errored_out, True)
        else:
            self.fail('Expected ValueError')

    def test_error_code_continue(self):

        self.extcode.options['command'] = ['python', 'external_code_for_testing.py',
             'external_code_output.txt', '--delay', '-3']
        self.extcode.options['timeout'] = 1.0
        self.extcode.options['on_error'] = 'continue'

        self.extcode.options['external_input_files'] = ['external_code_for_testing.py', ]

        dev_null = open(os.devnull, 'w')
        self.top.setup(check=True, out_stream=dev_null)
        self.top.run()
        self.assertEqual(self.extcode.errored_out, True)

    def test_badcmd(self):

        # Set command to nonexistant path.
        self.extcode.options['command'] = ['no-such-command', ]

        self.top.setup(check=False)
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
        self.extcode.stderr = STDOUT

        self.top.setup(check=False)
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

        dev_null = open(os.devnull, 'w')
        self.top.setup(check=True, out_stream=dev_null)
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
        self.extcode.options['external_input_files'] = ['external_code_for_testing.py',]
        self.extcode.options['external_output_files'] = ['does_not_exist.txt',]
        self.extcode.options['command'] = ['python', 'external_code_for_testing.py', 'external_code_output.txt']

        self.top.setup(check=False)
        self.top.run()



if __name__ == "__main__":
    unittest.main()
