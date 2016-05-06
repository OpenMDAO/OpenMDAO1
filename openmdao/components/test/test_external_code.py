from __future__ import print_function

import unittest
import os
import sys
import tempfile
import shutil
import pkg_resources

from openmdao.api import Problem, Group, ExternalCode, AnalysisError
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

    def test_timeout_raise(self):

        self.extcode.options['command'] = ['python', 'external_code_for_testing.py',
             'external_code_output.txt', '--delay', '3']
        self.extcode.options['timeout'] = 1.0

        self.extcode.options['external_input_files'] = ['external_code_for_testing.py', ]

        dev_null = open(os.devnull, 'w')
        self.top.setup(check=True, out_stream=dev_null)
        try:
            self.top.run()
        except AnalysisError as exc:
            self.assertEqual(str(exc), 'Timed out after 1.0 sec.')
        else:
            self.fail('Expected AnalysisError')

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
            self.assertTrue('Traceback' in str(exc),
                            "no traceback found in '%s'" % str(exc))
            self.assertEqual(self.extcode.return_code, 1)
        else:
            self.fail('Expected RuntimeError')

    def test_error_code_soft(self):

        self.extcode.options['command'] = ['python', 'external_code_for_testing.py',
             'external_code_output.txt', '--delay', '-3']
        self.extcode.options['timeout'] = 1.0
        self.extcode.options['fail_hard'] = False

        self.extcode.options['external_input_files'] = ['external_code_for_testing.py', ]

        dev_null = open(os.devnull, 'w')
        self.top.setup(check=True, out_stream=dev_null)
        try:
            self.top.run()
        except AnalysisError as err:
            self.assertTrue("delay must be >= 0" in str(err),
                            "expected 'delay must be >= 0' to be in '%s'" % str(err))
            self.assertTrue('Traceback' in str(err),
                            "no traceback found in '%s'" % str(err))
        else:
            self.fail("AnalysisError expected")

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
        self.assertTrue('SOME_ENV_VAR_VALUE' in file_contents,
                        "'SOME_ENV_VAR_VALUE' missing from '%s'" % file_contents)


if __name__ == "__main__":
    unittest.main()
