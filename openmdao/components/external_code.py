"""
.. _`external_code.py`:
"""

import sys
import os

import numpy.distutils
from numpy.distutils.exec_command import find_executable

from openmdao.core.component import Component
from openmdao.util.options import OptionsDictionary
from openmdao.util.shell_proc import STDOUT, DEV_NULL, ShellProc

from six import iteritems, itervalues

class ExternalCode(Component):
    """
    Run an external code as a component

    Default stdin is the 'null' device, default stdout is the console, and
    default stderr is ``error.out``.

    Options
    -------
    fd_options['force_fd'] :  bool(False)
        Set to True to finite difference this system.
    fd_options['form'] :  str('forward')
        Finite difference mode. (forward, backward, central) You can also set to 'complex_step' to peform the complex step method if your components support it.
    fd_options['step_size'] :  float(1e-06)
        Default finite difference stepsize
    fd_options['step_type'] :  str('absolute')
        Set to absolute, relative
    options['check_external_outputs'] :  bool(True)
        Check that all input or output external files exist
    options['command'] :  list([])
        command to be executed
    options['env_vars'] :  dict({})
        Environment variables required by the command
    options['external_input_files'] :  list([])
        (optional) list of input file names to check the pressence of before solve_nonlinear
    options['external_output_files'] :  list([])
        (optional) list of input file names to check the pressence of after solve_nonlinear
    options['poll_delay'] :  float(0.0)
        Delay between polling for command completion. A value of zero will use an internally computed default
    options['timeout'] :  float(0.0)
        Maximum time to wait for command completion. A value of zero implies an infinite wait
    options['on_timeout'] :  str('raise')
        Timeout behavior, either "raise" an exception or "continue" running OpenMDAO
    options['on_error'] :  str('raise')
        Behavior on error returned from code, either "raise" an exception or "continue" running OpenMDAO


    """

    def __init__(self):
        super(ExternalCode, self).__init__()

        self.STDOUT   = STDOUT
        self.DEV_NULL = DEV_NULL

        # Input options for this Component
        self.options = OptionsDictionary()
        self.options.add_option('command', [], desc='command to be executed')
        self.options.add_option('env_vars', {}, desc='Environment variables required by the command')
        self.options.add_option('poll_delay', 0.0, lower=0.0,
            desc='Delay between polling for command completion. A value of zero will use an internally computed default')
        self.options.add_option('timeout', 0.0, lower=0.0,
                                desc='Maximum time to wait for command completion. A value of zero implies an infinite wait')
        self.options.add_option('check_external_outputs', True,
            desc='Check that all input or output external files exist')

        self.options.add_option( 'external_input_files', [],
            desc='(optional) list of input file names to check the pressence of before solve_nonlinear')
        self.options.add_option( 'external_output_files', [],
            desc='(optional) list of input file names to check the pressence of after solve_nonlinear')
        self.options.add_option('on_timeout', 'raise', values=['raise', 'continue'],
            desc='Timeout behavior, either "raise" an exception or "continue" running OpenMDAO')
        self.options.add_option('on_error', 'raise', values=['raise', 'continue'],
            desc='Behavior on error returned from code, either "raise" an exception or "continue" running OpenMDAO')

        # Outputs of the run of the component or items that will not work with the OptionsDictionary
        self.return_code = 0 # Return code from the command
        self.timed_out = False # True if the command timed-out
        self.stdin  = self.DEV_NULL
        self.stdout = None
        self.stderr = "error.out"

    def check_setup(self, out_stream=sys.stdout):
        """Write a report to the given stream indicating any potential problems found
        with the current configuration of this ``Problem``.

        Args
        ----
        out_stream : a file-like object, optional
        """

        # check for the command
        if not self.options['command']:
            out_stream.write( "The command cannot be empty")
        else:
            if isinstance(self.options['command'], str):
                program_to_execute = self.options['command']
            else:
                program_to_execute = self.options['command'][0]
            command_full_path = find_executable( program_to_execute )

            if not command_full_path:
                msg = "The command to be executed, '%s', cannot be found" % program_to_execute
                out_stream.write(msg)

        # Check for missing input files
        missing_files = self._check_for_files(input=True)
        for iotype, path in missing_files:
            msg = "The %s file %s is missing" % ( iotype, path )
            out_stream.write(msg)

    def solve_nonlinear(self, params, unknowns, resids):
        """Runs the component
        """

        self.return_code = -12345678
        self.timed_out = False
        self.errored_out = False

        if not self.options['command']:
            raise ValueError('Empty command list')

        # self.check_files(inputs=True)

        return_code = None
        error_msg = ''
        try:
            return_code, error_msg = self._execute_local()

            if return_code is None:
                self.timed_out = True
                if self.options['on_timeout'] == 'raise':
                    raise RuntimeError('Timed out')

            elif return_code:
                self.errored_out = True
                if self.options['on_error'] == 'raise':
                    if isinstance(self.stderr, str):
                        if os.path.exists(self.stderr):
                            stderrfile = open(self.stderr, 'r')
                            error_desc = stderrfile.read()
                            stderrfile.close()
                            err_fragment = "\nError Output:\n%s" % error_desc
                        else:
                            err_fragment = "\n[stderr %r missing]" % self.stderr
                    else:
                        err_fragment = error_msg

                    raise RuntimeError('return_code = %d%s' % (return_code,
                                                               err_fragment))

            if self.options['check_external_outputs']:
                missing_files = self._check_for_files(input=False)
                msg = ""
                for iotype, path in missing_files:
                    msg +=  "%s file %s is missing\n" % (iotype, path)

                if msg:
                    raise RuntimeError( "Missing files: %s" % msg )
                # self.check_files(inputs=False)
        finally:
            self.return_code = -999999 if return_code is None else return_code

    def _check_for_files(self, input=True):
        """
        Check that all 'specific' input external files exist.

        input: bool
            If True, check inputs. Else check outputs
        """

        missing_files = []

        if input:
            files = self.options['external_input_files']
        else:
            files = self.options['external_output_files']

        for path in files:
            if not os.path.exists(path):
                missing_files.append(('input', path))

        return missing_files

    def _execute_local(self):
        """ Run command. """
        #self._logger.info('executing %s...', self.options['command'])
        # start_time = time.time()

        # check to make sure command exists
        if isinstance(self.options['command'], str):
            program_to_execute = self.options['command']
        else:
            program_to_execute = self.options['command'][0]

        # suppress message from find_executable function, we'll handle it
        numpy.distutils.log.set_verbosity(-1)

        command_full_path = find_executable( program_to_execute )
        if not command_full_path:
            raise ValueError("The command to be executed, '%s', cannot be found" % program_to_execute)

        command_for_shell_proc = self.options['command']
        if sys.platform == 'win32':
            command_for_shell_proc = ['cmd.exe', '/c' ] + command_for_shell_proc

        self._process = \
            ShellProc(command_for_shell_proc, self.stdin,
                      self.stdout, self.stderr, self.options['env_vars'])
        #self._logger.debug('PID = %d', self._process.pid)

        try:
            return_code, error_msg = \
                self._process.wait(self.options['poll_delay'], self.options['timeout'])
        finally:
            self._process.close_files()
            self._process = None

        # et = time.time() - start_time
        #if et >= 60:  #pragma no cover
            #self._logger.info('elapsed time: %.1f sec.', et)

        return (return_code, error_msg)
