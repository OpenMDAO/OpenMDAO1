
import sys
import os

import numpy.distutils
from numpy.distutils.exec_command import find_executable

from openmdao.core.system import AnalysisError
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
        Finite difference mode. (forward, backward, central) You can also set
        to 'complex_step' to peform the complex step method if your components
        support it.
    fd_options['step_size'] :  float(1e-06)
        Default finite difference stepsize
    fd_options['step_type'] :  str('absolute')
        Set to absolute, relative
    fd_options['extra_check_partials_form'] :  None or str
        Finite difference mode: ("forward", "backward", "central", "complex_step")
        During check_partial_derivatives, you can optionally do a
        second finite difference with a different mode.
    fd_options['linearize'] : bool(False)
        Set to True if you want linearize to be called even though you are using FD.

    options['command'] :  list([])
        Command to be executed. Command must be a list of command line args.
    options['env_vars'] :  dict({})
        Environment variables required by the command
    options['external_input_files'] :  list([])
        (optional) list of input file names to check the existence of before solve_nonlinear
    options['external_output_files'] :  list([])
        (optional) list of input file names to check the existence of after solve_nonlinear
    options['poll_delay'] :  float(0.0)
        Delay between polling for command completion. A value of zero will use
        an internally computed default.
    options['timeout'] :  float(0.0)
        Maximum time in seconds to wait for command completion. A value of zero
        implies an infinite wait. If the timeout interval is exceeded, an
        AnalysisError will be raised.
    options['fail_hard'] :  bool(True)
        Behavior on error returned from code, either raise a 'hard' error (RuntimeError) if True
        or a 'soft' error (AnalysisError) if False.


    """

    def __init__(self):
        super(ExternalCode, self).__init__()

        self.STDOUT   = STDOUT
        self.DEV_NULL = DEV_NULL

        # Input options for this Component
        self.options = OptionsDictionary()
        self.options.add_option('command', [], desc='command to be executed')
        self.options.add_option('env_vars', {},
                           desc='Environment variables required by the command')
        self.options.add_option('poll_delay', 0.0, lower=0.0,
            desc='Delay between polling for command completion. A value of zero will use an internally computed default')
        self.options.add_option('timeout', 0.0, lower=0.0,
                                desc='Maximum time to wait for command completion. A value of zero implies an infinite wait')
        self.options.add_option( 'external_input_files', [],
            desc='(optional) list of input file names to check the existence of before solve_nonlinear')
        self.options.add_option( 'external_output_files', [],
            desc='(optional) list of input file names to check the existence of after solve_nonlinear')
        self.options.add_option('fail_hard', True,
            desc="If True, external code errors raise a 'hard' exception (RuntimeError).  Otherwise raise a 'soft' exception (AnalysisError).")

        # Outputs of the run of the component or items that will not work with the OptionsDictionary
        self.return_code = 0 # Return code from the command
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
        cmd = [c for c in self.options['command'] if c.strip()]
        if not cmd:
            out_stream.write( "The command cannot be empty")
        else:
            program_to_execute = self.options['command'][0]
            command_full_path = find_executable( program_to_execute )

            if not command_full_path:
                out_stream.write("The command to be executed, '%s', "
                                 "cannot be found" % program_to_execute)

        # Check for missing input files
        missing = self._check_for_files(self.options['external_input_files'])
        if missing:
            out_stream.write("The following input files are missing at setup "
                             " time: %s" % missing)

    def solve_nonlinear(self, params, unknowns, resids):
        """Runs the component
        """

        self.return_code = -12345678

        if not self.options['command']:
            raise ValueError('Empty command list')

        if self.options['fail_hard']:
            err_class = RuntimeError
        else:
            err_class = AnalysisError

        return_code = None

        try:
            missing = self._check_for_files(self.options['external_input_files'])
            if missing:
                raise err_class("The following input files are missing: %s"
                                % sorted(missing))
            return_code, error_msg = self._execute_local()

            if return_code is None:
                raise AnalysisError('Timed out after %s sec.' %
                                     self.options['timeout'])

            elif return_code:
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

                raise err_class('return_code = %d%s' % (return_code,
                                                        err_fragment))

            missing = self._check_for_files(self.options['external_output_files'])
            if missing:
                raise err_class("The following output files are missing: %s"
                                % sorted(missing))

        finally:
            self.return_code = -999999 if return_code is None else return_code

    def _check_for_files(self, files):
        """ Check that specified files exist. """
        return [path for path in files if not os.path.exists(path)]

    def _execute_local(self):
        """ Run command. """

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

        try:
            return_code, error_msg = \
                self._process.wait(self.options['poll_delay'], self.options['timeout'])
        finally:
            self._process.close_files()
            self._process = None

        return (return_code, error_msg)
