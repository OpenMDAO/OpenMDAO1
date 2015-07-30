"""
.. _`external_code.py`:
"""

import sys
import os
from distutils.spawn import find_executable

from openmdao.core.component import Component
from openmdao.core.options import OptionsDictionary
from openmdao.util import shellproc


class ExternalCode(Component):
    """Run an external code as a component

    Default stdin is the 'null' device, default stdout is the console, and
    default stderr is ``error.out``.
    """

    STDOUT   = shellproc.STDOUT
    DEV_NULL = shellproc.DEV_NULL

    def __init__(self):
        super(ExternalCode, self).__init__()

        self.STDOUT   = shellproc.STDOUT
        self.DEV_NULL = shellproc.DEV_NULL

        # Input options for this Component
        self.options = OptionsDictionary()
        self.options.add_option('command', [], desc='command to be executed')
        self.options.add_option('env_vars', {}, desc='Environment variables required by the command')
        self.options.add_option('poll_delay', 0.0, desc='''Delay between polling for command completion. 
            A value of zero will use an internally computed default''')
        self.options.add_option('timeout', 0.0, desc='''Maximum time to wait for command 
            completion. A value of zero implies an infinite wait''')
        self.options.add_option('check_external_outputs', True, desc='Check that all input or output external files exist')

        self.options.add_option( 'external_files', [], 
                desc='list of dicts for external files used by this component.')

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
        missing_files = self.check_input_output_files(inputs=True)
        for iotype, path in missing_files:
            msg = "The %s file %s is missing" % ( iotype, path )
            out_stream.write(msg)

    def solve_nonlinear(self, params, unknowns, resids):
        """Runs the component
        """

        self.return_code = -12345678
        self.timed_out = False

        if not self.options['command']:
            raise ValueError('Empty command list')

        # self.check_files(inputs=True)

        return_code = None
        error_msg = ''
        try:
            return_code, error_msg = self._execute_local()

            if return_code is None:
                # if self._stop:
                #     raise RuntimeError('Run stopped')
                # else:
                self.timed_out = True
                raise RuntimeError('Timed out')

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
                    
                raise RuntimeError('return_code = %d%s' % (return_code, err_fragment))

            if self.options['check_external_outputs']:
                missing_files = self.check_input_output_files(inputs=False)
                msg = ""
                for iotype, path in missing_files:
                    msg +=  "%s file %s is missing\n" % (iotype, path)

                if msg:
                    raise RuntimeError( "Missing files: %s" % msg )
                # self.check_files(inputs=False)
        finally:
            self.return_code = -999999 if return_code is None else return_code

    def check_input_output_files(self, inputs):
        """
        Check that all 'specific' input or output external files exist.
        If an external file path specifies a pattern, it is *not* checked.

        inputs: bool
            If True, check inputs; otherwise outputs.
        """

        missing_files = []

        # External files.
        for metadata in self.options['external_files']:
            path = metadata['path']
            for ch in '*?[':
                if ch in path:
                    break
            else:
                if inputs:
                    if not metadata.get('input', False):
                        continue
                else:
                    if not metadata.get('output', False):
                        continue
                if not os.path.exists(path):
                    iotype = 'input' if inputs else 'output'
                    missing_files.append((iotype, path))
                    #raise RuntimeError('missing %s file %r' % (iotype, path))
        # Stdin, stdout, stderr.
        if inputs and self.stdin and self.stdin != self.DEV_NULL:
            if not os.path.exists(self.stdin):
                missing_files.append(('stdin',self.stdin))
                #raise RuntimeError('missing stdin file %r' % self.stdin)
        if not inputs and self.stdout and self.stdout != self.DEV_NULL:
            if not os.path.exists(self.stdout):
                missing_files.append(('stdout',self.stdout))
                # raise RuntimeError('missing stdout file %r' % self.stdout)

        if not inputs and self.stderr \
                      and self.stderr != self.DEV_NULL \
                      and self.stderr != self.STDOUT :
            if not os.path.exists(self.stderr):
                missing_files.append(('stderr',self.stderr))
                # raise RuntimeError('missing stderr file %r' % self.stderr)

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
        command_full_path = find_executable( program_to_execute )

        if not command_full_path:
            raise ValueError("The command to be executed, '%s', cannot be found" % program_to_execute)
            
        self._process = \
            shellproc.ShellProc(self.options['command'], self.stdin,
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



