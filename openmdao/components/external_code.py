"""
.. _`external_code.py`:
"""

import sys
import os
import time
from copy import deepcopy

from distutils.spawn import find_executable

from openmdao.core.component import Component, _NotSet
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

        self.options = OptionsDictionary()
        self.options.add_option('command', [], desc='command to be executed')
        self.options.add_option('env_vars', {}, desc='Environment variables required by the command')
        self.options.add_option('poll_delay', 0.0, desc='''Delay between polling for command completion. 
            A value of zero will use an internally computed default''')
        self.options.add_option('timeout', 0.0, desc='''Maximum time to wait for command 
            completion. A value of zero implies an infinite wait''')
        self.options.add_option('check_external_outputs', True, desc='Check that all input or output external files exist')

        self.options.add_option( 'external_files', [], desc='FileMetadata objects for external files used'
                               ' by this component.')

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


        # TODO




        # # All outputs must have surrogates assigned
        # # either explicitly or through the default surrogate
        # if self.default_surrogate is None:
        #     no_sur = []
        #     for name in self._surrogate_output_names:
        #         surrogate = self._unknowns_dict[name].get('surrogate')
        #         if surrogate is None:
        #             no_sur.append(name)
        #     if len(no_sur) > 0:
        #         msg = ("No default surrogate model is defined and the following"
        #                " outputs do not have a surrogate model:\n%s\n"
        #                "Either specify a default_surrogate, or specify a "
        #                "surrogate model for all outputs."
        #                % no_sur)
        #         out_stream.write(msg)

    def solve_nonlinear(self, params, unknowns, resids):
        """Runs the component
        """

        self.return_code = -12345678
        self.timed_out = False

        if not self.options['command']:
            raise ValueError('Empty command list')

        self.check_files(inputs=True)

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
                self.check_files(inputs=False)
        finally:
            self.return_code = -999999 if return_code is None else return_code

    def check_files(self, inputs):
        """
        Check that all 'specific' input or output external files exist.
        If an external file path specifies a pattern, it is *not* checked.

        inputs: bool
            If True, check inputs; otherwise outputs.
        """
        # External files.
        for metadata in self.options['external_files']:
            path = metadata.path
            for ch in ('*?['):
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
                    raise RuntimeError('missing %s file %r' % (iotype, path))
        # Stdin, stdout, stderr.
        if inputs and self.stdin and self.stdin != self.DEV_NULL:
            if not os.path.exists(self.stdin):
                raise RuntimeError('missing stdin file %r' % self.stdin)
        if not inputs and self.stdout and self.stdout != self.DEV_NULL:
            if not os.path.exists(self.stdout):
                raise RuntimeError('missing stdout file %r' % self.stdout)

        if not inputs and self.stderr \
                      and self.stderr != self.DEV_NULL \
                      and self.stderr != self.STDOUT :
            if not os.path.exists(self.stderr):
                raise RuntimeError('missing stderr file %r' % self.stderr)

    def _execute_local(self):
        """ Run command. """
        #self._logger.info('executing %s...', self.options['command'])
        start_time = time.time()

        # check to make sure command exists
        if isinstance(self.options['command'], basestring):
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

        et = time.time() - start_time
        #if et >= 60:  #pragma no cover
            #self._logger.info('elapsed time: %.1f sec.', et)

        return (return_code, error_msg)

    def stop(self):
        """ Stop the external code. """
        self._stop = True
        if self._process:
            self._process.terminate()

    # def copy_inputs(self, inputs_dir, patterns):
    #     """
    #     Copy inputs from `inputs_dir` that match `patterns`.

    #     inputs_dir: string
    #         Directory to copy files from. Relative paths are evaluated from
    #         the component's execution directory.

    #     patterns: list or string
    #         One or more :mod:`glob` patterns to match against.

    #     This can be useful for resetting problem state.
    #     """
    #     self._logger.info('copying initial inputs from %s...', inputs_dir)
    #     with self.dir_context:
    #         if not os.path.exists(inputs_dir):
    #             self.raise_exception("inputs_dir '%s' does not exist" \
    #                                  % inputs_dir, RuntimeError)
    #         self._copy(inputs_dir, patterns)

    # def copy_results(self, results_dir, patterns):
    #     """
    #     Copy files from `results_dir` that match `patterns`.

    #     results_dir: string
    #         Directory to copy files from. Relative paths are evaluated from
    #         the component's execution directory.

    #     patterns: list or string
    #         One or more :mod:`glob` patterns to match against.

    #     This can be useful for workflow debugging when the external
    #     code takes a long time to execute.
    #     """
    #     self._logger.info('copying precomputed results from %s...', results_dir)
    #     with self.dir_context:
    #         if not os.path.exists(results_dir):
    #             self.raise_exception("results_dir '%s' does not exist" \
    #                                  % results_dir, RuntimeError)
    #         self._copy(results_dir, patterns)

    # def _copy(self, directory, patterns):
    #     """
    #     Copy files from `directory` that match `patterns`
    #     to the current directory and ensure they are writable.

    #     directory: string
    #         Directory to copy files from.

    #     patterns: list or string
    #         One or more :mod:`glob` patterns to match against.
    #     """
    #     if isinstance(patterns, basestring):
    #         patterns = [patterns]

    #     for pattern in patterns:
    #         pattern = os.path.join(directory, pattern)
    #         for src_path in sorted(glob.glob(pattern)):
    #             dst_path = os.path.basename(src_path)
    #             self._logger.debug('    %s', src_path)
    #             shutil.copy(src_path, dst_path)
    #             # Ensure writable.
    #             mode = os.stat(dst_path).st_mode
    #             mode |= stat.S_IWUSR
    #             os.chmod(dst_path, mode)


