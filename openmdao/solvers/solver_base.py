""" Base class for linear and nonlinear solvers."""

from __future__ import print_function
from openmdao.recorders.recording_manager import RecordingManager
from openmdao.util.options import OptionsDictionary


class SolverBase(object):
    """ Common base class for Linear and Nonlinear solver. Should not be used
    by users. Always inherit from `LinearSolver` or `NonlinearSolver`."""

    def __init__(self):
        self.iter_count = 0
        self.options = OptionsDictionary()
        desc = 'Set to 0 to disable printing, set to 1 to print the ' \
               'residual to stdout each iteration, set to 2 to print ' \
               'subiteration residuals as well.'
        self.options.add_option('iprint', 0, values=[0, 1, 2], desc=desc)
        self.options.add_option('err_on_maxiter', False,
            desc='If True, raise an AnalysisError if not converged at maxiter.')
        self.recorders = RecordingManager()
        self.local_meta = None

    def setup(self, sub):
        """ Solvers override to define post-setup initiailzation.

        Args
        ----
        sub: `System`
            System that owns this solver.
        """
        pass

    def cleanup(self):
        """ Clean up resources prior to exit. """
        self.recorders.close()

    def print_norm(self, solver_string, pathname, iteration, res, res0,
                   msg=None, indent=0, solver='NL'):
        """ Prints out the norm of the residual in a neat readable format.

        Args
        ----
        solver_string: string
            Unique string to identify your solver type (e.g., 'LN_GS' or
            'NEWTON').

        pathname: dict
            Parent system pathname.

        iteration: int
            Current iteration number

        res: float
            Absolute residual value.

        res0: float
            Baseline initial residual for relative comparison.

        msg: string, optional
            Message that indicates convergence.

        ident: int, optional
            Additional indentation levels for subiterations.

        solver: string, optional
            Solver type if not LN or NL (mostly for line search operations.)
        """
        if pathname=='':
            name = 'root'
        else:
            name = 'root.' + pathname

        # Find indentation level
        level = pathname.count('.')
        # No indentation for driver; top solver is no indentation.
        level = level + indent

        indent = '   ' * level
        if msg is not None:
            form = indent + '[%s] %s: %s   %d | %s'
            print(form % (name, solver, solver_string, iteration, msg))
            return

        form = indent + '[%s] %s: %s   %d | %.9g %.9g'
        print(form % (name, solver, solver_string, iteration, res, res/res0))

    def print_all_convergence(self):
        """ Turns on iprint for this solver and all subsolvers. Override if
        your solver has subsolvers."""
        self.options['iprint'] = 1

    def generate_docstring(self):
        """
        Generates a numpy-style docstring for a user-created System class.

        Returns
        -------
        docstring : str
                string that contains a basic numpy docstring.

        """
        #start the docstring off
        docstrings = ['    \"\"\"']

        #Put options into docstring
        firstTime = 1

        for key, value in sorted(vars(self).items()):
            if type(value)==OptionsDictionary:
                if firstTime:  #start of Options docstring
                    docstrings.extend(['','    Options','    -------'])
                    firstTime = 0
                docstrings.append(value._generate_docstring(key))

        #finish up docstring
        docstrings.extend(['    \"\"\"',''])
        return '\n'.join(docstrings)


class LinearSolver(SolverBase):
    """ Base class for all linear solvers. Inherit from this class to create a
    new custom linear solver.

    Options
    -------
    options['iprint'] :  int(0)
        Set to 0 to disable printing, set to 1 to print the residual to stdout
        each iteration, set to 2 to print subiteration residuals as well.
    """

    def add_recorder(self, recorder):
        """Appends the given recorder to this solver's list of recorders.

        Args
        ----
        recorder: `BaseRecorder`
            A recorder object.
        """
        self.recorders.append(recorder)

    def solve(self, rhs, system, mode):
        """ Solves the linear system for the problem in self.system. The
        full solution vector is returned. This function must be defined
        when inheriting.

        Args
        ----
        rhs : ndarray
            Array containing the right-hand side for the linear solve. Also
            possibly a 2D array with multiple right-hand sides.

        system : `System`
            Parent `System` object.

        mode : string
            Derivative mode, can be 'fwd' or 'rev'.

        Returns
        -------
        ndarray : Solution vector
        """
        pass


class MultLinearSolver(LinearSolver):
    """Base class for ScipyGMRES and DirectSolver.  Adds a mult method.
    """

    def mult(self, arg):
        """ Applies Jacobian matrix. Mode is determined by the
        system. This is a GMRES callback and is called by DirectSolver.solve.

        Args
        ----
        arg : ndarray
            Incoming vector

        Returns
        -------
        ndarray : Matrix vector product of arg with jacobian
        """

        system = self.system
        mode = self.mode

        voi = self.voi
        if mode == 'fwd':
            sol_vec, rhs_vec = system.dumat[voi], system.drmat[voi]
        else:
            sol_vec, rhs_vec = system.drmat[voi], system.dumat[voi]

        # Set incoming vector
        sol_vec.vec[:] = arg

        # Start with a clean slate
        rhs_vec.vec[:] = 0.0
        system.clear_dparams()

        system._sys_apply_linear(mode, self.system._do_apply, vois=(voi,))

        #print("arg", arg)
        #print("result", rhs_vec.vec)
        return rhs_vec.vec


class NonLinearSolver(SolverBase):
    """ Base class for all nonlinear solvers. Inherit from this class to create a
    new custom nonlinear solver.

    Options
    -------
    options['iprint'] :  int(0)
        Set to 0 to disable printing, set to 1 to print the residual to stdout
        each iteration, set to 2 to print subiteration residuals as well.
    """

    def __init__(self):
        """ Initialize the default supports for nl solvers."""
        super(NonLinearSolver, self).__init__()

        # What this solver supports
        self.supports = OptionsDictionary(read_only=True)
        self.supports.add_option('uses_derivatives', False)

    def add_recorder(self, recorder):
        """Appends the given recorder to this solver's list of recorders.

        Args
        ----
        recorder: `BaseRecorder`
            A recorder object.
        """
        self.recorders.append(recorder)

    def solve(self, params, unknowns, resids, system, metadata=None):
        """ Drive all residuals in self.system and all subsystems to zero.
        This includes all implicit components. This function must be defined
        when inheriting.

        Args
        ----
        params : `VecWrapper`
            `VecWrapper` containing parameters. (p)

        unknowns : `VecWrapper`
            `VecWrapper` containing outputs and states. (u)

        resids : `VecWrapper`
            `VecWrapper` containing residuals. (r)

        system : `System`
            Parent `System` object.

        metadata : dict, optional
            Dictionary containing execution metadata (e.g. iteration coordinate).
        """
        pass


class LineSearch(SolverBase):
    """ Base class for all linesearch subsolvers. Line search is used by
    other solvers such as the Newton solver. Inherit from this class to
    create a new custom line search.

    Options
    -------
    options['iprint'] :  int(0)
        Set to 0 to disable printing, set to 1 to print the residual to stdout
        each iteration, set to 2 to print subiteration residuals as well.
    """

    def solve(self, params, unknowns, resids, system, solver, alpha, fnorm,
              fnorm0, metadata=None):
        """ Take the gradient calculated by the parent solver and figure out
        how far to go.

        Args
        ----
        params : `VecWrapper`
            `VecWrapper` containing parameters. (p)

        unknowns : `VecWrapper`
            `VecWrapper` containing outputs and states. (u)

        resids : `VecWrapper`
            `VecWrapper` containing residuals. (r)

        system : `System`
            Parent `System` object.

        metadata : dict, optional
            Dictionary containing execution metadata (e.g. iteration coordinate).

        solver : `Solver`
            Parent solver instance.

        alpha : float
            Initial over-relaxation factor as used in parent solver.

        fnorm : float
            Initial norm of the residual for absolute tolerance check.

        fnorm0 : float
            Initial norm of the residual for relative tolerance check.
        """
        pass
