""" OpenMDAO LinearSolver that explicitly solves the linear system using
linalg.solve or scipy LU factor/solve. Inherits from MultLinearSolver just
for the mult function."""

from collections import OrderedDict

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from openmdao.solvers.solver_base import MultLinearSolver


class DirectSolver(MultLinearSolver):
    """ OpenMDAO LinearSolver that explicitly solves the linear system using
    linalg.solve. The user can choose to have the jacobian assembled
    directly or through matrix-vector product.

    Options
    -------
    options['iprint'] :  int(0)
        Set to 0 to disable printing, set to 1 to print the residual to
        stdout each iteration, set to 2 to print subiteration residuals as
        well.
    options['mode'] :  str('auto')
        Derivative calculation mode, set to 'fwd' for forward mode, 'rev' for
        reverse mode, or 'auto' to let OpenMDAO determine the best mode.
    options['jacobian_method'] : str('MVP')
        Method to assemble the jacobian to solve. Select 'MVP' to build the
        Jacobian by calling apply_linear with columns of identity. Select
        'assemble' to build the Jacobian by taking the calculated Jacobians in
        each component and placing them directly into a clean identity matrix.
    options['solve_method'] : str('LU')
        Solution method, either 'solve' for linalg.solve, or 'LU' for
        linalg.lu_factor and linalg.lu_solve.
    """

    def __init__(self):
        super(DirectSolver, self).__init__()
        self.options.remove_option("err_on_maxiter")
        self.options.add_option('mode', 'auto', values=['fwd', 'rev', 'auto'],
                       desc="Derivative calculation mode, set to 'fwd' for " +
                       "forward mode, 'rev' for reverse mode, or 'auto' to " +
                       "let OpenMDAO determine the best mode.",
                       lock_on_setup=True)

        self.options.add_option('jacobian_method', 'MVP', values=['MVP', 'assemble'],
                                desc="Method to assemble the jacobian to solve. " +
                                "Select 'MVP' to build the Jacobian by calling " +
                                "apply_linear with columns of identity. Select " +
                                "'assemble' to build the Jacobian by taking the " +
                                "calculated Jacobians in each component and placing " +
                                "them directly into a clean identity matrix.")
        self.options.add_option('solve_method', 'LU', values=['LU', 'solve'],
                                desc="Solution method, either 'solve' for linalg.solve, " +
                                "or 'LU' for linalg.lu_factor and linalg.lu_solve.")

        self.jacobian = None
        self.lup = None
        self.mode = None
        self.icache = {}

    def setup(self, system):
        """ Initialization. Allocate Jacobian and set up some helpers.

        Args
        ----
        system: `System`
            System that owns this solver.
        """

        # Only need to setup if we are assembling the whole jacobian
        if self.options['jacobian_method'] == 'MVP':
            return

        # Note, we solve a slightly modified version of the unified
        # derivatives equations in OpenMDAO.
        # (dR/du) * (du/dr) = -I
        u_vec = system.unknowns
        self.jacobian = -np.eye(u_vec.vec.size)

        # Clear the index cache
        self.icache = {}

    def solve(self, rhs_mat, system, mode):
        """ Solves the linear system for the problem in self.system. The
        full solution vector is returned.

        Args
        ----
        rhs_mat : dict of ndarray
            Dictionary containing one ndarry per top level quantity of
            interest. Each array contains the right-hand side for the linear
            solve.

        system : `System`
            Parent `System` object.

        mode : string
            Derivative mode, can be 'fwd' or 'rev'.

        Returns
        -------
        dict of ndarray : Solution vectors
        """

        self.system = system

        if self.mode is None:
            self.mode = mode

        sol_buf = OrderedDict()

        for voi, rhs in rhs_mat.items():
            self.voi = None

            if system._jacobian_changed:
                self.jacobian = self._assemble_jacobian(rhs, mode)
                system._jacobian_changed = False

                if self.options['solve_method'] == 'LU':
                    self.lup = lu_factor(self.jacobian)

            if self.options['solve_method'] == 'LU':
                deriv = lu_solve(self.lup, rhs)
            else:
                deriv = np.linalg.solve(self.jacobian, rhs)

            self.system = None
            sol_buf[voi] = deriv

        return sol_buf

    def _assemble_jacobian(self, rhs, mode):
        """ Assemble Jacobian.

        Args
        ----
        rhs : ndarray
            An ndarray containomg the right-hand side for this linear
            solve.

        system : `System`
            Parent `System` object.

        mode : string
            Derivative mode, can be 'fwd' or 'rev'.

        Returns
        -------
        ndarray : Jacobian Matrix
        """
        system = self.system

        # OpenMDAO does matrix vector product.
        if self.options['jacobian_method'] == 'MVP':

            self.mode = mode

            n_edge = len(rhs)
            ident = np.eye(n_edge)

            partials = np.empty((n_edge, n_edge))

            for i in range(n_edge):
                partials[:, i] = self.mult(ident[:, i])

        # Assemble the Jacobian
        else:

            # Must clear the jacobian if we switch modes
            if self.mode != mode:
                self.setup(system)
            self.mode = mode

            sys_name = system.name + '.'
            partials = self.jacobian
            u_vec = system.unknowns
            icache = self.icache
            conn = system.connections
            sys_prom_name = system._sysdata.to_prom_name

            for sub in system.components(recurse=True, include_self=True):

                jac = sub._jacobian_cache

                # This method won't work on components where apply_linear
                # is overridden.
                if jac is None:
                    msg = "The 'assemble' jacobian_method is not supported when " + \
                         "'apply_linear' is used on a component (%s)." % sub.pathname
                    raise RuntimeError(msg)

                sub_u = sub.unknowns
                sub_name = sub.pathname

                for key in jac:
                    o_var, i_var = key
                    key2 = (sub_name, key)

                    # We cache the location of each variable in our jacobian
                    if key2 not in icache:

                        o_var_abs = '.'.join((sub_name, o_var))
                        i_var_abs = '.'.join((sub_name, i_var))
                        i_var_pro = sys_prom_name[i_var_abs]
                        o_var_pro = sys_prom_name[o_var_abs]

                        # States are fine ...
                        if i_var in sub.states:
                            pass

                        #... but inputs need to find their source.
                        elif i_var_pro not in u_vec:

                            # Param is not relevant
                            if i_var_abs not in conn:
                                continue

                            i_var_src = conn[i_var_abs][0]
                            i_var_pro = sys_prom_name[i_var_src]

                        o_start, o_end = u_vec._dat[o_var_pro].slice
                        i_start, i_end = u_vec._dat[i_var_pro].slice

                        icache[key2] = (o_start, o_end, i_start, i_end)

                    else:
                        (o_start, o_end, i_start, i_end) = icache[key2]

                    if mode=='fwd':
                        partials[o_start:o_end, i_start:i_end] = jac[o_var, i_var]
                    else:
                        partials[i_start:i_end, o_start:o_end] = jac[o_var, i_var].T

        return partials