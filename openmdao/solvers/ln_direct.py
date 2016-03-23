""" OpenMDAO LinearSolver that explicitly solves the linear system using
linalg.solve. Inherits from MultLinearSolver just for the mult function."""

from collections import OrderedDict

import numpy as np

from openmdao.solvers.solver_base import MultLinearSolver


class DirectSolver(MultLinearSolver):
    """ OpenMDAO LinearSolver that explicitly solves the linear system using
    linalg.solve. The user can choose to have the jacobian assemblled
    directly or throuugh matrix-vector product.

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
        each component and placing them directly into a clean identity matrix."
    """

    def __init__(self):
        super(DirectSolver, self).__init__()
        self.options.remove_option("err_on_maxiter")
        self.options.add_option('mode', 'auto', values=['fwd', 'rev', 'auto'],
                       desc="Derivative calculation mode, set to 'fwd' for " +
                       "forward mode, 'rev' for reverse mode, or 'auto' to " +
                       "let OpenMDAO determine the best mode.")
        self.options.add_option('jacobian_method', 'MVP', values=['MVP', 'assemble'],
                                desc="Method to assemble the jacobian to solve. " +
                                "Select 'MVP' to build the Jacobian by calling " +
                                "apply_linear with columns of identity. Select " +
                                "'assemble' to build the Jacobian by taking the " +
                                "calculated Jacobians in each component and placing " +
                                "them directly into a clean identity matrix.")

        self.jacobian = None
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

        # TODO: When to record?
        self.system = system

        if self.mode is None:
            self.mode = mode

        sol_buf = OrderedDict()

        # TODO: This solver could probably work with multiple RHS
        for voi, rhs in rhs_mat.items():
            self.voi = None

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
                conn = system.connections

                partials = self.jacobian
                u_vec = system.unknowns
                var_names = u_vec.keys()

                for sub in system.components(recurse=True, include_self=False):

                    jac = sub._jacobian_cache

                    # This method won't work on components where apply_linear
                    # is overridden.
                    if jac is None:
                        msg = "The 'assemble' jacobian_method is not supported when " + \
                             "'apply_linear' is used on a component (%s)." % sub.pathname
                        raise RuntimeError(msg)

                    sub_u = sub.unknowns
                    sub_name = sub.pathname
                    icache = self.icache

                    for key in jac:
                        o_var, i_var = key
                        key2 = (sub_name, key)

                        # Derivs wrt states, but states live in u_vec
                        if i_var in sub.states:
                            sub_p = sub_u
                        else:
                            sub_p = sub.params

                        # Skip anything not relevant
                        if o_var not in sub_u or i_var not in sub_p:
                            continue

                        # We cache the location of each variable in our jacobian
                        if key2 not in icache:
                            o_var_pro = sub_u.metadata(o_var)['top_promoted_name']
                            meta = sub_p.metadata(i_var)
                            i_var_pro = meta['top_promoted_name']

                            # States are fine ...
                            if i_var in sub.states:
                                pass

                            #... but inputs need to find their source.
                            elif i_var_pro not in var_names:
                                if i_var_pro in conn:
                                    i_var_src = conn[i_var_pro][0]
                                else:
                                    i_var_abs = meta['pathname']
                                    i_var_src = conn[i_var_abs][0]

                                # Promoted/Absolute gets kind of ridiculous
                                # sometimes
                                if i_var_src not in u_vec:
                                    for name in u_vec:
                                        meta = u_vec.metadata(name)
                                        if meta['pathname'] == i_var_src:
                                            i_var_src = name
                                            break

                                i_var_pro = u_vec.metadata(i_var_src)['top_promoted_name']

                            # Map names back to this solver level. Need to do
                            # this when Directsolver is in a sub group.
                            if i_var_pro.startswith(sys_name):
                                i_var_pro = i_var_pro[(len(sys_name)):]
                            if o_var_pro.startswith(sys_name):
                                o_var_pro = o_var_pro[(len(sys_name)):]

                            o_start, o_end = u_vec._dat[o_var_pro].slice
                            i_start, i_end = u_vec._dat[i_var_pro].slice

                            icache[key2] = (o_start, o_end, i_start, i_end)

                        else:
                            (o_start, o_end, i_start, i_end) = icache[key2]

                        if mode=='fwd':
                            partials[o_start:o_end, i_start:i_end] = jac[o_var, i_var]
                        else:
                            partials[i_start:i_end, o_start:o_end] = jac[o_var, i_var].T

            deriv = np.linalg.solve(partials, rhs)

            self.system = None
            sol_buf[voi] = deriv

        return sol_buf
