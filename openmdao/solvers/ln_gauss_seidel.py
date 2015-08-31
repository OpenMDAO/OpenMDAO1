""" OpenMDAO LinearSolver that uses linear Gauss Seidel."""

from __future__ import print_function

from six import iteritems

from openmdao.core.component import Component
from openmdao.solvers.solver_base import LinearSolver


class LinearGaussSeidel(LinearSolver):
    """ LinearSolver that uses linear Gauss Seidel.
    """

    def __init__(self):
        super(LinearGaussSeidel, self).__init__()

        opt = self.options
        opt.add_option('atol', 1e-12,
                       desc='Absolute convergence tolerance.')
        opt.add_option('rtol', 1e-10,
                       desc='Absolute convergence tolerance.')
        opt.add_option('maxiter', 1,
                       desc='Maximum number of iterations.')
        opt.add_option('mode', 'auto', values=['fwd', 'rev', 'auto'],
                       desc="Derivative calculation mode, set to 'fwd' for " + \
                       "forward mode, 'rev' for reverse mode, or 'auto' to " + \
                       "let OpenMDAO determine the best mode.")

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

        dumat = system.dumat
        drmat = system.drmat
        dpmat = system.dpmat

        system.clear_dparams()
        for names in system._relevance.vars_of_interest():
            for name in names:
                if name in dumat:
                    dumat[name].vec[:] = 0.0
        dumat[None].vec[:] = 0.0

        vois = rhs_mat.keys()
        # John starts with the following. It is not necessary, but
        # uncommenting it helps to debug when comparing print outputs to his.
        #for voi in vois:
        #    drmat[voi].vec[:] = -rhs_mat[voi]

        gs_outputs = {}
        sol_buf = {}

        f_norm0, f_norm = 1.0, 1.0
        self.iter_count = 0
        while self.iter_count < self.options['maxiter'] and \
              f_norm > self.options['atol'] and \
              f_norm/f_norm0 > self.options['rtol']:

            if mode == 'fwd':

                for sub in system.subsystems(local=True):

                    for voi in vois:
                        #print('pre scatter', sub.pathname, dpmat[voi].vec, dumat[voi].vec, drmat[voi].vec)
                        system._transfer_data(sub.name, deriv=True, var_of_interest=voi)
                        #print('pre apply', sub.pathname, dpmat[voi].vec, dumat[voi].vec, drmat[voi].vec)

                        gs_outputs[voi] = [x for x in dumat[voi]
                                           if x not in sub.dumat[voi]]


                    if isinstance(sub, Component):

                        # Components need to reverse sign and add 1 on diagonal
                        # for explicit unknowns
                        system._sub_apply_linear_wrapper(sub, mode, vois, ls_inputs=system._ls_inputs,
                                                         gs_outputs=gs_outputs)

                    else:
                        # Groups and all other systems just call their own
                        # apply_linear.
                        sub.apply_linear(mode, ls_inputs=system._ls_inputs, vois=vois,
                                         gs_outputs=gs_outputs)

                    #for voi in vois:
                       # print('post apply', dpmat[voi].vec, dumat[voi].vec, drmat[voi].vec)

                    for voi in vois:
                        drmat[voi].vec *= -1.0
                        drmat[voi].vec += rhs_mat[voi]
                        dpmat[voi].vec[:] = 0.0

                    sub.solve_linear(sub.dumat, sub.drmat,vois, mode=mode)
                    #for voi in vois:
                        #print('post solve', dpmat[voi].vec, dumat[voi].vec, drmat[voi].vec)

                for voi in vois:
                    sol_buf[voi] = dumat[voi].vec

            else:

                for sub in reversed(list(system.subsystems(local=True))):
                    for voi in vois:
                        dumat[voi].vec *= 0.0

                        #print('pre scatter', sub.pathname, voi, dpmat[voi].vec, dumat[voi].vec, drmat[voi].vec)
                        system._transfer_data(sub.name, mode='rev', deriv=True, var_of_interest=voi)
                        #print('post scatter', sub.pathname, voi, dpmat[voi].vec, dumat[voi].vec, drmat[voi].vec)

                        dumat[voi].vec *= -1.0
                        dumat[voi].vec += rhs_mat[voi]

                    sub.solve_linear(sub.dumat, sub.drmat, vois, mode=mode)
                    #for voi in vois:
                        #print('post solve', dpmat[voi].vec, dumat[voi].vec, drmat[voi].vec)

                    for voi in vois:
                        gs_outputs[voi] = [x for x in dumat[voi]
                                           if x not in sub.dumat[voi]]

                    if isinstance(sub, Component):

                        # Components need to reverse sign and add 1 on diagonal
                        # for explicit unknowns
                        system._sub_apply_linear_wrapper(sub, mode, vois, ls_inputs=system._ls_inputs,
                                                         gs_outputs=gs_outputs)

                    else:
                        # Groups and all other systems just call their own
                        # apply_linear.
                        sub.apply_linear(mode, ls_inputs=system._ls_inputs, vois=vois,
                                         gs_outputs=gs_outputs)

                    #for voi in vois:
                        #print('post apply', system.dpmat[voi].vec, dumat[voi].vec, drmat[voi].vec)

                for voi in vois:
                    sol_buf[voi] = drmat[voi].vec


            self.iter_count += 1
            if self.options['maxiter'] == 1:
                f_norm = 0.0
            else:
                f_norm = self._norm(system, mode, rhs_mat)

            if self.options['iprint'] > 0:
                self.print_norm('LN_GS', self.local_meta, self.iter_count,
                                f_norm, f_norm0, indent=1, solver='LN')

        return sol_buf

    def _norm(self, system, mode, rhs_mat):
        """ Computes the norm of the linear residual

        Args
        ----
        system : `System`
            Parent `System` object.

        mode : string
            Derivative mode, can be 'fwd' or 'rev'.

        rhs_mat : dict of ndarray
            Dictionary containing one ndarry per top level quantity of
            interest. Each array contains the right-hand side for the linear
            solve.
        """

        if mode == 'fwd':
            rhs_vec = system.drmat
        else:
            rhs_vec = system.dumat

        ls_inputs = system._ls_inputs
        gs_outputs = {}
        for voi in ls_inputs:
            gs_outputs[voi] = [x for x in system.dumat[voi]]

        system.apply_linear(mode, ls_inputs=ls_inputs, vois=rhs_mat.keys(),
                            gs_outputs=gs_outputs)

        norm = 0.0
        for voi, rhs in iteritems(rhs_mat):
            rhs_vec[voi].vec[:] -= rhs
            norm += rhs_vec[voi].norm()**2

        return norm**0.5
