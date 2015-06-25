""" OpenMDAO LinearSolver that uses linear Gauss Seidel."""

from __future__ import print_function

from openmdao.core.component import Component
from openmdao.solvers.solverbase import LinearSolver


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
        opt.add_option('mode', 'fwd', values=['fwd', 'rev', 'auto'],
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
        #drmat[None].vec[:] = -rhs

        vois = rhs_mat.keys()
        sol_buf = {}
        ls_inputs = {}
        norm0, norm = 1.0, 1.0
        counter = 0
        while counter < self.options['maxiter'] and \
              norm > self.options['atol'] and \
              norm/norm0 > self.options['rtol']:

            if mode == 'fwd':

                for name, sub in system.subsystems(local=True):

                    #for voi in vois:
                        #print(name, dpmat[voi].keys(), dumat[voi].keys())

                    for voi in vois:
                        #print('pre scatter', dpmat[voi].vec, dumat[voi].vec, drmat[voi].vec)
                        system._transfer_data(name, deriv=True, var_of_interest=voi)
                        #print('pre apply', dpmat[voi].vec, dumat[voi].vec, drmat[voi].vec)

                        ls_inputs[voi] = [x for x in dpmat[voi] if x not in sub.dpmat[voi]]

                    if isinstance(sub, Component):

                        # Components need to reverse sign and add 1 on diagonal
                        # for explicit unknowns
                        system._sub_apply_linear_wrapper(sub, mode, vois, ls_inputs=ls_inputs)

                    else:
                        # Groups and all other systems just call their own
                        # apply_linear.
                        sub.apply_linear(mode, ls_inputs=ls_inputs, vois=vois)

                    #for voi in vois:
                        #print('post apply', dpmat[voi].vec, dumat[voi].vec, drmat[voi].vec)

                    for voi in vois:
                        drmat[voi].vec *= -1.0
                        drmat[voi].vec += rhs_mat[voi]

                    sub.solve_linear(sub.dumat, sub.drmat,vois, mode=mode)
                    #for voi in vois:
                        #print('post solve', dpmat[voi].vec, dumat[voi].vec, drmat[voi].vec)

                for voi in vois:
                    sol_buf[voi] = dumat[voi].vec

            else:

                rev_systems = [sys for sys in system.subsystems(local=True)]

                for subsystem in reversed(rev_systems):
                    name, sub = subsystem
                    #for voi in vois:
                        #print(name, dpmat[voi].keys(), dumat[voi].keys())

                    for voi in vois:
                        dumat[voi].vec *= 0.0

                        #print('pre scatter', dpmat[voi].vec, dumat[voi].vec, drmat[voi].vec)
                        system._transfer_data(name, mode='rev', deriv=True, var_of_interest=voi)
                        #print('post scatter', dpmat[voi].vec, dumat[voi].vec, drmat[voi].vec)

                        dumat[voi].vec *= -1.0
                        dumat[voi].vec += rhs_mat[voi]

                    sub.solve_linear(sub.dumat, sub.drmat, vois, mode=mode)
                    #for voi in vois:
                        #print('post solve', dpmat[voi].vec, dumat[voi].vec, drmat[voi].vec)

                    for voi in vois:
                        ls_inputs[voi] = [x for x in dpmat[voi].keys() \
                                          if x not in sub.dpmat[voi].keys()]

                    if isinstance(sub, Component):

                        # Components need to reverse sign and add 1 on diagonal
                        # for explicit unknowns
                        system._sub_apply_linear_wrapper(sub, mode, vois, ls_inputs=ls_inputs)

                    else:
                        # Groups and all other systems just call their own
                        # apply_linear.
                        sub.apply_linear(mode, ls_inputs=ls_inputs, vois=vois)


                    #for voi in vois:
                        #print('post apply', dpmat[voi].vec, dumat[voi].vec, drmat[voi].vec)

                for voi in vois:
                    sol_buf[voi] = drmat[voi].vec


            counter += 1
            if self.options['maxiter'] == 1:
                norm = 0.0
            else:
                norm = self._norm(system, mode, rhs_mat)
                print('Residual:', norm)

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
            rhs_vec = system.dumat
        else:
            rhs_vec = system.drmat

        ls_inputs = {}
        for voi in rhs_mat:
            ls_inputs[voi] = system._all_params(voi)

        system.apply_linear(mode, ls_inputs=ls_inputs, vois=rhs_mat.keys())

        norm = 0.0
        for voi, rhs in rhs_mat.items():
            rhs_vec[voi].vec[:] -= rhs
            norm += rhs_vec[voi].norm()**2

        return norm
