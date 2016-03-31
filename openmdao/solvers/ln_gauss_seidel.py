""" OpenMDAO LinearSolver that uses linear Gauss Seidel."""

from __future__ import print_function

from six import iteritems, itervalues
from collections import OrderedDict

from openmdao.core.system import AnalysisError
from openmdao.solvers.solver_base import LinearSolver


class LinearGaussSeidel(LinearSolver):
    """ LinearSolver that uses linear Gauss Seidel.

    Options
    -------
    options['atol'] :  float(1e-12)
        Absolute convergence tolerance.
    options['err_on_maxiter'] : bool(False)
        If True, raise an AnalysisError if not converged at maxiter.
    options['iprint'] :  int(0)
        Set to 0 to disable printing, set to 1 to print the residual to stdout each iteration, set to 2 to print subiteration residuals as well.
    options['maxiter'] :  int(1)
        Maximum number of iterations.
    options['mode'] :  str('auto')
        Derivative calculation mode, set to 'fwd' for forward mode, 'rev' for reverse mode, or 'auto' to let OpenMDAO determine the best mode.
    options['rtol'] :  float(1e-10)
        Absolute convergence tolerance.

    """

    def __init__(self):
        super(LinearGaussSeidel, self).__init__()

        opt = self.options
        opt.add_option('atol', 1e-12, lower=0.0,
                       desc='Absolute convergence tolerance.')
        opt.add_option('rtol', 1e-10, lower=0.0,
                       desc='Absolute convergence tolerance.')
        opt.add_option('maxiter', 1, lower=1,
                       desc='Maximum number of iterations.')
        opt.add_option('mode', 'auto', values=['fwd', 'rev', 'auto'],
                       desc="Derivative calculation mode, set to 'fwd' for " +
                       "forward mode, 'rev' for reverse mode, or 'auto' to " +
                       "let OpenMDAO determine the best mode.",
                       lock_on_setup=True)
        opt.add_option('single_voi_relevance_reduction',
                        False, values=[True, False],
                        desc="If True, use relevance reduction even for"
                              " individual variables of interest. This "
                              "may increase performance but will use "
                              "more memory.",
                        lock_on_setup=True)

        self.print_name = 'LN_GS'

    def setup(self, group):
        """ Solvers override to define post-setup initiailzation.

        Args
        ----
        group: `Group`
            Group that owns this solver.
        """
        super(LinearGaussSeidel, self).setup(group)

        self._vois = [None]
        for vois in group._probdata.relevance.vars_of_interest():
            self._vois.extend(vois)

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
        gs_outputs = system._get_gs_outputs(mode, self._vois)
        relevance = system._probdata.relevance
        fwd = mode == 'fwd'

        system.clear_dparams()
        for voi in rhs_mat:
            dumat[voi].vec[:] = 0.0

        vois = rhs_mat.keys()
        # John starts with the following. It is not necessary, but
        # uncommenting it helps to debug when comparing print outputs to his.
        # for voi in vois:
        #    drmat[voi].vec[:] = -rhs_mat[voi]

        sol_buf = OrderedDict()

        f_norm0, f_norm = 1.0, 1.0
        self.iter_count = 0
        maxiter = self.options['maxiter']
        while self.iter_count < maxiter and f_norm > self.options['atol'] \
                  and f_norm/f_norm0 > self.options['rtol']:

            if fwd:

                for sub in itervalues(system._subsystems):

                    for voi in vois:
                        #print('pre scatter', sub.pathname, 'dp', dpmat[voi].vec,
                        #      'du', dumat[voi].vec, 'dr', drmat[voi].vec)
                        system._transfer_data(sub.name, deriv=True,
                                              var_of_interest=voi)
                        #print('pre apply', sub.pathname, 'dp', dpmat[voi].vec,
                        #      'du', dumat[voi].vec, 'dr', drmat[voi].vec)

                    # we need to loop over all subsystems in order to make
                    # the necessary collective calls to scatter, but only
                    # active subsystems do anything else
                    if not sub.is_active():
                        continue

                    # print(sub.name, sorted(gs_outputs['fwd'][sub.name][None]))

                    # Groups and all other systems just call their own
                    # apply_linear.
                    sub._sys_apply_linear(mode, system._do_apply, vois=vois,
                                          gs_outputs=gs_outputs['fwd'][sub.name])

                    # for voi in vois:
                    #    print('post apply', dpmat[voi].vec, dumat[voi].vec, drmat[voi].vec)

                    for voi in vois:
                        drmat[voi].vec *= -1.0
                        drmat[voi].vec += rhs_mat[voi]
                        dpmat[voi].vec[:] = 0.0

                    with sub._dircontext:
                        sub.solve_linear(sub.dumat, sub.drmat, vois, mode=mode)

                    # for voi in vois:
                    #    print('post solve', dpmat[voi].vec, dumat[voi].vec, drmat[voi].vec)

                for voi in vois:
                    sol_buf[voi] = dumat[voi].vec

            else:

                for sub in reversed(list(itervalues(system._subsystems))):

                    active = sub.is_active()

                    for voi in vois:
                        if active:
                            dumat[voi].vec *= 0.0

                        #print('pre scatter', sub.pathname, voi, dpmat[voi].vec, dumat[voi].vec, drmat[voi].vec)
                        system._transfer_data(sub.name, mode='rev', deriv=True, var_of_interest=voi)
                        #print('post scatter', sub.pathname, voi, dpmat[voi].vec, dumat[voi].vec, drmat[voi].vec)

                        if active:
                            dumat[voi].vec *= -1.0
                            dumat[voi].vec += rhs_mat[voi]

                    # we need to loop over all subsystems in order to make
                    # the necessary collective calls to scatter, but only
                    # active subsystems do anything else
                    if not active:
                        continue

                    with sub._dircontext:
                        sub.solve_linear(sub.dumat, sub.drmat, vois, mode=mode)
                    #for voi in vois:
                        #print('post solve', dpmat[voi].vec, dumat[voi].vec, drmat[voi].vec)

                    #print(sub.name, sorted(gs_outputs['rev'][sub.name][None]))

                    # Groups and all other systems just call their own
                    # apply_linear.
                    sub._sys_apply_linear(mode, system._do_apply, vois=vois,
                                         gs_outputs=gs_outputs['rev'][sub.name])

                    #for voi in vois:
                        #print('post apply', system.dpmat[voi].vec, dumat[voi].vec, drmat[voi].vec)

                for voi in vois:
                    sol_buf[voi] = drmat[voi].vec

            self.iter_count += 1
            if maxiter == 1:
                f_norm = 0.0
            else:
                f_norm = self._norm(system, mode, rhs_mat)

            if self.options['iprint'] > 0:
                self.print_norm(self.print_name, system.pathname, self.iter_count,
                                f_norm, f_norm0, indent=1, solver='LN')

        if maxiter > 1 and self.iter_count >= maxiter:
            msg = 'FAILED to converge after %d iterations' % self.iter_count
            failed = True
        else:
            failed = False

        if self.options['iprint'] > 0:
            if not failed:
                msg = 'converged'

            self.print_norm(self.print_name, system.pathname, self.iter_count, f_norm,
                            f_norm0, indent=1, solver='LN', msg=msg)

        if failed and self.options['err_on_maxiter']:
            raise AnalysisError("Solve in '%s': LinearGaussSeidel %s" %
                                (system.pathname, msg))

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

        # we used to build gs_outputs up using dumat, but dumat is already
        # identical to gs_outputs in the vois we care about, so just use it.
        system._sys_apply_linear(mode, system._do_apply, vois=rhs_mat.keys(),
                                gs_outputs=system.dumat)

        if mode == 'fwd':
            rhs_vec = system.drmat
        else:
            rhs_vec = system.dumat

        norm = 0.0
        for voi, rhs in iteritems(rhs_mat):
            rhs_vec[voi].vec[:] -= rhs
            norm += rhs_vec[voi].norm()**2

        return norm**0.5
