""" OpenMDAO LinearSolver that uses linear Gauss Seidel."""

from __future__ import print_function

# pylint: disable=E0611, F0401
import numpy as np

from openmdao.components.paramcomp import ParamComp
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
        opt.add_option('maxiter', 100,
                       desc='Maximum number of iterations.')
        opt.add_option('mode', 'fwd', values=['fwd', 'rev', 'auto'],
                       desc="Derivative calculation mode, set to 'fwd' for " + \
                       "forward mode, 'rev' for reverse mode, or 'auto' to " + \
                       "let OpenMDAO determine the best mode.")

    def solve(self, rhs, system, mode):
        """ Solves the linear system for the problem in self.system. The
        full solution vector is returned.

        Parameters
        ----------
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

        dumat = system.dumat
        drmat = system.drmat
        dpmat = system.dpmat

        system.clear_dparams()
        for names in system._relevance.vars_of_interest():
            for name in names:
                if name in dumat:
                    dumat[name].vec[:] = 0.0
                    drmat[name].vec[:] = 0.0
        dumat[None].vec[:] = 0.0
        drmat[None].vec[:] = 0.0

        #FIXME: Just want to get LGS working by itself before considering matmat
        voi = None

        if mode == 'fwd':

            for name, sub in system.subsystems(local=True):

                print(name, dpmat[voi].keys(), dumat[voi].keys())

                print('pre scatter', dpmat[voi].vec, dumat[voi].vec, drmat[voi].vec)
                system._transfer_data(name, deriv=True)

                #dresids.vec[:] = 0.0
                print('pre apply', dpmat[voi].vec, dumat[voi].vec, drmat[voi].vec)

                if isinstance(sub, Component):

                    # Components need to reverse sign and add 1 on diagonal
                    # for explicit unknowns
                    system._sub_apply_linear_wrapper(sub, mode, voi)

                else:
                    # Groups and all other systems just call their own
                    # apply_linear.
                    sub.apply_linear(sub.params, sub.unknowns, sub.dpmat,
                                     sub.dumat, sub.drmat, mode)

                print('post apply', dpmat[voi].vec, dumat[voi].vec, drmat[voi].vec)

                drmat[voi].vec *= -1.0
                drmat[voi].vec += rhs

                sub.solve_linear(sub.drmat[voi].vec, sub.dumat[voi], sub.drmat[voi],
                                 mode=mode)
                print('post solve', dpmat[voi].vec, dumat[voi].vec, drmat[voi].vec)

            return dumat[voi].vec

        else:

            rev_systems = [sys for sys in system.subsystems(local=True)]

            for subsystem in reversed(rev_systems):
                name, sub = subsystem
                print(name, dpmat[None].keys(), dumat[None].keys())

                dunknowns.vec *= -1.0
                dunknowns.vec += rhs

                print('pre solve', dpmat[None].vec, dumat[None].vec, drmat[None].vec)
                sub.solve_linear(sub.dunknowns.vec, sub.dunknowns, sub.dresids,
                                 mode=mode)

                print('pre apply', dpmat[None].vec, dumat[None].vec, drmat[None].vec)

                if isinstance(sub, Component):

                    # Components need to reverse sign and add 1 on diagonal
                    # for explicit unknowns
                    system._sub_apply_linear_wrapper(sub, mode)

                else:
                    # Groups and all other systems just call their own
                    # apply_linear.
                    sub.apply_linear(sub.params, sub.unknowns, sub.dparams,
                                     sub.dunknowns, sub.dresids, mode)

                print('pre scatter', dpmat[None].vec, dumat[None].vec, drmat[None].vec)
                system._transfer_data(name, mode='rev', deriv=True)
                print('post scatter', dpmat[None].vec, dumat[None].vec, drmat[None].vec)

            return dresids.vec
