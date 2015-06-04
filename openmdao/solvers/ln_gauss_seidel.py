""" OpenMDAO LinearSolver that uses linear Gauss Seidel."""

from __future__ import print_function

# pylint: disable=E0611, F0401
import numpy as np

from openmdao.components.paramcomp import ParamComp
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

        dunknowns = system.dunknowns
        dparams = system.dparams
        dresids = system.dresids

        dunknowns.vec[:] = 0.0
        dresids.vec[:] = 0.0
        system.clear_dparams()

        if mode == 'fwd':

            dunknowns.vec[:] = rhs

            for subsystem in system.subsystems(local=True):
                name, sub = subsystem
                #print(name, dparams.keys(), dunknowns.keys())

                #print('pre scatter', dparams.vec, dunknowns.vec, dresids.vec)
                system._varmanager._transfer_data(name, deriv=True)

                #print('pre apply', dparams.vec, dunknowns.vec, dresids.vec)
                sub.apply_linear(sub.params, sub.unknowns, sub.dparams,
                                 sub.dunknowns, sub.dresids, mode)
                #print('post apply', dparams.vec, dunknowns.vec, dresids.vec)

                sub.solve_linear(rhs, sub.dunknowns, sub.dresids, mode=mode)
                #print('post solve', dparams.vec, dunknowns.vec, dresids.vec)

            return dresids.vec

        else:

            dresids.vec[:] = rhs

            rev_systems = [sys for sys in system.subsystems(local=True)]

            for subsystem in reversed(rev_systems):
                name, sub = subsystem
                print(name, dparams.keys(), dunknowns.keys())

                print('pre apply', dparams.vec, dunknowns.vec, dresids.vec)
                sub.apply_linear(sub.params, sub.unknowns, sub.dparams,
                                 sub.dunknowns, sub.dresids, mode)

                print('pre scatter', dparams.vec, dunknowns.vec, dresids.vec)
                system._varmanager._transfer_data(name, mode='rev', deriv=True)

                print('pre solve', dparams.vec, dunknowns.vec, dresids.vec)
                #sub.solve_linear(rhs, sub.dunknowns, sub.dresids, mode=mode)
                print('post solve', dparams.vec, dunknowns.vec, dresids.vec)

            return dunknowns.vec
