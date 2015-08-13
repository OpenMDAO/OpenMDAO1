""" OpenMDAO LinearSolver that uses PetSC KSP to solve for a system's
derivatives. This solver can be used under MPI."""

from __future__ import print_function

from openmdao.solvers.solver_base import LinearSolver


class PetscKSP(LinearSolver):
    """ OpenMDAO LinearSolver that uses PetSC KSP to solve for a system's
    derivatives. This solver can be used under MPI."""