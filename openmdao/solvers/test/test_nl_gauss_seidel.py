""" Unit test for the Nonlinear Gauss Seidel nonlinear solver. """

import unittest

from openmdao.core.problem import Problem
from openmdao.solvers.scipy_gmres import ScipyGMRES
from openmdao.test.sellar import SellarNoDerivatives