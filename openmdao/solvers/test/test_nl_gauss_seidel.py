""" Unit test for the Nonlinear Gauss Seidel nonlinear solver. """

import unittest

from openmdao.components.paramcomp import ParamComp
from openmdao.core.group import Group
from openmdao.core.problem import Problem
from openmdao.solvers.scipy_gmres import ScipyGMRES
from openmdao.test.simplecomps import SimpleCompDerivMatVec