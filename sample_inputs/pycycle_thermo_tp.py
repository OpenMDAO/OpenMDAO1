import numpy as np

from openmdao.core import Component, Assembly, Group
from openmdao.components import ExprComp, LinearSystem, ResidComp
from openmdao.solver.linear import LinGS, ScipyGMRES

from pycycle.chemeq_tp import N2LS_TP, PI2N_TP
from pycycle.totals import TotalRHS, TotalCalcs
from pycycle.solver import CEAGaussSeidel

class SetTotalTP(Assembly):

    def __init_(self, themo_data=props.janaf, dtype=np.float, test_resid=False):

        self.thermo = thermo = props.Thermo(thermo_data)

        self.num_prod = thermo.num_prod
        self.num_element = thermo.num_element

        self.test_resid = test_resid
        self.dtype = dtype

        #chemical equilibrium calculations
        chem_eq = self.add('chem_eq', Group())
        if not test_resid: #don't put a solver in if you want to test residual derivatives
            chem_eq.solve_nonlinear = CEAGaussSeidel()
            chem_eq.solve_linear = ScipyGMRES()

        chem_eq.add('n2ls', N2LS_TP(thermo, dtype), promote="*")
        chem_eq.add('ls2pi', LinearSystem(self.num_element+1,
                                          A_name="ch", x_name="pi", b_name="rhs")
                    promote="*")
        chem_eq.add('pi2n', PI2N_TP(thermo, dtype), promote="*")
        chem_eq.add('n_resid', ResidComp('dLn=0', 'n'))

        #total Property Calculations done after Chem_eq is converged
        self.add('TP2ls', TotalRHS(thermo, dtype), promote="*")

        # LinearSystem should implement its own solve_linear method
        self.add('ls2t', LinearSystem(self.num_element+1,
                         A_name="lhs_TP", x_name="result_T", b_name="rhs_T"),
                 promote="*")

        self.add('ls2p', LinearSystem(self.num_element+1,
                         A_name="lhs_TP", x_name="result_P", b_name="rhs_P"),
                 promote="*")
        self.add('tp2tot', TotalCalcs(thermo), promote="*")

        self.solve_linear=LinGS()
