import numpy as np

from openmdao.core import Component, Assembly, Group
from openmdao.components import ExprComp, LinearSystem, ResidComp, UnitComp
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

        chem_eq.add('n2ls', N2LS_TP(thermo, dtype), promote=("n", 'init_prod_amounts', 'b', 'mu', 'rhs', 'ch'))
        chem_eq.add('ls2pi', LinearSystem(self.num_element+1,
                                          A_name="ch", x_name="pi", b_name="rhs")
                    promote="*")
        chem_eq.add('pi2n', PI2N_TP(thermo, dtype), promote="*")
        chem_eq.add('n_resid', ResidComp('dLn=0', 'n'))

        #total Property Calculations done after Chem_eq is converged
        self.add('TP2ls', TotalRHS(thermo, dtype), promote=('n', 'b', 'ch', 'rhs_T', 'rhs_P', 'lhs_TP'))

        # LinearSystem should implement its own solve_linear method
        self.add('ls2t', LinearSystem(self.num_element+1,
                         A_name="lhs_TP", x_name="result_T", b_name="rhs_T"),
                 promote="*")

        self.add('ls2p', LinearSystem(self.num_element+1,
                         A_name="lhs_TP", x_name="result_P", b_name="rhs_P"),
                 promote="*")
        self.add('tp2tot', TotalCalcs(thermo), promote=("n",'result_T','result_P','gamma'))

        self.solve_linear=LinGS()

        #convert units for boundary variables
        #Boundary inputs
        self.add('Tt_english', UnitComp(in_name='Tt', in_unit='degR',
                                        out_name="Tt_SI", out_unit="degK"), promote="*")
        self.add('Pt_english', UnitComp(in_name='Pt', in_unit='lbf/inch**2',
                                        out_name="Pt_SI", out_unit="bar"), promote="*")

        self.connect('Tt_SI', ('n2ls:Tt','TP2ls:Tt','tp2tot:Tt'))
        self.connect('Pt_SI', ('n2ls:Pt','tp2tot:Pt'))

        #Boundary Outputs
        self.add('ht_english', UnitComp('ht_SI', 'cal/g', 'ht', 'Btu/lbm'), promote="*")
        self.add('S_english', UnitComp('S_SI', 'cal/(g*degK)', 'S', 'Btu/(lbm*degR)'), promote="*")
        self.add('Cp_english', UnitComp('Cp_SI', 'cal/(g*degK)', 'Cp', 'Btu/(lbm*degR)'), promote="*")
        self.add('Cv_english', UnitComp('Cv_SI', 'cal/(g*degK)', 'Cv', 'Btu/(lbm*degR)'), promote="*")
        self.add('rhot_english', UnitComp('rhot_SI', "g/cm**3", 'rhot', "lbm/ft**3"), promote="*")

        self.connect('tp2tot:ht','ht_SI')
        self.connect('tp2tot:S','S_SI')
        self.connect('tp2tot:Cp','Cp_SI')
        self.connect('tp2tot:Cv','Cv_SI')
        self.connect('tp2tot:rhot','rhot_SI')
