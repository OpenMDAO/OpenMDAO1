"""
SSBJ test case implementation
see http://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19980234657.pdf
"""
from __future__ import print_function
import numpy as np

from openmdao.api import ExecComp, IndepVarComp
from openmdao.api import Group, Problem
from openmdao.api import NLGaussSeidel, ScipyGMRES

from disciplines.aerodynamics import Aerodynamics
from disciplines.performance import Performance
from disciplines.propulsion import Propulsion
from disciplines.structure import Structure
from disciplines.common import PolynomialFunction
# pylint: disable=C0103

class SSBJ_MDA(Group):
    """
    SSBJ Analysis with aerodynamics, performance, propulsion and structure disciplines.
    """
    def __init__(self, scalers, polyFunc):
        super(SSBJ_MDA, self).__init__()

        #Design variables
        self.add('z_ini',
                 IndepVarComp('z', np.array([1.0,1.0,1.0,1.0,1.0,1.0])),
                 promotes=['*'])
        self.add('x_aer_ini', IndepVarComp('x_aer', 1.0), promotes=['*'])
        self.add('x_str_ini',
                 IndepVarComp('x_str', np.array([1.0,1.0])),
                 promotes=['*'])
        self.add('x_pro_ini', IndepVarComp('x_pro', 1.0), promotes=['*'])

        #Disciplines
        sap_group = Group()
        sap_group.add('Struc', Structure(scalers,polyFunc), promotes=['*'])
        sap_group.add('Aero', Aerodynamics(scalers,polyFunc), promotes=['*'])
        sap_group.add('Propu',Propulsion(scalers,polyFunc),promotes=['*'])

        sap_group.nl_solver = NLGaussSeidel()
        sap_group.nl_solver.options['atol'] = 1.0e-3
        sap_group.ln_solver = ScipyGMRES()
        self.add('sap', sap_group, promotes=['*'])

        self.add('Perfo', Performance(scalers), promotes=['*'])

        #Constraints
        self.add('con_Theta_sup', ExecComp('con_Theta_up = Theta*'+\
                                           str(scalers['Theta'])+'-1.04'), promotes=['*'])
        self.add('con_Theta_inf', ExecComp('con_Theta_low = 0.96-Theta*'+\
                                           str(scalers['Theta'])), promotes=['*'])
        for i in range(5):
            self.add('con_Sigma'+str(i+1), ExecComp('con_sigma'+str(i+1)+'=sigma['+str(i)+']*'+\
                                                    str(scalers['sigma'][i])+'-1.9',
                                                    sigma=np.zeros(5)), promotes=['*'])
        self.add('con_Dpdx', ExecComp('con_dpdx=dpdx*'+str(scalers['dpdx'])+'-1.04'),
                 promotes=['*'])
        self.add('con1_ESF', ExecComp('con1_esf=ESF*'+str(scalers['ESF'])+'-1.5'),
                 promotes=['*'])
        self.add('con2_ESF', ExecComp('con2_esf=0.5-ESF*'+str(scalers['ESF'])),
                 promotes=['*'])
        self.add('con_Temp', ExecComp('con_temp=Temp*'+str(scalers['Temp'])+'-1.02'),
                 promotes=['*'])

        self.add('con_DT', ExecComp('con_dt=DT'), promotes=['*'])

def init_ssbj_mda():
    """
    Runs the analysis once.
    """
    prob = Problem()

    # Mean point is chosen for the design variables
    scalers = {}
    scalers['z']=np.array([0.05,45000.,1.6,5.5,55.0,1000.0])
    scalers['x_str']=np.array([0.25,1.0])
    scalers['x_pro']=0.5

    # Others variables are unknowns for the moment so Scale=1.0
    scalers['WT']=1.0
    scalers['Theta']=1.0
    scalers['L']=1.0
    scalers['WF']=1.0
    scalers['D']=1.0
    scalers['ESF']=1.0
    scalers['WE']=1.0
    scalers['fin']=1.0
    scalers['SFC']=1.0
    scalers['R']=1.0
    scalers['DT']=1.0
    scalers['Temp']=1.0
    scalers['dpdx']=1.0
    scalers['sigma']=np.array([1.0,1.0,1.0,1.0,1.0])

    pfunc = PolynomialFunction()
    prob.root = SSBJ_MDA(scalers, pfunc)
    prob.setup()

    #Initialization of acceptable values as initial values for the polynomial functions
    Z = prob.root.unknowns['z']*scalers['z']
    Wfo = 2000
    Wo = 25000
    We = 3*4360.0*(1.0**1.05)
    t = Z[0]*Z[5]/np.sqrt(Z[5]*Z[3])
    Wfw = (5.*Z[5]/18.)*(2.0/3.0*t)*(42.5)
    Fo = 1.0
    Wtotal = 80000.
    Wtot=1.1*Wtotal
    while abs(Wtot - Wtotal) > Wtotal*0.0001:
        Wtot = Wtotal
        Ww  = Fo*(.0051*((Wtot*6.0)**0.557)*Z[5]**.649*Z[3]**.5*Z[0]**-.4*((1.0+0.25)**.1)*\
                  ((np.cos(Z[4]*np.pi/180))**-1)*((.1875*Z[5])**.1))
        Wtotal = Wo + Ww + Wfo + Wfw + We

    prob.root.unknowns['WT'] = Wtotal
    prob.root.sap.params['Aero.WT'] = Wtotal
    prob.root.sap.params['Struc.L'] = Wtotal
    prob.root.unknowns['L'] = Wtotal

    prob.run()

    #Uptade the scalers dictionary
    for key in scalers.iterkeys():
        if key not in ['z', 'x_str', 'x_pro']:
            scalers[key] = prob.root.unknowns[key]

    return scalers, pfunc

if __name__ == "__main__":
    scalers, _ = init_ssbj_mda()
    print(scalers)
