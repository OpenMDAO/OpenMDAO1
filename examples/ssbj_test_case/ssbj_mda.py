import sys
import numpy as np

from openmdao.api import ExecComp, IndepVarComp
from openmdao.api import Group
from openmdao.api import NLGaussSeidel, ScipyGMRES
from openmdao.api import Newton
from openmdao.api import Problem

from disciplines.aerodynamics import Aerodynamics
from disciplines.performance import Performance
from disciplines.propulsion import Propulsion
from disciplines.structure import Structure
from disciplines.common import PolynomialFunction

class SSBJ_MDA(Group):

    def __init__(self, scalers, polyFunc, scaled=True, fd=False):
        super(SSBJ_MDA, self).__init__()

        self.scaled=scaled
        #Design variables
        if self.scaled:
            self.add('z_ini', IndepVarComp('z', np.array([1.0,1.0,1.0,1.0,1.0,1.0])), promotes=['*'])
            self.add('x_aer_ini', IndepVarComp('x_aer', 1.0), promotes=['*'])
            self.add('x_str_ini', IndepVarComp('x_str', np.array([1.0,1.0])), promotes=['*'])
            self.add('x_pro_ini', IndepVarComp('x_pro', 1.0), promotes=['*'])
        else:
            self.add('z_ini', IndepVarComp('z', np.array([0.05,45000.,1.6,5.5,55.0,1000.0])), promotes=['*'])
            self.add('x_aer_ini', IndepVarComp('x_aer', 1.0), promotes=['*'])
            self.add('x_str_ini', IndepVarComp('x_str', np.array([0.25,1.0])), promotes=['*'])
            self.add('x_pro_ini', IndepVarComp('x_pro', 0.5), promotes=['*'])

        #Disciplines
        g = Group()
        g.add('Struc', Structure(scalers,polyFunc), promotes=['*'])
        g.add('Aero', Aerodynamics(scalers,polyFunc), promotes=['*'])
        g.add('Propu',Propulsion(scalers,polyFunc),promotes=['*'])

        g.nl_solver = NLGaussSeidel()
        g.nl_solver.options['atol'] = 1.0e-3
        g.ln_solver = ScipyGMRES()
        #or use default LinearGaussSeidel solver but with an increased maxiter
        #g.ln_solver.options['maxiter']=100
        self.add('sap', g, promotes=['*'])

        self.add('Perfo', Performance(scalers), promotes=['*'])

        #Constraints
        self.add('con_Theta_sup',ExecComp('con_Theta_up = Theta*'+str(scalers['Theta'])+'-1.04'), promotes=['*'])
        self.add('con_Theta_inf',ExecComp('con_Theta_low = 0.96-Theta*'+str(scalers['Theta'])), promotes=['*'])
        self.add('con_Sigma1',ExecComp('con_sigma1=sigma[0]*'+str(scalers['sigma'][0])+'-1.9',sigma=np.zeros(5)),promotes=['*'])
        self.add('con_Sigma2',ExecComp('con_sigma2=sigma[1]*'+str(scalers['sigma'][1])+'-1.9',sigma=np.zeros(5)),promotes=['*'])
        self.add('con_Sigma3',ExecComp('con_sigma3=sigma[2]*'+str(scalers['sigma'][2])+'-1.9',sigma=np.zeros(5)),promotes=['*'])
        self.add('con_Sigma4',ExecComp('con_sigma4=sigma[3]*'+str(scalers['sigma'][3])+'-1.9',sigma=np.zeros(5)),promotes=['*'])
        self.add('con_Sigma5',ExecComp('con_sigma5=sigma[4]*'+str(scalers['sigma'][4])+'-1.9',sigma=np.zeros(5)),promotes=['*'])
        self.add('con_Dpdx',ExecComp('con_dpdx=dpdx*'+str(scalers['dpdx'])+'-1.04'),promotes=['*'])
        self.add('con1_ESF',ExecComp('con1_esf=ESF*'+str(scalers['ESF'])+'-1.5'),promotes=['*'])
        self.add('con2_ESF',ExecComp('con2_esf=0.5-ESF*'+str(scalers['ESF'])),promotes=['*'])
        self.add('con_Temp',ExecComp('con_temp=Temp*'+str(scalers['Temp'])+'-1.02'),promotes=['*'])

        self.add('con_DT',ExecComp('con_dt=DT'),promotes=['*'])

        if fd:
            self.fd_options['force_fd'] = True
            self.fd_options['form']='central'
            self.fd_options['step_type']= 'relative'
            self.fd_options['step_size']= 1e-8

def init_ssbj_mda(scaled=True):
    P=Problem()
    #scalers constants True or False, if False Scaling['all']=1.0
    #Mean point is chosen for the design variables
    scalers={}
    if scaled:
        scalers['z']=np.array([0.05,45000.,1.6,5.5,55.0,1000.0])
        scalers['x_str']=np.array([0.25,1.0])
        scalers['x_pro']=0.5
    else:
        scalers['z']=np.array([1.0,1.0,1.0,1.0,1.0,1.0])
        scalers['x_str']=np.array([1.0,1.0])
        scalers['x_pro']=1.0

    #Others variables are unknowns for the moment so Scale=1.0
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

    pf = PolynomialFunction()
    P.root=SSBJ_MDA(scalers, pf, scaled=scaled)
    P.setup()
    #Initialization of acceptable values as initial values for the polynomial functions
    Z=P.root.unknowns['z']*scalers['z']
    Wfo = 2000
    Wo  = 25000
    We  = 3*4360.0*(1.0**1.05)
    t   = Z[0]*Z[5]/np.sqrt(Z[5]*Z[3])
    Wfw = (5.*Z[5]/18.)*(2.0/3.0*t)*(42.5)
    Fo  = 1.0
    Wtotal=80000; Wtot=1.1*Wtotal;
    while abs(Wtot - Wtotal) > Wtotal*0.0001:
        Wtot = Wtotal;
        Ww  = Fo*(.0051*((Wtot*6.0)**0.557)*Z[5]**.649*Z[3]**.5*Z[0]**-.4*((1.0+0.25)**.1)*((np.cos(Z[4]*np.pi/180))**-1)*((.1875*Z[5])**.1))
        Wtotal = Wo + Ww + Wfo + Wfw + We

    P.root.unknowns['WT']=Wtotal
    P.root.sap.params['Aero.WT']=Wtotal
    P.root.sap.params['Struc.L']=Wtotal
    P.root.unknowns['L']=Wtotal

    #run
    P.run()
    #Uptade the scalers dictionary
    for key in scalers.iterkeys():
        if key not in ['z','x_str','x_pro']:
            if scaled:
                scalers[key]=P.root.unknowns[key]
            else:
                scalers[key]=1.0
    if not scaled:
        scalers['sigma']=np.ones((5,))

    return scalers, pf

if __name__ == "__main__":
    scaler, pf = init_ssbj_mda()
    print(scaler)