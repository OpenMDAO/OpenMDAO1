import numpy as np
from openmdao.api import Component, Problem, Group, ExecComp, IndepVarComp

from disciplines.aerodynamics import Aerodynamics
from disciplines.performance import Performance
from disciplines.propulsion import Propulsion
from disciplines.structure import Structure
from disciplines.common import PolynomialFunction

class SSBJ_IDF_MDA(Group):

    def __init__(self, scalers, polyFunc, scaled=True, fd=False):
        super(SSBJ_IDF_MDA, self).__init__()

        self.scaled=scaled
        #Design variables
        if self.scaled:
            self.add('z_ini', IndepVarComp('z', np.array([1.2,  1.,  1.,  1.,  1.,  1.])), promotes=['*'])
            self.add('x_aer_ini', IndepVarComp('x_aer', 1.), promotes=['*'])
            self.add('x_str_ini', IndepVarComp('x_str', np.array([1. ,  1.])), promotes=['*'])
            self.add('x_pro_ini', IndepVarComp('x_pro', 1.), promotes=['*'])
        else:
            self.add('z_ini', IndepVarComp('z', np.array([0.05,45000.,1.6,5.5,55.0,1000.0])), promotes=['*'])
            self.add('x_aer_ini', IndepVarComp('x_aer', 1.0), promotes=['*'])
            self.add('x_str_ini', IndepVarComp('x_str', np.array([0.25,1.0])), promotes=['*'])
            self.add('x_pro_ini', IndepVarComp('x_pro', 0.5), promotes=['*'])

        self.add('L_ini', IndepVarComp('L', 0.888), promotes=['*'])
        self.add('WE_ini', IndepVarComp('WE', 1.490), promotes=['*'])
        self.add('WT_ini', IndepVarComp('WT', 0.888), promotes=['*'])
        self.add('Theta_ini', IndepVarComp('Theta', 0.997), promotes=['*'])
        self.add('ESF_ini', IndepVarComp('ESF', 1.463), promotes=['*'])
        self.add('D_ini', IndepVarComp('D', 0.457), promotes=['*'])
        #Disciplines
        self.add('Struc', Structure(scalers, polyFunc,fd=False))
        self.add('Aero', Aerodynamics(scalers, polyFunc,fd=False))
        self.add('Propu', Propulsion(scalers, polyFunc,fd=False))
        self.add('Perfo', Performance(scalers,fd=False))

        #Connections
        #shared variables z
        self.connect('z', 'Struc.z')
        self.connect('z', 'Aero.z')
        self.connect('z', 'Propu.z')
        self.connect('z', 'Perfo.z')
        #local variables
        self.connect('x_str', 'Struc.x_str')
        self.connect('x_aer', 'Aero.x_aer')
        self.connect('x_pro', 'Propu.x_pro')
        #coupling variables
        self.connect('L', 'Struc.L')
        self.connect('WE', 'Struc.WE')
        self.connect('WT', 'Aero.WT')
        self.connect('Theta', 'Aero.Theta')
        self.connect('ESF', 'Aero.ESF')
        self.connect('D','Propu.D')
        #Objective function
        self.add('Obj', ExecComp('obj=-R'), promotes=['obj'])
        #Connections
        self.connect('Perfo.R','Obj.R')
        self.connect('Propu.SFC','Perfo.SFC')
        self.connect('Aero.fin','Perfo.fin')
        self.connect('Struc.WT','Perfo.WT')
        self.connect('Struc.WF','Perfo.WF')
        #Coupling constraints
        self.add('con_Str_Aer_WT',ExecComp('con_str_aer_wt = (WTi-WT)**2',WTi=1.0),promotes=['con_str_aer_wt'])
        self.connect('Struc.WT','con_Str_Aer_WT.WT')
        self.connect('Aero.WT','con_Str_Aer_WT.WTi')
        
        comp_con_str_aer_theta = ExecComp('con_str_aer_theta = (Thetai-Theta)**2')
        self.add('con_Str_Aer_Theta',comp_con_str_aer_theta, promotes=['con_str_aer_theta'])
        self.connect('Struc.Theta','con_Str_Aer_Theta.Theta')
        self.connect('Aero.Theta','con_Str_Aer_Theta.Thetai')
        self.add('con_Aer_Str_L',ExecComp('con_aer_str_l = (Li-L)**2'), promotes=['con_aer_str_l'])
        self.connect('Aero.L','con_Aer_Str_L.L')
        self.connect('Struc.L','con_Aer_Str_L.Li')
        self.add('con_Aer_Pro_D',ExecComp('con_aer_pro_d = (Di-D)**2'), promotes=['con_aer_pro_d'])
        self.connect('Aero.D','con_Aer_Pro_D.D')
        self.connect('Propu.D','con_Aer_Pro_D.Di')
        
        self.add('con_Pro_Aer_ESF', ExecComp('con_pro_aer_esf = (ESFi-ESF)**2'), promotes=['con_pro_aer_esf'])
        self.connect('Propu.ESF','con_Pro_Aer_ESF.ESF')
        self.connect('Aero.ESF','con_Pro_Aer_ESF.ESFi')
        
        self.add('con_Pro_Str_WE',ExecComp('con_pro_str_we = (WEi-WE)**2'), promotes=['con_pro_str_we'])
        self.connect('Propu.WE','con_Pro_Str_WE.WE')
        self.connect('Struc.WE','con_Pro_Str_WE.WEi')
        #Local constraints
        self.add('con_Theta_sup',ExecComp('con_Theta_up = Theta*'+str(scalers['Theta'])+'-1.04'),promotes=['con_Theta_up'])
        self.connect('Aero.Theta','con_Theta_sup.Theta')
        self.add('con_Theta_inf',ExecComp('con_Theta_low = 0.96-Theta*'+str(scalers['Theta'])),promotes=['con_Theta_low'])
        self.connect('Aero.Theta','con_Theta_inf.Theta')
        self.add('con_Sigma1',ExecComp('con_sigma1=sigma[0]*'+str(scalers['sigma'][0])+'-1.9',sigma=np.zeros(5)),promotes=['con_sigma1'])
        self.connect('Struc.sigma','con_Sigma1.sigma')
        self.add('con_Sigma2',ExecComp('con_sigma2=sigma[1]*'+str(scalers['sigma'][0])+'-1.9',sigma=np.zeros(5)),promotes=['con_sigma2'])
        self.connect('Struc.sigma','con_Sigma2.sigma')
        self.add('con_Sigma3',ExecComp('con_sigma3=sigma[2]*'+str(scalers['sigma'][0])+'-1.9',sigma=np.zeros(5)),promotes=['con_sigma3'])
        self.connect('Struc.sigma','con_Sigma3.sigma')
        self.add('con_Sigma4',ExecComp('con_sigma4=sigma[3]*'+str(scalers['sigma'][0])+'-1.9',sigma=np.zeros(5)),promotes=['con_sigma4'])
        self.connect('Struc.sigma','con_Sigma4.sigma')
        self.add('con_Sigma5',ExecComp('con_sigma5=sigma[4]*'+str(scalers['sigma'][0])+'-1.9',sigma=np.zeros(5)),promotes=['con_sigma5'])
        self.connect('Struc.sigma','con_Sigma5.sigma')
        self.add('con_Dpdx',ExecComp('con_dpdx=dpdx*'+str(scalers['dpdx'])+'-1.04'),promotes=['con_dpdx'])
        self.connect('Aero.dpdx','con_Dpdx.dpdx')
        self.add('con_ESF',ExecComp('con_esf=ESF*'+str(scalers['ESF'])+'-1.5'),promotes=['con_esf'])
        self.connect('Aero.ESF','con_ESF.ESF')
        self.add('con_Temp',ExecComp('con_temp=Temp*'+str(scalers['Temp'])+'-1.0'),promotes=['con_temp'])
        self.connect('Propu.Temp','con_Temp.Temp')
        self.add('con_DT',ExecComp('con_dt=DT'),promotes=['con_dt'])
        self.connect('Propu.DT','con_DT.DT')

        if fd:
            self.fd_options['force_fd'] = True
            self.fd_options['form']='central'
            self.fd_options['step_type']= 'relative'
            self.fd_options['step_size']= 1e-2
