from __future__ import print_function
import numpy as np
from openmdao.api import Component
from common import PolynomialFunction, WBE

class Propulsion(Component):

    def __init__(self, scalers, polyFunc, fd=False):
        super(Propulsion, self).__init__()
        # Global Design Variable z=(t/c,h,M,AR,Lambda,Sref)
        self.add_param('z', val=np.zeros(6))
        # Local Design Variable x_pro=T
        self.add_param('x_pro', val=1.0)
        # Coupling parameters
        self.add_param('D', val=1.0)
        # Coupling output
        self.add_output('SFC', val=1.0)
        self.add_output('WE', val=1.0)
        self.add_output('ESF', val=1.0)
        self.add_output('DT', val=1.0)
        self.add_output('Temp', val=1.0)
        # scalers values
        self.scalers = scalers
        # Polynomial function initialized with given constant values
        self.pf = polyFunc
        # Finite differences
        if fd:
            self.fd_options['force_fd'] = True
            self.fd_options['form'] = 'central'
            self.fd_options['step_type'] = 'relative'
            self.fd_options['step_size'] = 1e-8

    def solve_nonlinear(self, params, unknowns, resids):
        #Variables scaling
        Z = params['z']*self.scalers['z']
        Xpro = params['x_pro']*self.scalers['x_pro']
        #print("Z={} Xpro={}".format(params['z'], params['x_pro'])) 
        Tbar = abs(Xpro) * 16168.6
        unknowns['Temp'] = self.pf.eval(
            [Z[2], Z[1], abs(Xpro)],
            [2, 4, 2], [.25]*3, "Temp")/self.scalers['Temp']
        ESF = (params['D']*self.scalers['D']/3.0)/Tbar
        unknowns['ESF'] = ESF/self.scalers['ESF']
        SFC = 1.1324 + 1.5344*Z[2] - 3.2956E-05*Z[1] - 1.6379E-04*Tbar \
            - 0.31623*Z[2]**2 + 8.2138E-06*Z[2]*Z[1] - 10.496E-5*Tbar*Z[2] \
            - 8.574E-11*Z[1]**2 + 3.8042E-9*Tbar*Z[1] + 1.06E-8*Tbar**2
        unknowns['SFC'] = SFC/self.scalers['SFC']
        WE = 3.0*WBE*abs(ESF)**1.05
        unknowns['WE'] = WE/self.scalers['WE']
        TUAbar = 11484.0 + 10856.0 * Z[2] - 0.50802 * Z[1] \
            + 3200.2*(Z[2]**2) - 0.29326 * Z[2] * Z[1] + 6.8572E-6 * Z[1]**2
        DT = Tbar/TUAbar - 1.0
        unknowns['DT'] = DT/self.scalers['DT']

    def linearize(self, params, unknowns, resids):
        J = {}
        #Changement de variable
        Z = params['z']*self.scalers['z']
        Xpro = params['x_pro']*self.scalers['x_pro']
        Tbar = abs(Xpro) * 16168.6
        ESF = (params['D']*self.scalers['D']/3.0)/Tbar
        TUAbar = 11484.0 + 10856.0 * Z[2] - 0.50802 * Z[1] \
            + 3200.2 * Z[2]**2 - 0.29326 * Z[2] * Z[1] + 6.8572E-6 * Z[1]**2
        ##############SFC
        dSFCdT = -1.6379e-4*16168.6-10.496e-5*16168.6*Z[2]\
            +3.8042e-9*16168.6*Z[1]+2.0*1.06e-8*16168.6**2*Xpro
        J['SFC', 'x_pro'] = dSFCdT/self.scalers['SFC']*self.scalers['x_pro']
        dSFCdh = -3.2956e-5+8.2138e-6*Z[2]-2.0*8.574e-11*Z[1]+3.8042e-9*Tbar
        dSFCdM = 1.5344-2.0*0.31623*Z[2]+8.2138e-6*Z[1]-10.496e-5*Tbar
        J['SFC', 'z'] = np.zeros((1, 6))
        J['SFC', 'z'][0, 1] = dSFCdh/self.scalers['SFC']*self.scalers['z'][1]
        J['SFC', 'z'][0, 2] = dSFCdM/self.scalers['SFC']*self.scalers['z'][2]
        J['SFC', 'D'] = np.array([[0]])
        ###############ESF
        dESFdT = (-params['D']*self.scalers['D']/3.0)/(16168.6*Xpro**2)
        J['ESF', 'x_pro'] = np.array(
            [[dESFdT/self.scalers['ESF']*self.scalers['x_pro']]])
        J['ESF', 'z'] = np.zeros((1, 6))
        dESFdD = (1.0/3.0)/Tbar
        J['ESF', 'D'] = np.array([[dESFdD/self.scalers['ESF']*self.scalers['D']]])
        ###############WE
        dWEdT = 3.0*WBE*1.05*ESF**0.05*dESFdT
        J['WE', 'x_pro'] = np.array([[dWEdT/self.scalers['WE']*self.scalers['x_pro']]])
        J['WE', 'z'] = np.zeros((1, 6))
        dWEdD = 3.0*WBE*1.05*ESF**0.05*dESFdD
        J['WE', 'D'] = np.array([[dWEdD/self.scalers['WE']*self.scalers['D']]])
        ##############DT
        dDTdT = 16168.6/TUAbar
        J['DT', 'x_pro'] = np.array([[dDTdT/self.scalers['DT']*self.scalers['x_pro']]])
        dDTdh = -(-0.50802-0.29326*Z[2]+2.0*6.8572e-6*Z[1])*TUAbar**-2*Tbar
        dDTdM = -(10856.0+2.0*3200.2*Z[2]-0.29326*Z[1])*TUAbar**-2*Tbar
        J['DT', 'z'] = np.zeros((1, 6))
        J['DT', 'z'][0, 1] = dDTdh/self.scalers['DT']*self.scalers['z'][1]
        J['DT', 'z'][0, 2] = dDTdM/self.scalers['DT']*self.scalers['z'][2]
        J['DT', 'D'] = np.array([[0.0]])
        #############Temp
        S_shifted, Ai, Aij = self.pf.eval([Z[2], Z[1], abs(Xpro)], [2, 4, 2],
                                          [.25]*3, "Temp", deriv=True)
        if abs(Xpro)/self.pf.d['Temp'][2]<=1.25 and abs(Xpro)/self.pf.d['Temp'][2]>=0.75:						  
            dSTdT = 1.0/self.pf.d['Temp'][2]
        else:
            dSTdT = 0.0
        dSTdT2 = 2.0*S_shifted[0, 2]*dSTdT
        dTempdT = Ai[2]*dSTdT+0.5*Aij[2, 2]*dSTdT2 \
            +Aij[0, 2]*S_shifted[0, 0]*dSTdT+Aij[1, 2]*S_shifted[0, 1]*dSTdT
        J['Temp', 'x_pro'] = np.array(
            [[dTempdT/self.scalers['Temp']*self.scalers['x_pro']]]).reshape((1, 1))
        J['Temp', 'z'] = np.zeros((1, 6))
        if Z[1]/self.pf.d['Temp'][1]<=1.25 and Z[2]/self.pf.d['Temp'][1]>=0.75: 	
            dShdh = 1.0/self.pf.d['Temp'][1]
        else:
            dShdh = 0.0
        dShdh2 = 2.0*S_shifted[0, 1]*dShdh
        dTempdh = Ai[1]*dShdh+0.5*Aij[1, 1]*dShdh2 \
            + Aij[0, 1]*S_shifted[0, 0]*dShdh+Aij[2, 1]*S_shifted[0, 2]*dShdh
        if Z[2]/self.pf.d['Temp'][0]<=1.25 and Z[2]/self.pf.d['Temp'][0]>=0.75: 	
            dSMdM = 1.0/self.pf.d['Temp'][0]
        else:
            dSMdM = 0.0
        dSMdM2 = 2.0*S_shifted[0, 0]*dSMdM
        dTempdM = Ai[0]*dSMdM+0.5*Aij[0, 0]*dSMdM2 \
            +Aij[1, 0]*S_shifted[0, 1]*dSMdM+Aij[2, 0]*S_shifted[0, 2]*dSMdM
        J['Temp', 'z'][0, 1] = dTempdh/self.scalers['Temp']*self.scalers['z'][1]
        J['Temp', 'z'][0, 2] = dTempdM/self.scalers['Temp']*self.scalers['z'][2]
        J['Temp', 'D'] = np.array([[0.0]])
        return J

if __name__ == "__main__": # pragma: no cover

    from openmdao.api import Component, Problem, Group, IndepVarComp
    scalers = {}
    scalers['z'] = np.array([0.05, 45000., 1.6, 5.5, 55.0, 1000.0])
    scalers['x_pro'] = 0.5
    scalers['Temp'] = 1.0
    scalers['ESF'] = 1.0
    scalers['SFC'] = 1.12328
    scalers['WE'] = 5748.915355
    scalers['DT'] = 0.278366
    scalers['D'] = 12193.7018
    top = Problem()
    top.root = Group()
    top.root.add('z_in', IndepVarComp('z', np.array([1.2  ,  1.333,  0.875,  0.45 ,  1.27 ,  1.5])),
                 promotes=['*'])
    top.root.add('x_pro_in', IndepVarComp('x_pro', 0.3126), promotes=['*'])
    top.root.add('D_in', IndepVarComp('D', 0.457), promotes=['*'])
    pf = PolynomialFunction()
    top.root.add('Pro1', Propulsion(scalers,pf), promotes=['*'])
    top.setup()
    top.run()
    J1 = top.root.Pro1.linearize(top.root.Pro1.params,
                                top.root.Pro1.unknowns,
                                top.root.Pro1.resids)
    J2 = top.root.Pro1.fd_jacobian(top.root.Pro1.params,
                                   top.root.Pro1.unknowns,
                                   top.root.Pro1.resids)
    errAbs = []
    for i in range(len(J2.keys())):
        errAbs.append(J1[J2.keys()[i]] - J2[J2.keys()[i]])
        print ('ErrAbs_'+str(J2.keys()[i])+'=',
               J1[J2.keys()[i]]-J2[J2.keys()[i]])
        print (J1[J2.keys()[i]].shape == J2[J2.keys()[i]].shape)
