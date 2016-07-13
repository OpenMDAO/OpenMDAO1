"""
SSBJ test case - http://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19980234657.pdf
Python implementation and OpenMDAO integration developed by
Sylvain Dubreuil and Remi Lafage of ONERA, the French Aerospace Lab.
"""
from __future__ import print_function
import numpy as np

from openmdao.api import Component, Problem, Group, IndepVarComp
from common import PolynomialFunction, CDMIN
# pylint: disable=C0103

class Aerodynamics(Component):
    def __init__(self, scalers, pfunc):
        super(Aerodynamics, self).__init__()
        # Global Design Variable z=(t/c,h,M,AR,Lambda,Sref)
        self.add_param('z', val=np.zeros(6))
        # Local Design Variable x_aer=Cf
        self.add_param('x_aer', val=1.0)
        # Coupling parameters
        self.add_param('WT', val=1.0)
        self.add_param('Theta', val=1.0)
        self.add_param('ESF', val=1.0)
        # Coupling output
        self.add_output('L', val=1.0)
        self.add_output('D', val=1.0)
        self.add_output('fin', val=1.0)
        self.add_output('dpdx', val=1.0)
        # scalers values
        self.scalers = scalers
        # Polynomial function initialized with given
        # constant values
        self.pf = pfunc

    def solve_nonlinear(self, params, unknowns, resids):
        #Variables scaling
        Z = params['z']*self.scalers['z']
        WT = params['WT']*self.scalers['WT']
        ESF = params['ESF']*self.scalers['ESF']
        Theta = params['Theta']*self.scalers['Theta']
        #Computation
        if Z[1] <= 36089.0:
            V = 1116.39 * Z[2] * np.sqrt(abs(1.0 - 6.875E-6*Z[1]))
            rho = 2.377E-3 * (1. - 6.875E-6*Z[1])**4.2561
        else:
            V = 968.1 * abs(Z[2])
            rho = 2.377E-3 * 0.2971 * np.exp((36089.0 - Z[1]) / 20806.7)
        CL = WT / (0.5*rho*(V**2)*Z[5])
        Fo2 = self.pf.eval([ESF, abs(params['x_aer'])], [1, 1], [.25]*2, "Fo2")

        CDmin = CDMIN*Fo2 + 3.05*abs(Z[0])**(5.0/3.0) \
                * abs(np.cos(Z[4]*np.pi/180.0))**1.5
        if Z[2] >= 1:
            k = abs(Z[3]) * (abs(Z[2])**2-1.0) * np.cos(Z[4]*np.pi/180.) \
            / (4.* abs(Z[3])* np.sqrt(abs(Z[4]**2 - 1.) - 2.))
        else:
            k = (0.8 * np.pi * abs(Z[3]))**-1

        Fo3 = self.pf.eval([Theta], [5], [.25], "Fo3")
        CD = (CDmin + k * CL**2) * Fo3
        unknowns['L'] = params['WT']
        D = CD * 0.5 * rho * V**2 * Z[5]
        #Unknowns
        unknowns['D'] = D/self.scalers['D']
        fin = WT/D
        unknowns['fin'] = fin/self.scalers['fin']
        unknowns['dpdx'] = self.pf.eval([Z[0]], [1], [.25], "dpdx")/self.scalers['dpdx']

    def linearize(self, params, unknowns, resids):
        J = {}
        #Variables scaling
        Z = params['z']*self.scalers['z']
        WT = params['WT']*self.scalers['WT']
        ESF = params['ESF']*self.scalers['ESF']
        Theta = params['Theta']*self.scalers['Theta']
        #Computation of some terms necessary to the jacobian computation
        if Z[1] <= 36089.0:
            V = 1116.39 * Z[2] * np.sqrt(abs(1.0 - 6.875E-6 * Z[1]))
            rho = 2.377E-3 * (1. - 6.875E-6*Z[1])**4.2561
        else:
            V = 968.1 * abs(Z[2])
            rho = 2.377E-3*0.2971*np.exp((36089.0 - Z[1])/20806.7)
        CL = WT / (0.5*rho*(V**2)*Z[5])
        Fo2 = self.pf.eval([ESF, abs(params['x_aer'])], [1, 1], [.25]*2, "Fo2")

        CDmin = CDMIN * Fo2 + 3.05 * abs(Z[0])**(5.0/3.0) \
                * abs(np.cos(Z[4]*np.pi/180.0))**1.5
        if Z[2] >= 1.:
            k = abs(Z[3]) * (abs(Z[2])**2-1.0) * np.cos(Z[4]*np.pi/180.) \
                / (4. * abs(Z[3])* np.sqrt(abs(Z[4]**2 - 1.) - 2.))
        else:
            k = (0.8 * np.pi * abs(Z[3]))**-1

        Fo3 = self.pf.eval([Theta], [5], [.25], "Fo3")
        CD = (CDmin + k * CL**2) * Fo3
        D = CD * 0.5 * rho * V**2 * Z[5]
        #L=WT => dL/d.=0.0
        J['L', 'x_aer'] = np.array([[0.0]])
        J['L', 'z'] = np.zeros((1, 6))
        J['L', 'WT'] = np.array([[1.0]])
        J['L', 'Theta'] = np.array([[0.0]])
        J['L', 'ESF'] = np.array([[0.0]])
        #D
        S_shifted, Ai, Aij = self.pf.eval([ESF, abs(params['x_aer'])],
                                          [1, 1], [.25]*2, "Fo2", deriv=True)
        if abs(params['x_aer'])/self.pf.d['Fo2'][1]>=0.75 and \
           abs(params['x_aer'])/self.pf.d['Fo2'][1]<=1.25:	  
            dSCfdCf = 1.0/self.pf.d['Fo2'][1]
        else:
            dSCfdCf = 0.0
        dSCfdCf2 = 2.0*S_shifted[0, 1]*dSCfdCf
        dFo1dCf = Ai[1]*dSCfdCf+0.5*Aij[1, 1]*dSCfdCf2+Aij[0, 1]*S_shifted[0, 1]*dSCfdCf
        dDdCf = 0.5*rho*V**2*Z[5]*Fo3*CDMIN*dFo1dCf
        J['D', 'x_aer'] = np.array([[dDdCf/self.scalers['D']]]).reshape((1, 1))
        dDdtc = 0.5*rho*V**2*Z[5]*5.0/3.0*3.05*Fo3*Z[0]**(2./3.)*np.cos(Z[4]*np.pi/180.)**(3./2.)
        if Z[1] <= 36089.0:
            drhodh = 2.377E-3 * 4.2561 * 6.875E-6* (1. - 6.875E-6 * Z[1])**3.2561
            dVdh = 6.875E-6*1116.39*Z[2]/2* (1.0 - 6.875E-6 * Z[1])**-0.5
        else:
            drhodh = 2.377E-3 * 0.2971 * (-1.0)/20806.7 *np.exp((36089.0 - Z[1]) / 20806.7)
            dVdh = 0.0
        dVdh2 = 2.0*dVdh*V
        dCDdh = -k*Fo3*CL*WT/(0.5*Z[5])*(V**-2*rho**-2*drhodh+rho**-1*V**-3*dVdh)
        dDdh = 0.5*Z[5]*(drhodh*CDmin*V**2+rho*dCDdh*V**2+rho*CDmin*dVdh2)
        if Z[1] <= 36089.0:
            dVdM = 1116.39*(1.0 - 6.875E-6 * Z[1])**-0.5
        else:
            dVdM = 968.1
        if Z[2] >= 1:
            dkdM = abs(Z[3]) * (2.0*abs(Z[2])) * np.cos(Z[4]*np.pi/180.) \
                / (4. * abs(Z[3])* np.sqrt(abs(Z[4]**2 - 1.) - 2.))
        else:
            dkdM = 0.0
        dVdM2 = 2.0*V*dVdM
        dCLdM = -2.0*WT/(0.5*Z[5])*rho**-1*V**-3*dVdM
        dCDdM = Fo3*(2.0*k*CL*dCLdM+CL**2*dkdM)
        dDdM = 0.5*rho*Z[5]*(CD*dVdM2+V**2*dCDdM)
        if Z[2] >= 1:
            dkdAR = 0.0
        else:
            dkdAR = -1.0/(0.8 * np.pi * abs(Z[3])**2)
        dCDdAR = Fo3*CL**2*dkdAR
        dDdAR = 0.5*rho*Z[5]*V**2*dCDdAR
        dCDmindLambda = -3.05*3.0/2.0*Z[0]**(5.0/3.0)\
            *np.cos(Z[4]*np.pi/180.)**0.5*np.pi/180.*np.sin(Z[4]*np.pi/180.)
        if Z[2] >= 1:
            u = (Z[2]**2-1.)*np.cos(Z[4]*np.pi/180.)
            up = -np.pi/180.0*(Z[2]**2-1.)*np.sin(Z[4]*np.pi/180.)
            v = 4.0*np.sqrt(Z[4]**2-1.0)-2.0
            vp = 4.0*Z[4]*(Z[4]**2-1.0)**-0.5
            dkdLambda = (up*v-u*vp)/v**2
        else:
            dkdLambda = 0.0
        dCDdLambda = Fo3*(dCDmindLambda+CL**2*dkdLambda)
        dDdLambda = 0.5*rho*Z[5]*V**2*dCDdLambda
        dCLdSref2 = 2.0*CL*-WT/(0.5*rho*V**2*Z[5]**2)
        dCDdSref = Fo3*k*dCLdSref2
        dDdSref = 0.5*rho*V**2*(CD+Z[5]*dCDdSref)
        J['D', 'z'] = np.array([[dDdtc/self.scalers['D'], dDdh/self.scalers['D'],
                                 dDdM/self.scalers['D'], dDdAR/self.scalers['D'],
                                 dDdLambda/self.scalers['D'],
                                 dDdSref/self.scalers['D']]])*self.scalers['z']
        dDdWT = Fo3*k*2.0*WT/(0.5*rho*V**2*Z[5])
        J['D', 'WT'] = np.array([[dDdWT/self.scalers['D']*self.scalers['WT']]])
        S_shifted, Ai, Aij = self.pf.eval([Theta], [5], [.25], "Fo3", deriv=True)
        if Theta/self.pf.d['Fo3'][0]>=0.75 and Theta/self.pf.d['Fo3'][0]<=1.25: 
            dSThetadTheta = 1.0/self.pf.d['Fo3'][0]
        else:
            dSThetadTheta = 0.0
        dSThetadTheta2 = 2.0*S_shifted[0, 0]*dSThetadTheta
        dFo3dTheta = Ai[0]*dSThetadTheta + 0.5*Aij[0, 0]*dSThetadTheta2
        dCDdTheta = dFo3dTheta*(CDmin+k*CL**2)
        dDdTheta = 0.5*rho*V**2*Z[5]*dCDdTheta
        J['D', 'Theta'] = np.array(
            [[dDdTheta/self.scalers['D']*self.scalers['Theta']]]).reshape((1, 1))
        S_shifted, Ai, Aij = self.pf.eval([ESF, abs(params['x_aer'])],
                                          [1, 1], [.25]*2, "Fo2", deriv=True)
        if ESF/self.pf.d['Fo2'][0]>=0.75 and ESF/self.pf.d['Fo2'][0]<=1.25: 							  
            dSESFdESF = 1.0/self.pf.d['Fo2'][0]
        else:
            dSESFdESF = 0.0
        dSESFdESF2 = 2.0*S_shifted[0, 0]*dSESFdESF
        dFo2dESF = Ai[0]*dSESFdESF+0.5*Aij[0, 0]*dSESFdESF2 \
                   + Aij[1, 0]*S_shifted[0, 1]*dSESFdESF
        dCDdESF = Fo3*CDMIN*dFo2dESF
        dDdESF = 0.5*rho*V**2*Z[5]*dCDdESF
        J['D', 'ESF'] = np.array(
            [[dDdESF/self.scalers['D']*self.scalers['ESF']]]).reshape((1, 1))
        ##################dpdx=pf(t/c)
        J['dpdx', 'x_aer'] = np.array([[0.0]])
        J['dpdx', 'z'] = np.zeros((1, 6))
        S_shifted, Ai, Aij = self.pf.eval([Z[0]], [1], [.25], "dpdx", deriv=True)
        if Z[0]/self.pf.d['dpdx'][0]>=0.75 and Z[0]/self.pf.d['dpdx'][0]<=1.25:
            dStcdtc = 1.0/self.pf.d['dpdx'][0]
        else:
            dStcdtc = 0.0
        dStcdtc2 = 2.0*S_shifted[0, 0]*dStcdtc
        ddpdxdtc = Ai[0]*dStcdtc+0.5*Aij[0, 0]*dStcdtc2
        J['dpdx', 'z'][0, 0] = ddpdxdtc*self.scalers['z'][0]/self.scalers['dpdx']
        J['dpdx', 'WT'] = np.array([[0.0]])
        J['dpdx', 'Theta'] = np.array([[0.0]])
        J['dpdx', 'ESF'] = np.array([[0.0]])
        #################fin=L/D
        J['fin', 'x_aer'] = np.array(
            [[-dDdCf*WT/D**2/self.scalers['WT']*self.scalers['D']]]).reshape((1, 1))
        J['fin', 'z'] = np.array(
            [-J['D', 'z'][0]*WT/self.scalers['WT']/D**2*self.scalers['D']**2])
        J['fin', 'WT'] = np.array(
            [[(D-dDdWT*WT)/D**2/self.scalers['WT']*self.scalers['D'] \
              *self.scalers['WT']]]).reshape((1, 1))
        J['fin', 'Theta'] = np.array(
            [[(-dDdTheta*WT)/D**2/self.scalers['WT']*self.scalers['D'] \
              *self.scalers['Theta']]]).reshape((1, 1))
        J['fin', 'ESF'] = np.array(
            [[(-dDdESF*WT)/D**2/self.scalers['WT']\
              *self.scalers['D']*self.scalers['ESF']]]).reshape((1, 1))
        return J

if __name__ == "__main__": # pragma: no cover

    scalers = {}
    scalers['z'] = np.array([0.05, 45000., 1.6, 5.5, 55.0, 1000.0])
    scalers['WT'] = 49909.58578
    scalers['ESF'] = 1.0
    scalers['Theta'] = 0.950978
    scalers['D'] = 12193.7018
    scalers['fin'] = 4.093062
    scalers['dpdx'] = 1.0

    top = Problem()
    top.root = Group()
    top.root.add('z_in', IndepVarComp('z', np.array([1.2  ,  1.333,  0.875,  0.45 ,  1.27 ,  1.5])),
                 promotes=['*'])
    top.root.add('x_aer_in', IndepVarComp('x_aer', 0.75), promotes=['*'])
    top.root.add('WT_in', IndepVarComp('WT', 0.89), promotes=['*'])
    top.root.add('Theta_in', IndepVarComp('Theta', 0.9975), promotes=['*'])
    top.root.add('ESF_in', IndepVarComp('ESF', 1.463), promotes=['*'])
    top.root.add('Aer1', Aerodynamics(scalers, PolynomialFunction()), promotes=['*'])
    top.setup()
    top.run()
    J1 = top.root.Aer1.linearize(top.root.Aer1.params,
                                 top.root.Aer1.unknowns,
                                 top.root.Aer1.resids)
    J2 = top.root.Aer1.fd_jacobian(top.root.Aer1.params,
                                   top.root.Aer1.unknowns,
                                   top.root.Aer1.resids)
    errAbs = []
    for i in range(len(J2.keys())):
        errAbs.append(J1[J2.keys()[i]] - J2[J2.keys()[i]])
        print ('errAbs_'+str(J2.keys()[i])+'=',
               J1[J2.keys()[i]]-J2[J2.keys()[i]])
        print (J1[J2.keys()[i]].shape == J2[J2.keys()[i]].shape)
