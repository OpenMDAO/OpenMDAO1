from __future__ import print_function
import numpy as np
from openmdao.api import Component
from disciplines.common import PolynomialFunction, WFO, WO, NZ

class Structure(Component):
#       """
#       Structure component
#       Input:
#               - Local: taper, x
#               - Shared: AR, sweep, tc, Sref
#               - Aerodynamics: L
#               - Propulsion: WE
#       Output: WT, WF, twist, sigma1->5
#       """
    def __init__(self, scalers, polyFunc, fd=False):
        super(Structure, self).__init__()
        # Global Design Variable z=(t/c,h,M,AR,Lambda,Sref)
        self.add_param('z', val=np.zeros(6))

        # Local Design Variable x_str=(lambda,section caisson)
        self.add_param('x_str', val=np.zeros(2))

        # Coupling parameters
        self.add_param('L', val=1.0)
        self.add_param('WE', val=1.0)
        # Coupling output
        self.add_output('WT', val=1.0)
        self.add_output('Theta', val=1.0)
        self.add_output('WF', val=1.0)
        self.add_output('sigma', val=np.zeros(5))
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
        Xstr = params['x_str']*self.scalers['x_str']
        L = params['L']*self.scalers['L']
        #Computation
        t = Z[0]*Z[5]/(np.sqrt(abs(Z[5]*Z[3])))
        b = np.sqrt(abs(Z[5]*Z[3]))/2.0
        R = (1.0+2.0*Xstr[0])/(3.0*(1.0+Xstr[0]))
        Theta = self.pf.eval([abs(Xstr[1]), b, R, L],
                             [2, 4, 4, 3], [0.25]*4, "twist")
        ##Uncomment to use Fo1=pf(x)
        Fo1 = self.pf.eval([Xstr[1]], [1], [.008], "Fo1")
        #Fo1=1.0
        WT_hat = L
        WW = Fo1 * (0.0051 * abs(WT_hat*NZ)**0.557 * \
                    abs(Z[5])**0.649 * abs(Z[3])**0.5 * abs(Z[0])**(-0.4) \
                    * abs(1.0+Xstr[0])**0.1 * (0.1875*abs(Z[5]))**0.1 \
                    / abs(np.cos(Z[4]*np.pi/180.)))
        WFW = 5.0/18.0 * abs(Z[5]) * 2.0/3.0 * t * 42.5
        WF = WFW + WFO
        WT = WO + WW + WF + params['WE']*self.scalers['WE']
        Sigma0 = self.pf.eval([Z[0], L, Xstr[1], b, R],
                              [4, 1, 4, 1, 1], [0.1]*5, "sigma[1]")
        Sigma1 = self.pf.eval([Z[0], L, Xstr[1], b, R],
                              [4, 1, 4, 1, 1], [0.15]*5, "sigma[2]")
        Sigma2 = self.pf.eval([Z[0], L, Xstr[1], b, R],
                              [4, 1, 4, 1, 1], [0.2]*5, "sigma[3]")
        Sigma3 = self.pf.eval([Z[0], L, Xstr[1], b, R],
                              [4, 1, 4, 1, 1], [0.25]*5, "sigma[4]")
        Sigma4 = self.pf.eval([Z[0], L, Xstr[1], b, R],
                              [4, 1, 4, 1, 1], [0.30]*5, "sigma[5]")
        #Unknowns
        unknowns['Theta'] = Theta/self.scalers['Theta']
        unknowns['WF'] = WF/self.scalers['WF']
        unknowns['WT'] = WT/self.scalers['L']
        unknowns['sigma'][0] = Sigma0/self.scalers['sigma'][0]
        unknowns['sigma'][1] = Sigma1/self.scalers['sigma'][1]
        unknowns['sigma'][2] = Sigma2/self.scalers['sigma'][2]
        unknowns['sigma'][3] = Sigma3/self.scalers['sigma'][3]
        unknowns['sigma'][4] = Sigma4/self.scalers['sigma'][4]

    def linearize(self, params, unknowns, resids):
        #Variables scaling
        Z = params['z']*self.scalers['z']
        Xstr = params['x_str']*self.scalers['x_str']
        L = params['L']*self.scalers['L']
        #Uncomment to use Fo1=pf(x)
        Fo1 = self.pf.eval([Xstr[1]], [1], [.008], "Fo1")
        #Fo1=1.0
        """ Jacobian for Structure discipline """
        J = {}
        ######################WT#############################
        dWtdlambda = 0.1*Fo1/np.cos(Z[4]*np.pi/180.)*0.0051 \
            *(abs(L)*NZ)**0.557*abs(Z[5])**0.649 \
            * abs(Z[3])**0.5 * abs(Z[0])**(-0.4) \
            * (1.0+Xstr[0])**-0.9 * (0.1875*abs(Z[5]))**0.1
        A = (0.0051 * abs(L*NZ)**0.557 * abs(Z[5])**0.649 \
             * abs(Z[3])**0.5 * abs(Z[0])**(-0.4) * abs(1.0+Xstr[0])**0.1 \
             * (0.1875*abs(Z[5]))**0.1 / abs(np.cos(Z[4]*np.pi/180.)))
        #uncomment to use Fo1=self.pf(x)
        S_shifted, Ai, Aij = self.pf.eval([Xstr[1]], [1], [.008],
                                          "Fo1", deriv=True)
        dWtdx = A*(Ai[0]/self.pf.d['Fo1'][0] \
                   + Aij[0, 0]/self.pf.d['Fo1'][0]*S_shifted[0, 0])
        # if Fo1=1.0
        #dWtdx=0.0
        J['WT', 'x_str'] = np.array([[dWtdlambda/self.scalers['L'],
                                      dWtdx/self.scalers['L']]])*self.scalers['x_str']
        dWTdtc = -0.4*Fo1/np.cos(Z[4]*np.pi/180.)*0.0051 \
            * abs(L*NZ)**0.557 * abs(Z[5])**0.649 \
            * abs(Z[3])**0.5*abs(Z[0])**(-1.4)*abs(1.0+Xstr[0])**0.1 \
            * (0.1875*abs(Z[5]))**0.1  +  212.5/27.*Z[5]**(3.0/2.0)/np.sqrt(Z[3])
        dWTdh = 0.0
        dWTdM = 0.0
        dWTdAR = 0.5*Fo1/np.cos(Z[4]*np.pi/180.)* 0.0051 \
            * abs(L*NZ)**0.557 * abs(Z[5])**0.649 \
            * abs(Z[3])**-0.5*abs(Z[0])**(-0.4)*abs(1.0+Xstr[0])**0.1 \
            * (0.1875*abs(Z[5]))**0.1 + 212.5/27.*Z[5]**(3.0/2.0) \
            * Z[0] * -0.5*Z[3]**(-3.0/2.0)
        dWTdLambda = Fo1*np.pi/180.*np.sin(Z[4]*np.pi/180.)/np.cos(Z[4]*np.pi/180.)**2 \
            * 0.0051 * abs(L*NZ)**0.557 * abs(Z[5])**0.649 \
            * abs(Z[3])**0.5*abs(Z[0])**(-0.4)*abs(1.0+Xstr[0])**0.1 \
            * (0.1875*abs(Z[5]))**0.1
        dWTdSref = 0.749*Fo1/np.cos(Z[4]*np.pi/180.)*0.1875**(0.1)*0.0051 \
            * abs(L*NZ)**0.557*abs(Z[5])**-0.251 \
            *abs(Z[3])**0.5*abs(Z[0])**(-0.4)*abs(1.0+Xstr[0])**0.1 \
            + 637.5/54.*Z[5]**(0.5)*Z[0]/np.sqrt(Z[3])
        J['WT', 'z'] = np.array([[dWTdtc/self.scalers['L'],
                                  dWTdh/self.scalers['L'],
                                  dWTdM/self.scalers['L'],
                                  dWTdAR/self.scalers['L'],
                                  dWTdLambda/self.scalers['L'],
                                  dWTdSref/self.scalers['L']]])*self.scalers['z']
        dWTdL = 0.557*Fo1/np.cos(Z[4]*np.pi/180.)*0.0051 * abs(L)**-0.443 \
            *NZ**0.557* abs(Z[5])**0.649 * abs(Z[3])**0.5 \
            * abs(Z[0])**(-0.4) * abs(1.0+Xstr[0])**0.1 * (0.1875*abs(Z[5]))**0.1
        J['WT', 'L'] = np.array([[dWTdL]])
        dWTdWE = 1.0
        J['WT', 'WE'] = np.array([[dWTdWE]])/self.scalers['L']*self.scalers['WE']
        ######################WF#############################
        dWFdlambda = 0.0
        dWFdx = 0.0
        J['WF', 'x_str'] = np.array([[dWFdlambda/self.scalers['WF'],
                                      dWFdx/self.scalers['WF']]]) \
            *self.scalers['x_str']
        dWFdtc = 212.5/27.*Z[5]**(3.0/2.0)/np.sqrt(Z[3])
        dWFdh = 0.0
        dWFdM = 0.0
        dWFdAR = 212.5/27.*Z[5]**(3.0/2.0) * Z[0] * -0.5*Z[3]**(-3.0/2.0)
        dWFdLambda = 0.0
        dWFdSref = 637.5/54.*Z[5]**(0.5)*Z[0]/np.sqrt(Z[3])
        J['WF', 'z'] = np.array([[dWFdtc/self.scalers['WF'],
                                  dWFdh/self.scalers['WF'],
                                  dWFdM/self.scalers['WF'],
                                  dWFdAR/self.scalers['WF'],
                                  dWFdLambda/self.scalers['WF'],
                                  dWFdSref/self.scalers['WF']]])\
            *self.scalers['z']
        dWFdL = 0.0
        J['WF', 'L'] = np.array([[dWFdL]])/self.scalers['WF']*self.scalers['L']
        dWFdWE = 0.0
        J['WF', 'WE'] = np.array([[dWFdWE]])/self.scalers['WF']*self.scalers['WE']
        ##################Theta#############################
        b = np.sqrt(abs(Z[5]*Z[3]))/2.0
        R = (1.0+2.0*Xstr[0])/(3.0*(1.0+Xstr[0]))
        S_shifted, Ai, Aij = self.pf.eval([abs(Xstr[1]), b, R, L],
                                          [2, 4, 4, 3],
                                          [0.25]*4, "twist", deriv=True)
        dSRdlambda = 1.0/self.pf.d['twist'][2]*1.0/(3.0*(1.0+Xstr[0])**2)
        dSRdlambda2 = 2.0*S_shifted[0, 2]*dSRdlambda
        dThetadlambda = Ai[2]*dSRdlambda + 0.5*Aij[2, 2]*dSRdlambda2 \
            + Aij[0, 2]*S_shifted[0, 0]*dSRdlambda \
            + Aij[1, 2]*S_shifted[0, 1]*dSRdlambda\
            + Aij[3, 2]*S_shifted[0, 3]*dSRdlambda
        dSxdx = 1.0/self.pf.d['twist'][0]
        dSxdx2 = 2.0*S_shifted[0, 0]*dSxdx
        dThetadx = Ai[0]*dSxdx + 0.5*Aij[0, 0]*dSxdx2 \
            + Aij[1, 0]*S_shifted[0, 1]*dSxdx \
            + Aij[2, 0]*S_shifted[0, 2]*dSxdx \
            + Aij[3, 0]*S_shifted[0, 3]*dSxdx
        J['Theta', 'x_str'] = np.array([[dThetadlambda[0, 0]/self.scalers['Theta'],
                                         dThetadx[0, 0]/self.scalers['Theta']]])\
            *self.scalers['x_str']
        dThetadtc = 0.0
        dThetadh = 0.0
        dThetadM = 0.0
        dSbdAR = 1.0/self.pf.d['twist'][1]*(np.sqrt(Z[5])/4.0*Z[3]**-0.5)
        dSbdAR2 = 2.0*S_shifted[0, 1]*dSbdAR
        dThetadAR = Ai[1]*dSbdAR+0.5*Aij[1, 1]*dSbdAR2 \
            + Aij[0, 1]*S_shifted[0, 0]*dSbdAR \
            + Aij[2, 1]*S_shifted[0, 2]*dSbdAR \
            + Aij[3, 1]*S_shifted[0, 3]*dSbdAR
        dThetadLambda = 0.0
        dSbdSref = 1.0/self.pf.d['twist'][1]*(np.sqrt(Z[3])/4.0*Z[5]**-0.5)
        dSbdSref2 = 2.0*S_shifted[0, 1]*dSbdSref
        dThetadSref = Ai[1]*dSbdSref + 0.5*Aij[1, 1]*dSbdSref2 \
            + Aij[0, 1]*S_shifted[0, 0]*dSbdSref \
            + Aij[2, 1]*S_shifted[0, 2]*dSbdSref \
            + Aij[3, 1]*S_shifted[0, 3]*dSbdSref

        J['Theta', 'z'] = np.array([[dThetadtc/self.scalers['Theta'],
                                     dThetadh/self.scalers['Theta'],
                                     dThetadM/self.scalers['Theta'],
                                     dThetadAR/self.scalers['Theta'],
                                     dThetadLambda/self.scalers['Theta'],
                                     dThetadSref/self.scalers['Theta']]])*self.scalers['z']
        dSLdL = 1.0/self.pf.d['twist'][3]
        dSLdL2 = 2.0*S_shifted[0, 3]*dSLdL
        dThetadL = Ai[3]*dSLdL + 0.5*Aij[3, 3]*dSLdL2 \
            + Aij[0, 3]*S_shifted[0, 0]*dSLdL \
            + Aij[1, 3]*S_shifted[0, 1]*dSLdL \
            + Aij[2, 3]*S_shifted[0, 2]*dSLdL
        J['Theta', 'L'] = (np.array([[dThetadL]]) \
                           / self.scalers['Theta']*self.scalers['L']).reshape((1, 1))
        dThetadWE = 0.0
        J['Theta', 'WE'] = np.array([[dThetadWE]])/self.scalers['Theta']*self.scalers['WE']

        ###############sigma###############################
        b = np.sqrt(abs(Z[5]*Z[3]))/2.0
        R = (1.0+2.0*Xstr[0])/(3.0*(1.0+Xstr[0]))
        S_shifted, Ai, Aij = self.pf.eval([Z[0], L, Xstr[1], b, R],
                                          [4, 1, 4, 1, 1], [0.1]*5,
                                          "sigma[1]", deriv=True)
        dSRdlambda = 1.0/self.pf.d['sigma[1]'][4]*1.0/(3.0*(1.0+Xstr[0])**2)
        dSRdlambda2 = 2.0*S_shifted[0, 4]*dSRdlambda
        dsigma1dlambda = Ai[4]*dSRdlambda + 0.5*Aij[4, 4]*dSRdlambda2 \
            + Aij[0, 4]*S_shifted[0, 0]*dSRdlambda \
            + Aij[1, 4]*S_shifted[0, 1]*dSRdlambda \
            + Aij[2, 4]*S_shifted[0, 2]*dSRdlambda \
            + Aij[3, 4]*S_shifted[0, 3]*dSRdlambda
        dSxdx = 1.0/self.pf.d['sigma[1]'][2]
        dSxdx2 = 2.0*S_shifted[0, 2]*dSxdx
        dsigma1dx = Ai[2]*dSxdx+0.5*Aij[2, 2]*dSxdx2 \
            + Aij[0, 2]*S_shifted[0, 0]*dSxdx \
            + Aij[1, 2]*S_shifted[0, 1]*dSxdx \
            + Aij[3, 2]*S_shifted[0, 3]*dSxdx \
            + Aij[4, 2]*S_shifted[0, 4]*dSxdx

        S_shifted, Ai, Aij = self.pf.eval([Z[0], L, Xstr[1], b, R],
                                          [4, 1, 4, 1, 1], [0.15]*5,
                                          "sigma[2]", deriv=True)
        dSRdlambda = 1.0/self.pf.d['sigma[2]'][4]*1.0/(3.0*(1.0+Xstr[0])**2)
        dSRdlambda2 = 2.0*S_shifted[0, 4]*dSRdlambda
        dsigma2dlambda = Ai[4]*dSRdlambda \
            + 0.5*Aij[4, 4]*dSRdlambda2 \
            + Aij[0, 4]*S_shifted[0, 0]*dSRdlambda \
            + Aij[1, 4]*S_shifted[0, 1]*dSRdlambda \
            + Aij[2, 4]*S_shifted[0, 2]*dSRdlambda \
            + Aij[3, 4]*S_shifted[0, 3]*dSRdlambda
        dSxdx = 1.0/self.pf.d['sigma[2]'][2]
        dSxdx2 = 2.0*S_shifted[0, 2]*dSxdx
        dsigma2dx = Ai[2]*dSxdx + 0.5*Aij[2, 2]*dSxdx2 \
            + Aij[0, 2]*S_shifted[0, 0]*dSxdx \
            + Aij[1, 2]*S_shifted[0, 1]*dSxdx \
            + Aij[3, 2]*S_shifted[0, 3]*dSxdx \
            + Aij[4, 2]*S_shifted[0, 4]*dSxdx

        S_shifted, Ai, Aij = self.pf.eval([Z[0], L, Xstr[1], b, R],
                                          [4, 1, 4, 1, 1], [0.2]*5,
                                          "sigma[3]", deriv=True)
        dSRdlambda = 1.0/self.pf.d['sigma[3]'][4]*1.0/(3.0*(1.0+Xstr[0])**2)
        dSRdlambda2 = 2.0*S_shifted[0, 4]*dSRdlambda
        dsigma3dlambda = Ai[4]*dSRdlambda+0.5*Aij[4, 4]*dSRdlambda2 \
            + Aij[0, 4]*S_shifted[0, 0]*dSRdlambda \
            + Aij[1, 4]*S_shifted[0, 1]*dSRdlambda \
            + Aij[2, 4]*S_shifted[0, 2]*dSRdlambda \
            + Aij[3, 4]*S_shifted[0, 3]*dSRdlambda
        dSxdx = 1.0/self.pf.d['sigma[3]'][2]
        dSxdx2 = 2.0*S_shifted[0, 2]*dSxdx
        dsigma3dx = Ai[2]*dSxdx+0.5*Aij[2, 2]*dSxdx2 \
            + Aij[0, 2]*S_shifted[0, 0]*dSxdx \
            + Aij[1, 2]*S_shifted[0, 1]*dSxdx \
            + Aij[3, 2]*S_shifted[0, 3]*dSxdx \
            + Aij[4, 2]*S_shifted[0, 4]*dSxdx

        S_shifted, Ai, Aij = self.pf.eval([Z[0], L, Xstr[1], b, R],
                                          [4, 1, 4, 1, 1], [0.25]*5,
                                          "sigma[4]", deriv=True)
        dSRdlambda = 1.0/self.pf.d['sigma[4]'][4]*1.0/(3.0*(1.0+Xstr[0])**2)
        dSRdlambda2 = 2.0*S_shifted[0, 4]*dSRdlambda
        dsigma4dlambda = Ai[4]*dSRdlambda \
            + 0.5*Aij[4, 4]*dSRdlambda2 \
            + Aij[0, 4]*S_shifted[0, 0]*dSRdlambda \
            + Aij[1, 4]*S_shifted[0, 1]*dSRdlambda \
            + Aij[2, 4]*S_shifted[0, 2]*dSRdlambda \
            + Aij[3, 4]*S_shifted[0, 3]*dSRdlambda
        dSxdx = 1.0/self.pf.d['sigma[4]'][2]
        dSxdx2 = 2.0*S_shifted[0, 2]*dSxdx
        dsigma4dx = Ai[2]*dSxdx+0.5*Aij[2, 2]*dSxdx2 \
            + Aij[0, 2]*S_shifted[0, 0]*dSxdx \
            + Aij[1, 2]*S_shifted[0, 1]*dSxdx \
            + Aij[3, 2]*S_shifted[0, 3]*dSxdx \
            + Aij[4, 2]*S_shifted[0, 4]*dSxdx
        S_shifted, Ai, Aij = self.pf.eval([Z[0], L, Xstr[1], b, R],
                                          [4, 1, 4, 1, 1], [0.3]*5,
                                          "sigma[5]", deriv=True)
        dSRdlambda = 1.0/self.pf.d['sigma[5]'][4]*1.0/(3.0*(1.0+Xstr[0])**2)
        dSRdlambda2 = 2.0*S_shifted[0, 4]*dSRdlambda
        dsigma5dlambda = Ai[4]*dSRdlambda+0.5*Aij[4, 4]*dSRdlambda2 \
            + Aij[0, 4]*S_shifted[0, 0]*dSRdlambda \
            + Aij[1, 4]*S_shifted[0, 1]*dSRdlambda \
            + Aij[2, 4]*S_shifted[0, 2]*dSRdlambda \
            + Aij[3, 4]*S_shifted[0, 3]*dSRdlambda
        dSxdx = 1.0/self.pf.d['sigma[5]'][2]
        dSxdx2 = 2.0*S_shifted[0, 2]*dSxdx
        dsigma5dx = Ai[2]*dSxdx + 0.5*Aij[2, 2]*dSxdx2 \
            + Aij[0, 2]*S_shifted[0, 0]*dSxdx \
            + Aij[1, 2]*S_shifted[0, 1]*dSxdx \
            + Aij[3, 2]*S_shifted[0, 3]*dSxdx \
            + Aij[4, 2]*S_shifted[0, 4]*dSxdx

        J['sigma', 'x_str'] = np.array(
            [[dsigma1dlambda[0, 0]/self.scalers['sigma'][0],
              dsigma1dx[0, 0]/self.scalers['sigma'][0]],
             [dsigma2dlambda[0, 0]/self.scalers['sigma'][1],
              dsigma2dx[0, 0]/self.scalers['sigma'][1]],
             [dsigma3dlambda[0, 0]/self.scalers['sigma'][2],
              dsigma3dx[0, 0]/self.scalers['sigma'][2]],
             [dsigma4dlambda[0, 0]/self.scalers['sigma'][3],
              dsigma4dx[0, 0]/self.scalers['sigma'][3]],
             [dsigma5dlambda[0, 0]/self.scalers['sigma'][4],
              dsigma5dx[0, 0]/self.scalers['sigma'][4]]])*self.scalers['x_str']

        S_shifted, Ai, Aij = self.pf.eval([Z[0], L, Xstr[1], b, R],
                                          [4, 1, 4, 1, 1], [0.1]*5,
                                          "sigma[1]", deriv=True)
        dStcdtc = 1.0/self.pf.d['sigma[1]'][0]
        dStcdtc2 = 2.0*S_shifted[0, 0]*dStcdtc
        dsigma1dtc = Ai[0]*dStcdtc+0.5*Aij[0, 0]*dStcdtc2 \
            + Aij[1, 0]*S_shifted[0, 1]*dStcdtc \
            + Aij[2, 0]*S_shifted[0, 2]*dStcdtc \
            + Aij[3, 0]*S_shifted[0, 3]*dStcdtc \
            + Aij[4, 0]*S_shifted[0, 4]*dStcdtc
        dsigma1dh = 0.0
        dsigma1dM = 0.0
        dSbdAR = 1.0/self.pf.d['sigma[1]'][3]*(np.sqrt(Z[5])/4.0*Z[3]**-0.5)
        dSbdAR2 = 2.0*S_shifted[0, 3]*dSbdAR
        dsigma1dAR = Ai[3]*dSbdAR+0.5*Aij[3, 3]*dSbdAR2 \
            + Aij[0, 3]*S_shifted[0, 0]*dSbdAR \
            + Aij[1, 3]*S_shifted[0, 1]*dSbdAR \
            + Aij[2, 3]*S_shifted[0, 2]*dSbdAR \
            + Aij[4, 3]*S_shifted[0, 4]*dSbdAR
        dsigma1dLambda = 0.0
        dSbdSref = 1.0/self.pf.d['sigma[1]'][3]*(np.sqrt(Z[3])/4.0*Z[5]**-0.5)
        dSbdSref2 = 2.0*S_shifted[0, 3]*dSbdSref
        dsigma1dSref = Ai[3]*dSbdSref + 0.5*Aij[3, 3]*dSbdSref2 \
            + Aij[0, 3]*S_shifted[0, 0]*dSbdSref \
            + Aij[1, 3]*S_shifted[0, 1]*dSbdSref \
            + Aij[2, 3]*S_shifted[0, 2]*dSbdSref \
            + Aij[4, 3]*S_shifted[0, 4]*dSbdSref
        S_shifted, Ai, Aij = self.pf.eval([Z[0], L, Xstr[1], b, R],
                                          [4, 1, 4, 1, 1], [0.15]*5,
                                          "sigma[2]", deriv=True)
        dStcdtc = 1.0/self.pf.d['sigma[2]'][0]
        dStcdtc2 = 2.0*S_shifted[0, 0]*dStcdtc
        dsigma2dtc = Ai[0]*dStcdtc+0.5*Aij[0, 0]*dStcdtc2 \
            + Aij[1, 0]*S_shifted[0, 1]*dStcdtc \
            + Aij[2, 0]*S_shifted[0, 2]*dStcdtc \
            + Aij[3, 0]*S_shifted[0, 3]*dStcdtc \
            + Aij[4, 0]*S_shifted[0, 4]*dStcdtc
        dsigma2dh = 0.0
        dsigma2dM = 0.0
        dSbdAR = 1.0/self.pf.d['sigma[2]'][3]*(np.sqrt(Z[5])/4.0*Z[3]**-0.5)
        dSbdAR2 = 2.0*S_shifted[0, 3]*dSbdAR
        dsigma2dAR = Ai[3]*dSbdAR+0.5*Aij[3, 3]*dSbdAR2 \
            + Aij[0, 3]*S_shifted[0, 0]*dSbdAR \
            + Aij[1, 3]*S_shifted[0, 1]*dSbdAR \
            + Aij[2, 3]*S_shifted[0, 2]*dSbdAR \
            + Aij[4, 3]*S_shifted[0, 4]*dSbdAR
        dsigma2dLambda = 0.0
        dSbdSref = 1.0/self.pf.d['sigma[2]'][3]*(np.sqrt(Z[3])/4.0*Z[5]**-0.5)
        dSbdSref2 = 2.0*S_shifted[0, 3]*dSbdSref
        dsigma2dSref = Ai[3]*dSbdSref + 0.5*Aij[3, 3]*dSbdSref2 \
            + Aij[0, 3]*S_shifted[0, 0]*dSbdSref \
            + Aij[1, 3]*S_shifted[0, 1]*dSbdSref \
            + Aij[2, 3]*S_shifted[0, 2]*dSbdSref \
            + Aij[4, 3]*S_shifted[0, 4]*dSbdSref

        S_shifted, Ai, Aij = self.pf.eval([Z[0], L, Xstr[1], b, R],
                                          [4, 1, 4, 1, 1], [0.20]*5,
                                          "sigma[3]", deriv=True)
        dStcdtc = 1.0/self.pf.d['sigma[3]'][0]
        dStcdtc2 = 2.0*S_shifted[0, 0]*dStcdtc
        dsigma3dtc = Ai[0]*dStcdtc+0.5*Aij[0, 0]*dStcdtc2 \
            + Aij[1, 0]*S_shifted[0, 1]*dStcdtc \
            + Aij[2, 0]*S_shifted[0, 2]*dStcdtc \
            + Aij[3, 0]*S_shifted[0, 3]*dStcdtc \
            + Aij[4, 0]*S_shifted[0, 4]*dStcdtc
        dsigma3dh = 0.0
        dsigma3dM = 0.0
        dSbdAR = 1.0/self.pf.d['sigma[3]'][3]*(np.sqrt(Z[5])/4.0*Z[3]**-0.5)
        dSbdAR2 = 2.0*S_shifted[0, 3]*dSbdAR
        dsigma3dAR = Ai[3]*dSbdAR+0.5*Aij[3, 3]*dSbdAR2 \
            + Aij[0, 3]*S_shifted[0, 0]*dSbdAR \
            + Aij[1, 3]*S_shifted[0, 1]*dSbdAR \
            + Aij[2, 3]*S_shifted[0, 2]*dSbdAR \
            + Aij[4, 3]*S_shifted[0, 4]*dSbdAR
        dsigma3dLambda = 0.0
        dSbdSref = 1.0/self.pf.d['sigma[3]'][3]*(np.sqrt(Z[3])/4.0*Z[5]**-0.5)
        dSbdSref2 = 2.0*S_shifted[0, 3]*dSbdSref
        dsigma3dSref = Ai[3]*dSbdSref+0.5*Aij[3, 3]*dSbdSref2 \
            + Aij[0, 3]*S_shifted[0, 0]*dSbdSref \
            + Aij[1, 3]*S_shifted[0, 1]*dSbdSref \
            + Aij[2, 3]*S_shifted[0, 2]*dSbdSref \
            + Aij[4, 3]*S_shifted[0, 4]*dSbdSref

        S_shifted, Ai, Aij = self.pf.eval([Z[0], L, Xstr[1], b, R],
                                          [4, 1, 4, 1, 1], [0.25]*5,
                                          "sigma[4]", deriv=True)
        dStcdtc = 1.0/self.pf.d['sigma[4]'][0]
        dStcdtc2 = 2.0*S_shifted[0, 0]*dStcdtc
        dsigma4dtc = Ai[0]*dStcdtc+0.5*Aij[0, 0]*dStcdtc2 \
            + Aij[1, 0]*S_shifted[0, 1]*dStcdtc \
            + Aij[2, 0]*S_shifted[0, 2]*dStcdtc \
            + Aij[3, 0]*S_shifted[0, 3]*dStcdtc \
            + Aij[4, 0]*S_shifted[0, 4]*dStcdtc
        dsigma4dh = 0.0
        dsigma4dM = 0.0
        dSbdAR = 1.0/self.pf.d['sigma[4]'][3]*(np.sqrt(Z[5])/4.0*Z[3]**-0.5)
        dSbdAR2 = 2.0*S_shifted[0, 3]*dSbdAR
        dsigma4dAR = Ai[3]*dSbdAR + 0.5*Aij[3, 3]*dSbdAR2 \
            + Aij[0, 3]*S_shifted[0, 0]*dSbdAR \
            + Aij[1, 3]*S_shifted[0, 1]*dSbdAR \
            + Aij[2, 3]*S_shifted[0, 2]*dSbdAR \
            + Aij[4, 3]*S_shifted[0, 4]*dSbdAR
        dsigma4dLambda = 0.0
        dSbdSref = 1.0/self.pf.d['sigma[4]'][3]*(np.sqrt(Z[3])/4.0*Z[5]**-0.5)
        dSbdSref2 = 2.0*S_shifted[0, 3]*dSbdSref
        dsigma4dSref = Ai[3]*dSbdSref + 0.5*Aij[3, 3]*dSbdSref2 \
            + Aij[0, 3]*S_shifted[0, 0]*dSbdSref \
            + Aij[1, 3]*S_shifted[0, 1]*dSbdSref \
            + Aij[2, 3]*S_shifted[0, 2]*dSbdSref \
            + Aij[4, 3]*S_shifted[0, 4]*dSbdSref

        S_shifted, Ai, Aij = self.pf.eval([Z[0], L, Xstr[1], b, R],
                                          [4, 1, 4, 1, 1], [0.3]*5,
                                          "sigma[5]", deriv=True)
        dStcdtc = 1.0/self.pf.d['sigma[5]'][0]
        dStcdtc2 = 2.0*S_shifted[0, 0]*dStcdtc
        dsigma5dtc = Ai[0]*dStcdtc+0.5*Aij[0, 0]*dStcdtc2 \
            + Aij[1, 0]*S_shifted[0, 1]*dStcdtc \
            + Aij[2, 0]*S_shifted[0, 2]*dStcdtc \
            + Aij[3, 0]*S_shifted[0, 3]*dStcdtc \
            + Aij[4, 0]*S_shifted[0, 4]*dStcdtc
        dsigma5dh = 0.0
        dsigma5dM = 0.0
        dSbdAR = 1.0/self.pf.d['sigma[5]'][3]*(np.sqrt(Z[5])/4.0*Z[3]**-0.5)
        dSbdAR2 = 2.0*S_shifted[0, 3]*dSbdAR
        dsigma5dAR = Ai[3]*dSbdAR + 0.5*Aij[3, 3]*dSbdAR2 \
            + Aij[0, 3]*S_shifted[0, 0]*dSbdAR \
            + Aij[1, 3]*S_shifted[0, 1]*dSbdAR \
            + Aij[2, 3]*S_shifted[0, 2]*dSbdAR \
            + Aij[4, 3]*S_shifted[0, 4]*dSbdAR
        dsigma5dLambda = 0.0
        dSbdSref = 1.0/self.pf.d['sigma[5]'][3]*(np.sqrt(Z[3])/4.0*Z[5]**-0.5)
        dSbdSref2 = 2.0*S_shifted[0, 3]*dSbdSref
        dsigma5dSref = Ai[3]*dSbdSref + 0.5*Aij[3, 3]*dSbdSref2 \
            + Aij[0, 3]*S_shifted[0, 0]*dSbdSref \
            + Aij[1, 3]*S_shifted[0, 1]*dSbdSref \
            + Aij[2, 3]*S_shifted[0, 2]*dSbdSref \
            + Aij[4, 3]*S_shifted[0, 4]*dSbdSref

        J['sigma', 'z'] = np.array(
            [[dsigma1dtc[0, 0]/self.scalers['sigma'][0],
              dsigma1dh/self.scalers['sigma'][0],
              dsigma1dM/self.scalers['sigma'][0],
              dsigma1dAR[0, 0]/self.scalers['sigma'][0],
              dsigma1dLambda/self.scalers['sigma'][0],
              dsigma1dSref[0, 0]/self.scalers['sigma'][0]],
             [dsigma2dtc[0, 0]/self.scalers['sigma'][1],
              dsigma2dh/self.scalers['sigma'][1],
              dsigma2dM/self.scalers['sigma'][1],
              dsigma2dAR[0, 0]/self.scalers['sigma'][1],
              dsigma2dLambda/self.scalers['sigma'][1],
              dsigma2dSref[0, 0]/self.scalers['sigma'][1]],
             [dsigma3dtc[0, 0]/self.scalers['sigma'][2],
              dsigma3dh/self.scalers['sigma'][2],
              dsigma3dM/self.scalers['sigma'][2],
              dsigma3dAR[0, 0]/self.scalers['sigma'][2],
              dsigma3dLambda/self.scalers['sigma'][2],
              dsigma3dSref[0, 0]/self.scalers['sigma'][2]],
             [dsigma4dtc[0, 0]/self.scalers['sigma'][3],
              dsigma4dh/self.scalers['sigma'][3],
              dsigma4dM/self.scalers['sigma'][3],
              dsigma4dAR[0, 0]/self.scalers['sigma'][3],
              dsigma4dLambda/self.scalers['sigma'][3],
              dsigma4dSref[0, 0]/self.scalers['sigma'][3]],
             [dsigma5dtc[0, 0]/self.scalers['sigma'][4],
              dsigma5dh/self.scalers['sigma'][4],
              dsigma5dM/self.scalers['sigma'][4],
              dsigma5dAR[0, 0]/self.scalers['sigma'][4],
              dsigma5dLambda/self.scalers['sigma'][4],
              dsigma5dSref[0, 0]/self.scalers['sigma'][4]]])*self.scalers['z']
        S_shifted, Ai, Aij = self.pf.eval([Z[0], L, Xstr[1], b, R],
                                          [4, 1, 4, 1, 1], [0.1]*5,
                                          "sigma[1]", deriv=True)
        dSLdL = 1.0/self.pf.d['sigma[1]'][1]
        dSLdL2 = 2.0*S_shifted[0, 1]*dSLdL
        dsigma1dL = Ai[1]*dSLdL + 0.5*Aij[1, 1]*dSLdL2 \
            + Aij[0, 1]*S_shifted[0, 0]*dSLdL \
            + Aij[2, 1]*S_shifted[0, 2]*dSLdL \
            + Aij[3, 1]*S_shifted[0, 3]*dSLdL \
            + Aij[4, 1]*S_shifted[0, 4]*dSLdL

        S_shifted, Ai, Aij = self.pf.eval([Z[0], L, Xstr[1], b, R],
                                          [4, 1, 4, 1, 1], [0.15]*5,
                                          "sigma[2]", deriv=True)
        dSLdL = 1.0/self.pf.d['sigma[2]'][1]
        dSLdL2 = 2.0*S_shifted[0, 1]*dSLdL
        dsigma2dL = Ai[1]*dSLdL+0.5*Aij[1, 1]*dSLdL2 \
            + Aij[0, 1]*S_shifted[0, 0]*dSLdL \
            + Aij[2, 1]*S_shifted[0, 2]*dSLdL \
            + Aij[3, 1]*S_shifted[0, 3]*dSLdL \
            + Aij[4, 1]*S_shifted[0, 4]*dSLdL

        S_shifted, Ai, Aij = self.pf.eval([Z[0], L, Xstr[1], b, R],
                                          [4, 1, 4, 1, 1], [0.2]*5,
                                          "sigma[3]", deriv=True)
        dSLdL = 1.0/self.pf.d['sigma[3]'][1]
        dSLdL2 = 2.0*S_shifted[0, 1]*dSLdL
        dsigma3dL = Ai[1]*dSLdL + 0.5*Aij[1, 1]*dSLdL2 \
            + Aij[0, 1]*S_shifted[0, 0]*dSLdL \
            + Aij[2, 1]*S_shifted[0, 2]*dSLdL \
            + Aij[3, 1]*S_shifted[0, 3]*dSLdL \
            + Aij[4, 1]*S_shifted[0, 4]*dSLdL

        S_shifted, Ai, Aij = self.pf.eval([Z[0], L, Xstr[1], b, R],
                                          [4, 1, 4, 1, 1], [0.25]*5,
                                          "sigma[4]", deriv=True)
        dSLdL = 1.0/self.pf.d['sigma[4]'][1]
        dSLdL2 = 2.0*S_shifted[0, 1]*dSLdL
        dsigma4dL = Ai[1]*dSLdL + 0.5*Aij[1, 1]*dSLdL2 \
            + Aij[0, 1]*S_shifted[0, 0]*dSLdL \
            + Aij[2, 1]*S_shifted[0, 2]*dSLdL \
            + Aij[3, 1]*S_shifted[0, 3]*dSLdL \
            + Aij[4, 1]*S_shifted[0, 4]*dSLdL

        S_shifted, Ai, Aij = self.pf.eval([Z[0], L, Xstr[1], b, R],
                                          [4, 1, 4, 1, 1], [0.3]*5,
                                          "sigma[5]", deriv=True)
        dSLdL = 1.0/self.pf.d['sigma[5]'][1]
        dSLdL2 = 2.0*S_shifted[0, 1]*dSLdL
        dsigma5dL = Ai[1]*dSLdL + 0.5*Aij[1, 1]*dSLdL2 \
            + Aij[0, 1]*S_shifted[0, 0]*dSLdL \
            + Aij[2, 1]*S_shifted[0, 2]*dSLdL \
            + Aij[3, 1]*S_shifted[0, 3]*dSLdL \
            + Aij[4, 1]*S_shifted[0, 4]*dSLdL

        J['sigma','L'] = np.array(
            [[dsigma1dL/self.scalers['sigma'][0]*self.scalers['L']],
             [dsigma2dL/self.scalers['sigma'][1]*self.scalers['L']],
             [dsigma3dL/self.scalers['sigma'][2]*self.scalers['L']],
             [dsigma4dL/self.scalers['sigma'][3]*self.scalers['L']],
             [dsigma5dL/self.scalers['sigma'][4]*self.scalers['L']]]).reshape((5, 1))

        J['sigma','WE'] = np.zeros((5, 1))

        return J


if __name__ == "__main__": # pragma: no cover
    from openmdao.api import Component, Problem, Group, IndepVarComp
    scalers = {}
    scalers['z'] = np.array([0.05, 45000., 1.6, 5.5, 55.0, 1000.0])
    scalers['x_str'] = np.array([0.25, 1.0])
    scalers['L'] = 49909.58578
    scalers['Theta'] = 0.950978
    scalers['WF'] = 7306.20261
    scalers['WT'] = 49909.58578
    scalers['WE'] = 5748.915355
    scalers['sigma'] = np.array([1.12255, 1.08170213, 1.0612766,
                                 1.04902128, 1.04085106])
    top=Problem()
    top.root=Group()
    top.root.add('z_in', IndepVarComp('z',
                                      np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])),
                 promotes=['*'])
    top.root.add('x_str_in', IndepVarComp('x_str', np.array([1.0, 1.0])),
                 promotes=['*'])
    top.root.add('L_in', IndepVarComp('L', 1.0), promotes=['*'])
    top.root.add('WE_in', IndepVarComp('WE', 1.0), promotes=['*'])
    top.root.add('Str1', Structure(scalers), promotes=['*'])
    top.setup()
    pf=PolynomialFunction()
    top.run()
    J1=top.root.Str1.jacobian(top.root.Str1.params,
                              top.root.Str1.unknowns,
                              top.root.Str1.resids)
    J2=top.root.Str1.fd_jacobian(top.root.Str1.params,
                                 top.root.Str1.unknowns,
                                 top.root.Str1.resids)
    errAbs=[]
    for i in range(len(J2.keys())):
        errAbs.append(J1[J2.keys()[i]]-J2[J2.keys()[i]])
        print ('ErrAbs_'+str(J2.keys()[i])+'=',J1[J2.keys()[i]]-J2[J2.keys()[i]])
        print (J1[J2.keys()[i]].shape==J2[J2.keys()[i]].shape)
