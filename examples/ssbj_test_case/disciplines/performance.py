from __future__ import print_function
import numpy as np
from openmdao.api import Component
from disciplines.common import PolynomialFunction

class Performance(Component):

    def __init__(self, scalers, fd=False):
        super(Performance, self).__init__()
        # Global Design Variable z=(t/c,h,M,AR,Lambda,Sref)
        self.add_param('z', val=np.zeros(6))
        # Local Design Variable x_per=null
        # Coupling parameters
        self.add_param('WT', val=1.0)
        self.add_param('WF', val=1.0)
        self.add_param('fin', val=1.0)
        self.add_param('SFC', val=1.0)
        # Coupling output
        self.add_output('R', val=1.0)
        self.add_output('Rm', val=1.0)
        # scalers values
        self.scalers = scalers
        # Finite differences
        if fd:
            self.fd_options['force_fd'] = True
            self.fd_options['force_fd'] = True
            self.fd_options['form'] = 'central'
            self.fd_options['step_type'] = 'relative'
            self.fd_options['step_size'] = 1e-8

    def solve_nonlinear(self, params, unknowns, resids):
        #Variables scaling
        Z = params['z']*self.scalers['z']
        fin = params['fin']*self.scalers['fin']
        SFC = params['SFC']*self.scalers['SFC']
        WT = params['WT']*self.scalers['WT']
        WF = params['WF']*self.scalers['WF']
        if Z[1] <= 36089.:
            theta = 1.0-6.875E-6*Z[1]
        else:
            theta = 0.7519
        R = 661.0*np.sqrt(theta)*Z[2]*fin/SFC*np.log(abs(WT/(WT-WF)))
        unknowns['R'] = R/self.scalers['R']
        unknowns['Rm'] = -R/self.scalers['R']

    def linearize(self, params, unknowns, resids):
        J = {}
        #Changement de variable
        Z = params['z']*self.scalers['z']
        fin = params['fin']*self.scalers['fin']
        SFC = params['SFC']*self.scalers['SFC']
        WT = params['WT']*self.scalers['WT']
        WF = params['WF']*self.scalers['WF']
        if Z[1] <= 36089:
            theta = 1.0-6.875E-6*Z[1]
        else:
            theta = 0.7519
        ########R
        if Z[1] <= 36089.:
            dRdh = -0.5*661.0*theta**-0.5*6.875e-6*Z[2]*fin \
                   /SFC*np.log(abs(WT/(WT-WF)))
        else:
            dRdh = 0.0
        dRdM = 661.0*np.sqrt(theta)*fin/SFC*np.log(abs(WT/(WT-WF)))
        J['R', 'z'] = np.zeros((1, 6))
        J['R', 'z'][0, 1] = np.array([dRdh/self.scalers['R'] *45000.0])
        J['R', 'z'][0, 2] = np.array([dRdM/self.scalers['R'] *1.6])
        dRdfin = 661.0*np.sqrt(theta)*Z[2]/SFC*np.log(abs(WT/(WT-WF)))
        J['R', 'fin'] = np.array([[dRdfin/self.scalers['R']*self.scalers['fin']]])
        dRdSFC = -661.0*np.sqrt(theta)*Z[2]*fin/SFC**2*np.log(abs(WT/(WT-WF)))
        J['R', 'SFC'] = np.array([[dRdSFC/self.scalers['R']*self.scalers['SFC']]])
        dRdWT = 661.0*np.sqrt(theta)*Z[2]*fin/SFC*-WF/(WT*(WT-WF))
        J['R', 'WT'] = np.array([[dRdWT/self.scalers['R']*self.scalers['WT']]])
        dRdWF = 661.0*np.sqrt(theta)*Z[2]*fin/SFC*1.0/(WT-WF)
        J['R', 'WF'] = np.array([[dRdWF/self.scalers['R']*self.scalers['WF']]])
        ########Rm
        J['Rm', 'z'] = -J['R', 'z']
        J['Rm', 'fin'] = -J['R', 'fin']
        J['Rm', 'SFC'] = -J['R', 'SFC']
        J['Rm', 'WT'] = -J['R', 'WT']
        J['Rm', 'WF'] = -J['R', 'WF']
        return J

if __name__ == "__main__": # pragma: no cover

    from openmdao.api import Problem, Group, IndepVarComp
    scalers = {}
    scalers['z'] = np.array([0.05, 45000., 1.6, 5.5, 55.0, 1000.0])
    scalers['fin'] = 4.093062
    scalers['SFC'] = 1.12328
    scalers['WF'] = 7306.20261
    scalers['WT'] = 49909.58578
    scalers['R'] = 528.91363
    top = Problem()
    top.root = Group()
    top.root.add('z_in', IndepVarComp('z', np.array([1.0, 0.6, 1.0,
                                                     1.0, 1.0, 1.0])),
                 promotes=['*'])
    top.root.add('WT_in', IndepVarComp('WT', 0.8), promotes=['*'])
    top.root.add('WF_in', IndepVarComp('WF', 1.0), promotes=['*'])
    top.root.add('fin_in', IndepVarComp('fin', 1.0), promotes=['*'])
    top.root.add('SFC_in', IndepVarComp('SFC', 1.0), promotes=['*'])
    top.root.add('Per1', Performance(scalers), promotes=['*'])
    top.setup()
    pf = PolynomialFunction()
    top.run()
    J1 = top.root.Per1.linearize(top.root.Per1.params,
                                top.root.Per1.unknowns,
                                top.root.Per1.resids)
    J2 = top.root.Per1.fd_linearize(top.root.Per1.params,
                                   top.root.Per1.unknowns,
                                   top.root.Per1.resids)
    errAbs = []
    for i in range(len(J2.keys())):
        ErrAbs.append(J[J2.keys()[i]] - J2[J2.keys()[i]])
        print ('ErrAbs_'+str(J2.keys()[i])+'=',
               J[J2.keys()[i]]-J2[J2.keys()[i]])
        print (J[J2.keys()[i]].shape == J2[J2.keys()[i]].shape)
