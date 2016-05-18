# SSBJ test case - MDF formulation
# Initial Author : Sylvain Dubreuil

from sys import argv
import numpy as np

from openmdao.api import Problem
from openmdao.api import SqliteRecorder
from openmdao.api import ScipyOptimizer

from ssbj_mda import init_ssbj_mda, SSBJ_MDA

#Construction of optimization problem
scaled = True
scalers, pf = init_ssbj_mda(scaled=scaled)

P = Problem()
P.root = SSBJ_MDA(scalers, pf, scaled=scaled)

#Optimizer options
P.driver = ScipyOptimizer()
optimizer ='SLSQP'
P.driver.options['optimizer'] = optimizer
P.driver.options['tol'] = 1.0e-10
P.driver.options['maxiter'] = 70

#Design variables
if scaled:
    P.driver.add_desvar('z', lower=np.array([0.2, 0.666, 0.875,
                                             0.45, 0.72, 0.5]),
                         upper=np.array([1.8, 1.333, 1.125, 1.45, 1.27, 1.5]))
    P.driver.add_desvar('x_str', lower=np.array([0.4, 0.75]),
                        upper=np.array([1.6, 1.25]))
    P.driver.add_desvar('x_aer', lower=0.75, upper=1.25)
    P.driver.add_desvar('x_pro', lower=0.18, upper=1.81)
else:
    P.driver.add_desvar('z', lower=np.array([0.01,30000., 1.4, 2.5, 40., 500.0]),
                         upper=np.array([0.09, 60000., 1.8, 8.5, 70., 1500.0]))
    P.driver.add_desvar('x_str', lower=np.array([0.1, 0.75]),
                                 upper=np.array([0.4,1.25]))
    P.driver.add_desvar('x_aer', lower=0.75, upper=1.25)
    P.driver.add_desvar('x_pro', lower=0.1, upper=1.0)

#Objective function
#P.driver.add_objective('Rm')
P.driver.add_objective('R', scaler=-1.)
#Constraints
P.driver.add_constraint('con_dt', upper=0.0)
P.driver.add_constraint('con_Theta_up', upper=0.0)
P.driver.add_constraint('con_Theta_low', upper=0.0)
P.driver.add_constraint('con_sigma1', upper=0.0)
P.driver.add_constraint('con_sigma2', upper=0.0)
P.driver.add_constraint('con_sigma3', upper=0.0)
P.driver.add_constraint('con_sigma4', upper=0.0)
P.driver.add_constraint('con_sigma5', upper=0.0)
P.driver.add_constraint('con_dpdx', upper=0.0)
P.driver.add_constraint('con1_esf', upper=0.0)
P.driver.add_constraint('con2_esf', upper=0.0)
P.driver.add_constraint('con_temp', upper=0.0)

#Recorder
if "--plot" in argv:
    recorder = SqliteRecorder('MDF.sqlite')
    recorder.options['record_params'] = True
    recorder.options['record_metadata'] = True
    P.driver.add_recorder(recorder)

#Run optimization
P.setup()

P.run()
P.cleanup()

print 'Z_opt=',P.root.unknowns['z']*scalers['z']
print 'X_str_opt=', P['x_str']*scalers['x_str']
print 'X_aer_opt=', P['x_aer']
print 'X_pro_opt=', P['x_pro']*scalers['x_pro']
print 'R_opt=', P['R']*scalers['R']
print scalers

if "--plot" in argv:
    import matplotlib.pylab as plt
    import sqlitedict
    import re

    db = sqlitedict.SqliteDict( 'MDF.sqlite', 'openmdao')
    ###Plot some results
    plt.figure()

    pattern = re.compile(optimizer+'/\d+$')
    i = 0
    for k, v in db.iteritems():
        if re.match(pattern, k):
            plt.plot(i, db[k]['Unknowns']['R'],'r+')
            i += 1

    resz = []
    resXaer = []
    resXpro = []
    resXstr = []

    for k, v in db.iteritems():
        if re.match(pattern, k):
            resz.append(db[k]['Parameters']['sap.Aero.z'])
            resXaer.append(db[k]['Parameters']['sap.Aero.x_aer'])
            resXpro.append(db[k]['Parameters']['sap.Propu.x_pro'])
            resXstr.append(db[k]['Parameters']['sap.Struc.x_str'])

    resz = np.array(resz)
    resXaer = np.array(resXaer)
    resXpro = np.array(resXpro)
    resXstr = np.array(resXstr)

    plt.figure()
    for i in range(6):
        plt.plot(resz[:, i])

    plt.figure()
    plt.plot(resXaer[:])
    plt.plot(resXpro[:])
    plt.plot(resXstr[:, 0])
    plt.plot(resXstr[:, 1])

    plt.show()

