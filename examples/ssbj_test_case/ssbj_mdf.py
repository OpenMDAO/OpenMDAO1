# SSBJ test case - MDF formulation
# Initial Author : Sylvain Dubreuil

from sys import argv
import numpy as np

from openmdao.api import Problem
from openmdao.api import SqliteRecorder
from openmdao.api import ScipyOptimizer, pyOptSparseDriver

from ssbj_mda import init_ssbj_mda, SSBJ_MDA

#Construction of optimization problem
scaled = True
scalers, pf = init_ssbj_mda(scaled=scaled)

P = Problem()
P.root = SSBJ_MDA(scalers, pf, scaled=scaled)

#Optimizer options
P.driver = ScipyOptimizer()
#P.driver = pyOptSparseDriver()
optimizer ='SLSQP'
P.driver.options['optimizer'] = optimizer

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

if "--plot" in argv:
    import matplotlib.pylab as plt
    import sqlitedict
    import re

    db = sqlitedict.SqliteDict( 'MDF.sqlite', 'openmdao')
    plt.figure()

    pattern = re.compile('rank0:'+optimizer+'/\d+$')
    r = []
    for k, v in db.iteritems():
        if re.match(pattern, k):
            r.append(v['Unknowns']['R']*scalers['R'])
    plt.plot(r)
    plt.show()

