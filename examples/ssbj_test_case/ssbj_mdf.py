"""
SSBJ test case implementation - MDF formulation
see http://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19980234657.pdf
Initial Author : Sylvain Dubreuil
"""
from sys import argv
import re
import numpy as np
import matplotlib.pylab as plt
import sqlitedict

from openmdao.api import Problem
from openmdao.api import SqliteRecorder
from openmdao.api import ScipyOptimizer #, pyOptSparseDriver

from ssbj_mda import init_ssbj_mda, SSBJ_MDA
# pylint: disable=C0103

# Optimization problem
scalers, pfunc = init_ssbj_mda()
prob = Problem()
prob.root = SSBJ_MDA(scalers, pfunc)

#Optimizer options
prob.driver = ScipyOptimizer()
#P.driver = pyOptSparseDriver()
optimizer ='SLSQP'
prob.driver.options['optimizer'] = optimizer

#Design variables
prob.driver.add_desvar('z', lower=np.array([0.2, 0.666, 0.875,
                                         0.45, 0.72, 0.5]),
                         upper=np.array([1.8, 1.333, 1.125, 1.45, 1.27, 1.5]))
prob.driver.add_desvar('x_str', lower=np.array([0.4, 0.75]),
                    upper=np.array([1.6, 1.25]))
prob.driver.add_desvar('x_aer', lower=0.75, upper=1.25)
prob.driver.add_desvar('x_pro', lower=0.18, upper=1.81)

#Objective function
prob.driver.add_objective('R', scaler=-1.)

#Constraints
prob.driver.add_constraint('con_dt', upper=0.0)
prob.driver.add_constraint('con_Theta_up', upper=0.0)
prob.driver.add_constraint('con_Theta_low', upper=0.0)
prob.driver.add_constraint('con_sigma1', upper=0.0)
prob.driver.add_constraint('con_sigma2', upper=0.0)
prob.driver.add_constraint('con_sigma3', upper=0.0)
prob.driver.add_constraint('con_sigma4', upper=0.0)
prob.driver.add_constraint('con_sigma5', upper=0.0)
prob.driver.add_constraint('con_dpdx', upper=0.0)
prob.driver.add_constraint('con1_esf', upper=0.0)
prob.driver.add_constraint('con2_esf', upper=0.0)
prob.driver.add_constraint('con_temp', upper=0.0)

#Recorder
db_name = 'MDF.sqlite'
if "--plot" in argv:
    recorder = SqliteRecorder(db_name)
    recorder.options['record_params'] = True
    recorder.options['record_metadata'] = True
    prob.driver.add_recorder(recorder)

#Run optimization
prob.setup()
prob.run()
prob.cleanup()

print 'Z_opt=',prob.root.unknowns['z']*scalers['z']
print 'X_str_opt=', prob['x_str']*scalers['x_str']
print 'X_aer_opt=', prob['x_aer']
print 'X_pro_opt=', prob['x_pro']*scalers['x_pro']
print 'R_opt=', prob['R']*scalers['R']

if "--plot" in argv:
    db = sqlitedict.SqliteDict(db_name, 'openmdao')
    plt.figure()

    pattern = re.compile('rank0:'+optimizer+r'/\d+$')
    r = []
    for k, v in db.iteritems():
        if re.match(pattern, k):
            r.append(v['Unknowns']['R']*scalers['R'])
    plt.plot(r)
    plt.show()

