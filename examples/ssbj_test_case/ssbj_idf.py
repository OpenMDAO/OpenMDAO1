"""
SSBJ test case - http://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19980234657.pdf
Python implementation and OpenMDAO integration developed by
Sylvain Dubreuil and Remi Lafage of ONERA, the French Aerospace Lab.
"""
from sys import argv
import re
import numpy as np
import matplotlib.pylab as plt
import sqlitedict

from openmdao.api import Problem, ScipyOptimizer, SqliteRecorder #, pyOptSparseDriver
from ssbj_idf_mda import SSBJ_IDF_MDA
from ssbj_mda import init_ssbj_mda
# pylint: disable=C0103

# Optimization problem
scalers, pf = init_ssbj_mda()

prob = Problem()
prob.root = SSBJ_IDF_MDA(scalers, pf)

# Optimizer options
prob.driver = ScipyOptimizer()
#prob.driver = pyOptSparseDriver()
optimizer = 'SLSQP'
prob.driver.options['optimizer'] = optimizer

#Design variables
prob.driver.add_desvar('z', lower=np.array([0.2, 0.666,0.875,0.45,0.72,0.5]),
                     upper=np.array([1.8,1.333,1.125,1.45,1.27,1.5]))
prob.driver.add_desvar('x_str', lower=np.array([0.4,0.75]), upper=np.array([1.6,1.25]))
prob.driver.add_desvar('x_aer', lower=0.75, upper=1.25)
prob.driver.add_desvar('x_pro', lower=0.18, upper=1.81)

prob.driver.add_desvar('Theta')
prob.driver.add_desvar('L')
prob.driver.add_desvar('WE')
prob.driver.add_desvar('WT')
prob.driver.add_desvar('ESF')
prob.driver.add_desvar('D')

# Objective function
prob.driver.add_objective('obj')

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
prob.driver.add_constraint('con_esf', upper=0.0)
prob.driver.add_constraint('con_temp', upper=0.0)

#Coupling constraints
#Threshold for the coupling (constraints define as (x_in-x_out)**2<epsilon)
epsilon=1e-6
prob.driver.add_constraint('con_str_aer_wt',upper=epsilon)
prob.driver.add_constraint('con_str_aer_theta',upper=epsilon)
prob.driver.add_constraint('con_aer_str_l',upper=epsilon)
prob.driver.add_constraint('con_aer_pro_d',upper=epsilon)
prob.driver.add_constraint('con_pro_aer_esf',upper=epsilon)
prob.driver.add_constraint('con_pro_str_we',upper=epsilon)
#Recorder
if "--plot" in argv:
    recorder = SqliteRecorder('IDF.sqlite')
    recorder.options['record_params'] = True
    recorder.options['record_metadata'] = True
    prob.driver.add_recorder(recorder)

#Run optimization
prob.setup()
prob.run()
prob.cleanup()

print 'Z_opt=', prob['z']*scalers['z']
print 'X_str_opt=', prob['x_str']*scalers['x_str']
print 'X_aer_opt=', prob['x_aer']
print 'X_pro_opt=', prob['x_pro']*scalers['x_pro']
print 'R_opt=', -prob['obj']*scalers['R']

if "--plot" in argv:
    db = sqlitedict.SqliteDict( 'IDF.sqlite', 'openmdao')
    plt.figure()

    pattern = re.compile('rank0:'+optimizer+'/\\d+$')
    r = []
    for k, v in db.iteritems():
        if re.match(pattern, k):
            r.append(v['Unknowns']['Perfo.R']*scalers['R'])

    plt.plot(r)
    plt.show()



