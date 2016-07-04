# SSBJ test case - IDF formulation
# Initial Author : Sylvain Dubreuil

from sys import argv
import numpy as np

from openmdao.api import Problem, ScipyOptimizer, SqliteRecorder, pyOptSparseDriver

from ssbj_idf_mda import SSBJ_IDF_MDA
from ssbj_mda import init_ssbj_mda


#Construction of optimization problem
scaled = True
scalers, pf = init_ssbj_mda(scaled=scaled)

P=Problem()
P.root=SSBJ_IDF_MDA(scalers, pf, scaled=scaled,fd=False)

#Optimizer options
P.driver = ScipyOptimizer()
#P.driver = pyOptSparseDriver()
optimizer = 'SLSQP'
P.driver.options['optimizer'] = optimizer
#Design variables
if scaled:
    P.driver.add_desvar('z', lower=np.array([0.2, 0.666,0.875,0.45,0.72,0.5]),
                         upper=np.array([1.8,1.333,1.125,1.45,1.27,1.5]))
    P.driver.add_desvar('x_str', lower=np.array([0.4,0.75]), upper=np.array([1.6,1.25]))
    P.driver.add_desvar('x_aer', lower=0.75, upper=1.25)
    P.driver.add_desvar('x_pro', lower=0.18, upper=1.81)
else:
    P.driver.add_desvar('z', lower=np.array([0.01,30000.,1.4,2.5,40.,500.0]),
                         upper=np.array([0.09,60000.,1.8,8.5,70.,1500.0]))
    P.driver.add_desvar('x_str', lower=np.array([0.1,0.75]), upper=np.array([0.4,1.25]))
    P.driver.add_desvar('x_aer', lower=0.75, upper=1.25)
    P.driver.add_desvar('x_pro', lower=0.1, upper=1.0)

P.driver.add_desvar('Theta')
P.driver.add_desvar('L')
P.driver.add_desvar('WE')
P.driver.add_desvar('WT')
P.driver.add_desvar('ESF')
P.driver.add_desvar('D')
#Objective function
P.driver.add_objective('obj')
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
P.driver.add_constraint('con_esf', upper=0.0)
P.driver.add_constraint('con_temp', upper=0.0)
#Coupling constraints
#Threshold for the coupling (constraints define as (x_in-x_out)**2<epsilon)
epsilon=1e-6
P.driver.add_constraint('con_str_aer_wt',upper=epsilon)
P.driver.add_constraint('con_str_aer_theta',upper=epsilon)
P.driver.add_constraint('con_aer_str_l',upper=epsilon)
P.driver.add_constraint('con_aer_pro_d',upper=epsilon)
P.driver.add_constraint('con_pro_aer_esf',upper=epsilon)
P.driver.add_constraint('con_pro_str_we',upper=epsilon)
#Recorder
if "--plot" in argv:
    recorder = SqliteRecorder('IDF.sqlite')
    recorder.options['record_params'] = True
    recorder.options['record_metadata'] = True
    P.driver.add_recorder(recorder)

#Run optimization
P.setup()
P.run()
P.cleanup()

print 'Z_opt=', P['z']*scalers['z']
print 'X_str_opt=', P['x_str']*scalers['x_str']
print 'X_aer_opt=', P['x_aer']
print 'X_pro_opt=', P['x_pro']*scalers['x_pro']
print 'R_opt=', -P['obj']*scalers['R']

if "--plot" in argv:
    import matplotlib.pylab as plt
    import sqlitedict
    import re
    
    db = sqlitedict.SqliteDict( 'IDF.sqlite', 'openmdao')
    plt.figure()

    pattern = re.compile('rank0:'+optimizer+'/\d+$')
    r = []
    for k, v in db.iteritems():
        if re.match(pattern, k):
            r.append(v['Unknowns']['Perfo.R']*scalers['R'])
            
    plt.plot(r)
    plt.show()



