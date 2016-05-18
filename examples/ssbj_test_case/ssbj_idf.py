# SSBJ test case - IDF formulation
# Initial Author : Sylvain Dubreuil

import numpy as np
#import matplotlib.pyplot as plt

from openmdao.api import Problem, ScipyOptimizer, SqliteRecorder

from ssbj_idf_mda import SSBJ_IDF_MDA
from ssbj_mda import init_ssbj_mda


#Construction of optimization problem
scaled = True
scalers, pf = init_ssbj_mda(scaled=scaled)

P=Problem()
P.root=SSBJ_IDF_MDA(scalers, pf, scaled=scaled)

#Optimizer options
P.driver = ScipyOptimizer()
optimizer = 'SLSQP'
P.driver.options['optimizer'] = optimizer
P.driver.options['tol'] = 1.0e-10
P.driver.options['maxiter']=50
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
#recorder = SqliteRecorder('Test')
#P.driver.add_recorder(recorder)
#Run optimization
P.setup()
P.run()

#view_tree(P.root)

##Close recorder and read recorded values in a dictionnary
#P.driver.recorders[0].close()
#import sqlitedict
#import re
#db = sqlitedict.SqliteDict( 'Test', 'openmdao')
##Plot some results
#pattern = re.compile(optimizer+'/\d+$')
#i = 0
#for k, v in db.iteritems():
    #if re.match(pattern, k):
        #plt.plot(i,db[k]['Unknowns']['con_str_aer_wt'],'r+')
        #plt.plot(i,db[k]['Unknowns']['con_str_aer_theta'],'g+')
        #plt.plot(i,db[k]['Unknowns']['con_aer_pro_d'],'y+')
        #plt.plot(i,db[k]['Unknowns']['con_pro_aer_esf'],'k+')
        #plt.plot(i,db[k]['Unknowns']['con_pro_str_we'],'c+')
        #i+=1

print 'Z_opt=', P['z']*scalers['z']
print 'X_str_opt=', P['x_str']*scalers['x_str']
print 'X_aer_opt=', P['x_aer']
print 'X_pro_opt=', P['x_pro']*scalers['x_pro']
print 'R_opt=', P['obj']*scalers['R']

print 'con_str_aer_wt=', P['con_str_aer_wt']
print 'con_str_aer_theta=', P['con_str_aer_theta']
print 'con_aer_str_l=', P['con_aer_str_l']
print 'con_aer_pro_d=', P['con_aer_pro_d']
print 'con_pro_aer_esf=', P['con_pro_aer_esf']
print 'con_pro_str_we=', P['con_pro_str_we']
