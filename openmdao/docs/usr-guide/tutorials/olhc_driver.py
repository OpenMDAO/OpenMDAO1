from openmdao.api import IndepVarComp, Group, Problem, ScipyOptimizer, ExecComp, DumpRecorder
from openmdao.test.paraboloid import Paraboloid

from openmdao.drivers.latinhypercube_driver import LatinHypercubeDriver, OptimizedLatinHypercubeDriver

top = Problem()
root = top.root = Group()

root.add('p1', IndepVarComp('x', 50.0), promotes=['*'])
root.add('p2', IndepVarComp('y', 50.0), promotes=['*'])
root.add('comp', Paraboloid(), promotes=['*'])

top.driver = LatinHypercubeDriver(10)
top.driver.add_desvar('x', low=-50.0, high=50.0)
top.driver.add_desvar('y', low=-50.0, high=50.0)

top.driver.add_objective('f_xy')

recorder = DumpRecorder('paraboloid')
recorder.options['record_params'] = True
recorder.options['record_unknowns'] = False
recorder.options['record_resids'] = False
top.driver.add_recorder(recorder)

top.setup(check=False)
top.run()

top.driver.recorders[0].close()