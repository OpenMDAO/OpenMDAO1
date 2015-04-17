import numpy as np

from openmdao.core import Component, Assembly, Group
from openmdao.util import ExprComp

class Parab(Component):

    def __init__(self):

        self.add_input('x', default=np.ones(10,), units="BTU/lbm")
        self.add_input('x', size=10, type=np.array)

        self.add_output('y', default=1.0)

    def get_var_idx(self): # needed only for parallel
        comm = self.comm

        return {'x':, [1,3,4,10,9,7,8,12,52,18]}


    def execute(self, ins, outs):
        outs['z'] = ins['x']**2

class Adder(Component):

    def __init__(self):
        super(Adder, self).__init__()

        self.add_input('x', val=1.0, size=1)
        self.add_output('y', val=1.0, size=1)
        self.add_state('u', val=1.0, size=1)

    def execute(self, ins, outs):
        outs['z'] = ins['x']+2


class Sim(Assembly):

    def __init__(self):

        super(Sim, self).__init__()

        p1 = self.add(Parab(), name='parab1')
        p2 = self.add(Parab(), name='parab2')
        p3 = self.add(Adder(), name='parab3')

        self.alias('parab1.y', 'y')
        self.alias('parab1.x', 'x')

        self.connect('y','parab2.x')

        #this actually creates a new component with an output named "y" at this level of the system hierarchy
        #    This component should be non-namespacing so that a variable called 'y' in this Assembly
        p_expr = self.add(ExprComp('z=3*y+2*x'))

        self.connect('z', 'parab3.x')


if __name__ == "__main__":

    from openmdao.core import Assembly
    from openmdao.drivers import Cobyla

    top = Assembly()

    s = top.root = Sim(name_space="")

    s.sub_systems  == ['parab1', 'parab2', 'parab3']

    s.auto_workflow()

    s.workflow == ['parab1', 'parab2' ,'parab3']

    #top.driver = Cobyla()
    #top.setup()
    top.setup_comps()
    top.setup_variables() # all systems repor their outputs from the bottom to the top
    top.setup_sizes()
    top.setup_vectors() #this is where each sub-system will get its own vector wrapper that only has access to variables within its own scope
    top.setup_scatters()


    top.run()
