import numpy as np

from openmdao.core import Component, NameSpace, Group

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
        super(Component,self).__init__()

        self.add_input('x', val=1.0, size=1)

        self.add_output('y', val=1.0, size=1)

        self.add_state('u', val=1.0, size=1)

    def execute(self, ins, outs):
        outs['z'] = ins['x']+2


class Sim(Group):

    def __init__(self):

        super(Sim, self).__init__()

        p1 = self.add('parab1', Parab(), name_space=False)

        p2 = self.add('parab2', Parab(),  name_space=False)

        p3 = self.add('parab3', Adder(),  name_space=False)

        self.connect('parab1.x', 'z0')

        self.create_passthrough('parab.x2','y')
        self.connect('x', 'Fl_O.y', target_units="Btu/lbm")
        # self.connect(name="y", 'parab1.y', 'parab2.x')

        self.connect('z0+3*y', 'parab.x')

        self.connect('parab1.y', 'parab3.x', name='x1')

if __name__ == "__main__":

    from openmdao.core import Assembly
    from openmdao.drivers import Cobyla

    top = Assembly()

    s = top.root = Sim(name_space="")

    s.sub_systems  == ['parab1', 'parab2', 'parab3']

    s.auto_workflow()

    s.workflow == ['parab1', 'parab2' ,'parab3']

    #top.driver = Cobyla()
    top.setup()

    top.run()
