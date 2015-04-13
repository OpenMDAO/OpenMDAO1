from openmdao.core import Component, NameSpace, Group

class Parab(Component):

    def __init__(self):
        super(Component,self).__init__()

        self.add_input('x', val=1.0, size=1)

        self.add_output('y', val=1.0, size=1)

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

        self.add('parab1', Parab()
            promote_ins = {'x':'x'},
            promote_outs = {'y': 'z0'}
        )

        self.add('parab2', Parab()
            promote_ins = {'x':'z0',},
            promote_outs = {'y':'y1'}
        )

        self.add('parab3', Adder(),
            promote_ins = {'x':'z0'},
            promote_outs = {'y':'y2'}
        )

        # self.connect('parab1.y', 'parab2.x', name='x1')
        # self.conenct('parab1.y', 'parab3.x', name='x1')

if __name__ == "__main__":

    from openmdao.core import Assembly
    from openmdao.drivers import Cobyla

    top = Assembly()

    s = top.root = Sim()

    #top.driver = Cobyla()
    top.setup()

    top.run()
