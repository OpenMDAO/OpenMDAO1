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

        self.add('parab1', Parab(), auto_alias=True)

        self.add('parab2', Parab(), auto_alias=True)

        self.add('parab3', Adder(), auto_alias=True)

        self.alias('parab1.x', 'z0')

        self.alias('parab1.y', 'y')
        self.alias('parab2.x', 'y')
        self.connect(name="y", 'parab1.y', 'parab2.x')

        self.connect('z0+3*y', 'parab.x')

        # self.connect('parab1.y', 'parab3.x', name='x1')

if __name__ == "__main__":

    from openmdao.core import Assembly
    from openmdao.drivers import Cobyla

    top = Assembly()

    s = top.root = Sim(name_space="")

    s.sequence  == ['parab1', 'parab2', 'parab3']
    s.sequence  == ['parab2', 'parab1', 'parab3']

    s.auto_workflow()

    s.workflow == ['parab1', 'parab2' ,'parab3']

    #top.driver = Cobyla()
    top.setup()

    top.run()
