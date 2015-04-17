
from openmdao.core.system import System

__undefined = object()

class Component(System):
    def __init__(self):
        super(Component, self).__init__()
        
        # by default, don't promote any vars up to our parent
        self.promotes = ()

    def add_input(name, val=__undefined):
        pass

    def add_output(name, val=__undefined):
        pass
