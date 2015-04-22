
from openmdao.core.system import System


class Component(System):
    def __init__(self):
        super(Component, self).__init__()

        # by default, don't promote any vars up to our parent
        self.promotes = ()

    def add_input(self, name, val=None):
        pass

    def add_output(self, name, val=None):
        pass

    def add_state(self, name, val=None):
        pass
