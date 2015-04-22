from collections import OrderedDict

from openmdao.core.system import System

class Group(System):
    def __init__(self):
        super(Group, self).__init__()
        self.subsystems = OrderedDict()
        self.local_subsystems = OrderedDict()

    def add(self, name, system):
        self.subsystems[name] = system
