
from openmdao.core.system import System

class Group(System):
    def __init__(self):
        super(Group, self).__init__()
        self.subsystems = []
