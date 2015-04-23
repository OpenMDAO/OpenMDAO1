from collections import OrderedDict

from openmdao.core.system import System

class Group(System):
    def __init__(self):
        super(Group, self).__init__()
        self._subsystems = OrderedDict()
        self._local_subsystems = OrderedDict()

    def add(self, name, system, promotes=None):
        if promotes is not None:
            system.promotes = promotes
        self._subsystems[name] = system

    def subsystems(self):
        """ returns iterator over subsystems """
        return self._subsystems.iteritems()

    def variables(self):
        params = OrderedDict()
        unknowns = OrderedDict()
        states = OrderedDict()

        for name, system in self.subsystems():
            params.update()
