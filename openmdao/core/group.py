from collections import OrderedDict

from openmdao.core.system import System

class Group(System):
    def __init__(self):
        super(Group, self).__init__()
        self._subsystems = OrderedDict()
        self._local_subsystems = OrderedDict()
        self._src = {}

    def add(self, name, system, promotes=None):
        if promotes is not None:
            system.promotes = promotes
        self._subsystems[name] = system
        system.name = name

    def connect(self, src, target):
        self._src[target] = src

    def subsystems(self):
        """ returns iterator over subsystems """
        return self._subsystems.iteritems()

    def variables(self):
        params = OrderedDict()
        unknowns = OrderedDict()
        states = OrderedDict()

        for name, sub in self.subsystems():
            subparams, subunks, substates = sub.variables()
            for p, meta in subparams.items():
                params[self.var_pathname(p, sub)] = meta

            for u, meta in subunks.items():
                unknowns[self.var_pathname(u, sub)] = meta

            for s, meta in substates.items():
                states[self.var_pathname(s, sub)] = meta

        return params, unknowns, states

    def var_pathname(self, name, subsystem):
        if subsystem.promoted(name):
            return name
        return subsystem.name+':'+name

    def setup_vectors(self, parent_vm=None):
        params, unknowns, states = self.variables()
        if parent_vm is None:
            self.varmanager = VarManager(params, unknowns, states)
        else:
            self.varmanager = SubVarManager(parent_vm,
                                            self.name,
                                            self.promotes,
                                            params,
                                            unknowns,
                                            states)

        for sub in self.subsystems():
            sub.setup_vectors(self.varmanager)
