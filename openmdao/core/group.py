from collections import OrderedDict

from openmdao.core.system import System
from openmdao.core.varmanager import VarManager, VarViewManager

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
        return system

    def connect(self, src, target):
        self._src[target] = src

    def subsystems(self):
        """ returns iterator over subsystems """
        return self._subsystems.iteritems()

    def variables(self):
        params = OrderedDict()
        outputs = OrderedDict()
        states = OrderedDict()

        for name, sub in self.subsystems():
            subparams, suboutputs, substates = sub.variables()
            for p, meta in subparams.items():
                pname = self.var_pathname(p, sub)
                source = self._src.get(pname)
                if source is not None:
                    meta['_source_'] = self.var_pathname(source, self)
                params[self.var_pathname(p, sub)] = meta

            for u, meta in suboutputs.items():
                outputs[self.var_pathname(u, sub)] = meta

            for s, meta in substates.items():
                states[self.var_pathname(s, sub)] = meta

        return params, outputs, states

    def connections(self):
        """ returns iterator over connections """
        conns = self._src.copy()
        for name, subsystem in self.subsystems():
            if isinstance(subsystem, Group):
                for tgt, src in subsystem.connections():
                    src_name = self.var_pathname(src, subsystem)
                    tgt_name = self.var_pathname(tgt, subsystem)
                    conns[tgt_name] = src_name
        return conns.items()

    def var_pathname(self, name, subsystem):
        if subsystem.promoted(name):
            return name
        if len(subsystem.name) > 0:
            return subsystem.name+':'+name
        else:
            return name

    def setup_vectors(self, parent_vm=None):
        params, outputs, states = self.variables()
        if parent_vm is None:
            self.varmanager = VarManager(params, outputs, states)
        else:
            self.varmanager = VarViewManager(parent_vm,
                                             self.name,
                                             self.promotes,
                                             params,
                                             outputs,
                                             states)

        for name, sub in self.subsystems():
            sub.setup_vectors(self.varmanager)
