""" Defines the base class for a Group in OpenMDAO."""

from collections import OrderedDict

from openmdao.core.system import System
from openmdao.core.varmanager import VarManager, VarViewManager

class Group(System):
    """A system that contains other systems"""
    
    def __init__(self):
        super(Group, self).__init__()

        self._subsystems = OrderedDict()
        self._local_subsystems = OrderedDict()
        self._src = {}
        
        # These point to (du,df) or (df,du) depending on mode.
        self.sol_vec = None
        self.rhs_vec = None

    def __contains__(self, name):
        return name in self._subsystems

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

    def subgroups(self):
        """ returns iterator over subgroups """
        for name, subsystem in self._subsystems.items():
            if isinstance(subsystem, Group):         
                yield name, subsystem

    def setup_variables(self):
        """Return params and unkowns for all susbsystems"""
        # TODO: check for the same var appearing more than once in unknowns

        comps = {}
        for name, sub in self.subsystems():           
            subparams, subunknowns = sub.setup_variables()
            for p, meta in subparams.items():
                meta = meta.copy()
                if '_source_' in meta:
                    meta['_source_'] = self.var_pathname(meta['_source_'], sub)
                else:
                    pname = self.var_pathname(p, sub)
                    source = self._src.get(pname)
                    if source is not None:
                        parts = source.split(':', 1)
                        if parts[0] in self._subsystems:
                            src_sys = self._subsystems[parts[0]]
                            vname = parts[1]
                            meta['_source_'] = self.var_pathname(vname, src_sys)
                        else:
                            meta['_source_'] = source

                self._params[self.var_pathname(p, sub)] = meta

            for u, meta in subunknowns.items():
                self._unknowns[self.var_pathname(u, sub)] = meta

        return self._params, self._unknowns

    def assign_parameters(self, params, unknowns):
        """Map absolute system names to the absolute names of the
        parameters they control
        """
        
        param_owners = {}
        
        for name, subgroup in self.subgroups():
            param_owners.update(subgroup.assign_parameters(params, unknowns))
        
        for par, unk in self._src.items():
            par_pathname = params[par]['pathname']
            unk_pathname = unknowns[unk]['pathname']
            
            par_parts = par_pathname.split(':')
            unk_parts = unk_pathname.split(':')
            
            common_parts = []
            i = 0
            while(par_parts[i] == unk_parts[i]):
                common_parts.append(par_parts[i])
            common_path = ':'.join(common_parts)
            
            param_owners[par_pathname] = common_path
            
        return param_owners    
        
            
        # TODO: implement this:
        """
            a group owns a scatter if:
                if group owns the connection and it was made at the 'appropriate' level
                    i.e. the lowest level at which the connection can be made

        =====================

                G2.connect(C1:y, G1:C2:x)   vs   G4.connect(G2:C1:y, G2:G1:C2:x)

                - G2 unknowns will have metadata for C1:y
                    abs path of C1:y will be G2:C1;y
                - G2 params will have metadata for G1:C2:x
                    abs path of G1:c2:x will be G2:G1:C2:x
                - the system that is responsible for scatter is G2 (common path)
                    that's us.. so we 'own' the param


                - G4 unknowns will have G2:C1:y with metadata
                    abs path of G2:C1:y will be G2:C1:y
                - G4 params will have G2:G1:C2:x
                    abs path will be same
                - the system that is responsible for the scatter is G2 (common path)
                    that's NOT us, we don't 'own' the param
                    we know that the reponsible system is 'G2'
                    G2 has already provided it's variable info...


        =====================

        """
        return {}

    def connections(self):
        """ returns iterator over connections """
        conns = self._src.copy()
        for name, subsystem in self.subgroups():
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

    def setup_vectors(self, parent_vm=None, param_owners=None):
        # TODO: move first-time only stuff to Problem
        if parent_vm is None:
            param_owners = self.assign_parameters(params, unknowns)
            my_params = param_owners.get(self.pathname, [])

            self.varmanager = VarManager(params, unknowns, my_params)
        else:
            my_params = param_owners.get(self.pathname, [])
            self.varmanager = VarViewManager(parent_vm,
                                             self.name,
                                             self.promotes,
                                             params,
                                             unknowns,
                                             my_params)

        for name, sub in self.subsystems():
            sub.setup_vectors(self.varmanager, param_owners)

    def setup_paths(self, parent_path):
        """Set the absolute pathname of each System in the
        tree.
        """
        super(Group, self).setup_paths(parent_path)
        for name, sub in self.subsystems():
            sub.setup_paths(self.pathname)
