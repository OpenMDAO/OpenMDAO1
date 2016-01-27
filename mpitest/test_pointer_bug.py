""" Test for a bug reported by Rob."""

from __future__ import print_function, division
import sys
import numpy as np

from openmdao.api import Group, Component, IndepVarComp, ParallelGroup, \
                         PetscImpl, Problem
from openmdao.core.mpi_wrap import MPI
from openmdao.test.mpi_util import MPITestCase

if MPI:
    from openmdao.core.petsc_impl import PetscImpl as impl
else:
    from openmdao.api import BasicImpl as impl


class PointerProblem(Problem):
    def __init__(self, root=None, driver=None, impl=None, comm=None):
        super(PointerProblem,self).__init__(root, driver, impl, comm)

    def setup(self, check=True, out_stream=sys.stdout):
        self.root.setup_solver(self.driver)
        super(PointerProblem,self).setup(check=check, out_stream=out_stream)


class Trajectory(Group):
    def __init__(self,*args,**kwargs):
        super(Trajectory, self).__init__(*args,**kwargs)
        self.phases = {}

    def add_phase(self,phase,promotes=None):
        self.add(phase.name,phase,promotes)
        self.phases[phase.name] = phase

    def setup_solver(self,driver):
        for phase_name in self.phases:
            self.phases[phase_name].setup_solver(driver)


class PhaseTimeComp(Component):
    def __init__(self):
        super(PhaseTimeComp, self).__init__()
        self.fd_options['force_fd'] = True
        self.add_output(name='tf', val=1.0)

    def solve_nonlinear(self,params,unknowns,resids):
        pass


class FlatEarthEOM(Component):
    def __init__(self, num_nodes):
        super(FlatEarthEOM,self).__init__()
        self.num_nodes = num_nodes
        self.eom_states = []
        self.add_param('vx', eom_state=True)
        self.fd_options['force_fd'] = True

    def solve_nonlinear(self, params, unknowns, resids):
        pass

    def add_param(self, name, eom_state=False, **kwargs):
        kwargs['eom_state'] = True
        super(FlatEarthEOM, self).add_param(name,shape=(self.num_nodes,),**kwargs)

        # If we added a state variable, add the appropriate outputs to the EOM automatically
        if eom_state:
            self.eom_states.append( {'name':name})
            self.add_output(name='dXdt:{0}'.format(name),
                            shape=(self.num_nodes,))


class DefectComp(Component):
    def __init__(self, ncn, eom_states):
        super(DefectComp, self).__init__()
        self.fd_options['force_fd'] = True

        self.state_names = [ state['name'] for state in eom_states ]
        nin = ncn-1

        for state_name in self.state_names:
            self.add_param(name='X_c:{0}'.format(state_name), val=np.zeros(ncn))
            self.add_output(name='defect:{0}'.format(state_name), val=np.zeros(nin))

    def solve_nonlinear(self,params,unknowns,resids):
        pass


class StateInterpolatorComp(Component):
    def __init__(self,ncn, eom_states):

        super(StateInterpolatorComp, self).__init__()
        self.fd_options['force_fd'] = True

        state_names = [ state['name'] for state in eom_states ]

        for state_name in state_names:
            self.add_param(name='X_c:{0}'.format(state_name), val=np.zeros(ncn))

    def solve_nonlinear(self,params,unknowns,resids):
        pass


class FlatEarthRHS(Group):
    def __init__(self, num_nodes):
        super(FlatEarthRHS,self).__init__()
        self.eom_states = []
        self.add(name='eom', system=FlatEarthEOM(num_nodes))

    def add(self, name, system, promotes=None):
        super(FlatEarthRHS, self).add(name,system,promotes)
        self.eom_states.extend( system.eom_states )


class CollocationPhase(Group):
    def __init__(self,name,rhs,num_seg,seg_ncn=3,rel_lengths=1):
        super(CollocationPhase,self).__init__()

        self._eom_states = rhs.eom_states
        self.trajectory = None
        self.name = name
        self._segments = []
        parallel_segment_group = ParallelGroup()

        for i in range(num_seg):
            seg_name = '{0}'.format(i)
            seg = CollocationSegment(index=i, rhs=rhs,
                                     num_cardinal_nodes=2,
                                     rel_length=1)
            parallel_segment_group.add(name=seg_name,
                                       system=seg)
            self._segments.append(seg)

        self.add(name='segments',system=parallel_segment_group)

        # 3. Add the state and dynamic control param comps and muxing components
        eom_state_names = ['X_c:{0}'.format(state['name']) for state in rhs.eom_states]
        for i,state in enumerate(self._eom_states):
            self.add( name='eom_state_ivar_comp:{0}'.format(state['name']),
                      system=IndepVarComp(name=eom_state_names[i],
                                          val=np.zeros((3))),
                      promotes=[eom_state_names[i]])

        for i, seg in enumerate(self._segments):
            idxs_states = range(0, 2)

            for state in self._eom_states:
                state_name = state['name']
                self.connect( 'X_c:{0}'.format(state_name), 'segments.{0:d}.X_c:{1}'.format(i, state_name),
                              src_indices=idxs_states)

    def setup_solver(self,driver):
        for i,eom_state in enumerate(self._eom_states):
            driver.add_desvar(name='{0}.X_c:{1}'.format(self.name,eom_state['name']))

            # Add the state defects as constraints
            for i in range(len(self._segments)):
                driver.add_constraint(name='{0}.segments.{1:d}.defect:{2}'.format(self.name,i,eom_state['name']),
                                      equals=0.0)


class CollocationSegment(Group):
    def __init__(self, index, rhs, num_cardinal_nodes=2, rel_length=1.0):
        super(CollocationSegment, self).__init__()

        # 4. state interpolator
        self.add(name='state_interp',
                 system=StateInterpolatorComp(num_cardinal_nodes,rhs.eom_states),
                 promotes=['X_c:{0}'.format(state['name']) for state in rhs.eom_states])

        # 6. defects
        defect_promotes = ['defect:{0}'.format(state['name']) for state in rhs.eom_states]
        self.add(name='defect',
                 system=DefectComp(num_cardinal_nodes, rhs.eom_states),
                 promotes=defect_promotes)

        # 3. rhs_c
        for state in rhs.eom_states:
            s = state['name']
            self.connect('X_c:{0}'.format(s),['defect.X_c:{0}'.format(s)])

def flat_earth(num_seg=3, seg_ncn=3):

    prob = PointerProblem(root=Trajectory(), impl=PetscImpl)
    traj = prob.root
    rhs = FlatEarthRHS(1)
    phase0 = CollocationPhase(name='phase0', rhs=rhs, num_seg=num_seg, seg_ncn=seg_ncn,
                              rel_lengths=1)
    traj.add_phase(phase0)
    return prob, phase0


class MPIPointerBug1(MPITestCase):

    N_PROCS = 2

    def test_index_bug(self):
        # This was a bug reported by Rob Falck where the problem wouldn't
        # setup because of a petsc index error. The problem seems to stem
        # from a design variable that points to a promoted variable in the
        # parallel systems which is a seeming input-input matchup.

        # This test will fail if the setup raises an error. Otherwise it'll
        # pass. The output is not tested here as the problem iss a bit bogus
        # in the simplified form without the pointer infrastructure.

        prob,phase0 = flat_earth(num_seg=2, seg_ncn=2)
        prob.setup(check=False)
        prob.run()
        prob.cleanup()

if __name__ == '__main__':
    from openmdao.test.mpi_util import mpirun_tests
    mpirun_tests()