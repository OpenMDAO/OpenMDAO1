import unittest
import sys
import time
import numpy as np
from openmdao.core.problem import Problem
from openmdao.core.group import Group
from openmdao.core.parallel_group import ParallelGroup
from openmdao.core.component import Component
from openmdao.components.indep_var_comp import IndepVarComp
from openmdao.core.mpi_wrap import MPI, MultiProcFailCheck

from openmdao.test.converge_diverge import ConvergeDiverge
from openmdao.test.example_groups import ExampleGroup

if MPI: # pragma: no cover
    from openmdao.core.petsc_impl import PetscImpl as impl
else:
    from openmdao.core import BasicImpl as impl

def assertMetadataRecorded(self, expected):
    raise NotImplementedError()

class ABCDArrayComp(Component):

    def __init__(self, arr_size=9, delay=0.01):
        super(ABCDArrayComp, self).__init__()
        self.add_param('a', np.ones(arr_size, float))
        self.add_param('b', np.ones(arr_size, float))
        self.add_param('in_string', '')
        self.add_param('in_list', [])

        self.add_output('c', np.ones(arr_size, float))
        self.add_output('d', np.ones(arr_size, float))
        self.add_output('out_string', '')
        self.add_output('out_list', [])

        self.delay = delay

    def solve_nonlinear(self, params, unknowns, resids):
        time.sleep(self.delay)

        unknowns['c'] = params['a'] + params['b']
        unknowns['d'] = params['a'] - params['b']

        unknowns['out_string'] = params['in_string'] + '_' + self.name
        unknowns['out_list']   = params['in_list'] + [1.5]

def test_driver_records_metadata(self):
    size = 3

    prob = Problem(Group(), impl=impl)

    G1 = prob.root.add('G1', ParallelGroup())
    G1.add('P1', IndepVarComp('x', np.ones(size, float) * 1.0))
    G1.add('P2', IndepVarComp('x', np.ones(size, float) * 2.0))

    prob.root.add('C1', ABCDArrayComp(size))

    prob.root.connect('G1.P1.x', 'C1.a')
    prob.root.connect('G1.P2.x', 'C1.b')

    prob.driver.add_recorder(self.recorder)

    self.recorder.options['record_metadata'] = True
    prob.setup(check=False)

    self.recorder.close()
    

    expected = (
            list(prob.root.params.iteritems()),
            list(prob.root.unknowns.iteritems()),
            list(prob.root.resids.iteritems()),
    )

    self.assertMetadataRecorded(expected)

def test_driver_doesnt_records_metadata(self):
    size = 3

    prob = Problem(Group(), impl=impl)

    G1 = prob.root.add('G1', ParallelGroup())
    G1.add('P1', IndepVarComp('x', np.ones(size, float) * 1.0))
    G1.add('P2', IndepVarComp('x', np.ones(size, float) * 2.0))

    prob.root.add('C1', ABCDArrayComp(size))

    prob.root.connect('G1.P1.x', 'C1.a')
    prob.root.connect('G1.P2.x', 'C1.b')

    prob.driver.add_recorder(self.recorder)

    self.recorder.options['record_metadata'] = False
    prob.setup(check=False)

    self.recorder.close()

    self.assertMetadataRecorded(None)
