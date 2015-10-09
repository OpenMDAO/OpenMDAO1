import time

import numpy as np

from openmdao.core.problem import Problem
from openmdao.test.converge_diverge import ConvergeDiverge
from openmdao.test.example_groups import ExampleGroup

def assertMetadataRecorded(self, expected):
    raise NotImplementedError()

def test_driver_records_metadata(self):
    prob = Problem()
    prob.root = ConvergeDiverge()
    prob.driver.add_recorder(self.recorder)
    self.recorder.options['record_metadata'] = True
    prob.setup(check=False)
    self.recorder.close()

    expected_params = list(prob.root.params.iteritems())
    expected_unknowns = list(prob.root.unknowns.iteritems())
    expected_resids = list(prob.root.resids.iteritems())

    self.assertMetadataRecorded((expected_params, expected_unknowns, expected_resids))

def test_driver_doesnt_record_metadata(self):
    prob = Problem()
    prob.root = ConvergeDiverge()
    prob.driver.add_recorder(self.recorder)
    self.recorder.options['record_metadata'] = False
    prob.setup(check=False)
    self.recorder.close()

    self.assertMetadataRecorded(None)

def test_root_solver_records_metadata(self):
    prob = Problem()
    prob.root = ConvergeDiverge()
    prob.root.nl_solver.add_recorder(self.recorder)
    self.recorder.options['record_metadata'] = True
    prob.setup(check=False)
    self.recorder.close()

    expected_params = list(prob.root.params.iteritems())
    expected_unknowns = list(prob.root.unknowns.iteritems())
    expected_resids = list(prob.root.resids.iteritems())
    
    self.assertMetadataRecorded((expected_params, expected_unknowns, expected_resids))

def test_root_solver_doesnt_record_metadata(self):
    prob = Problem()
    prob.root = ConvergeDiverge()
    prob.root.nl_solver.add_recorder(self.recorder)
    self.recorder.options['record_metadata'] = False
    prob.setup(check=False)
    self.recorder.close()

    self.assertMetadataRecorded(None)

def test_subsolver_records_metadata(self):
    prob = Problem()
    prob.root = ExampleGroup()
    prob.root.G2.G1.nl_solver.add_recorder(self.recorder)
    self.recorder.options['record_metadata'] = True
    prob.setup(check=False)
    self.recorder.close()

    expected_params = list(prob.root.params.iteritems())
    expected_unknowns = list(prob.root.unknowns.iteritems())
    expected_resids = list(prob.root.resids.iteritems())

    self.assertMetadataRecorded((expected_params, expected_unknowns, expected_resids))

def test_subsolver_doesnt_record_metadata(self):
    prob = Problem()
    prob.root = ExampleGroup()
    prob.root.G2.G1.nl_solver.add_recorder(self.recorder)
    self.recorder.options['record_metadata'] = False
    prob.setup(check=False)
    self.recorder.close()

    self.assertMetadataRecorded(None)
