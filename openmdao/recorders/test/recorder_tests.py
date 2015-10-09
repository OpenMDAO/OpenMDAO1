import time
from openmdao.core.problem import Problem
from openmdao.test.converge_diverge import ConvergeDiverge
from openmdao.test.example_groups import ExampleGroup
from openmdao.core.mpi_wrap import MPI
import numpy as np

def get_local_vars(system, vec):
    return filter(lambda x: MPI.COMM_WORLD.rank == system._owning_rank[x[0]], vec)

def run_problem(problem):
    t0 = time.time()
    problem.run()
    t1 = time.time()

    return t0, t1

def assertIterationDataRecorded(self, expected, tolerance, problem=None):
    raise NotImplementedError()

def test_only_resids_recorded(self):
    prob = Problem()
    prob.root = ConvergeDiverge()
    prob.driver.add_recorder(self.recorder)
    self.recorder.options['record_params'] = False
    self.recorder.options['record_unknowns'] = False
    self.recorder.options['record_resids'] = True
    prob.setup(check=False)

    t0, t1 = run_problem(prob)
    self.recorder.close()
    
    coordinate = ['Driver', (1, )]

    expected_resids = [
        ("comp1.y1", 0.0),
        ("comp1.y2", 0.0),
        ("comp2.y1", 0.0),
        ("comp3.y1", 0.0),
        ("comp4.y1", 0.0),
        ("comp4.y2", 0.0),
        ("comp5.y1", 0.0),
        ("comp6.y1", 0.0),
        ("comp7.y1", 0.0),
        ("p.x", 0.0)
    ]

    self.assertIterationDataRecorded(((coordinate, (t0, t1), None, None, expected_resids),), self.eps)
   
def test_only_unknowns_recorded(self):
    prob = Problem()
    prob.root = ConvergeDiverge()
    prob.driver.add_recorder(self.recorder)
    prob.setup(check=False)
    
    t0, t1 = run_problem(prob)
    self.recorder.close()
    
    coordinate = ['Driver', (1, )]

    expected_unknowns = [
        ("comp1.y1", 8.0),
        ("comp1.y2", 6.0),
        ("comp2.y1", 4.0),
        ("comp3.y1", 21.0),
        ("comp4.y1", 46.0),
        ("comp4.y2", -93.0),
        ("comp5.y1", 36.8),
        ("comp6.y1", -46.5),
        ("comp7.y1", -102.7),
        ("p.x", 2.0)
    ]

    self.assertIterationDataRecorded(((coordinate, (t0, t1), None, expected_unknowns, None),), self.eps)

def test_only_params_recorded(self):
    prob = Problem()
    prob.root = ConvergeDiverge()
    prob.driver.add_recorder(self.recorder)
    self.recorder.options['record_params'] = True
    self.recorder.options['record_resids'] = False
    self.recorder.options['record_unknowns'] = False
    prob.setup(check=False)

    t0, t1 = run_problem(prob)
    self.recorder.close()

    coordinate = ['Driver', (1,)]
    expected_params = [
        ("comp1.x1", 2.0),
        ("comp2.x1", 8.0),
        ("comp3.x1", 6.0),
        ("comp4.x1", 4.0),
        ("comp4.x2", 21.0),
        ("comp5.x1", 46.0),
        ("comp6.x1", -93.0),
        ("comp7.x1", 36.8),
        ("comp7.x2", -46.5)
    ]
    
    self.assertIterationDataRecorded(((coordinate, (t0, t1), expected_params, None, None),), self.eps)

def test_basic(self):
    prob = Problem()
    prob.root = ConvergeDiverge()
    prob.driver.add_recorder(self.recorder)
    self.recorder.options['record_params'] = True
    self.recorder.options['record_resids'] = True
    prob.setup(check=False)

    t0, t1 = run_problem(prob)
    self.recorder.close()
    
    coordinate = ['Driver', (1, )]

    expected_params = [
        ("comp1.x1", 2.0),
        ("comp2.x1", 8.0),
        ("comp3.x1", 6.0),
        ("comp4.x1", 4.0),
        ("comp4.x2", 21.0),
        ("comp5.x1", 46.0),
        ("comp6.x1", -93.0),
        ("comp7.x1", 36.8),
        ("comp7.x2", -46.5)
    ]

    expected_unknowns = [
        ("comp1.y1", 8.0),
        ("comp1.y2", 6.0),
        ("comp2.y1", 4.0),
        ("comp3.y1", 21.0),
        ("comp4.y1", 46.0),
        ("comp4.y2", -93.0),
        ("comp5.y1", 36.8),
        ("comp6.y1", -46.5),
        ("comp7.y1", -102.7),
        ("p.x", 2.0)
    ]

    expected_resids = [
        ("comp1.y1", 0.0),
        ("comp1.y2", 0.0),
        ("comp2.y1", 0.0),
        ("comp3.y1", 0.0),
        ("comp4.y1", 0.0),
        ("comp4.y2", 0.0),
        ("comp5.y1", 0.0),
        ("comp6.y1", 0.0),
        ("comp7.y1", 0.0),
        ("p.x", 0.0)
    ]

    if MPI and self.recorder._parallel:
        expected_params = get_local_vars(prob.root, expected_params)
        expected_unknowns  = get_local_vars(prob.root, expected_unknowns)
        expected_resids = get_local_vars(prob.root, expected_resids)

    if self.recorder._parallel or prob.root.comm.rank == 0:
        self.assertIterationDataRecorded(((coordinate, (t0, t1), expected_params, expected_unknowns, expected_resids),), self.eps)

def test_includes(self):
    prob = Problem()
    prob.root = ConvergeDiverge()
    prob.driver.add_recorder(self.recorder)
    self.recorder.options['includes'] = ['comp1.*']
    self.recorder.options['record_params'] = True
    self.recorder.options['record_resids'] = True
    prob.setup(check=False)
    t0, t1 = run_problem(prob)
    self.recorder.close()
    
    coordinate = ['Driver', (1,)]

    expected_params = [
        ("comp1.x1", 2.0)
    ]
    expected_unknowns = [
        ("comp1.y1", 8.0),
        ("comp1.y2", 6.0)
    ]
    expected_resids = [
        ("comp1.y1", 0.0),
        ("comp1.y2", 0.0)
    ]
    
    self.assertIterationDataRecorded(((coordinate, (t0, t1), expected_params, expected_unknowns, expected_resids),), self.eps)

def test_includes_and_excludes(self):
    prob = Problem()
    prob.root = ConvergeDiverge()
    prob.driver.add_recorder(self.recorder)
    self.recorder.options['includes'] = ['comp1.*']
    self.recorder.options['excludes'] = ["*.y2"]
    self.recorder.options['record_params'] = True
    self.recorder.options['record_resids'] = True
    prob.setup(check=False)
    t0, t1 = run_problem(prob)
    self.recorder.close()

    coordinate = ['Driver', (1,)]

    expected_params = [
        ("comp1.x1", 2.0)
    ]
    expected_unknowns = [
        ("comp1.y1", 8.0)
    ]
    expected_resids = [
        ("comp1.y1", 0.0)
    ]

    self.assertIterationDataRecorded(((coordinate, (t0, t1), expected_params, expected_unknowns, expected_resids),), self.eps)

def test_solver_record(self):
    prob = Problem()
    prob.root = ConvergeDiverge()
    prob.root.nl_solver.add_recorder(self.recorder)
    self.recorder.options['record_params'] = True
    self.recorder.options['record_resids'] = True
    prob.setup(check=False)
    t0, t1 = run_problem(prob)
    self.recorder.close()
    
    coordinate = ['Driver', (1,), "root", (1,)]

    expected_params = [
        ("comp1.x1", 2.0),
        ("comp2.x1", 8.0),
        ("comp3.x1", 6.0),
        ("comp4.x1", 4.0),
        ("comp4.x2", 21.0),
        ("comp5.x1", 46.0),
        ("comp6.x1", -93.0),
        ("comp7.x1", 36.8),
        ("comp7.x2", -46.5)
    ]
    expected_unknowns = [
        ("comp1.y1", 8.0),
        ("comp1.y2", 6.0),
        ("comp2.y1", 4.0),
        ("comp3.y1", 21.0),
        ("comp4.y1", 46.0),
        ("comp4.y2", -93.0),
        ("comp5.y1", 36.8),
        ("comp6.y1", -46.5),
        ("comp7.y1", -102.7),
        ("p.x", 2.0)
    ]
    expected_resids = [
        ("comp1.y1", 0.0),
        ("comp1.y2", 0.0),
        ("comp2.y1", 0.0),
        ("comp3.y1", 0.0),
        ("comp4.y1", 0.0),
        ("comp4.y2", 0.0),
        ("comp5.y1", 0.0),
        ("comp6.y1", 0.0),
        ("comp7.y1", 0.0),
        ("p.x", 0.0)
    ]

    self.assertIterationDataRecorded(((coordinate, (t0, t1), expected_params, expected_unknowns, expected_resids),), self.eps)

def test_sublevel_record(self):

    prob = Problem()
    prob.root = ExampleGroup()
    prob.root.G2.G1.nl_solver.add_recorder(self.recorder)
    self.recorder.options['record_params'] = True
    self.recorder.options['record_resids'] = True
    prob.setup(check=False)
    t0, t1 = run_problem(prob)
    self.recorder.close()

    coordinate = ['Driver', (1,), "root", (1,), "G2", (1,), "G1", (1,)]

    expected_params = [
        ("C2.x", 5.0)
    ]
    expected_unknowns = [
        ("C2.y", 10.0)
    ]
    expected_resids = [
        ("C2.y", 0.0)
    ]
    
    self.assertIterationDataRecorded(((coordinate, (t0, t1), expected_params, expected_unknowns, expected_resids),), self.eps)

def test_multilevel_record(self):
    prob = Problem()
    prob.root = ExampleGroup()
    prob.root.G2.G1.nl_solver.add_recorder(self.recorder)
    prob.driver.add_recorder(self.recorder)
    self.recorder.options['record_params'] = True
    self.recorder.options['record_resids'] = True
    prob.setup(check=False)
    t0, t1 = run_problem(prob)
    self.recorder.close()

    solver_coordinate = ['Driver', (1,), "root", (1,), "G2", (1,), "G1", (1,)]
    
    g1_expected_params = [
        ("C2.x", 5.0)
    ]
    g1_expected_unknowns = [
        ("C2.y", 10.0)
    ]
    g1_expected_resids = [
        ("C2.y", 0.0)
    ]

    g1_expected = (g1_expected_params, g1_expected_unknowns, g1_expected_resids)

    driver_coordinate = ['Driver', (1,)]

    driver_expected_params = [
        ("G3.C3.x", 10.0)
    ]

    driver_expected_unknowns = [
        ("G2.C1.x", 5.0),
        ("G2.G1.C2.y", 10.0),
        ("G3.C3.y", 20.0),
        ("G3.C4.y", 40.0),
    ]

    driver_expected_resids = [
        ("G2.C1.x", 0.0),
        ("G2.G1.C2.y", 0.0),
        ("G3.C3.y", 0.0),
        ("G3.C4.y", 0.0),
    ]
 
    expected = []
    expected.append((solver_coordinate, (t0, t1), g1_expected_params, g1_expected_unknowns, g1_expected_resids))
    expected.append((driver_coordinate, (t0, t1), driver_expected_params, driver_expected_unknowns, driver_expected_resids))

    self.assertIterationDataRecorded(expected, self.eps)
