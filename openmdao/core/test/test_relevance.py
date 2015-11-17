import unittest
from six import itervalues

from openmdao.api import ExecComp, IndepVarComp, Problem, Group


class TestLinearGaussSeidel(unittest.TestCase):
    def setUp(self):
        self.p = Problem(Group())
        root = self.p.root

        root.add('P1', IndepVarComp('x', 2.0))
        root.add('P2', IndepVarComp('x', 2.0))

        root.add('C2', ExecComp('y = 2.0*x'))
        root.add('C3', ExecComp('y = 2.0*x'))
        root.add('C4', ExecComp('y = 2.0*x'))
        root.add('C5', ExecComp('y = 2.0*x'))
        root.add('C6', ExecComp('y = 2.0*x'))
        root.add('C7', ExecComp('y = 2.0*x'))
        root.add('C8', ExecComp('y = 2.0*x1 + 3.0*x2'))

        root.connect('P1.x', 'C2.x')
        root.connect('P1.x', 'C3.x')
        root.connect('P2.x', 'C4.x')
        root.connect('P2.x', 'C5.x')

        root.connect('C2.y', 'C8.x1')
        root.connect('C4.y', 'C8.x2')

        root.connect('C3.y', 'C6.x')
        root.connect('C5.y', 'C7.x')

    def test_relevant(self):
        p = self.p
        root = p.root

        p.driver.add_desvar('P1.x')
        p.driver.add_objective('C8.y')

        p.setup(check=False)

        rels = {
            'P1.x': ['P1.x', 'C2.x', 'C2.y', 'C8.x1', 'C8.y'],
            'C8.y': ['P1.x', 'C2.x', 'C2.y', 'C8.x1', 'C8.y']
        }

        not_rels = {
            'P1.x': ['P2.x', 'C4.x', 'C4.y', 'C5.x', 'C5.y', 'C7.x', 'C7.y', 'C8.x2'],
            'C8.y': ['P2.x', 'C4.x', 'C4.y', 'C5.x', 'C5.y', 'C7.x', 'C7.y', 'C8.x2']
        }

        for voi, voi_rels in rels.items():
            for voi_rel in voi_rels:
                self.assertTrue(root._probdata.relevance.is_relevant(voi, voi_rel),
                                msg="%s should be True" % voi_rel)

        for voi, voi_nrels in not_rels.items():
            for voi_nrel in voi_nrels:
                self.assertFalse(root._probdata.relevance.is_relevant(voi, voi_nrel),
                                 msg="%s should be False" % voi_nrel)

    def test_relevant_systems(self):
        p = self.p
        root = p.root

        p.driver.add_desvar('P1.x')
        p.driver.add_objective('C8.y')

        p.setup(check=False)

        rel_systems = ['P1', 'C2', 'C8']
        for s in itervalues(root._subsystems):
            if s.pathname in rel_systems:
                self.assertTrue(root._probdata.relevance.is_relevant_system('P1.x', s),
                                msg="%s should be relevant" % s.pathname)
                self.assertTrue(root._probdata.relevance.is_relevant_system('C8.y', s),
                                msg="%s should be relevant" % s.pathname)
            else:
                self.assertFalse(root._probdata.relevance.is_relevant_system('P1.x', s),
                                 msg="%s should be irrelevant" % s.pathname)
                self.assertFalse(root._probdata.relevance.is_relevant_system('C8.y', s),
                                 msg="%s should be irrelevant" % s.pathname)
