import unittest
from six import text_type

import numpy as np

from openmdao.api import Problem, Group, IndepVarComp, ExecComp

class TestPromConns(unittest.TestCase):
    def test_prom_conns(self):
        # this test mimics some of the connections found in test_nozzle in pycycle. The bug was that
        # an unknown that was connected to one parameter
        # (desVars.Ps_exhaust to nozzle.press_calcs.Ps_exhaust), was not being connected to the
        # other parameters ('nozzle.ideal_flow.chem_eq.n2ls.P', 'nozzle.ideal_flow.mach_calc.Ps',
        # and 'nozzle.ideal_flow.props.tp2props.P') that were connected via input-input connections
        # to nozzle.press_calcs.Ps_exhaust.

        prob = Problem(root=Group())
        root = prob.root

        desVars = root.add('desVars', IndepVarComp('Ps_exhaust', 1.0), promotes=('Ps_exhaust',))
        nozzle  = root.add('nozzle', Group())

        press_calcs = nozzle.add('press_calcs', ExecComp('out=Ps_exhaust'), promotes=('Ps_exhaust',))
        ideal_flow  = nozzle.add('ideal_flow', Group())

        chem_eq     = ideal_flow.add('chem_eq', Group(), promotes=('P',))
        props       = ideal_flow.add('props', Group(), promotes=('P',))
        mach_calc   = ideal_flow.add('mach_calc', ExecComp('out=Ps'), promotes=('Ps',))

        n2ls        = chem_eq.add('n2ls', ExecComp('out=P'), promotes=('P',))

        tp2props    = props.add('tp2props', ExecComp('out=P'), promotes=('P',))

        ideal_flow.connect('Ps', 'P')
        nozzle.connect('Ps_exhaust', 'ideal_flow.Ps')
        root.connect('Ps_exhaust', 'nozzle.Ps_exhaust')

        prob.setup(check=False)

        expected_targets = set(['nozzle.ideal_flow.chem_eq.n2ls.P',
                                'nozzle.press_calcs.Ps_exhaust',
                                'nozzle.ideal_flow.mach_calc.Ps',
                                'nozzle.ideal_flow.props.tp2props.P'])
        self.assertEqual(set(prob.root.connections), expected_targets)

        for tgt in expected_targets:
            self.assertTrue('desVars.Ps_exhaust' in prob.root.connections[tgt])

if __name__ == '__main__':
    unittest.main()
