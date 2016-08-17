import unittest

from openmdao.api import Problem, Group, IndepVarComp, ExecComp, ScipyOptimizer

from openmdao.test.util import assert_rel_error

from openmdao.examples.beam_tutorial import BeamTutorial




class TestExamples(unittest.TestCase):



    def test_beam_tutorial_viewtree(self):

        top = Problem()
        top.root = BeamTutorial()

        top.driver = ScipyOptimizer()
        top.driver.options['optimizer'] = 'SLSQP'
        top.driver.options['tol'] = 1.0e-8
        top.driver.options['maxiter'] = 10000 #maximum number of solver iterations
        top.driver.options['disp'] = False

        #room length and width bounds
        top.driver.add_desvar('ivc_rlength.room_length', lower=5.0*12.0, upper=50.0*12.0) #domain: 1in <= length <= 50ft
        top.driver.add_desvar('ivc_rwidth.room_width', lower=5.0*12.0, upper=30.0*12.0) #domain: 1in <= width <= 30ft

        top.driver.add_objective('d_neg_area.neg_room_area') #minimize negative area (or maximize area)

        top.driver.add_constraint('d_len_minus_wid.length_minus_width', lower=0.0) #room_length >= room_width
        top.driver.add_constraint('d_deflection.deflection', lower=720.0) #deflection >= 720
        top.driver.add_constraint('d_bending.bending_stress_ratio', upper=0.5) #bending < 0.5
        top.driver.add_constraint('d_shear.shear_stress_ratio', upper=1.0/3.0) #shear < 1/3

        top.setup(check=False)
        from openmdao.api import view_tree
        view_tree(top, show_browser=False)
        import os.path

        self.assertTrue(os.path.isfile('partition_tree_n2.html'))
        os.remove('partition_tree_n2.html')




if __name__ == "__main__":
    unittest.main()
