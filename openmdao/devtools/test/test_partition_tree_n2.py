import os.path
import unittest
from tempfile import mkdtemp

from openmdao.api import Problem, Group, IndepVarComp, ExecComp, ScipyOptimizer, SqliteRecorder
from openmdao.api import view_model

from openmdao.test.util import assert_rel_error

from openmdao.examples.beam_tutorial import BeamTutorial

class TestExamples(unittest.TestCase):

    def test_beam_tutorial_viewmodel(self):

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
        view_model(top, show_browser=False)

        self.assertTrue(os.path.isfile('partition_tree_n2.html'))
        os.remove('partition_tree_n2.html')

    def test_beam_tutorial_viewmodel_using_data_from_sqlite_case_recorder_file(self):

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

        tempdir = mkdtemp()
        case_recorder_filename = "tmp.sql"
        filename = os.path.join(tempdir, case_recorder_filename)
        recorder = SqliteRecorder(filename)
        top.driver.add_recorder(recorder)

        top.setup(check=False)

        top.run()
        view_model(filename, show_browser=False)

        self.assertTrue(os.path.isfile('partition_tree_n2.html'))
        os.remove('partition_tree_n2.html')


    def test_beam_tutorial_viewmodel_using_data_from_hdf5_case_recorder_file(self):

        SKIP = False
        try:
            from openmdao.recorders.hdf5_recorder import HDF5Recorder
            import h5py
        except ImportError:
            # Necessary for the file to parse
            from openmdao.recorders.base_recorder import BaseRecorder
            HDF5Recorder = BaseRecorder
            SKIP = True

        if SKIP:
            raise unittest.SkipTest("Could not import HDF5Recorder. Is h5py installed?")

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

        tempdir = mkdtemp()
        case_recorder_filename = "tmp.hdf5"
        filename = os.path.join(tempdir, case_recorder_filename)
        recorder = HDF5Recorder(filename)
        top.driver.add_recorder(recorder)

        top.setup(check=False)

        top.run()
        view_model(filename, show_browser=False)

        self.assertTrue(os.path.isfile('partition_tree_n2.html'))
        os.remove('partition_tree_n2.html')

    def test_viewmodel_using_bogus_recorder_file_type(self):

        tempdir = mkdtemp()
        case_recorder_filename = "tmp.bogus"
        filename = os.path.join(tempdir, case_recorder_filename)
        # Just make an empty file
        open(filename, 'a').close()

        try:
            view_model(filename, show_browser=False)
        except Exception as err:
            self.assertEqual(str(err),
                "The given filename is not one of the supported file formats: sqlite or hdf5")

if __name__ == "__main__":
    unittest.main()
