""" Beam sizing problem"""


from openmdao.api import Problem, ScipyOptimizer, Component, IndepVarComp, Group


#room_area = room_length * room_width                                (1)
#room_length >= room_width                                   (2)
#(29000000 * 228 * 384) / {5 * [(0.24305 * room_width)  + 4.83] * (room_length)3}>= 720  (3)
#(0.5*8.75) * [(0.24305 * room_width)  + 4.83] * (room_length)2 / (8 * 50,000 * 228) < 0.5   (4)
#0.5 * [(0.24305 * room_width)  + 4.83] * (room_length) / (17.1*50,000) < 1/3            (5)

#constants
E = 29000000 #modulus of elasticity (constant 29000000psi for ASTM A992 Grade 50 steel)
I = 228 #Ix = moment of Inertia (constant 228in4 for the W8x58 beam)
BEAM_WEIGHT_LBS_PER_IN = 58.0 / 12.0 #self weight of beam per unit length (58 lbs/ft or 4.83 lbs/in.)
DEAD_LOAD_PSI = 20.0 / 144 #The dead load is 20psf or 0.1389psi.
LIVE_LOAD_PSI = 50.0 / 144 #The live load is 50psf or 0.3472psi.
TOTAL_LOAD_PSI = DEAD_LOAD_PSI + LIVE_LOAD_PSI #total load
BEAM_HEIGHT_IN = 8.75 #inches
YIELD_STRENGTH_PSI = 50000 #The maximum yield strength Fy for ASTM A992 Grade 50 steel is 50,000 psi
CROSS_SECTIONAL_AREA_SQIN = 17.1 #sq in



#negate the area to turn from a maximization problem to a minimization problem
class NegativeArea(Component):

    def __init__(self):
        super(NegativeArea, self).__init__()

        self.add_param('room_width', val=0.0)
        self.add_param('room_length', val=0.0)
        self.add_output('neg_room_area', val=0.0)

    def solve_nonlinear(self, params, unknowns, resids):

        room_width = params['room_width']
        room_length = params['room_length']

        unknowns['neg_room_area'] = -(room_length * room_width)

    def linearize(self, params, unknowns, resids):
        J = {}

        room_width = params['room_width']
        room_length = params['room_length']

        J['neg_room_area','room_width'] = -room_length
        J['neg_room_area','room_length'] = -room_width

        return J


class LengthMinusWidth(Component):

    def __init__(self):
        super(LengthMinusWidth, self).__init__()

        self.add_param('room_width', val=0.0)
        self.add_param('room_length', val=0.0)
        self.add_output('length_minus_width', val=0.0)

    def solve_nonlinear(self, params, unknowns, resids):

        room_width = params['room_width']
        room_length = params['room_length']

        unknowns['length_minus_width'] = room_length - room_width

    def linearize(self, params, unknowns, resids):
        J = {}

        room_width = params['room_width']
        room_length = params['room_length']

        J['length_minus_width','room_width'] = -1.0
        J['length_minus_width','room_length'] = 1.0

        return J

class Deflection(Component):

    def __init__(self):
        super(Deflection, self).__init__()

        self.add_param('room_width', val=0.0)
        self.add_param('room_length', val=0.0)
        self.add_output('deflection', val=0.0)

    def solve_nonlinear(self, params, unknowns, resids):

        room_width = params['room_width']
        room_length = params['room_length']

        unknowns['deflection'] = (E * I * 384.0) / (5.0 * ((0.5 * TOTAL_LOAD_PSI * room_width)  + BEAM_WEIGHT_LBS_PER_IN) * room_length**3)


    def linearize(self, params, unknowns, resids):
        J = {}

        room_width = params['room_width']
        room_length = params['room_length']

        J['deflection','room_width'] = (-192.0 * E * I * TOTAL_LOAD_PSI) / ((5.0 * room_length**3) * (TOTAL_LOAD_PSI * room_width/2.0 + BEAM_WEIGHT_LBS_PER_IN)**2)
        J['deflection','room_length'] = (-1152.0 * E * I) / (5.0 * ((TOTAL_LOAD_PSI * room_width)/2.0 + BEAM_WEIGHT_LBS_PER_IN) * room_length**4)

        return J


class BendingStress(Component):

    def __init__(self):
        super(BendingStress, self).__init__()

        self.add_param('room_width', val=0.0)
        self.add_param('room_length', val=0.0)
        self.add_output('bending_stress_ratio', val=0.0)

    def solve_nonlinear(self, params, unknowns, resids):

        room_width = params['room_width']
        room_length = params['room_length']

        unknowns['bending_stress_ratio'] = (0.5*BEAM_HEIGHT_IN * ((0.5 * TOTAL_LOAD_PSI * room_width)  + BEAM_WEIGHT_LBS_PER_IN) * (room_length)**2) / (8.0 * YIELD_STRENGTH_PSI * I)

    def linearize(self, params, unknowns, resids):
        J = {}

        room_width = params['room_width']
        room_length = params['room_length']

        J['bending_stress_ratio','room_width'] = (room_length**2) * BEAM_HEIGHT_IN * (TOTAL_LOAD_PSI*room_width/2.0 + BEAM_WEIGHT_LBS_PER_IN) / (16.0 * I * YIELD_STRENGTH_PSI)
        J['bending_stress_ratio','room_length'] = (BEAM_WEIGHT_LBS_PER_IN + (TOTAL_LOAD_PSI*room_width/2.0)) * BEAM_HEIGHT_IN * room_length / (8.0 * I * YIELD_STRENGTH_PSI)

        return J

class ShearStress(Component):

    def __init__(self):
        super(ShearStress, self).__init__()

        self.add_param('room_width', val=0.0)
        self.add_param('room_length', val=0.0)
        self.add_output('shear_stress_ratio', val=0.0)

    def solve_nonlinear(self, params, unknowns, resids):

        room_width = params['room_width']
        room_length = params['room_length']

        unknowns['shear_stress_ratio'] = 0.5 * ((0.5 * TOTAL_LOAD_PSI * room_width)  + BEAM_WEIGHT_LBS_PER_IN) * (room_length) / (CROSS_SECTIONAL_AREA_SQIN * YIELD_STRENGTH_PSI)

    def linearize(self, params, unknowns, resids):
        J = {}

        room_width = params['room_width']
        room_length = params['room_length']

        J['shear_stress_ratio','room_width'] = TOTAL_LOAD_PSI * room_length / (4.0 * YIELD_STRENGTH_PSI * CROSS_SECTIONAL_AREA_SQIN)
        J['shear_stress_ratio','room_length'] = (BEAM_WEIGHT_LBS_PER_IN + (TOTAL_LOAD_PSI * room_width / 2.0))/(2.0 * YIELD_STRENGTH_PSI * CROSS_SECTIONAL_AREA_SQIN)

        return J


class BeamTutorial(Group):

    def __init__(self):
        super(BeamTutorial, self).__init__()

        #add design variables or IndepVarComp's
        self.add('ivc_rlength', IndepVarComp('room_length', 100.0))
        self.add('ivc_rwidth', IndepVarComp('room_width', 100.0))

        #add our custom components
        self.add('d_len_minus_wid', LengthMinusWidth())
        self.add('d_deflection', Deflection())
        self.add('d_bending', BendingStress())
        self.add('d_shear', ShearStress())
        self.add('d_neg_area', NegativeArea())

        #make connections from design variables to the Components
        self.connect('ivc_rlength.room_length','d_len_minus_wid.room_length')
        self.connect('ivc_rwidth.room_width','d_len_minus_wid.room_width')

        self.connect('ivc_rlength.room_length','d_deflection.room_length')
        self.connect('ivc_rwidth.room_width','d_deflection.room_width')

        self.connect('ivc_rlength.room_length','d_bending.room_length')
        self.connect('ivc_rwidth.room_width','d_bending.room_width')

        self.connect('ivc_rlength.room_length','d_shear.room_length')
        self.connect('ivc_rwidth.room_width','d_shear.room_width')

        self.connect('ivc_rlength.room_length','d_neg_area.room_length')
        self.connect('ivc_rwidth.room_width','d_neg_area.room_width')

if __name__ == "__main__":
    top = Problem()
    top.root = BeamTutorial()

    top.driver = ScipyOptimizer()
    top.driver.options['optimizer'] = 'SLSQP'
    top.driver.options['tol'] = 1.0e-8
    top.driver.options['maxiter'] = 10000 #maximum number of solver iterations

    #room length and width bounds
    top.driver.add_desvar('ivc_rlength.room_length', lower=5.0*12.0, upper=50.0*12.0) #domain: 1in <= length <= 50ft
    top.driver.add_desvar('ivc_rwidth.room_width', lower=5.0*12.0, upper=30.0*12.0) #domain: 1in <= width <= 30ft

    top.driver.add_objective('d_neg_area.neg_room_area') #minimize negative area (or maximize area)

    top.driver.add_constraint('d_len_minus_wid.length_minus_width', lower=0.0) #room_length >= room_width
    top.driver.add_constraint('d_deflection.deflection', lower=720.0) #deflection >= 720
    top.driver.add_constraint('d_bending.bending_stress_ratio', upper=0.5) #bending < 0.5
    top.driver.add_constraint('d_shear.shear_stress_ratio', upper=1.0/3.0) #shear < 1/3

    top.setup()
    top.run()

    print("\n")
    print( "Solution found")
    print("room area: %f in^2 (%f ft^2)" % (-top['d_neg_area.neg_room_area'], -top['d_neg_area.neg_room_area']/144.0))
    print("room width: %f in (%f ft)" % (top['ivc_rwidth.room_width'], top['ivc_rwidth.room_width']/12.0))
    print("room/beam length: %f in (%f ft)" % (top['ivc_rlength.room_length'], top['ivc_rlength.room_length']/12.0))
    print( "deflection: L/%f"  % (top['d_deflection.deflection']))
    print( "bending stress ratio: %f"  % (top['d_bending.bending_stress_ratio']))
    print( "shear stress ratio: %f"  % (top['d_shear.shear_stress_ratio']))

    loadingPlusBeam = ((0.5 * TOTAL_LOAD_PSI * top['ivc_rwidth.room_width']) + BEAM_WEIGHT_LBS_PER_IN) #PLI (pounds per linear inch)
    loadingNoBeam = ((0.5 * TOTAL_LOAD_PSI * top['ivc_rwidth.room_width'])) #PLI (pounds per linear inch)
    print( "loading (including self weight of beam): %fpli %fplf"  % (loadingPlusBeam, loadingPlusBeam*12.0))
    print( "loading (not including self weight of beam): %fpli %fplf"  % (loadingNoBeam, loadingNoBeam*12.0))
    print( "Finished!")
