""" Find the minimum delta-V for a Hohmann Transer from Low Earth Orbit (LEO)
to GEostationary Orbit (GEO)
"""

import numpy as np
from openmdao.api import Problem, Group, Component, IndepVarComp, ExecComp, \
                         ScipyOptimizer


class VCircComp(Component):
    """ Computes the circular orbit velocity given a radius and gravitational
    parameter.
    """

    def __init__(self, radius=6378.14+400, mu=398600.4418):
        super(VCircComp, self).__init__()

        # Derivative specification
        self.deriv_options['type'] = 'user'

        self.deriv_options['check_type'] = 'cs'
        self.deriv_options['check_step_size'] = 1.0e-16

        self.add_param('r',
                       val=radius,
                       desc='Radius from central body',
                       units='km')

        self.add_param('mu',
                       val=mu,
                       desc='Gravitational parameter of central body',
                       units='km**3/s**2')

        self.add_output('vcirc',
                        val=1.0,
                        desc='Circular orbit velocity at given radius '
                             'and gravitational parameter',
                        units='km/s')

    def solve_nonlinear(self, params, unknowns, resids):
        r = params['r']
        mu = params['mu']

        unknowns['vcirc'] = np.sqrt(mu/r)

    def linearize(self, params, unknowns, resids):
        r = params['r']
        mu = params['mu']
        vcirc = unknowns['vcirc']

        J = {}
        J['vcirc', 'mu'] = 0.5/(r*vcirc)
        J['vcirc', 'r'] = -0.5*mu/(vcirc*r**2)
        return J


class DeltaVComp(Component):

    def __init__(self):
        super(DeltaVComp, self).__init__()

        # Derivative specification
        self.deriv_options['type'] = 'user'

        self.deriv_options['check_type'] = 'cs'
        self.deriv_options['check_step_size'] = 1.0e-16

        self.add_param('v1', val=1.0, desc='Initial velocity', units='km/s')
        self.add_param('v2', val=1.0, desc='Final velocity', units='km/s')
        self.add_param('dinc', val=1.0, desc='Plane change', units='rad')

        # Note:  We're going to use trigonometric functions on dinc.  The
        # automatic unit conversion in OpenMDAO comes in handy here.

        self.add_output('delta_v', val=0.0, desc='Delta-V', units='km/s')

    def solve_nonlinear(self, params, unknowns, resids):

        v1 = params['v1']
        v2 = params['v2']
        dinc = params['dinc']

        unknowns['delta_v'] = np.sqrt(v1**2 + v2**2 - 2.0*v1*v2*np.cos(dinc))

    def linearize(self, params, unknowns, resids):
        v1 = params['v1']
        v2 = params['v2']
        dinc = params['dinc']

        J = {}
        J['delta_v', 'v1'] = 0.5/unknowns['delta_v'] * (2*v1 - 2*v2*np.cos(dinc))
        J['delta_v', 'v2'] = 0.5/unknowns['delta_v'] * (2*v2 - 2*v1*np.cos(dinc))
        J['delta_v', 'dinc'] = 0.5/unknowns['delta_v'] * (2*v1*v2*np.sin(dinc))

        return J


class TransferOrbitComp(Component):

    def __init__(self):
        super(TransferOrbitComp, self).__init__()

        # Derivative specification
        self.deriv_options['type'] = 'fd'

        self.deriv_options['check_type'] = 'cs'
        self.deriv_options['check_step_size'] = 1.0e-16

        self.add_param('mu',
                       val=398600.4418,
                       desc='Gravitational parameter of central body',
                       units='km**3/s**2')
        self.add_param('rp', val=7000.0, desc='periapsis radius', units='km')
        self.add_param('ra', val=42164.0, desc='apoapsis radius', units='km')

        self.add_output('vp', val=0.0, desc='periapsis velocity', units='km/s')
        self.add_output('va', val=0.0, desc='apoapsis velocity', units='km/s')

    def solve_nonlinear(self, params, unknowns, resids):

        mu = params['mu']
        rp = params['rp']
        ra = params['ra']

        a = (ra+rp)/2.0
        e = (a-rp)/a
        p = a*(1.0-e**2)
        h = np.sqrt(mu*p)

        unknowns['vp'] = h/rp
        unknowns['va'] = h/ra


if __name__ == '__main__':

    prob = Problem(root=Group())

    root = prob.root

    root.add('mu_comp', IndepVarComp('mu', val=0.0, units='km**3/s**2'),
             promotes=['mu'])

    root.add('r1_comp', IndepVarComp('r1', val=0.0, units='km'),
             promotes=['r1'])
    root.add('r2_comp', IndepVarComp('r2', val=0.0, units='km'),
             promotes=['r2'])

    root.add('dinc1_comp', IndepVarComp('dinc1', val=0.0, units='deg'),
             promotes=['dinc1'])
    root.add('dinc2_comp', IndepVarComp('dinc2', val=0.0, units='deg'),
             promotes=['dinc2'])

    root.add('leo', system=VCircComp())
    root.add('geo', system=VCircComp())

    root.add('transfer', system=TransferOrbitComp())

    root.connect('r1', ['leo.r', 'transfer.rp'])
    root.connect('r2', ['geo.r', 'transfer.ra'])

    root.connect('mu', ['leo.mu', 'geo.mu', 'transfer.mu'])

    root.add('dv1', system=DeltaVComp())

    root.connect('leo.vcirc', 'dv1.v1')
    root.connect('transfer.vp', 'dv1.v2')
    root.connect('dinc1', 'dv1.dinc')

    root.add('dv2', system=DeltaVComp())

    root.connect('transfer.va', 'dv2.v1')
    root.connect('geo.vcirc', 'dv2.v2')
    root.connect('dinc2', 'dv2.dinc')

    root.add('dv_total', system=ExecComp('delta_v=dv1+dv2',
                                         units={'delta_v': 'km/s',
                                                'dv1': 'km/s',
                                                'dv2': 'km/s'}),
             promotes=['delta_v'])

    root.connect('dv1.delta_v', 'dv_total.dv1')
    root.connect('dv2.delta_v', 'dv_total.dv2')

    root.add('dinc_total', system=ExecComp('dinc=dinc1+dinc2',
                                           units={'dinc': 'deg',
                                                  'dinc1': 'deg',
                                                  'dinc2': 'deg'}),
             promotes=['dinc'])

    root.connect('dinc1', 'dinc_total.dinc1')
    root.connect('dinc2', 'dinc_total.dinc2')

    prob.driver = ScipyOptimizer()

    prob.driver.add_desvar('dinc1', lower=0, upper=28.5)
    prob.driver.add_desvar('dinc2', lower=0, upper=28.5)
    prob.driver.add_constraint('dinc', lower=28.5, upper=28.5, scaler=1.0)
    prob.driver.add_objective('delta_v', scaler=1.0)

    # Setup the problem

    prob.setup()

    # Set initial values

    prob['mu'] = 398600.4418
    prob['r1'] = 6778.137
    prob['r2'] = 42164.0

    prob['dinc1'] = 0.0
    prob['dinc2'] = 28.5

    # Use run_once to evaluate the model at the initial guess.
    # This will give us the :math:`\Delta V` for performing
    # the entire plane change at apogee.
    prob.run_once()
    dv_all_apogee = prob['delta_v']

    # Go!
    prob.run()

    print('Impulse 1:')
    print('    Delta-V: {0:6.4f} km/s'.format(prob['dv1.delta_v']))
    print('    Inclination Change: {0:6.4f} deg'.format(prob['dinc1']))
    print('Impulse 2:')
    print('    Delta-V: {0:6.4f} km/s'.format(prob['dv2.delta_v']))
    print('    Inclination Change: {0:6.4f} deg'.format(prob['dinc2']))
    print('Total Delta-V: {0:6.4f} km/s'.format(prob['delta_v']))
    print('Total Plane Change: {0:6.4f} deg'.format(prob['dinc']))
    print('\nPerforming the plane change at apogee gives a '
          'Delta-V of {0:6.4f} km/s'.format(dv_all_apogee))
