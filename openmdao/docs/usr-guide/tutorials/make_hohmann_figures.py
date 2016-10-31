import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.patches as mpatches
import numpy as np
import matplotlib.pyplot as plt

def anomaly_mean_to_eccentric(M, e, tol=1.0E-12):
    """
    Given the mean anomaly and eccentricty of an orbit, iterate on and
    return the Eccentric anomaly.
    """
    E = 1.
    E0 = 0.
    for i in range(100):
        E = E0 + ((M+e*np.sin(E0)-E0)/(1-e*np.cos(E0)))
        if np.all(abs(E-E0) < tol):
            break
        E0 = E
    return E

def anomaly_eccentric_to_mean(E, e):
    """
    Given the eccentric anomaly, E, and eccentricity, e, return
    the mean anomaly
    """
    return E - e*np.sin(E)

def anomaly_true_to_eccentric(f, e):
    """
    Given the true anomaly (f) and eccentriciity (e) return the eccentric
    anomaly of the orbit.
    """
    return np.arctan2( np.sin(f) * np.sqrt(1-e**2) , ( e + np.cos(f) ) )

def anomaly_eccentric_to_true( E, e):
    """
    Given the eccentric anomaly of an orbit and its eccentricty, return the true anomaly.
    From Equation 4.2-12 in 'Fundamentals of Astrodynamics' by Bate, Mueller, and White
    """
    return 2*np.arctan2( np.sin(E/2.)*np.sqrt(1+e), np.cos(E/2.)*np.sqrt(1-e) )

def kep2cart(a,e,I,raan,argp,anom,mu,anom_type='true', stack=False):
    """
    Returns the cartesian coordinates of an object in this orbit at the
    specified anomaly.

    Reference:
        Fundamentals of Astrodynamics - Bate, Mueller, & White - 1971- Sec. 4.4-4.5
    """

    a = np.array(a)
    e = np.array(e)
    I = np.array(I)
    raan = np.array(raan)
    argp = np.array(argp)
    anom = np.array(anom)

    if anom_type == 'true':
        f = anom
        E = anomaly_true_to_eccentric( f, e)
    elif anom_type == 'mean':
        M = anom
        E = anomaly_mean_to_eccentric( M, e)
    elif anom_type == 'eccentric':
        E = anom
    else:
        raise ValueError('anomaly argument to kep2cart not understood: %s' % str(anom_type))

    Edot = np.sqrt(mu / a**3) / (1.0-e*np.cos(E))

    P = np.array( [ np.cos(argp)*np.cos(raan) - np.sin(argp) * np.cos(I) * np.sin(raan),
                   np.cos(argp)*np.sin(raan) + np.sin(argp) * np.cos(I) * np.cos(raan),
                   np.sin(argp)*np.sin(I) ])

    Q = np.array( [ -np.sin(argp)*np.cos(raan) - np.cos(argp) * np.cos(I) * np.sin(raan),
                   -np.sin(argp)*np.sin(raan) + np.cos(argp) * np.cos(I) * np.cos(raan),
                    np.sin(I)*np.cos(argp) ])

    P = P*np.ones([E.size,3])
    Q = Q*np.ones([E.size,3])

    A = a*(np.cos(E)-e)
    B = a*np.sqrt(1-e**2)*np.sin(E)
    C = -a*np.sin(E)*Edot
    D = a*np.sqrt(1-e**2)*np.cos(E)*Edot
    n = A.size

    A = A.reshape((1,n))
    B = B.reshape((1,n))
    C = C.reshape((1,n))
    D = D.reshape((1,n))

    r = (A.T*P).T + (B.T*Q).T
    v = (C.T*P).T + (D.T*Q).T

    if stack:
        rv = np.hstack((r.T,v.T))
        if n == 1:
            rv = np.ravel(rv)
        return rv

    return r.T,v.T

def plot_orbit(Xkep,Mu,fig=None,anom=None, proj='3d', **kwargs):
    sma, ecc, inc, raan, argp, mass = Xkep[:]
    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(111,aspect='equal',projection=proj,)
        ax.grid(True)

        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)

        x = 6378.137 * np.outer(np.cos(u), np.sin(v))
        y = 6378.137 * np.outer(np.sin(u), np.sin(v))
        z = 6378.137 * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='b', linewidth=0, alpha=0.2)
        fig.patch.set_visible(False)
        ax.patch.set_visible(False)
        ax.set_axis_off()
        ax._axis3don = False
    else:
        ax = fig.gca()

    if anom is None:
        anom = np.linspace(-np.pi,np.pi,100)

    orbit = kep2cart(sma,ecc,inc,raan,argp,anom=anom,mu=Mu,anom_type='true', stack=False)[0]
    #ax.plot(orbit[:,0],orbit[:,1],orbit[:,2])
    ax.plot(orbit[:,0],orbit[:,1],orbit[:,2], **kwargs)

    return fig


def main():
    Re = 6378.137
    mu = 0.3986592936294783e+15

    xkep_leo = np.array([ Re+400, 0.0, np.radians(28.5), 0.0, 0.0, 0.0])
    xkep_geo = np.array([ 41264.0, 0.0, np.radians(0.0), 0.0, 0.0, 0.0])

    rp = xkep_leo[0]
    ra = xkep_geo[0]
    a = (ra+rp)/2.0
    e = 1-rp/a

    xkep_xfer = np.array([ a, e, np.radians(15), 0.0, 0.0, 0.0])

    fig = plot_orbit(xkep_leo, mu, color='r', proj='3d', label='Low Earth Orbit (LEO)')
    fig = plot_orbit(xkep_geo, mu, fig, color='k', label='Geostationary Orbit (GEO)')
    fig = plot_orbit(xkep_xfer, mu, fig, anom=np.linspace(0,np.pi,100), color='b', ls='-', label='Geostationary Transfer Orbit (GTO)')

    ax = fig.gca()
    ax.set_xlim(-50000, 50000)
    ax.set_ylim(-50000, 50000)
    ax.set_zlim(-50000, 50000)
    ax.legend()
    ax.view_init(elev=90, azim=90,)
    ax.dist = 5

    ax.text(rp+9000, 0, 0, '$\Delta v_1$', fontsize=16)
    ax.text(-ra-3000, 0, 0, '$\Delta v_2$', fontsize=16)

    fig.savefig('hohmann_transfer.png')

    ######### Impulse 1 Diagram ############

    dv1_fig = plt.figure(tight_layout=True)

    ax = dv1_fig.add_subplot(111,aspect='equal')

    Re = 6378.137
    inc = np.radians(28.5)
    inc2 = np.radians(8)
    alt_leo = 400

    circle = mpatches.Circle((0,0), Re, ec="none", alpha=0.2)
    ax.add_patch(circle)

    # equator
    ax.plot([-Re, Re], [0.0, 0.0], 'k--', label='equator')

    # leo
    a = Re + alt_leo
    xs = np.array([ -a*np.cos(inc), a*np.cos(inc)])
    ys = np.array([ -a*np.sin(inc), a*np.sin(inc)])
    line1, = ax.plot(xs, ys, 'r-', label='LEO')

    # gto
    a2 = 10000
    xs = np.array([0, a2*np.cos(inc2)])
    ys = np.array([0, a2*np.sin(inc2)])
    line2, = ax.plot(xs, ys, 'b-', label='GTO')

    # vc1
    a3 = a*0.5
    ax.quiver([0], [0], a3*np.cos(inc), a3*np.sin(inc), scale=1, angles='xy', scale_units='xy')
    ax.text(a3*np.cos(inc)*0.3, a3*np.sin(inc)*0.3+600, '$v_c$', fontsize=16)

    # vp1
    a4 = a
    ax.quiver([0], [0], a4*np.cos(inc2), a4*np.sin(inc2), scale=1, angles='xy', scale_units='xy')
    ax.text(a4*np.cos(inc2)*0.4, a4*np.sin(inc2)*0.4+300, '$v_p$', fontsize=16)

    #angle_plot = get_angle_plot(line1, line2, offset = 1)
    ax.add_patch(mpatches.Arc((0,0), 14000, 14000, theta1=np.degrees(inc2), theta2=np.degrees(inc)))
    ax.text(7000*np.cos((inc+inc2)/2), 7000*np.sin((inc+inc2)/2), '$\Delta i_1$', fontsize=16)

    #dv1
    inc2 = np.radians(8)
    a5 = 5*2.0
    ax.quiver(a3*np.cos(inc), a3*np.sin(inc), 0.9*(a4*np.cos(inc2)-a3*np.cos(inc)), 0.9*(a4*np.sin(inc2)-a3*np.sin(inc)), scale=1, angles='xy', scale_units='xy')
    ax.text(a3*np.cos(inc)+800, a3*np.sin(inc), '$\Delta V_1$', fontsize=16)

    ax.set_xlim(-9000,9000)
    ax.set_ylim(-9000,9000)
    ax.legend()
    dv1_fig.patch.set_visible(False)
    ax.patch.set_visible(False)
    ax.set_axis_off()

    dv1_fig.savefig('hohmann_dv1.png')


    ########### Impulse 2 Diagram ##########

    dv2_fig = plt.figure(tight_layout=True)

    ax = dv2_fig.add_subplot(111,aspect='equal')

    Re = 6378.137
    inc = np.radians(0)
    inc2 = np.radians(-15)

    circle = mpatches.Circle((0,0), Re, ec="none", alpha=0.2)
    ax.add_patch(circle)

    # equator
    ax.plot([-Re, Re], [0.0, 0.0], 'k--', label='equator')

    # leo
    a = 42164
    xs = np.array([ -a*np.cos(inc), a*np.cos(inc)])
    ys = np.array([ -a*np.sin(inc), a*np.sin(inc)])
    ax.plot(xs, ys, 'k-', label='GEO')

    # gto
    a2 = 40000
    xs = np.array([-a2*np.cos(inc2), 0])
    ys = np.array([-a2*np.sin(inc2), 0])
    ax.plot(xs, ys, 'b-', label='GTO')

    # vc1
    a3 = a
    ax.quiver([0], [0], a3*np.cos(inc), a3*np.sin(inc), scale=1, angles='xy', scale_units='xy')
    ax.text(a3*np.cos(inc)*0.4, a3*np.sin(inc)*0.3+1200, '$v_c$', fontsize=16)

    # va
    a4 = a*0.4
    ax.quiver([0], [0], a4*np.cos(inc2), a4*np.sin(inc2), scale=1, angles='xy', scale_units='xy')
    ax.text(a4*np.cos(inc2)*0.4, a4*np.sin(inc2)*0.4-4000, '$v_a$', fontsize=16)

    ax.add_patch(mpatches.Arc((0,0), 80000, 80000, theta1=180+np.degrees(inc2), theta2=180))
    ax.text(-40000*np.cos((inc+inc2)/2), -40000*np.sin((inc+inc2)/2)-2000, '$\Delta i_2$', fontsize=16)

    #dv2
    a5 = 5*2.0
    ax.quiver(a4*np.cos(inc2), a4*np.sin(inc2), 0.8*(a3*np.cos(inc)-a4*np.cos(inc2)), 0.8*(a3*np.sin(inc)-a4*np.sin(inc2)), scale=1, angles='xy', scale_units='xy')
    ax.text(1.5*a4*np.cos(inc2), a4*np.sin(inc2)-2000, '$\Delta V_2$', fontsize=16)
    #

    #ax.grid(True)

    ax.set_xlim(-45000,45000)
    ax.set_ylim(-45000,45000)
    ax.legend()
    dv2_fig.patch.set_visible(False)
    ax.patch.set_visible(False)
    ax.set_axis_off()

    dv2_fig.savefig('hohmann_dv2.png')

    plt.show()


if __name__ == '__main__':
    main()