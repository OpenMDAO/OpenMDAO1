"""
Cokriging example from [Forrester 2007] to show
MultiFiMetaModel and MultiFiCoKrigingSurrogate usage
"""

import numpy as np

from openmdao.api import Component, Group, Problem, MultiFiMetaModel, MultiFiCoKrigingSurrogate, KrigingSurrogate

def model_hifi(x):
    return ((6*x-2)**2)*np.sin((6*x-2)*2)

def model_lofi(x):
    return 0.5*((6*x-2)**2)*np.sin((6*x-2)*2)+(x-0.5)*10. - 5

class Simulation(Group):

    def __init__(self, surrogate, nfi):
        super(Simulation, self).__init__()
        self.surrogate = surrogate

        mm = self.add("mm", MultiFiMetaModel(nfi=nfi))
        mm.add_param('x', val=0.)
        mm.add_output('f_x', val=(0.,0.), surrogate=surrogate)

if __name__ == "__main__":

    # Co-kriging with 2 levels of fidelity
    surrogate = MultiFiCoKrigingSurrogate()
    pbcok = Problem(Simulation(surrogate, nfi=2))
    pbcok.setup(check=False)

    doe_e = [0.0, 0.4, 0.6, 1.0]
    doe_c = [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9] + doe_e

    pbcok['mm.train:x'] = np.array(doe_e).reshape(len(doe_e),1)
    pbcok['mm.train:x_fi2'] = np.array(doe_c).reshape(len(doe_c),1)
    pbcok['mm.train:f_x'] = model_hifi(pbcok['mm.train:x'])
    pbcok['mm.train:f_x_fi2'] = model_lofi(pbcok['mm.train:x_fi2'])

    # train
    pbcok.run()

    ngrid = 100
    pred_cok = []
    inputs = np.linspace(0, 1, ngrid)
    for x in inputs:
        pbcok['mm.x'] = x
        pbcok.run()
        pred_cok.append(pbcok['mm.f_x'])

    pbcok_mu    = np.array([float(p[0]) for p in pred_cok])
    pbcok_sigma = np.array([float(p[1]) for p in pred_cok])

    ## "Co-kriging" with 1 level of fidelity a.k.a. kriging
    surrogate = MultiFiCoKrigingSurrogate()
    ## Kriging from openmdao
    #surrogate = KrigingSurrogate()

    pbk = Problem(Simulation(surrogate, nfi=1))
    pbk.setup()

    pbk['mm.train:x'] = np.array(doe_e).reshape(len(doe_e),1)
    pbk['mm.train:f_x'] = model_hifi(pbk['mm.train:x'])

    pbk.run() # train

    ngrid = 100
    pred_k = []
    inputs = np.linspace(0, 1, ngrid)
    for x in inputs:
        pbk['mm.x'] = x
        pbk.run()
        pred_k.append(pbk['mm.f_x'])

    pbk_mu    = np.array([float(p[0]) for p in pred_k])
    pbk_sigma = np.array([float(p[1]) for p in pred_k])

    check  = inputs
    actual = model_hifi(inputs)

    import pylab as plt

    plt.figure(2)
    plt.plot(check, actual, 'k', label='True f')
    plt.plot(doe_e, model_hifi(np.array(doe_e)),'ok',label="High Fi")
    plt.plot(doe_c, model_lofi(np.array(doe_c)),'or',label="Low Fi")
    plt.plot(check, pbcok_mu, 'g', label='Co-kriging')
    plt.plot(check, pbcok_mu + 2*pbcok_sigma, 'g', alpha=0.5, label='I95%')
    plt.plot(check, pbcok_mu - 2*pbcok_sigma, 'g', alpha=0.5)
    plt.fill_between(check, pbcok_mu + 2*pbcok_sigma,
                            pbcok_mu - 2*pbcok_sigma, facecolor='g', alpha=0.2)
    plt.plot(check, pbk_mu, 'b', label='Kriging')
    plt.plot(check, pbk_mu + 2*pbk_sigma, 'b', alpha=0.5, label='I95%')
    plt.plot(check, pbk_mu - 2*pbk_sigma, 'b', alpha=0.5)
    plt.fill_between(check, pbk_mu + 2*pbk_sigma,
                            pbk_mu - 2*pbk_sigma, facecolor='b', alpha=0.2)

    plt.legend(loc='best')
    plt.show()

    # RMSE CoKriging
    error = 0.
    for a,p in zip(actual, pbcok_mu):
        error += (a-p)**2
    error = (error/len(actual))
    print("RMSE Cokriging = %g" % error)

    # RMSE Kriging
    error = 0.
    for a,p in zip(actual, pbk_mu):
        error += (a-p)**2
    error = (error/len(actual))
    print("RMSE Kriging = %g" % error)
