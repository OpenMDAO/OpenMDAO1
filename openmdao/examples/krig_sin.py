from __future__ import print_function

import numpy as np

from openmdao.api import Group, Problem, MetaModel, \
    KrigingSurrogate, FloatKrigingSurrogate

class TrigMM(Group):
    ''' FloatKriging gives responses as floats '''

    def __init__(self):
        super(TrigMM, self).__init__()

        # Create meta_model for f_x as the response
        sin_mm = self.add("sin_mm", MetaModel())
        sin_mm.add_param('x', val=0.)
        sin_mm.add_output('f_x:float', val=0., surrogate=FloatKrigingSurrogate())
        sin_mm.add_output('f_x:norm_dist', val=(0.,0.), surrogate=KrigingSurrogate())


if __name__ == '__main__':

    prob = Problem()
    prob.root = TrigMM()
    prob.setup()

    #traning data is just set manually. No connected input needed, since
    #  we're assuming the data is pre-existing
    prob['sin_mm.train:x'] = np.linspace(0,10,20)
    prob['sin_mm.train:f_x:float'] = np.sin(prob['sin_mm.train:x'])
    prob['sin_mm.train:f_x:norm_dist'] = np.cos(prob['sin_mm.train:x'])

    prob['sin_mm.x'] = 2.1 #prediction happens at this value
    prob.run()
    print('float predicted:', '%3.4f'%prob['sin_mm.f_x:float']) #predicted value
    print('float actual: ', '%3.4f'%np.sin(2.1))
    print('norm_dist predicted:', '%3.4f,'%prob['sin_mm.f_x:norm_dist'][0], '%3.4e'%prob['sin_mm.f_x:norm_dist'][1]) #predicted value
    print('norm_dist actual: ', '%3.4f'%np.cos(2.1))
