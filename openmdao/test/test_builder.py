"""
Various functions to make it easier to build test models.
"""
from __future__ import print_function

import numpy

from openmdao.core.group import Group
from openmdao.core.component import Component
from openmdao.test.exec_comp_for_test import ExecComp4Test

class ABCDArrayComp(ExecComp4Test):
    def __init__(self, size, **kwargs):
        super(ABCDArrayComp, self).__init__(['c = a*1.1','d=b*0.9'],
                                            nl_delay=kwargs.get('nl_delay',.01),
                                            lin_delay=kwargs.get('lin_delay',.01),
                                            trace=kwargs.get('trace',False),
                                            a=numpy.zeros(size),
                                            b=numpy.zeros(size),
                                            c=numpy.zeros(size),
                                            d=numpy.zeros(size))

#class Summer()

def _child_name(child, i):
    if isinstance(child, Group):
        return 'G%d'%i
    return 'C%d'%i

def build_sequence(child_factory, num_children, conns=(), parent=None):
    if parent is None:
        parent = Group()

    cnames = []
    for i in range(num_children):
        child = child_factory()
        cname = _child_name(child, i)
        parent.add(cname, child)
        if i:
            for u,v in conns:
                parent.connect('.'.join((cnames[-1],u)),
                               '.'.join((cname,v)))
        cnames.append(cname)

    return parent


if __name__ == '__main__':
    from openmdao.core.problem import Problem
    from openmdao.devtools.debug import stats
    vec_size = 100000
    num_comps = 50

    g = build_sequence(lambda: ABCDArrayComp(vec_size),
                       num_comps, [('c', 'a'),('d','b')])
    p = Problem(root=g)
    p.setup()
    p.run()
    #g.dump(verbose=True)
    stats(g)
