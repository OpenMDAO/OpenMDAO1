from __future__ import print_function
import sys
import numpy as np
from six import iteritems

from openmdao.core.system import System
from openmdao.core.component import Component
from openmdao.core.group import Group
from openmdao.core.problem import Problem

from openmdao.components.paramcomp import ParamComp


def _find_all_components(group):
    """recurse down through the hiearchy and find all components"""

    comps = list(group.components())
    name = group.name

    sub_groups = group.subgroups()
    for name, sg in sub_groups:
        sub_comps = _find_all_components(sg)
        sub_comps = [(c.pathname, c) for name, c in sub_comps]
        if sub_comps:
            comps.extend(sub_comps)

    return comps

def check_partial_derivs(prob, mode="fwd", out_stream=sys.stdout):
    """Recurses down the system tree to find all components, then checks
    the partial derivatives of each one against finite difference approximations

    Parameters
    ----------
    sys: `Problem`
        The Problem instance whith components to check partial derivatives of
    mode: {'fwd', 'rev'}, optional
        mode used to compute the analytic derivatives
    out_stream: file_like
        where to write report data (default is sys.stdout). Can be None, if no
        report is needed

    Returns
    -------
    dict
        nested dictionary of derivative errors. First level key is component name
        second level key is the (unknown,param) pair defining the derivative.
    """

    if isinstance(prob, Problem):
        root = prob.root
        comps = _find_all_components(root)

        group_params = root._params_dict
        group_unknowns = root._unknowns_dict

        conns = root._varmanager.connections

        for pathname, comp in comps:
            if isinstance(comp, ParamComp):
                continue # no need to check these

            comp_params = comp._params_dict
            comp_unknowns = comp._unknowns_dict

            for p_name, meta in iteritems(comp_params):
                u_name = conns[p_name]
                x_base = root[u_name].copy()
                x_shape = x_base.shape
                delta_x = x_base * 1.001


        return

    else:
        raise ValueError("first argument must be a of type Problem, "
                         "but got %s"%(str(type(system))))

if __name__ == "__main__":
    from openmdao.test.simplecomps import SimpleArrayComp, SimpleCompDerivJac


    p = Problem()
    p.root = root = Group()
    root.add('c1', SimpleArrayComp())
    g1 = root.add('g1', Group())
    g1.add('c2', SimpleArrayComp(), promotes=['*',])

    g2 = root.add('g2', Group())
    sg1 = g2.add('sg1', Group())
    sg1.add('c3', SimpleCompDerivJac())

    root.add('p1', ParamComp('p', 1*np.ones(2)))
    root.add('p2', ParamComp('p', 2*np.ones(2)))
    root.add('p3', ParamComp('p', 3.0))
    root.connect('p1:p','c1:x')
    root.connect('p2:p','g1:x')
    root.connect('p3:p','g2:sg1:c3:x')
    p.setup()

    check_partial_derivs(p)
