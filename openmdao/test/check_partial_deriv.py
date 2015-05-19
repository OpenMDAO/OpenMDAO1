from __future__ import print_function
import sys
import numpy as np
import copy
from six import iteritems

from openmdao.core.system import System
from openmdao.core.component import Component
from openmdao.core.group import Group
from openmdao.core.problem import Problem
from openmdao.components.paramcomp import ParamComp


def pull_unknowns(prob):
    """grabs all the keys and values from an unknowns vector and returns a copy of it"""
    root = prob.root
    u_vars = root._unknowns_dict

    u_vals = {}
    for var_name, meta in iteritems(u_vars):
        rel_var_name = meta['relative_name']
        u_vals[rel_var_name] = copy.deepcopy(prob[rel_var_name])

    return u_vals

def push_unknowns(prob, u_vals):
    """pushes saved u vector values back into the problem"""
    root = prob.root
    u_vars = root._unknowns_dict

    for var_name, meta in iteritems(u_vars):
        rel_var_name = meta['relative_name']
        prob[rel_var_name] = u_vals[rel_var_name]
    pass

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
        prob.run()

        base_u_vals = pull_unknowns(prob)
        # push_unknowns(prob, base_u_vals)


        for pathname, comp in comps:
            if isinstance(comp, ParamComp):
                continue # no need to check these

            comp_params = comp._params_dict
            comp_unknowns = comp._unknowns_dict

            print(pathname)
            for p_name, meta in iteritems(comp_params):
                root_u_name = conns[p_name]
                x_base = root[root_u_name].copy()
                x_shape = x_base.shape
                delta_x = x_base * 1.001
                prob[root_u_name] = delta_x
                local_rel_p_name = comp_params[p_name]['relative_name']
                prob.run()
                for u_name, meta in iteritems(comp_unknowns):
                    global_rel_u_name = group_unknowns[u_name]['relative_name']
                    local_rel_u_name = comp_unknowns[u_name]['relative_name']
                    print("  ",local_rel_u_name, local_rel_p_name, prob[global_rel_u_name], base_u_vals[global_rel_u_name])
                push_unknowns(prob, base_u_vals)
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
