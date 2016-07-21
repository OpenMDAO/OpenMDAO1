import os
from openmdao.api import Problem
from examples.sellar_state_MDF_optimize import SellarStateConnection
from openmdao.api import view_tree

top = Problem()
top.root = SellarStateConnection()

top.setup(check=False)
current_dir = os.path.dirname(os.path.abspath(__file__))
view_tree(top, show_browser=False, offline=False, embed=True, outfile=os.path.join( current_dir, 'n2_sellar_state.html'))
