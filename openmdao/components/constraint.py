
from openmdao.component.exprcomp import ExprComp

class ConstraintComp(ExprComp):
    def __init__(self, cnst):
        # transform constraint expr into 'x - y = 0' form
        # expr = ???
        super(ConstraintComp, self).__init__(expr)
    
