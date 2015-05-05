
from openmdao.core.component import Component

class ExprComp(Component):
    def __init__(self, expr):
        super(ExprComp, self).__init__()

        # by default, promote all vars to parent
        self._promotes = ('*',)

