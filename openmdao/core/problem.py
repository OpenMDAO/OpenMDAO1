
from openmdao.core.system import System

class Problem(System):
    def __init__(self, root=None, driver=None):
        self.root = root
        self.driver = driver

    def run(self):
        pass
