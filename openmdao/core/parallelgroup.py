
from openmdao.core.group import Group

class ParallelGroup(Group):
    def apply_nonlinear(self, params, unknowns, resids):
        """ Evaluates the residuals of our children systems.

        Parameters
        ----------
        params : `VecWrapper`
            ``VecWrapper` ` containing parameters (p)

        unknowns : `VecWrapper`
            `VecWrapper`  containing outputs and states (u)

        resids : `VecWrapper`
            `VecWrapper`  containing residuals. (r)
        """

        # full scatter
        self._varmanager._transfer_data()

        for name, system in self.subsystems(local=True):
            view = self._views[name]
            system.apply_nonlinear(view.params, view.unknowns, view.resids)

    def children_solve_nonlinear(self):
        """Loops over our children systems and asks them to solve."""

        # full scatter
        self._varmanager._transfer_data()

        for name, system in self.subsystems(local=True):
            view = self._views[name]
            system.solve_nonlinear(view.params, view.unknowns, view.resids)
