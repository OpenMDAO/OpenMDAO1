import shelve
from openmdao.recorders.baserecorder import BaseRecorder
from openmdao.util.recordutil import format_iteration_coordinate


class ShelveRecorder(BaseRecorder):
    """
    A recorder that stores data using Python's shelve.

    Args
    ----
    out : str
        String containing the filename for the shelve file.

    **shelve_args
        Additional keyword args to be passed to shelve.open().
    """

    def __init__(self, out, **shelve_args):
        super(ShelveRecorder, self).__init__()
        self.out = shelve.open(out, **shelve_args)

    def record(self, params, unknowns, resids, metadata):
        """
        Stores the provided data in the shelve file using the iteration
        coordinate for the key.

        Args
        ----
        params : dict
            Dictionary containing parameters. (p)

        unknowns : dict
            Dictionary containing outputs and states. (u)

        resids : dict
            Dictionary containing residuals. (r)

        metadata : dict, optional
            Dictionary containing execution metadata (e.g. iteration coordinate).
        """

        iteration_coordinate = metadata['coord']
        group_name = format_iteration_coordinate(iteration_coordinate)

        f = self.out

        groupings = (
            ("/Parameters/", params),
            ("/Unknowns/", unknowns),
            ("/Residuals/", resids),
        )

        for label, values in groupings:
            local_name = group_name + label
            for key, val in values.items():
                f[local_name + key] = val
