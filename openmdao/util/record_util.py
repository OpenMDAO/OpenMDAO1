""" Utility functions related to recording or execution metadata. """
from six.moves import map, zip

from openmdao.core.mpi_wrap import MPI

def create_local_meta(metadata, name):
    """
    Creates the metadata dictionary for this level of execution.

    Args
    ----
    metadata : dict
        Dictionary containing the metadata passed from the parent level.

    name : str
        String to describe the current level of execution.
    """

    # Create new metadata if parent's isn't available
    if metadata is None:
        if MPI:
            coordinate = [MPI.COMM_WORLD.rank, name, (0,)]
        else:
            coordinate = [0, name, (0,)]
    else:
        coordinate = list(metadata['coord'])

        # The root group has no name, but we want the iteration coordinate to have one.
        if name == '' and len(coordinate) == 3:
            name = 'root'

        coordinate.extend([name, (0,)])

    return {
        'name': name,
        'coord': coordinate,
        'timestamp': None,
        'success': 1,
        'msg': '',
    }

def update_local_meta(local_meta, iteration):
    """
    Updates the local metadata dictionary to prepare for a new iteration.

    Args
    ----
    local_meta : dict
        Dictionary containing the execution metadata at the current level.

    parent : list
        List containing the parent iteration coordinate.

    iteration : tuple(int)
        Tuple of integers representing the current iteration and any sub-iterations.
    """
    # Construct needed structures
    iter_coord = local_meta['coord']

    # Last entry in the iteration coordinate should be the iteration number
    # for this level.
    if isinstance(iteration, int):
        iteration = (iteration,)

    iter_coord[-1] = iteration


def format_iteration_coordinate(coord):
    """
    Formats the iteration coordinate to a human-readable string.

    Args
    ----
    coord : list
        List containing the iteration coordinate.
    """

    iteration_coordinate = []

    for name, local_coord in zip(coord[1::2], coord[2::2]):
        iteration_coordinate.append(name)
        iteration_coordinate.append('-'.join(map(str, local_coord)))

    return ':'.join(["rank%d"%coord[0], '/'.join(iteration_coordinate)])
