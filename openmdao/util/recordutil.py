"""
Utility functions related to recording or execution metadata.
"""

from six.moves import map, zip

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
        parent_coordinate = []
        parent_meta = {}
    else:
        parent_coordinate, parent_meta = metadata['current']

    # The root group has no name, but we want the iteration coordinate to have one.
    if len(parent_coordinate) == 2 and name == '':
        name = 'root'

    local_meta = {'name': name, 'coord': parent_coordinate + [name, (0,)]}

    # Update entry in parent metadata
    parent_meta[name] = local_meta

    return local_meta

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
    child_iteration = {}
    iter_coord = local_meta['coord']

    # Last entry in the iteration coordinate should be the iteration number for this level.
    iter_coord[-1] = iteration

    # Update local metadata with the given information
    local_meta[iteration] = child_iteration
    local_meta['current'] = (iter_coord, child_iteration)

def format_iteration_coordinate(coord):
    """
    Formats the iteration coordinate to a human-readable string.

    Args
    ----
    coord : list
        List containing the iteration coordinate
    """

    separator = '/'
    iteration_number_separator = '-'

    tmp = []

    for name, iter_coord in zip(coord[::2], coord[1::2]):
        tmp.append(name)
        tmp.append(iteration_number_separator.join((map(str, iter_coord))))

    return separator.join(tmp)
