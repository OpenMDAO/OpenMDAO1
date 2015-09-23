""" Utility functions related to recording or execution metadata. """
from six.moves import map, zip


class _ExecutionMetadata(object):
    def __init__(self, name='', coord=None, timestamp=None):
        self.name = name
        self.coord = coord
        self.timestamp = timestamp

    def __getitem__(self, key):
        try:
            return getattr(self, key)
        except AttributeError:
            msg = "KeyError: '{0}'.".format(key)
            raise KeyError(msg)

    def __setitem__(self, key, value):
        if not hasattr(self, key):
            msg = "KeyError: '{0}'.".format(key)
            raise KeyError(msg)
        setattr(self, key, value)


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
    else:
        parent_coordinate = metadata['coord']

    # The root group has no name, but we want the iteration coordinate to have one.
    if len(parent_coordinate) == 2 and name == '':
        name = 'root'

    local_meta = _ExecutionMetadata(name, parent_coordinate + [name, (0,)])

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

    separator = '/'
    iteration_number_separator = '-'

    iteration_coordinate = []

    for name, local_coord in zip(coord[::2], coord[1::2]):
        iteration_coordinate.append(name)
        iter_str = map(str, local_coord)
        coord_str = iteration_number_separator.join(iter_str)
        iteration_coordinate.append(coord_str)

    return separator.join(iteration_coordinate)
