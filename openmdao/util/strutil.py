


def get_common_ancestor(name1, name2):
    """
    Returns
    -------
    str
        Absolute name of any common ancestor `System` containing
        both name1 and name2.  If none is found, returns ''.
    """
    common_parts = []
    for part1, part2 in zip(name1.split(':'), name2.split(':')):
        if part1 == part2:
            common_parts.append(part1)
        else:
            break

    if common_parts:
        return ':'.join(common_parts)
    else:
        return ''

