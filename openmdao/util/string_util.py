""" Sting utilities. """

import ast

#public symbols
__all__ = ['get_common_ancestor', 'nearest_child', 'name_relative_to', 'parse_for_vars']

def get_common_ancestor(name1, name2):
    """
    Args
    ----
    name1 : str
        Name of first system.

    name2 : str
        Name of second system.

    Returns
    -------
    str
        Absolute name of any common ancestor `System` containing
        both name1 and name2.  If none is found, returns ''.
    """
    common_parts = []
    for part1, part2 in zip(name1.split('.'), name2.split('.')):
        if part1 == part2:
            common_parts.append(part1)
        else:
            break

    if common_parts:
        return '.'.join(common_parts)
    else:
        return ''

def nearest_child(parent_abspath, child_abspath):
    """ Determine the relative name of the nearest child inside of the
    parent that contains or is the object specified by child_abspath.

    Args
    ----
    parent_abspath : str
        Asbolute path of the parent system.

    child_abspath : str
        Absolute path of the child variable or system.

    Returns
    -------
    str
        Name of the child relative to the parent.
    """
    start = len(parent_abspath)+1 if parent_abspath else 0
    return child_abspath[start:].split('.', 1)[0]

def name_relative_to(parent_abspath, child_abspath):
    """ Determine the relative name of a child path with respect to a parent
    system.

    Args
    ----
    parent_abspath : str
        Asbolute path of the parent.

    child_abspath : str
        Absolute path of the child.

    Returns
    -------
    str
        Name of the child relative to the parent.
    """
    start = len(parent_abspath)+1 if parent_abspath else 0
    return child_abspath[start:]

class ExprVarScanner(ast.NodeVisitor):
    """
    This node visitor collects all variable names found in the
    AST, and excludes names of functions.  Variables having
    dotted names are not supported.
    """
    def __init__(self, vnames=()):
        self.varnames = set()
        self._lookfor = vnames

    def visit_Name(self, node):
        self.varnames.add(node.id)

    def visit_Call(self, node):
        if not isinstance(node.func, ast.Name):
            self.visit(node.func)
        for arg in node.args:
            self.visit(arg)

    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Name) and node.value.id in self._lookfor:
            self.varnames.add(node.value.id)

def parse_for_vars(expr, vnames=()):
    """
    Args
    ----
    expr : str
        An expression string that we want to parse for variable names.

    Returns
    -------
    list of str
        Names of variables from the given string.
    """
    root = ast.parse(expr, mode='exec')
    scanner = ExprVarScanner(vnames)
    scanner.visit(root)
    return scanner.varnames
