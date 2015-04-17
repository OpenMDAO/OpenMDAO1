
from openmdao.core.group import Group

class Assembly(Group):

    def add(name, system, promotes=None):
        # check name validity
        # ??
        system.name = name
        if promotes is not None:
            system.promotes = promotes
