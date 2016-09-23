from abc import ABCMeta, abstractmethod, abstractproperty


class CaseReaderBase(object):
    """ The Abstract base class of all CaseReader implementations.



    """

    __metaclass__ = ABCMeta

    def __init__(self, filename):
        self._filename = filename
        self._parameters = None
        self._unknowns = None
        self._format_version = None
        self._case_keys = ()

    @abstractmethod
    def get_case(self, case_id):
        pass

    @abstractmethod
    def list_cases(self):
        pass

    @property
    def filename(self):
        return self._filename

    @property
    def format_version(self):
        """
        Returns
        -------
        The format version used in the given file.

        """
        return self._format_version

    @property
    def parameters(self):
        """
        Returns
        -------
        The parameters metadata in the given file, or None if
        it is not present.

        """
        return self._parameters

    @property
    def unknowns(self):
        """

        Returns
        -------
        The unknowns metadata in the given file, or None if it is not present.

        """
        return self._unknowns

    @property
    def num_cases(self):
        return len(self.list_cases())
