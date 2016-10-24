from abc import ABCMeta, abstractmethod, abstractproperty


class CaseReaderBase(object):
    """ The Abstract base class of all CaseReader implementations. """

    __metaclass__ = ABCMeta

    def __init__(self, filename):
        self._format_version = None
        self._filename = filename
        self._parameters = None
        self._unknowns = None
        self._case_keys = ()

    @abstractmethod
    def get_case(self, case_id):
        """
        Parameters
        ----------
        case_id : str or int
            If int, the index of the case to be read in the case iterations.
            If given as a string, it is the identifier of the case.

        Returns
        -------
        Case
            The case from the recorded file with the given identifier or index.

        """
        pass

    def list_cases(self):
        """ Return a tuple of the case string identifiers available in this
        instance of the CaseReader.
        """
        return self._case_keys

    @property
    def filename(self):
        """ The name of the file from which the case was created. """
        return self._filename

    @property
    def format_version(self):
        """ The format version used in the given file. """
        return self._format_version

    @property
    def parameters(self):
        """ The parameters metadata in the given file, or None if
        it is not present.
        """
        return self._parameters

    @property
    def unknowns(self):
        """ The unknowns metadata in the given file, or None if it
        is not present.
        """
        return self._unknowns

    @property
    def num_cases(self):
        """ The number of cases in the recorded file. """
        return len(self.list_cases())
