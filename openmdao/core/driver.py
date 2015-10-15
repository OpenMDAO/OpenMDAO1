""" Base class for Driver."""

from __future__ import print_function

from collections import OrderedDict
from itertools import chain
from six import iteritems
import warnings
import itertools
import numpy as np

from openmdao.core.mpi_wrap import MPI
from openmdao.core.options import OptionsDictionary
from openmdao.recorders.recording_manager import RecordingManager
from openmdao.util.record_util import create_local_meta, update_local_meta


class Driver(object):
    """ Base class for drivers in OpenMDAO. Drivers can only be placed in a
    Problem, and every problem has a Driver. Driver is the simplest driver that
    runs (solves using solve_nonlinear) a problem once.
    """

    def __init__(self):
        super(Driver, self).__init__()
        self.recorders = RecordingManager()

        # What this driver supports
        self.supports = OptionsDictionary(read_only=True)
        self.supports.add_option('inequality_constraints', True)
        self.supports.add_option('equality_constraints', True)
        self.supports.add_option('linear_constraints', True)
        self.supports.add_option('multiple_objectives', True)
        self.supports.add_option('two_sided_constraints', True)
        self.supports.add_option('integer_design_vars', True)

        # This driver's options
        self.options = OptionsDictionary()

        self._desvars = OrderedDict()
        self._objs = OrderedDict()
        self._cons = OrderedDict()

        self._voi_sets = []
        self._vars_to_record = None

        # We take root during setup
        self.root = None

        self.iter_count = 0
        self.dv_conversions = {}
        self.fn_conversions = {}

    def _setup(self, root):
        """ Updates metadata for params, constraints and objectives, and
        check for errors. Also determines all variables that need to be
        gathered for case recording.
        """
        self.root = root

        desvars = OrderedDict()
        objs = OrderedDict()
        cons = OrderedDict()

        item_tups = [
            ('Parameter', self._desvars, desvars),
            ('Objective', self._objs, objs),
            ('Constraint', self._cons, cons)
        ]

        for item_name, item, newitem in item_tups:
            for name, meta in iteritems(item):
                rootmeta = root.unknowns.metadata(name)

                if MPI and 'src_indices' in rootmeta: # pragma: no cover
                    raise ValueError("'%s' is a distributed variable and may "
                                     "not be used as a design var, objective, "
                                     "or constraint." % name)

                # Check validity of variable
                if name not in root.unknowns:
                    msg = "{} '{}' not found in unknowns."
                    msg = msg.format(item_name, name)
                    raise ValueError(msg)

                # Size is useful metadata to save
                if 'indices' in meta:
                    meta['size'] = len(meta['indices'])
                else:
                    meta['size'] = rootmeta['size']

                newitem[name] = meta

        self._desvars = desvars
        self._objs = objs
        self._cons = cons

        # Cache scalers for derivative calculation

        self.dv_conversions = {}
        for name, meta in iteritems(desvars):
            scaler = meta.get('scaler')
            if isinstance(scaler, np.ndarray):
                if all(scaler == 1.0):
                    continue
            elif scaler == 1.0:
                continue

            self.dv_conversions[name] = np.reciprocal(scaler)

        self.fn_conversions = {}
        for name, meta in iteritems(objs):
            scaler = meta.get('scaler')
            if isinstance(scaler, np.ndarray):
                if all(scaler == 1.0):
                    continue
            elif scaler == 1.0:
                continue

            self.fn_conversions[name] = scaler

        for name, meta in iteritems(cons):
            scaler = meta.get('scaler')
            if isinstance(scaler, np.ndarray):
                if all(scaler == 1.0):
                    continue
            elif scaler == 1.0:
                continue

            self.fn_conversions[name] = scaler

    def _map_voi_indices(self):
        poi_indices = {}
        qoi_indices = {}
        for name, meta in chain(iteritems(self._cons), iteritems(self._objs)):
            # set indices of interest
            if 'indices' in meta:
                qoi_indices[name] = meta['indices']

        for name, meta in iteritems(self._desvars):
            # set indices of interest
            if 'indices' in meta:
                poi_indices[name] = meta['indices']

        return poi_indices, qoi_indices

    def _of_interest(self, voi_list):
        """Return a list of tuples, with the given voi_list organized
        into tuples based on the previously defined grouping of VOIs.
        """
        vois = []
        remaining = set(voi_list)
        for voi_set in self._voi_sets:
            vois.append([])

        for i, voi_set in enumerate(self._voi_sets):
            for v in voi_list:
                if v in voi_set:
                    vois[i].append(v)
                    remaining.remove(v)

        vois = [tuple(x) for x in vois if x]

        for v in voi_list:
            if v in remaining:
                vois.append((v,))

        return vois

    def desvars_of_interest(self):
        """
        Returns
        -------
        list of tuples of str
            The list of design vars, organized into tuples according to
            previously defined VOI groups.
        """
        return self._of_interest(self._desvars)

    def outputs_of_interest(self):
        """
        Returns
        -------
        list of tuples of str
            The list of constraints and objectives, organized into tuples
            according to previously defined VOI groups.
        """
        return self._of_interest(list(chain(self._objs, self._cons)))

    def parallel_derivs(self, vnames):
        """
        Specifies that the named variables of interest are to be grouped
        together so that their derivatives can be solved for concurrently.

        Args
        ----
        vnames : iter of str
            The names of variables of interest that are to be grouped.
        """
        #make sure all vnames are desvars, constraints, or objectives
        found = set()
        for n in vnames:
            if not (n in self._desvars or n in self._objs or n in self._cons):
                raise RuntimeError("'%s' is not a param, objective, or "
                                   "constraint" % n)
        for grp in self._voi_sets:
            for vname in vnames:
                if vname in grp:
                    msg = "'%s' cannot be added to VOI set %s because it " + \
                          "already exists in VOI set: %s"
                    raise RuntimeError(msg % (vname, tuple(vnames), grp))

        param_intsect = set(vnames).intersection(self._desvars.keys())

        if param_intsect and len(param_intsect) != len(vnames):
            raise RuntimeError("%s cannot be grouped because %s are design "
                               "vars and %s are not." %
                               (vnames, list(param_intsect),
                                list(set(vnames).difference(param_intsect))))

        if MPI: # pragma: no cover
            self._voi_sets.append(tuple(vnames))
        else:
            warnings.warn("parallel derivs %s specified but not running under MPI")

    def add_recorder(self, recorder):
        """
        Adds a recorder to the driver.

        Args
        ----
        recorder : BaseRecorder
           A recorder instance.
        """
        self.recorders.append(recorder)

    def add_desvar(self, name, low=None, high=None, indices=None, adder=0.0, scaler=1.0):
        """
        Adds a parameter to this driver.

        Args
        ----
        name : string
           Name of the IndepVarComp in the root system.

        low : float or ndarray, optional
            Lower boundary for the param

        high : upper or ndarray, optional
            Lower boundary for the param

        indices : iter of int, optional
            If a param is an array, these indicate which entries are of
            interest for derivatives.

        adder : float or ndarray, optional
            Value to add to the model value to get the scaled value. Adder
            is first in precedence.

        scaler : float or ndarray, optional
            value to multiply the model value to get the scaled value. Scaler
            is second in precedence.
        """

        if low is None:
            low = -1e99
        elif isinstance(low, np.ndarray):
            low = low.flatten()

        if high is None:
            high = 1e99
        elif isinstance(high, np.ndarray):
            high = high.flatten()

        if isinstance(adder, np.ndarray):
            adder = adder.flatten()
        if isinstance(scaler, np.ndarray):
            scaler = scaler.flatten()

        # Scale the low and high values
        low = (low + adder)*scaler
        high = (high + adder)*scaler

        param = {}
        param['low'] = low
        param['high'] = high
        param['adder'] = adder
        param['scaler'] = scaler
        if indices:
            param['indices'] = np.array(indices, dtype=int)

        self._desvars[name] = param

    def add_param(self, name, low=None, high=None, indices=None, adder=0.0,
                  scaler=1.0):
        """
        Deprecated.  Use ``add_desvar`` instead.
        """
        warnings.simplefilter('always', DeprecationWarning)
        warnings.warn("Driver.add_param() is deprecated. Use add_desvar() instead.",
                      DeprecationWarning,stacklevel=2)
        warnings.simplefilter('ignore', DeprecationWarning)

        self.add_desvar(name, low=low, high=high, indices=indices, adder=adder,
                        scaler=scaler)

    def get_desvars(self):
        """ Returns a dict of possibly distributed parameters.

        Returns
        -------
        dict
            Keys are the param object names, and the values are the param
            values.
        """
        uvec = self.root.unknowns
        desvars = OrderedDict()

        for key, meta in iteritems(self._desvars):
            desvars[key] = self._get_distrib_var(key, meta, 'design var')

        return desvars

    def _get_distrib_var(self, name, meta, voi_type):
        uvec = self.root.unknowns
        comm = self.root.comm
        nproc = comm.size
        iproc = comm.rank

        if nproc > 1:
            owner = self.root._owning_ranks[name]
            if iproc == owner:
                flatval = uvec.flat[name]
            else:
                flatval = None
        else:
            owner = 0
            flatval = uvec.flat[name]

        if 'indices' in meta and not (nproc > 1 and owner != iproc):
            # Make sure our indices are valid
            try:
                flatval = flatval[meta['indices']]
            except IndexError:
                msg = "Index for {} '{}' is out of bounds. "
                msg += "Requested index: {}, "
                msg += "shape: {}."
                raise IndexError(msg.format(voi_type, name, meta['indices'],
                                            uvec.metadata(name)['shape']))

        if nproc > 1:
            flatval = comm.bcast(flatval, root=owner)

        scaler = meta['scaler']
        adder = meta['adder']

        if isinstance(scaler, np.ndarray) or isinstance(adder, np.ndarray) \
           or scaler != 1.0 or adder != 0.0:
            return (flatval + adder)*scaler
        else:
            return flatval

    def get_desvar_metadata(self):
        """ Returns a dict of parameter metadata.

        Returns
        -------
        dict
            Keys are the param object names, and the values are the param
            values.
        """
        return self._desvars

    def set_desvar(self, name, value):
        """ Sets a parameter.

        Args
        ----
        name : string
           Name of the IndepVarComp in the root system.

        val : ndarray or float
            value to set the parameter
        """
        if self.root.unknowns.flat[name].size == 0:
            return

        scaler = self._desvars[name]['scaler']
        adder = self._desvars[name]['adder']
        if isinstance(scaler, np.ndarray) or isinstance(adder, np.ndarray) \
           or scaler != 1.0 or adder != 0.0:
            value = value/scaler - adder
        else:
            value = value

        # Only set the indices we requested when we set the parameter.
        idx = self._desvars[name].get('indices')
        if idx is not None:
            self.root.unknowns[name][idx] = value
        else:
            self.root.unknowns[name] = value

    def add_objective(self, name, indices=None, adder=0.0, scaler=1.0):
        """ Adds an objective to this driver.

        Args
        ----
        name : string
            Promoted pathname of the output that will serve as the objective.

        indices : iter of int, optional
            If an objective is an array, these indicate which entries are of
            interest for derivatives.

        adder : float or ndarray, optional
            Value to add to the model value to get the scaled value. Adder
            is first in precedence.

        scaler : float or ndarray, optional
            value to multiply the model value to get the scaled value. Scaler
            is second in precedence.
        """

        if isinstance(adder, np.ndarray):
            adder = adder.flatten()
        if isinstance(scaler, np.ndarray):
            scaler = scaler.flatten()

        obj = {}
        obj['adder'] = adder
        obj['scaler'] = scaler
        if indices:
            obj['indices'] = indices
            if len(indices) > 1 and not self.supports['multiple_objectives']:
                raise RuntimeError("Multiple objective indices specified for "
                                   "variable '%s', but driver '%s' doesn't "
                                   "support multiple objectives." %
                                   (name, self.pathname))
        self._objs[name] = obj

    def get_objectives(self, return_type='dict'):
        """ Gets all objectives of this driver.

        Args
        ----
        return_type : string
            Set to 'dict' to return a dictionary, or set to 'array' to return a
            flat ndarray.

        Returns
        -------
        dict (for return_type 'dict')
            Key is the objective name string, value is an ndarray with the values.

        ndarray (for return_type 'array')
            Array containing all objective values in the order they were added.
        """
        uvec = self.root.unknowns
        objs = OrderedDict()

        for key, meta in iteritems(self._objs):
            objs[key] = self._get_distrib_var(key, meta, 'objective')

        return objs

    def add_constraint(self, name, lower=None, upper=None, equals=None,
                       linear=False, jacs=None, indices=None, adder=0.0,
                       scaler=1.0):
        """ Adds a constraint to this driver. For inequality constraints,
        `lower` or `upper` must be specified. For equality constraints, `equals`
        must be specified.

        Args
        ----
        name : string
            Promoted pathname of the output that will serve as the quantity to
            constrain.

        lower : float or ndarray, optional
             Constrain the quantity to be greater than this value.

        upper : float or ndarray, optional
             Constrain the quantity to be less than this value.

        equals : float or ndarray, optional
             Constrain the quantity to be equal to this value.

        linear : bool, optional
            Set to True if this constraint is linear with respect to all design
            variables so that it can be calculated once and cached.

        jacs : dict of functions, optional
            Dictionary of user-defined functions that return the flattened
            Jacobian of this constraint with repsect to the design vars of
            this driver, as indicated by the dictionary keys. Default is None
            to let OpenMDAO calculate all derivatives. Note, this is currently
            unsupported

        indices : iter of int, optional
            If a constraint is an array, these indicate which entries are of
            interest for derivatives.

        adder : float or ndarray, optional
            Value to add to the model value to get the scaled value. Adder
            is first in precedence.

        scaler : float or ndarray, optional
            value to multiply the model value to get the scaled value. Scaler
            is second in precedence.
        """

        if equals is not None and (lower is not None or upper is not None):
            msg = "Constraint '{}' cannot be both equality and inequality."
            raise RuntimeError(msg.format(name))
        if equals is not None and self.supports['equality_constraints'] is False:
            msg = "Driver does not support equality constraint '{}'."
            raise RuntimeError(msg.format(name))
        if equals is None and self.supports['inequality_constraints'] is False:
            msg = "Driver does not support inequality constraint '{}'."
            raise RuntimeError(msg.format(name))
        if lower is not None and upper is not None and self.supports['two_sided_constraints'] is False:
            msg = "Driver does not support 2-sided constraint '{}'."
            raise RuntimeError(msg.format(name))
        if lower is None and upper is None and equals is None:
            msg = "Constraint '{}' needs to define lower, upper, or equals."
            raise RuntimeError(msg.format(name))


        if isinstance(scaler, np.ndarray):
            scaler = scaler.flatten()
        if isinstance(adder, np.ndarray):
            adder = adder.flatten()
        if isinstance(lower, np.ndarray):
            lower = lower.flatten()
        if isinstance(upper, np.ndarray):
            upper = upper.flatten()
        if isinstance(equals, np.ndarray):
            equals = equals.flatten()

        # Scale the low and high values
        if lower is not None:
            lower = (lower + adder)*scaler
        if upper is not None:
            upper = (upper + adder)*scaler
        if equals is not None:
            equals = (equals + adder)*scaler

        con = {}
        con['lower'] = lower
        con['upper'] = upper
        con['equals'] = equals
        con['linear'] = linear
        con['adder'] = adder
        con['scaler'] = scaler
        con['jacs'] = jacs

        if indices:
            con['indices'] = indices
        self._cons[name] = con

    def get_constraints(self, ctype='all', lintype='all'):
        """ Gets all constraints for this driver.

        Args
        ----
        ctype : string
            Default is 'all'. Optionally return just the inequality constraints
            with 'ineq' or the equality constraints with 'eq'.

        lintype : string
            Default is 'all'. Optionally return just the linear constraints
            with 'linear' or the nonlinear constraints with 'nonlinear'.

        Returns
        -------
        dict
            Key is the constraint name string, value is an ndarray with the values.
        """
        uvec = self.root.unknowns
        cons = OrderedDict()

        for key, meta in iteritems(self._cons):

            if lintype == 'linear' and meta['linear'] == False:
                continue

            if lintype == 'nonlinear' and meta['linear']:
                continue

            if ctype == 'eq' and meta['equals'] is None:
                continue

            if ctype == 'ineq' and meta['equals'] is not None:
                continue

            scaler = meta['scaler']
            adder = meta['adder']

            cons[key] = self._get_distrib_var(key, meta, 'constraint')

        return cons

    def get_constraint_metadata(self):
        """ Returns a dict of constraint metadata.

        Returns
        -------
        dict
            Keys are the constraint object names, and the values are the param
            values.
        """
        return self._cons

    def run(self, problem):
        """ Runs the driver. This function should be overriden when inheriting.

        Args
        ----
        problem : `Problem`
            Our parent `Problem`.
        """
        system = problem.root

        # Metadata Setup
        self.iter_count += 1
        metadata = create_local_meta(None, 'Driver')
        system.ln_solver.local_meta = metadata
        update_local_meta(metadata, (self.iter_count,))

        # Solve the system once and record results.
        system.solve_nonlinear(metadata=metadata)

        self.recorders.record_iteration(system, metadata)

    def calc_gradient(self, indep_list, unknown_list, mode='auto',
                      return_format='array', sparsity=None):
        """ Returns the scaled gradient for the system that is slotted in
        self.root, scaled by all scalers that were specified when the desvars
        and constraints were added.

        Args
        ----
        indep_list : list of strings
            List of independent variable names that derivatives are to
            be calculated with respect to. All params must have a IndepVarComp.

        unknown_list : list of strings
            List of output or state names that derivatives are to
            be calculated for. All must be valid unknowns in OpenMDAO.

        mode : string, optional
            Deriviative direction, can be 'fwd', 'rev', 'fd', or 'auto'.
            Default is 'auto', which uses mode specified on the linear solver
            in root.

        return_format : string, optional
            Format for the derivatives, can be 'array' or 'dict'.

        sparsity : dict, optional
            Dictionary that gives the relevant design variables for each
            constraint. This option is only supported in the `dict` return
            format.

        Returns
        -------
        ndarray or dict
            Jacobian of unknowns with respect to params.
        """

        return self._problem.calc_gradient(indep_list, unknown_list, mode=mode,
                                           return_format=return_format,
                                           dv_scale=self.dv_conversions,
                                           cn_scale=self.fn_conversions,
                                           sparsity=sparsity)

    def generate_docstring(self):
        """
        Generates a numpy-style docstring for a user-created Driver class.

        Returns
        -------
        docstring : str
                string that contains a basic numpy docstring.
        """
        #start the docstring off
        docstring = '    \"\"\"\n'

        #Put options into docstring
        from openmdao.core.options import OptionsDictionary
        firstTime = 1
        #for py3.4, items from vars must come out in same order.
        v = OrderedDict(sorted(vars(self).items()))
        for key, value in v.items():
            if type(value)==OptionsDictionary:
                if key == "supports": continue
                if firstTime:  #start of Options docstring
                    docstring += '\n    Options\n    -------\n'
                    firstTime = 0
                for (name, val) in sorted(value.items()):
                    docstring += "    " + key + "['"
                    docstring += name + "']"
                    docstring += " :  " + type(val).__name__
                    docstring += "("
                    if type(val).__name__ == 'str': docstring += "'"
                    docstring += str(val)
                    if type(val).__name__ == 'str': docstring += "'"
                    docstring += ")\n"

                    desc = value._options[name]['desc']
                    if(desc):
                        docstring += "        " + desc + "\n"
        #finish up docstring
        docstring += '\n    \"\"\"\n'
        return docstring
