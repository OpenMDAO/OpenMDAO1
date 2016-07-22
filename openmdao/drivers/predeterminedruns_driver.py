"""
Baseclass for design-of-experiments Drivers that have pre-determined
parameter sets.
"""
from __future__ import print_function

import sys
import os
import traceback
import logging
from six.moves import zip
from six import next, PY3

import multiprocessing

import numpy

from openmdao.core.problem import _get_root_var
from openmdao.core.driver import Driver
from openmdao.util.record_util import create_local_meta, update_local_meta
from openmdao.util.array_util import evenly_distrib_idxs
from openmdao.core.mpi_wrap import MPI, debug, any_proc_is_true
from openmdao.core.system import AnalysisError

trace = os.environ.get('OPENMDAO_TRACE')

def worker(problem, response_vars, case_queue, response_queue, worker_id):
    """This is used to run parallel DOEs using multprocessing. It takes a case
    off of the case_queue, runs it, then puts responses on the response_queue.
    """
    # set env var so comps/recorders know they're running in a worker proc
    os.environ['OPENMDAO_WORKER_ID'] = str(worker_id)

    try:
        # on windows all of our args are pickled, which causes us to lose the
        # connections between our numpy views and their parent arrays, so force
        # the problem to setup() again.
        if sys.platform == 'win32':
            problem.setup(check=False)

        driver = problem.driver
        root = driver.root

        terminate = 0
        for case_id, case in iter(case_queue.get, 'STOP'):
            #logging.info("worker %d, case id %d, case %s" % (worker_id, case_id, case))

            if terminate:
                continue

            metadata = driver._prep_case(case, case_id)

            try:
                terminate, exc = driver._try_case(root, metadata)
                if terminate:
                    complete_case = (metadata, [])
                else:
                    complete_case = (metadata,
                             [_get_root_var(root, n) for n in response_vars])
            except:
                # we generally shouldn't get here, but just in case,
                # handle it so that the main process doesn't hang at the
                # end when it tries to join all of the concurrent processes.
                if metadata.get('msg'):
                    metadata['msg'] += "\n\n%s" % traceback.format_exc()
                else:
                    metadata['msg'] = traceback.format_exc()
                metadata['success'] = 0
                metadata['terminate'] = 1
                complete_case = (metadata, [])

            metadata['id'] = case_id
            response_queue.put(complete_case)
    except:
        logging.error(traceback.format_exc())
        raise

class PredeterminedRunsDriver(Driver):
    """
    Baseclass for design-of-experiments Drivers that have pre-determined
    parameter sets.

    Args
    ----
    num_par_doe : int, optional
        The number of DOE cases to run concurrently.  Defaults to 1.

    load_balance : bool, Optional
        If True and running under MPI, use rank 0 as master and load balance
        cases among all of the other ranks. Default is False.  If
        multiprocessing is being used instead of MPI, then cases are always
        load balanced.
    """

    def __init__(self, num_par_doe=1, load_balance=False):
        if type(self) == PredeterminedRunsDriver:
            raise Exception('PredeterminedRunsDriver is an abstract class')
        super(PredeterminedRunsDriver, self).__init__()

        self._num_par_doe = int(num_par_doe)
        self._par_doe_id = 0
        self._load_balance = load_balance

    def _setup_communicators(self, comm, parent_dir):
        """
        Assign a communicator to the root `System`.

        Args
        ----
        comm : an MPI communicator (real or fake)
            The communicator being offered by the Problem.

        parent_dir : str
            Absolute dir of parent `System`.
        """
        root = self.root

        if self._num_par_doe <= 1:
            self._num_par_doe = 1
            self._load_balance = False

        self._full_comm = comm

        # figure out which parallel DOE we are associated with
        if MPI and self._num_par_doe > 1:
            minprocs, maxprocs = root.get_req_procs()
            if self._load_balance:
                sizes, offsets = evenly_distrib_idxs(self._num_par_doe-1,
                                                     comm.size-1)
                sizes = [1]+list(sizes)
                offsets = [0]+[o+1 for o in offsets]
            else:
                sizes, offsets = evenly_distrib_idxs(self._num_par_doe,
                                                     comm.size)

            # a 'color' is assigned to each subsystem, with
            # an entry for each processor it will be given
            # e.g. [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
            color = []

            self._id_map = {}
            for i in range(self._num_par_doe):
                color.extend([i]*sizes[i])
                self._id_map[i] = (sizes[i], offsets[i])

            self._par_doe_id = color[comm.rank]

            if self._load_balance:
                self._casecomm = None
            else:
                casecolor = []
                for i in range(self._num_par_doe):
                    if sizes[i] > 0:
                        casecolor.append(1)
                        casecolor.extend([MPI.UNDEFINED]*(sizes[i]-1))

                # we need a comm that has all the 0 ranks of the subcomms so
                # we can gather multiple cases run as part of parallel DOE.
                if trace:
                    debug('%s: splitting casecomm, doe_id=%s' % ('.'.join((root.pathname,
                                                                   'driver')),
                                                        self._par_doe_id))
                self._casecomm = comm.Split(casecolor[comm.rank])
                if trace: debug('%s: casecomm split done' % '.'.join((root.pathname,
                                                               'driver')))

                if self._casecomm == MPI.COMM_NULL:
                    self._casecomm = None

            # create a sub-communicator for each color and
            # get the one assigned to our color/process
            if trace:
                debug('%s: splitting comm, doe_id=%s' % ('.'.join((root.pathname,
                                                               'driver')),
                                                    self._par_doe_id))
            comm = comm.Split(self._par_doe_id)
            if trace: debug('%s: comm split done' % '.'.join((root.pathname,
                                                           'driver')))
        else:
            self._casecomm = None

        # tell RecordingManager it needs to do a multicase gather
        self.recorders._casecomm = self._casecomm

        root._setup_communicators(comm, parent_dir)

    def get_req_procs(self):
        """
        Returns
        -------
        tuple
            A tuple of the form (min_procs, max_procs), indicating the
            min and max processors usable by this `Driver`.
        """
        minprocs, maxprocs = self.root.get_req_procs()

        if MPI:
            minprocs *= self._num_par_doe
        if MPI and maxprocs is not None:
            maxprocs *= self._num_par_doe

        return (minprocs, maxprocs)

    def run(self, problem):
        """Build a runlist and execute the Problem for each set of generated
        parameters.
        """
        self.iter_count = 0

        with problem.root._dircontext:
            if self._num_par_doe > 1:
                if MPI:
                    if self._load_balance:
                        self._run_lb(problem.root)
                    else:
                        self._run_par_doe(problem.root)
                else: # use multiprocessing
                    self._run_lb_multiproc(problem)
            else:
                self._run_serial()

    def _save_case(self, case, meta=None):
        if self._num_par_doe > 1:
            if self._load_balance:
                self.recorders.record_completed_case(self.root, case)
            else:
                self.recorders.record_iteration(self.root, meta,
                                                dummy=(case is None))
        else:
            self.recorders.record_iteration(self.root, meta)

    def _prep_case(self, case, iter_count):
        """Create metadata for the case and set design variables.
        """
        metadata = create_local_meta(None, 'Driver')
        update_local_meta(metadata, (iter_count,))
        for dv_name, dv_val in case:
            self.set_desvar(dv_name, dv_val)
        return metadata

    def _try_case(self, root, metadata):
        """Run a case and save exception info and mark the metadata
        if the case fails.
        """

        terminate = False
        exc = None

        metadata['terminate'] = 0

        try:
            root.solve_nonlinear(metadata=metadata)
        except AnalysisError:
            metadata['msg'] = traceback.format_exc()
            metadata['success'] = 0
        except Exception:
            metadata['success'] = 0
            # this will tell master to stop sending cases in lb case
            metadata['terminate'] = 1
            metadata['msg'] = traceback.format_exc()
            print(metadata['msg'])
            if not self._load_balance:
                exc = sys.exc_info()
                terminate = True

        return terminate, exc

    def _run_serial(self):
        """This runs a DOE in serial on a single process."""

        root = self.root

        for case in self._build_runlist():
            metadata = self._prep_case(case, self.iter_count)

            terminate, exc = self._try_case(root, metadata)

            if exc is not None:
                if PY3:
                    raise exc[0].with_traceback(exc[1], exc[2])
                else:
                    # exec needed here since otherwise python3 will
                    # barf with a syntax error  :(
                    exec('raise exc[0], exc[1], exc[2]')

            self._save_case(case, metadata)
            self.iter_count += 1

    def _run_par_doe(self, root):
        """This runs the DOE in parallel where cases are evenly distributed
        among all processes.
        """

        for case in self._get_case_w_nones(self._distrib_build_runlist()):
            if case is None: # dummy cases have case == None
                # must take part in collective Allreduce call
                any_proc_is_true(self._full_comm, False)

            else:  # case is not a dummy case
                metadata = self._prep_case(case, self.iter_count)

                terminate, exc = self._try_case(root, metadata)

                if any_proc_is_true(self._full_comm, terminate):
                    if exc:
                        if PY3:
                            raise exc[0].with_traceback(exc[1], exc[2])
                        else:
                            # exec needed here since otherwise python3 will
                            # barf with a syntax error  :(
                            exec('raise exc[0], exc[1], exc[2]')
                    else:
                        raise RuntimeError("an exception was raised by another MPI process.")

            self._save_case(case, metadata)
            self.iter_count += 1


    def _run_lb(self, root):
        """This runs the DOE in parallel with load balancing via MPI.  A new case
        is distributed to a worker process as soon as it finishes its
        previous case.  The rank 0 process is the 'master' process and does
        not run cases itself.  The master does nothing but distribute the
        cases to the workers and collect the results.
        """

        for case in self._distrib_lb_build_runlist():
            if self._full_comm.rank == 0:
                # we're the master rank and case is a completed case
                self._save_case(case)
            else:  # we're a worker
                metadata = self._prep_case(case, self.iter_count)

                self._try_case(root, metadata)

                # keep meta for worker to send to master
                self._last_meta = metadata

            self.iter_count += 1

    def _build_case(self, meta, uvars, pvars, numuvars, values):
        """
        Given values returned from a multiproc run, construct
        a case object that can be passed to recorders.
        """
        if meta['terminate']:
            print("Worker has requested termination. No more new "
                  "cases will be distributed. Worker traceback was:\n%s" %
                  meta['msg'])
            return None

        return {
                  'u':{n:v for n,v in zip(uvars, values)},
                  'p':{n:v for n,v in zip(pvars, values[numuvars:])},
                  'r':{},
                  'meta': meta
               }

    def _run_lb_multiproc(self, problem):
        """This runs the DOE in parallel with load balancing via
        multiprocessing.  A new case is distributed to a worker process as
        soon as it finishes its previous case.
        """
        root = problem.root

        uvars = list(self.recorders._vars_to_record['unames'])
        pvars = list(self.recorders._vars_to_record['pnames'])
        response_vars = uvars + pvars
        numuvars = len(uvars)

        runiter = self._build_runlist()

        # Create queues
        if sys.platform == 'win32':
            manager = multiprocessing.Manager()
            task_queue = manager.Queue()
            done_queue = manager.Queue()
        else:
            task_queue = multiprocessing.Queue()
            done_queue = multiprocessing.Queue()

        procs = []
        terminating = False

        # Start worker processes
        for i in range(self._num_par_doe):
            procs.append(multiprocessing.Process(target=worker,
                                                 args=(problem, response_vars,
                                                 task_queue, done_queue, i)))

        for proc in procs:
            proc.start()

        iter_count = 0
        num_active = 0
        empty = {}
        try:
            for proc in procs:
                # case is a generator, so must make a list to send
                case = list(next(runiter))
                task_queue.put((iter_count, case))
                iter_count += 1
                num_active += 1
        except StopIteration:
            pass
        else:
            try:
                while num_active > 0:
                    meta, values = done_queue.get()
                    #logging.info("RECEIVED: %d, %s" % (meta['id'], values[2]))
                    complete_case = self._build_case(meta, uvars, pvars,
                                                     numuvars, values)
                    num_active -= 1
                    if complete_case is None:
                        # there was a fatal error, don't run more cases
                        break

                    self.recorders.record_completed_case(root, complete_case)
                    case = list(next(runiter))
                    task_queue.put((iter_count, case))
                    iter_count += 1
                    num_active += 1
            except StopIteration:
                pass

        # tell all workers we're done
        for proc in procs:
            task_queue.put('STOP')

        for i in range(num_active):
            meta, values = done_queue.get()
            #logging.info("RECEIVED: %d, %s" % (meta['id'], values[0]))
            complete_case = self._build_case(meta, uvars, pvars,
                                             numuvars, values)
            if complete_case is None:
                # had a fatal error, don't record
                continue

            self.recorders.record_completed_case(root, complete_case)

        for proc in procs:
            proc.join()

    def _get_case_w_nones(self, it):
        """A wrapper around a case generator that returns None cases if
        any of the other members of the MPI comm have any cases left to run,
        so that we can prevent hanging calls to gather.
        """
        comm = self._casecomm

        if comm is None:
            for case in it:
                yield case
        else:

            cases_remain = numpy.array(1, dtype=int)

            while True:
                try:
                    case = next(it)
                except StopIteration:
                    case = None

                val = 1 if case is not None else 0
                comm.Allreduce(numpy.array(val, dtype=int),
                               cases_remain, op=MPI.SUM)

                if cases_remain > 0:
                    yield case
                else:
                    break

    def _distrib_build_runlist(self):
        """
        Returns an iterator over only those cases meant to execute
        in the current rank as part of a parallel DOE. _build_runlist
        will be called on all ranks, but only those cases targeted to
        this rank will run. Override this method
        (see LatinHypercubeDriver) if your DOE generator needs to
        create all cases on one rank and scatter them to other ranks.
        """
        for i, case in enumerate(self._build_runlist()):
            if (i % self._num_par_doe) == self._par_doe_id:
                yield case

    def _distrib_lb_build_runlist(self):
        """
        Runs a load balanced version of the runlist, with the master
        rank (0) sending a new case to each worker rank as soon as it
        has finished its last case.
        """
        comm = self._full_comm

        if self._full_comm.rank == 0:  # master rank
            runiter = self._build_runlist()
            received = 0
            sent = 0

            # cases left for each par doe
            cases = {n:{'count': 0, 'terminate': 0, 'p':{}, 'u':{}, 'r':{},
                        'meta':{'success': 1, 'msg': ''}}
                                    for n in self._id_map}

            # create a mapping of ranks to doe_ids, to handle those cases
            # where a single DOE is executed across multiple processes, i.e.,
            # for each process, we need to know which case it's working on.
            doe_ids = {}
            for doe_id, tup in self._id_map.items():
                size, offset = tup
                for i in range(size):
                    doe_ids[i+offset] = doe_id

            # seed the workers
            for i in range(1, self._num_par_doe):
                try:
                    # case is a generator, so must make a list to send
                    case = list(next(runiter))
                except StopIteration:
                    break
                size, offset = self._id_map[i]
                # send the case to all of the subprocs that will work on it
                for j in range(size):
                    if trace:
                        debug('Sending Seed case %d, %d' % (i, j))
                    comm.send(case, j+offset, tag=1)
                    if trace:
                        debug('Seed Case Sent %d, %d' % (i, j))
                    cases[i]['count'] += 1
                    sent += 1

            # send the rest of the cases
            if sent > 0:
                more_cases = True
                while True:
                    if trace: debug("Waiting on case")
                    worker, p, u, r, meta = comm.recv(tag=2)
                    if trace: debug("Case Recieved from Worker %d" % worker )

                    received += 1

                    caseinfo = cases[doe_ids[worker]]
                    caseinfo['count'] -= 1
                    caseinfo['p'].update(p)
                    caseinfo['u'].update(u)
                    caseinfo['r'].update(r)

                    # save certain parts of existing metadata so we don't hide failures
                    oldmeta = caseinfo['meta']
                    success = oldmeta['success']
                    if not success:
                        msg = oldmeta['msg']
                        oldmeta.update(meta)
                        oldmeta['success'] = success
                        oldmeta['msg'] = msg
                    else:
                        oldmeta.update(meta)

                    caseinfo['terminate'] += meta.get('terminate', 0)

                    if caseinfo['count'] == 0:
                        # we've received case from all procs with that doe_id
                        # so the case is complete.

                        # worker has experienced some critical error, so we'll
                        # stop sending new cases and start to wrap things up
                        if caseinfo['terminate'] > 0:
                            more_cases = False
                            print("Worker %d has requested termination. No more new "
                                  "cases will be distributed. Worker traceback was:\n%s" %
                                  (worker, meta['msg']))
                        else:

                            # Send case to recorders
                            yield caseinfo

                            if more_cases:
                                try:
                                    case = list(next(runiter))
                                except StopIteration:
                                    more_cases = False
                                else:
                                    # send a new case to every proc that works on
                                    # cases with the current worker
                                    doe = doe_ids[worker]
                                    size, offset = self._id_map[doe]
                                    cases[doe]['terminate'] = 0
                                    cases[doe]['meta'] = {'success': 1, 'msg': ''}
                                    for j in range(size):
                                        if trace:
                                            debug("Sending New Case to Worker %d" % worker )
                                        comm.send(case, j+offset, tag=1)
                                        if trace:
                                            debug("Case Sent to Worker %d" % worker )
                                        cases[doe]['count'] += 1
                                        sent += 1

                    # don't stop until we hear back from every worker process
                    # we sent a case to
                    if received == sent:
                        break

            # tell all workers to stop
            for rank in range(1, self._full_comm.size):
                if trace:
                    debug("Make Worker Stop on Rank %d" % rank )
                comm.send(None, rank, tag=1)
                if trace:
                    debug("Worker has Stopped on Rank %d" % rank )

        else:   # worker
            while True:
                # wait on a case from the master
                if trace: debug("Receiving Case from Master")

                case = comm.recv(source=0, tag=1)

                if trace: debug("Case Received from Master")
                if case is None: # we're done
                    break

                # yield the case so it can be executed
                yield case

                # get local vars from RecordingManager
                params, unknowns, resids = self.recorders._get_local_case_data(self.root)

                # tell the master we're done with that case and send local vars
                if trace: debug("Send Master Local Vars")

                comm.send((comm.rank, params, unknowns, resids, self._last_meta), 0, tag=2)

                if trace: debug("Local Vars Sent to Master")
