
import traceback

def concurrent_eval_lb(func, cases, comm, broadcast=False):
    """
    Runs a load balanced version of the given function, with the master
    rank (0) sending a new case to each worker rank as soon as it
    has finished its last case.

    Args
    ----

    func : function
        The function to execute in workers.

    cases : collection of function args
        Entries are assumed to be of the form (args, kwargs) where
        kwargs are allowed to be None and args should be a list or tuple.

    com : MPI communicator or None
        The MPI communicator that is shared between the master and workers.
        If None, the function will be executed serially.

    broadcast : bool, optional
        If True, the results will be broadcast out to the worker procs so
        that the return value of concurrent_eval_lb will be the full result
        list in every process.
    """
    if comm is not None:
        if comm.rank == 0:  # master rank
            results = _concurrent_eval_lb_master(cases, comm)
        else:
            results = _concurrent_eval_lb_worker(func, comm)

        if broadcast:
            results = comm.bcast(results, root=0)

    else: # serial execution
        results = []
        for args, kwargs in cases:
            try:
                if kwargs:
                    retval = func(*args, **kwargs)
                else:
                    retval = func(*args)
            except:
                err = traceback.format_exc()
                retval = None
            else:
                err = None
            results.append((retval, err))

    return results

def _concurrent_eval_lb_master(cases, comm):
    """
    This runs only on rank 0.  It sends cases to all of the workers and
    collects their results.
    """
    received = 0
    sent = 0

    results = []

    case_iter = iter(cases)

    # seed the workers
    for i in range(1, comm.size):
        try:
            case = next(case_iter)
        except StopIteration:
            break

        comm.send(case, i, tag=1)
        sent += 1

    # send the rest of the cases
    if sent > 0:
        while True:
            # wait for any worker to finish
            worker, retval, err = comm.recv(tag=2)

            received += 1

            # store results
            results.append((retval, err))

            # don't stop until we hear back from every worker process
            # we sent a case to
            if received == sent:
                break

            try:
                case = next(case_iter)
            except StopIteration:
                pass
            else:
                # send new case to the last worker that finished
                comm.send(case, worker, tag=1)
                sent += 1

    # tell all workers to stop
    for rank in range(1, comm.size):
        comm.send((None, None), rank, tag=1)

    return results

def _concurrent_eval_lb_worker(func, comm):
    while True:
        # wait on a case from the master
        args, kwargs = comm.recv(source=0, tag=1)

        if args is None: # we're done
            break

        try:
            if kwargs:
                retval = func(*args, **kwargs)
            else:
                retval = func(*args)
        except:
            err = traceback.format_exc()
            retval = None
        else:
            err = None

        # tell the master we're done with that case
        comm.send((comm.rank, retval, err), 0, tag=2)
