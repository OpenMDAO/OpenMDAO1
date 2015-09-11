

def problem_derivatives_test(unittest, problem, tol = 1e-5, skip_comps = False):
    """
    Runs partial derivates check on an OpenMDAO problem instance.
    Asserts that forward and reverse derivatives are within a specified
    relative tolerance.

    unittest -> unittest.TestCase : unit test instance

    problem  -> Problem : OpenMDAO problem instance to be tested

    tol: float -> Float : tolerance for relative error in the derivative checks

    skip_comps -> [String, ...] : skips components with these names

    """
    partials = problem.check_partial_derivatives(out_stream=None)
    for comp in partials:

        if skip_comps:
            skip = False
            for ignore in ["flow", "props", "chem_eq", "static"]:
                if ignore in comp:
                    skip = True
                    break
            if skip:
                continue
        
        derivs = partials[comp]
        for deriv in derivs.keys():
            absol = derivs[deriv]['abs error'] 
            err = derivs[deriv]['rel error']

            if max(absol) > 0: # zero abs error implies rel error = nan
                try:
                    unittest.assertLessEqual(max(err), tol)
                    # print "Deriv test passed:", comp, deriv, max(err) 
                except AssertionError as e:
                    print "Deriv test failed:", comp, deriv, max(err)   
                    raise e