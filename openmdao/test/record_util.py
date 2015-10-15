import inspect

def is_test(obj):
    if not inspect.isfunction(obj):
        return False

    return obj.__name__.startswith('test')

def is_assertion(obj):
    if not inspect.isfunction(obj):
        return False

    return obj.__name__.startswith('assert')

def collect_tests(testcase):
    return inspect.getmembers(testcase, is_test)

def collect_assertions(testcase):
    return inspect.getmembers(testcase, is_assertion)

def create_testcase(testclass, modules):
    for module in modules:
        tests = collect_tests(module)
        assertions = collect_assertions(module)

        for name, method in tests:
            setattr(testclass, name, method)

        for name, method in assertions:
            if not hasattr(testclass, name):
                setattr(testclass, name, method)

    return testclass
