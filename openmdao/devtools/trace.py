
import sys
import os
from functools import wraps

# from Eli Bendersky with some modifictations
# http://eli.thegreenplace.net/2012/08/22/easy-tracing-of-nested-function-calls-in-python
class TraceCalls(object):
    """ Use as a decorator on functions that should be traced. Several
        functions can be decorated - they will all be indented according
        to their call depth.

        Usage:
        @TraceCalls(env_vars=('OPENMDAO_TRACE',))
        def myfunc(n):
            # .. do stuff

        If any of the environment variables specified in env_vars are truthy
        at module import time, the decorator will be active. Otherwise the
        decorator will not be applied at all.

    """
    def __init__(self, stream=sys.stdout, indent_step=2, show_return=True,
                 env_vars=()):
        self.stream = stream
        self.indent_step = indent_step
        self.show_ret = show_return
        self.env_vars = env_vars

        # This is a class attribute since we want to share the indentation
        # level between different traced functions, in case they call
        # each other.
        TraceCalls.cur_indent = 0

    def __call__(self, fn):
        # don't do the wrap unless certain env vars are true
        dowrap = False
        for e in self.env_vars:
            if os.environ.get(e):
                dowrap = True
                break

        if dowrap or not self.env_vars:
            @wraps(fn)
            def wrapper(*args, **kwargs):
                indent = ' ' * TraceCalls.cur_indent
                argstr = ', '.join(
                    [repr(a) for a in args] +
                    ["%s=%s" % (a, repr(b)) for a, b in kwargs.items()])
                self.stream.write('%s%s(%s)\n' % (indent, fn.__name__, argstr))
                self.stream.flush()

                TraceCalls.cur_indent += self.indent_step
                ret = fn(*args, **kwargs)
                TraceCalls.cur_indent -= self.indent_step

                if self.show_ret:
                    self.stream.write('%s--> %s\n' % (indent, ret))
                    self.stream.flush()
                return ret
            return wrapper
        return fn
