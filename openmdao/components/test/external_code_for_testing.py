import os
import sys
import time


def main():
    """ Just an external program for testing ExternalCode. """

    if len(sys.argv) >= 2:
        delay = float(sys.argv[1])
        if delay < 0:
            raise ValueError('delay must be >= 0')
        time.sleep(delay)

    out = open('external_code_output.txt', 'w')
    out.write("test data\n")
    out.close()

if __name__ == '__main__': # pragma no cover
    main()

