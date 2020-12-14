from functools import reduce
from operator import mul
import time

def list_prod(l):
    return reduce(mul, l)

def format_seconds_elapsed(seconds):
    """Helper function to format elapsed time in seconds in a human-readable format
    broken up into days, hours, minutes, seconds, and milliseconds.

    Parameters
    ----------
    seconds : float
        Elapsed time in seconds.

    Returns
    -------
    time : str
        Formatted human-readable string describing the time.
    """
    def get_plural(val):
        return 's' if val > 1 else ''

    milliseconds = seconds * 1000

    days, milliseconds = divmod(milliseconds, 86400000)
    hours, milliseconds = divmod(milliseconds, 3600000)
    minutes, milliseconds = divmod(milliseconds, 60000)
    seconds, milliseconds = divmod(milliseconds, 1000)

    formatted = ''
    if days > 0:
        formatted += '{:.0f} day{} '.format(days, get_plural(days))
    if hours > 0:
        formatted += '{:.0f} hour{} '.format(hours, get_plural(hours))
    if minutes > 0:
        formatted += '{:.0f} minute{} '.format(minutes, get_plural(minutes))
    if seconds > 0:
        formatted += '{:.0f} second{} '.format(seconds, get_plural(seconds))
    formatted += '{:.0f} millisecond{}'.format(milliseconds, get_plural(milliseconds))
    return formatted


class Timer(object):
    """Context manager to record time elapsed for an operation and optionally print it
    out (formatted with :func:`format_seconds_elapsed`). After exiting the context, the
    time elapsed can be accessed through the ``time_elapsed`` attribute of the manager.

    Parameters
    ----------
    op_name : :class:`str`
        The name of the operation being timed. Elapsed time will be printed as
        ``"<op_name> took <elapsed time>."``.
    verbose : :class:`bool`, optional
        Whether or not to print the elapsed time. Default is ``True``.
    print_func : :class:`function`, optional
        The print function to use. Default is the builtin :func:`print`, which will print
        to stdout. The function should take in a single arg, which is the message to
        print.
    """
    def __init__(self, op_name, verbose=True, print_func=None):
        # Store args.
        self.op_name = op_name
        self.verbose = verbose
        self.print_func = print_func

    def __enter__(self):
        """Enter the context.
        """
        self.st = time.time()
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        """Exit the context.
        """
        self.time_elapsed = time.time() - self.st
        if self.verbose:
            if self.print_func is None:
                print(
                    '"{}" took {}.'.format(
                        self.op_name,
                        format_seconds_elapsed(self.time_elapsed)
                    )
                )
            else:
                self.print_func(
                    '"{}" took {}.'.format(
                        self.op_name,
                        format_seconds_elapsed(self.time_elapsed)
                    )
                )

