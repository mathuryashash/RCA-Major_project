"""Single source of the application version.

Both executables and both log files report this. Without it a bug report
says only "the app", and several builds a day with different behaviour are
indistinguishable once they leave this machine.
"""

__version__ = "1.4.1"
