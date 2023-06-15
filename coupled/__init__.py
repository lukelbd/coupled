#!/usr/bin/env python3
"""
Shared utilities for working with coupled model output.
"""
# Internal stuff
from functools import partial
from warnings import warn
_CoupledWarning = type('CoupledWarning', (UserWarning,), {})
_warn_coupled = partial(warn, category=_CoupledWarning, stacklevel=2)

# Import tools
from climopy import ureg, vreg, const  # noqa: F401
from .output import *  # noqa: F401, F403
from .templates import *  # noqa: F401, F403
from .internals import *  # noqa: F401, F403
