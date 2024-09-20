#!/usr/bin/env python3
"""
Shared utilities for working with coupled model output.
"""
# Internal stuff
from functools import partial
from warnings import warn
from icecream import ic, colorize, pprint
ic.configureOutput(
    outputFunction=lambda *args: print(colorize(*args)),
    argToStringFunction=lambda x: pprint.pformat(x, sort_dicts=False),
)
_CoupledWarning = type('CoupledWarning', (UserWarning,), {})
_warn_coupled = partial(warn, category=_CoupledWarning, stacklevel=2)

# Import tools
from climopy import ureg, vreg, const  # noqa: F401
from .datasets import *  # noqa: F401, F403
from .process import *  # noqa: F401, F403
from .reduce import *  # noqa: F401, F403
from .specs import *  # noqa: F401, F403
from .general import *  # noqa: F401, F403
from .templates import *  # noqa: F401, F403

# Add sigma units
ureg._on_redefinition = 'ignore'
ureg.define('sigma = 1 = σ = stdev = std')
ureg._on_redefinition = 'warn'
