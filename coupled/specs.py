#!/usr/bin/env python3
"""
Helper functions for parsing plot specifications.
"""
import collections
import inspect
import itertools
import re
import time  # noqa: F401

import climopy as climo  # noqa: F401
import numpy as np
import proplot as pplt
import xarray as xr
from climopy import ureg, vreg  # noqa: F401
from icecream import ic  # noqa: F401
try:
    from proplot.internals.rcsetup import _rc_nodots
except ImportError:
    from proplot.internals.settings import _rc_nodots

__all__ = ['get_path', 'get_label', 'get_labels', 'parse_spec', 'parse_specs']

# Scalar settings and regular expressions
CHAR_PER_EM = 1.8  # threshold for font wrapping
EM_PER_ABC = 8 / CHAR_PER_EM  # space for e.g. 'A.' (2x4 em-widths for centered titles)
REGEX_FLOAT = re.compile(  # allow exponential notation
    r'\A([-+]?[0-9._]+(?:[eE][-+]?[0-9_]+)?)\Z'
)
REGEX_SPLIT = re.compile(  # ignore signs and use '.' instead of '*' for product
    r'(?<=[^+./-])([+./-])(?=[^+./-])'
)

# Reduce labels to exclude from paths and models to exclude from dataset
# NOTE: New format will prefer either 'monthly' or 'annual'. To prevent overwriting
# old results keep 'slope' as the default (i.e. omitted) value when generating paths.
PATHS_EXCLUDE = {
    'project': 'cmip',  # excluded from path name if explicitly passed
    'ensemble': 'flagship',
    'source': 'eraint',
    'style': 'slope',  # preserve to prevent overwriting older figures
    'region': 'globe',
    'period': 'ann',
    # 'start': 0,  # always include in path name
    # 'stop': 150,  # always include in path name
}

# Argument sorting constants
# NOTE: Use logical top-down order for file naming and reduction instruction order
# and more complex bottom-up human-readable order for automatic label generation.
ORDER_LOGICAL = (
    'name',
    'facets',  # model facets information
    'project',
    'institute',  # e.g. 'flagship'
    'model',
    'experiment',
    'ensemble',
    'version',  # feedback version index
    'source',
    'style',
    'startstop',
    'start',
    'stop',
    'region',  # special coordinates
    'period',
    'initial',
    'remove',
    'detrend',
    'error',
    'correct',
    'season',
    'month',
    'time',  # space and time
    'plev',
    'area',
    'volume',
    'lat',
    'lon',
    'spatial',  # always at the end
    'method',
)
ORDER_READABLE = (
    'error',  # scalar coordinates
    'correct',
    'facets',  # model facets information
    'project',
    'institute',
    'model',
    'lon',  # space and time
    'lat',
    'area',
    'volume',
    'plev',
    'time',
    'period',  # NOTE: this is outdated
    'initial',
    'remove',
    'detrend',
    'season',
    'month',
    'startstop',
    'start',
    'stop',
    'ensemble',  # model experiment
    'experiment',
    'region',  # feedback version index
    'style',
    'source',
    'version',
    'name',
    'suffix',
    'spatial',  # always at the end
    'method',
)

# General long and short labels
# NOTE: These are used for figure path generation and for figure axes
# and attributes. See below for details.
GENERAL_LABELS = {
    ('project', 'ceres'): 'CERES',
    ('project', 'cmip'): 'CMIP',
    ('project', 'cmip5'): 'CMIP5',
    ('project', 'cmip6'): 'CMIP6',
    ('project', 'cmip56'): 'matching CMIP5',
    ('project', 'cmip65'): 'matching CMIP6',
    ('project', 'cmip55'): 'non-matching CMIP5',
    ('project', 'cmip66'): 'non-matching CMIP6',
    ('project', 'cmip655'): 'matching CMIP',  # almost matching
    ('project', 'cmip6556'): 'matching CMIP',
    ('project', 'cmip5665'): 'matching CMIP',
    ('project', 'cmip6655'): 'non-matching CMIP',
    ('project', 'cmip5566'): 'non-matching CMIP',
    ('institute', 'flagship'): 'institute-flagship',
    ('institute', 'avg'): 'institute-average',
    ('institute', 'wgt'): None,
    ('institute', None): None,
    ('experiment', 'picontrol'): 'control',
    ('experiment', 'abrupt4xco2'): r'4$\times$CO$_2$',
    ('source', 'eraint'): 'Davis et al.',
    ('source', 'zelinka'): 'Zelinka et al.',
    ('style', 'slope'): 'annual',
    ('style', 'annual'): 'annual',
    ('style', 'monthly'): 'monthly',
    ('style', 'ratio'): 'ratio',
    ('startstop', (None, None)): None,  # TODO: repair kludge
    ('startstop', (0, 150)): 'full',
    ('startstop', (1, 150)): 'full',
    ('startstop', (2, 150)): 'full',
    ('startstop', (0, 20)): 'early',
    ('startstop', (1, 20)): 'early',
    ('startstop', (2, 20)): 'early',
    ('startstop', (0, 50)): 'year 0--50',
    ('startstop', (1, 50)): 'year 1--50',
    ('startstop', (2, 50)): 'year 2--50',
    ('startstop', (20, 150)): 'late',
    ('startstop', (100, 150)): 'year 100--150',
    ('startstop', (0, 23)): None,  # merged scalar data
    ('startstop', (0, 24)): None,  # merged scalar data
    ('region', 'globe'): 'global-$T$',
    ('region', 'point'): 'local-$T$',
    ('region', 'latitude'): 'zonal-$T$',
    ('region', 'hemisphere'): 'hemispheric-$T$',
    ('spatial', 'slope'): 'spatial regression',  # NOTE: also see _apply_double
    ('spatial', 'proj'): 'spatial projection',
    ('spatial', 'corr'): 'spatial correlation',
    ('spatial', 'cov'): 'spatial covariance',
    ('spatial', 'rsq'): 'spatial variance explained',
    ('initial', 'init'): None,  # TODO: revisit this
    ('initial', 'jan'): None,  # avoid adding labels to merged distributions
    ('period', 'full'): None,
    ('period', '20yr'): None,
    ('period', '23yr'): None,  # scalar data
    ('period', '24yr'): None,  # scalar data
    ('period', '50yr'): None,
    ('remove', 'climate'): None,
    ('remove', 'average'): None,
    ('detrend', ''): None,  # TODO: revisit this
    ('detrend', 'x'): None,  # avoid adding labels to merged distributions
    ('detrend', 'y'): None,
    ('detrend', 'xy'): None,
    ('detrend', 'yx'): None,
    ('detrend', 'i'): None,  # avoid adding labels to merged distributions
    ('detrend', 'j'): None,
    ('detrend', 'ij'): None,
    ('detrend', 'ji'): None,
    ('error', 'regression'): None,
    ('error', 'internal'): 'internal',
    ('correct', ''): None,  # TODO: revisit this
    ('correct', 'x'): None,  # avoid adding labels to merged distributions
    ('correct', 'y'): None,
    ('correct', 'r'): None,
    ('plev', 'int'): None,
    ('plev', 'avg'): 'column',
    ('area', None): 'local',  # only used with identical=False
    ('area', 'avg'): 'global',  # only used with identical=False
    ('area', 'globe'): 'global',
    ('area', 'trop'): 'tropical',  # only used with identical=False
    ('area', 'atl'): 'Atlantic',
    ('area', 'natl'): 'North Atlantic',
    ('area', 'tatl'): 'tropical Atlantic',
    ('area', 'pac'): 'Pacific',
    ('area', 'ipac'): 'Pacific',
    ('area', 'npac'): 'North Pacific',
    ('area', 'wpac'): 'West Pacific',
    ('area', 'epac'): 'East Pacific',
    ('area', 'tpac'): 'tropical Pacific',
    ('area', 'pool'): 'warm pool',
    ('area', 'nina'): 'warm pool',
    ('area', 'nino'): 'cold tongue',
    ('area', 'nino3'): 'cold tongue',
    ('area', 'nino4'): 'cold tongue',
    ('area', 'so'): 'Southern Ocean',
    ('area', 'se'): 'southern extratropical',
    ('area', 'ne'): 'northern extratropical',
    ('area', 'sh'): 'southern hemisphere',
    ('area', 'nh'): 'northern hemisphere',
    ('lat', 'absmin'): 'minimum',
    ('lat', 'absmax'): 'maximum',
    ('lat', 'min'): 'minimum',
    ('lat', 'max'): 'maximum',
    ('lon', 'int'): None,
    ('lon', 'avg'): None,
    ('dist', 'sigma1'): 'regression standard error',
    ('dist', 'sigma2'): 'bootstrapped standard deviation',
    ('dist', 'range1'): '95% regression uncertainty',
    ('dist', 'range2'): '95% bootstrapped uncertainty',
}

# General path labels
# NOTE: These are used for figure path generation and for figure
# axes and attributes. See above for details.
GENERAL_PATHS = {
    ('institute', 'avg'): 'inst',
    ('institute', 'flagship'): 'flag',
    ('institute', None): 'model',
    ('experiment', 'picontrol'): 'pictl',
    ('experiment', 'abrupt4xco2'): '4xco2',
    ('startstop', (None, None)): None,  # TODO: repair kludge
    ('startstop', (0, 150)): 'full',
    ('startstop', (1, 150)): 'full1',
    ('startstop', (2, 150)): 'full2',
    ('startstop', (0, 20)): 'early',
    ('startstop', (1, 20)): 'early1',
    ('startstop', (2, 20)): 'early2',
    ('startstop', (0, 50)): 'early50',
    ('startstop', (20, 150)): 'late',
    ('startstop', (100, 150)): 'late50',
    ('startstop', (0, 23)): '23yr',  # merged scalar data
    ('startstop', (0, 24)): '24yr',  # merged scalar data
    ('region', 'point'): 'loc',
    ('region', 'latitude'): 'lat',
    ('region', 'hemisphere'): 'hemi',
    ('region', 'globe'): 'globe',
    ('region', 'apoint'): 'apt',
    ('region', 'alatitude'): 'alat',
    ('region', 'ahemisphere'): 'ahemi',
    ('region', 'aglobe'): 'aglobe',
    ('initial', 'init'): None,
    ('remove', 'climate'): 'clim',
    ('remove', 'average'): None,
    ('detrend', ''): 'raw',
    ('detrend', 'x'): 'rawy',
    ('detrend', 'y'): 'rawx',
    ('detrend', 'xy'): None,
    ('detrend', 'i'): 'trendx',
    ('detrend', 'j'): 'trendy',
    ('detrend', 'ij'): 'trend',
    ('detrend', 'ji'): 'trend',
    ('error', 'regression'): None,
    ('error', 'internal'): 'int',
    ('correct', ''): None,
    ('correct', 'x'): 'adjx',
    ('correct', 'y'): 'adjy',
    ('correct', 'r'): 'adj',
    ('plev', 'avg'): 'avg',  # always report
    ('volume', 'avg'): 'avg',  # always report
    ('area', 'avg'): 'avg',  # always report
    ('lat', 'absmin'): 'min',
    ('lat', 'absmax'): 'max',
    ('lat', 'min'): 'min',
    ('lat', 'max'): 'max',
    ('lon', 'int'): None,
    ('lon', 'avg'): 'avg',  # always report
}

# Ensemble reduceion translations
# NOTE: These are used only explicitly with e.g. rows=(..., 'method') and
# cols=(..., 'methods'). Should revisit in future.
METHOD_LONGS = {
    ('method', 'avg'): 'multi-model averages',
    ('method', 'med'): 'multi-model medians',
    ('method', 'std'): 'inter-model standard deviation',
    ('method', 'pct'): 'inter-model percentile range',  # see _reduce_data()
    ('method', 'var'): 'inter-model variance',
    ('method', 'skew'): 'inter-model skewness',
    ('method', 'kurt'): 'inter-model kurtosis',
    ('method', 'pctile'): 'inter-model percentile range',  # see _reduce_data()
    ('method', 'slope'): 'inter-model regressions',
    ('method', 'proj'): 'inter-model projections',
    ('method', 'corr'): 'inter-model correlations',
    ('method', 'rsq'): 'ensemble variance explained',
    ('method', 'diff'): 'ensemble composite differences',
    ('method', 'dist'): None,
}
METHOD_SHORTS = {
    ('method', 'avg'): None,  # TODO: revisit this
    ('method', 'med'): None,
    ('method', 'dist'): None,
    ('method', 'std'): 'standard deviation',
    ('method', 'pct'): 'percentile range',
    ('method', 'var'): 'variance',
    ('method', 'skew'): 'skewness',
    ('method', 'kurt'): 'kurtosis',
    ('method', 'pctile'): 'percentile range',
    ('method', 'slope'): 'regressions',  # NOTE: only possible 'restrict'
    ('method', 'proj'): 'projections',
    ('method', 'corr'): 'correlations',
    ('method', 'rsq'): 'variance explained',
    ('method', 'diff'): 'composite differences',
}

# Period and time translations
# NOTE: These are used in reduce.py _select_time(). In future should support more
# generalized seasonal selections of continuous months (see observed reduce_time).
PERIOD_LONGS = {
    'ann': 'annual',
    'djf': 'boreal winter',
    'mam': 'boreal spring',
    'jja': 'boreal summer',
    'son': 'boreal autumn',
    'jan': 'January',
    'feb': 'February',
    'mar': 'March',
    'apr': 'April',
    'may': 'May',
    'jun': 'June',
    'jul': 'July',
    'aug': 'August',
    'sep': 'September',
    'oct': 'October',
    'nov': 'November',
    'dec': 'December',
}
PERIOD_SHORTS = {
    'ann': 'annual',
    'djf': 'DJF',
    'mam': 'MAM',
    'jja': 'JJA',
    'son': 'SON',
    'jan': 'Jan',
    'feb': 'Feb',
    'mar': 'Mar',
    'apr': 'Apr',
    'may': 'May',
    'jun': 'Jun',
    'jul': 'Jul',
    'aug': 'Aug',
    'sep': 'Sep',
    'oct': 'Oct',
    'nov': 'Nov',
    'dec': 'Dec',
}
PERIOD_LONGS = {
    (key, value): label
    for key in ('initial', 'period', 'season', 'month')
    for value, label in PERIOD_LONGS.items()
}
PERIOD_SHORTS = {
    (key, value): label
    for key in ('initial', 'period', 'season', 'month')
    for value, label in PERIOD_SHORTS.items()
}

# Institute translations
# NOTE: These are based on 'institute_id' metadata and include ISO 3166 country
# codes from https://en.wikipedia.org/wiki/List_of_ISO_3166_country_codes
INSTITUTE_LABELS = {  # map ids to (and alphabetize by) more familiar abbreviations
    'AS-RCEC': 'AS (TW)',  # 'AS',  # Taipei Academia Sinica, Taiwan
    'AWI': 'AWI (DE)',  # Alfred Wegener Institute, Germany
    'BCC': 'BCC (CN)',  # Beijing Climate Center, China
    'BNU': 'BNU (CN)',  # Beijing N. University, China
    'CAMS': 'CAMS (CN)',  # Chinese Academy of Meteorological Sciences, China
    'CAS': 'CAS (CN)',  # 'FGOALS',  # Chinese Academy of Sciences, China
    'CCCma': 'CCCma (CA)',  # 'CanESM',  # Can. Cen. Clim. Modelling + Analaysis, Canada
    'CMCC': 'CMCC (IT)',  # 'CMCC-Esm',  # Cen. Euro-Med. Cambiamenti Climatici, Italy
    'CNRM-CERFACS': 'CNRM (FR)',  # Cen. National de Recherches Meteorologiques, France
    'CSIRO': 'CSIRO (AU)',  # 'ACCESS',  # Commonwealth Sci. + Ind. Research Org.
    'E3SM-Project': 'E3SM (US)',  # various USA institutes
    'EC-Earth-Consortium': 'EC-Earth (EU)',  # various European institutes
    'FIO': 'FIO (CN)',  # First Institute for Oceanography, China
    'NOAA': 'GFDL (US)',  # 'NOAA',  # Geophysical Fluid Dynamics Laboratory, USA
    'NASA': 'GISS (US)',  # 'NASA',  # Goddard Institute for Space Studies, USA
    'CCCR-IITM': 'IITM (IN)',  # Indian Institute of Tropical Meteorology, India
    'INM': 'INM (RU)',  # Institute for Numerical Mathematics, Moscow
    'IPSL': 'IPSL (FR)',  # Institut Pierre Simon Laplace, France
    'KIOST': 'KIOST (KR)',  # Korea Institute of Ocean Science & Technology, Korea
    'NIMS-KMA': 'KMA (KR)',  # 'KACE',  # Korea Meteorological Administration, Korea
    'MIROC': 'MIROC (JP)',  # jaMstec/nIes/R-ccs/aOri Consortium, Japan
    'MOHC': 'MOHC (UK)',  # 'HadGEM',  # Met Office Hadley Centre, UK
    'MPI-M': 'MPI (DE)',  # 'MPI-ESM',  # Max Planck Institut, Germany
    'MRI': 'MRI (JP)',  # Meteorological Research Institute, Japan
    'NCAR': 'NCAR (US)',  # 'CESM',  # National Center for Atmospheric Research, USA
    'NCC': 'NCC (NO)',  # 'NorESM',  # NorESM Climate modeling Consortium, Norway
    'NUIST': 'NUIST (CN)',  # 'NESM',  # Nanjing U. of Information Sci. and Tech., China
    'SNU': 'SNU (KR)',  # 'SAM',  # Seoul National University, Korea
    'THU': 'THU (CN)',  # 'CIESM',  # Beijing Tsinghua University, China
    'UA': 'UA (US)',  # 'MCM-UA',  # University of Arizonta, USA
}
INSTITUTE_PATHS = {
    ('institute', key): value[:-5].lower()  # exclude country code
    for key, value in INSTITUTE_LABELS.items()
}
INSTITUTE_SHORTS = {
    ('institute', key): value[:-5]  # exclude country code
    for key, value in INSTITUTE_LABELS.items()
}
INSTITUTE_LONGS = {
    ('institute', key): value
    for key, value in INSTITUTE_LABELS.items()
}

# Combine translation dictionaries
# NOTE: Put period first so e.g. (initial, jan): None can override month name.
TRANSLATE_PATHS = {
    **GENERAL_PATHS,
    **INSTITUTE_PATHS,
}
TRANSLATE_SHORTS = {
    **METHOD_SHORTS,
    **PERIOD_SHORTS,
    **GENERAL_LABELS,
    **INSTITUTE_SHORTS,
}
TRANSLATE_LONGS = {
    # **METHOD_LONGS,  # TODO: revisit
    **METHOD_SHORTS,  # TODO: revisit
    **PERIOD_LONGS,
    **GENERAL_LABELS,
    **INSTITUTE_LONGS,
}

# Ignored string components
# NOTE: This is used to merge variable names with common parts
SHARED_LABELS = [
    'flux',  # transport terms
    'energy',
    'transport',
    'convergence',
    'feedback',  # feedback terms,
    'forcing',
    'spatial',
    'pattern',
    'TOA',  # boundary terms
    'surface',
    'atmospheric',
    'top-of-atmosphere',
    r'2$\times$CO$_2$',  # scaling terms
    r'4$\times$CO$_2$',
]


def _pop_kwargs(kwargs, *args):
    """
    Return keyword arguments for specific keys or function.

    Parameters
    ----------
    kwargs : dict
        The input keyword arguments.
    *args : str or callable or xarray.Dataset
        The options or functions.
    """
    from .datasets import FACETS_LEVELS, VERSION_LEVELS
    from observed.feedbacks import TRANSLATE_PARAMS
    keys = []
    keys_levels = sorted(set(name for name, _ in TRANSLATE_PARAMS.values()))
    keys_levels.extend((*FACETS_LEVELS, *VERSION_LEVELS, 'statistic'))
    keys_levels.extend(('area', 'volume', 'remove', 'detrend', 'correct'))
    for arg in args:
        if callable(arg):
            arg = inspect.signature(arg)
        if isinstance(arg, xr.Dataset):  # special reductions
            keys.extend(name for idx in arg.indexes.values() for name in idx.names)
            keys.extend(keys_levels)
        elif isinstance(arg, inspect.Signature):  # key=default arguments
            opts = [key for key, obj in arg.parameters.items() if obj.default is not obj.empty]  # noqa: E501
            keys.extend(opts)
        else:  # string or sequence
            opts = (arg,) if isinstance(arg, str) else tuple(arg)
            keys.extend(opts)
    kw_args = {key: kwargs.pop(key) for key in keys if key in kwargs}
    return kw_args


def _expand_lists(*args):
    """
    Return lists with matched lengths.

    Parameters
    ----------
    *args : list
        The lists to match.
    """
    lengths = list(map(len, args))
    length = set(lengths) - {1}
    args = list(args)  # modifable
    if len(length) > 1:
        values = '\n'.join(f'{len(arg)}: {arg!r}' for arg in args)
        raise ValueError(f'Mixed lengths {lengths} for values\n{values}.')
    length = length.pop() if length else None
    for i, items in enumerate(args):
        if length and len(items) == 1:
            args[i] = length * items
    return args


def _expand_parts(key, value):
    """
    Expand reduction value along mathematical operations.

    Parameters
    ----------
    key : str
        The reduction key.
    value : object
        The reduction value.
    """
    # NOTE: This translates e.g. startstop=('20-0', '150-20') to the anomaly
    # pairs [('20', '150'), ('-', '-'), ('0', '20')] for better processing.
    parts, values = [], value if isinstance(value, tuple) else (value,)
    for value in values:
        iparts, ivalues = [], (value,)  # support 'startstop' anomaly tuples
        if isinstance(value, str) and key not in ('model', 'institute'):
            ivalues = REGEX_SPLIT.split(value)
        for part in ivalues:
            if not isinstance(part, str):
                pass
            elif part.lower() == 'none':
                part = None  # e.g. experiment=abrupt4xco2-picontrol, stop=20-None,
            elif part.isdecimal():
                part = int(part)
            elif REGEX_FLOAT.match(part):
                part = float(part)
            iparts.append(part)
        parts.append(iparts)
    return [part[0] if len(part) == 1 else part for part in zip(*parts)]


def _group_parts(kwargs, keep_operators=False):
    """
    Return grouped reduction keyword arguments.

    Parameters
    ----------
    kwargs : dict
        The reduction keyword args.
    keep_operators : bool, optional
        Whether to keep pre-industrial start and stop operators.
    """
    # WARNING: Need process_data() to accomodate situations where e.g. generating
    # several 'scatter plots' that get concatenated into a bar plot of 'late minus
    # early vs. internal' regression coefficients, in which case reduce_general()
    # will translate denominator to 'full internal minus full internal' i.e.
    # zero, and case where we are doing e.g. spatial version of this regression,
    # where reduce_general() gets regression coefficients before the subtraction
    # operation so there is no need to worry about denominator. Solution is to
    # keep same number of operators (so numerator and denominator have same number
    # and thus can be combined) but replace with just '+' i.e. a dummy average.
    kwargs = kwargs.copy()
    experiment = kwargs.get('experiment')
    if 'stop' in kwargs:
        kwargs.setdefault('start', None)
    if 'start' in kwargs:
        kwargs.setdefault('stop', None)
    if 'start' in kwargs and 'stop' in kwargs:
        start, stop = kwargs.pop('start'), kwargs.pop('stop')
        if keep_operators and experiment == 'picontrol':
            num1 = sum(map(start.count, '+-')) if isinstance(start, str) else 0
            num2 = sum(map(stop.count, '+-')) if isinstance(stop, str) else 0
            start = '+'.join(itertools.repeat(f'{start or 0}', num1 + 1))
            stop = '+'.join(itertools.repeat(f'{stop or 150}', num2 + 1))
        kwargs['startstop'] = (start, stop)
    return kwargs


def _merge_labels(*labels, identical=False):
    """
    Helper function to combine separate labels.

    Parameters
    ----------
    *labels : str
        The labels to be merged.
    identical : bool, optional
        Whether to keep only identical or non-identical portions.
    """
    # NOTE: For typical feedback breakdown this returns e.g. 'feedback' when identical
    # is true and ('net', 'shortwave', 'longwave') when identical is false. Also
    # returns empty string when identical is true but all labels are different, and
    # empty strings when identical is false but all labels (more than one) are the same.
    labels = [label or '' for label in labels]
    shared = [item for item in SHARED_LABELS if all(item in label for label in labels)]
    regex = '|'.join(r'\s*' + re.escape(item) for item in shared)
    regex = re.compile(f'({regex})')
    if identical:
        if len(set(labels)) == 1:  # note always true if labels is singleton
            labels = labels[0]
        else:
            labels = [''.join(regex.findall(labels[0])).strip() for label in labels]
            labels = labels[0] if all(label == labels[0] for label in labels) else ''
    else:
        if len(set(labels)) == 1:  # note always true if labels is singleton
            labels = [''] * len(labels)
        else:
            labels = [regex.sub('', label).strip() for label in labels]
    return labels


def _split_label(
    label, refwidth=None, fontsize=None, shift=None, scale=None, nmax=None,
):
    """
    Helper function to split the label into separate lines.

    Parameters
    ----------
    label : str
        The input label.
    refwidth : unit-spec, optional
        The reference maximum width.
    fontsize : unit-spec, optional
        The font size used for scaling.
    shift : bool, optional
        Whether to shift the width to account for a-b-c labels.
    scale : float, optional
        Additional scaling applied to the label width calculation.
    nmax : int, optional
        The maximum number of breaks to use.
    """
    # TODO: Consider existing matplotlib auto-wrapping utilities (e.g. tight layout)
    # NOTE: This counts characters toward fixed em-width and optionally offsets
    # to prevent centered axes title from overlapping left-aligned a-b-c label
    label = label or ''
    label = label.replace('\n', ' ') + ' '  # remove previous breaks
    imasks, imaths = [], []  # space within math
    ispace = np.array([i for i, c in enumerate(label) if c == ' '])
    icheck = ispace.astype(float)
    for m in re.finditer(r'\$[^$]+\$', label):  # skip inner math
        idx, jdx = m.span()  # index range providing '$math$'
        tex = label[idx + 1:jdx - 1]  # content inside '$math$'
        tex = re.sub(r'\\[A-Za-z]+\{([^}]*)\}', r'\1', tex)  # replace e.g. \mathrm{}
        tex = re.sub(r'[_^]\{?[0-9A-Za-z+-]*\}?', '#', tex)  # replace exponents
        tex = re.sub(r'\s*\\[.,:;]\s*', '', tex)  # replace spaces between terms
        tex = re.sub(r'\\[A-Za-z]+', 'x', tex)  # replace tex symbols with char
        tex = re.sub(r'[ ,.:;{}()[\]\\]', '', tex)  # remove dots and tex chars
        span = jdx - idx - 1  # actual character span
        mask1 = (icheck >= idx) & (icheck < jdx)
        mask2 = icheck >= jdx
        imaths.extend(range(idx, jdx))  # record math zones
        imasks.append((mask1, mask2, span - len(tex)))  # adjust by span differences
    for m in re.finditer(r'[- ,.:;iltjfI{}()[\]]', label):
        idx = m.start()
        if m.start() in imaths:
            continue
        mask1 = np.zeros(icheck.shape, dtype=bool)
        mask2 = icheck >= idx
        imasks.append((mask1, mask2, 0.5))  # adjust by width
    for mask1, mask2, adjust in imasks:  # get line break positions
        icheck[mask1] = np.nan  # no breaks here
        icheck[mask2] -= adjust
    size_ = fontsize or pplt.rc['font.size']
    size_ = pplt.utils._fontsize_to_pt(size_) / pplt.rc['font.size']
    width = refwidth or pplt.rc['subplots.refwidth']
    width = pplt.units(width, 'in', 'em')
    width -= EM_PER_ABC * size_ if shift else 0
    width *= CHAR_PER_EM * (scale or 1)
    splits = width * np.arange(1, 20) / size_
    seen, chars, count = set(), list(label), 0  # convert string to list
    for split in splits:
        idxs, = np.where(icheck <= split)
        if not np.any(icheck > split):  # no spaces or end-of-string over this
            continue
        if not idxs.size:  # including empty icheck
            continue
        if nmax and count >= nmax:
            continue
        if idxs[-1] not in seen:  # avoid infinite loop and jump to next threshold
            seen.add(idx := idxs[-1])
            splits -= split - icheck[idx]  # closer to threshold by this
            chars[ispace[idx]] = '\n'
            count += 1  # update newline count
    return ''.join(chars[:-1])  # ignore trailing space


def get_heading(label, prefix=None, suffix=None):
    """
    Return the label formatted as a heading.

    Parameters
    ----------
    label : str
        The display label.
    prefix, suffix : optional
        The label prefix and suffix.
    """
    kwargs = {}
    if '{prefix}' in label:
        kwargs['prefix'], prefix = prefix and f'{prefix} ' or '', None
    if '{suffix}' in label:
        kwargs['suffix'], suffix = suffix and f' {suffix}' or '', None
    label = label.format(**kwargs)
    if prefix and not label[:2].isupper():
        label = label[:1].lower() + label[1:]
    if prefix:
        prefix = prefix[:1].upper() + prefix[1:]
    elif label:
        label = label[:1].upper() + label[1:]
    elif suffix:
        suffix = suffix[:1].upper() + suffix[1:]
    parts = (prefix, label, suffix)
    return ' '.join(filter(None, parts))


def get_path(dataset, *kws_process):
    """
    Convert reduction operators into path suitable for saving.

    Parameters
    ----------
    dataset : xarray.Dataset
        The source dataset.
    *kws_process : dict
        The `process_data` keywords.

    Returns
    -------
    path : str
        The parts joined with underscores and dashes.
    """
    # NOTE: This omits keywords that correspond to default values (for those that
    # have fixed defaults, i.e. not 'style' or 'experiment'). In latter case may get
    # distinction between e.g. passing experiment=None and experiment='abrupt4xco2'
    # even if that is in fact the default when saving path names. Should revisit.
    labels = []
    kws_process = [(kw,) if isinstance(kw, dict) else tuple(kw) for kw in kws_process]
    kws_process = [kw.copy() for kws in kws_process for kw in kws]
    kws_process = list(map(_group_parts, kws_process))
    for key in ORDER_LOGICAL:
        seen, parts = set(), []
        values = [kw[key] for kw in kws_process if key in kw]
        values = [value for value in values if value not in seen and not seen.add(value)]  # noqa: E501
        if len(values) == 1 and values[0] == PATHS_EXCLUDE.get(key, None):
            continue
        for value in values:  # across all subplots and tuples
            label = get_label(key, value, 'path', dataset=dataset)
            if not label:  # e.g. for 'None' reduction
                continue
            if label not in parts:  # e.g. 'avg' to be ignored
                parts.append(label)
        if parts and parts not in labels:
            labels.append(parts)  # sort these so newer orders overwrite
    result = '_'.join('-'.join(sorted(parts)) for parts in labels)
    return result


def get_label(key, value, mode=None, dataset=None, experiment=True):
    """
    Return a label for the variable or selection.

    Parameters
    ----------
    key, value : str
        The reduce coordinate and selection.
    mode : {'long', 'short', 'path'}, optional
        The label mode used to translate selections.
    restrict : str or sequence, optional
        The reduce keys that should have non-none labels.
    dataset : xarray.Dataset, optional
        The source dataset. Required for `name` labels.
    experiment : bool, optional : str, optional
        Whether to use ``'internal'`` and ``'forced'`` labels.

    Returns
    -------
    label : str
        The variable or selection label.
    """
    # NOTE: This function is used for axis label prefixes, legend entry prefixes, row
    # and column labels, and figure and path titles. It is never used for the 'method'
    # key because this is added to the data short and long names during application,
    # which are subsequently used for axis labels, legend entries, and colorbar labels
    from .feedbacks import ALIAS_VARIABLES, VARIABLE_ALIASES
    from .process import get_result  # avoid recursive import
    mode = mode or 'long'  # {{{
    dataset = dataset or xr.Dataset({})
    translates = {'path': TRANSLATE_PATHS, 'short': TRANSLATE_SHORTS, 'long': TRANSLATE_LONGS}  # noqa: E501
    if mode not in translates:
        raise ValueError(f'Invalid label mode {mode!r}.')
    translate_operator = {'+': 'plus', '-': 'minus', '*': 'times', '/': 'over'}
    translate_part = translates.get(mode, {})
    options = [key for idx in dataset.indexes.values() for key in idx.names]
    parts = []
    for part in _expand_parts(key, value):
        if key == 'name':  # retrieve without calculating using get_result(attr=attr)
            if part and '|' in part:
                *_, part = part.split('|')  # numerator in pattern regression
            part = ALIAS_VARIABLES.get(part, part)
            if not part:
                label = None
            elif part in translate_operator:
                label = part if mode == 'path' else translate_operator[part]
            elif part in (*options, *dataset.coords):
                label = part  # e.g. get_label('name', 'experiment')
            elif mode == 'path':
                label = VARIABLE_ALIASES.get(part, part)
            elif mode == 'short':
                label = get_result(dataset, part, 'short_name')
            else:
                label = get_result(dataset, part, 'long_name')
        elif part is None or isinstance(part, (str, tuple)):  # can have 'None' labels
            if part and part[0] == 'a' and part != 'ann' and key in ('region', 'period'):  # noqa: E501
                part = part if mode == 'path' else part[1:]  # abrupt-only label
            if part and part == (None, None):
                part = None
            if part and part[0] in translate_operator:
                label = part[0] if mode == 'path' else translate_operator[part[0]]
            else:
                label = translate_part.get((key, part), part)
            if experiment and mode != 'path' and part == 'picontrol':
                label = 'internal'
            if experiment and mode != 'path' and part == 'abrupt4xco2':
                label = 'forced'
            if isinstance(label, tuple) and any(_ is None for _ in label):
                label = ''
            elif isinstance(label, tuple):  # i.e. 'startstop' without shorthand
                label = '-'.join(format(lab, 's' if isinstance(lab, str) else '04.0f') for lab in label)  # noqa: E501
        else:
            unit = get_result(dataset, key, 'units')
            if not isinstance(part, ureg.Quantity):
                part = ureg.Quantity(part, unit)
            part = part.to(unit)
            if part.units == ureg.degE and part > 180 * ureg.deg:
                part = part - 360 * ureg.deg
            if mode == 'path':
                label = f'{part:~.0f}'  # e.g. include degree sign
            else:
                label = f'{part:~P.0f}'.replace(' ', r'$\,$')
        if label is None:  # e.g. skip 'avg' label
            continue
        deg = '\N{DEGREE SIGN}'
        for neg, pos in ('SN', 'WE'):
            if '-' in label and f'{deg}{pos}' in label:
                label = label.replace('-', '').replace(f'{deg}{pos}', f'{deg}{neg}')
        if mode == 'path':  # extra processing
            for symbol in (deg, *translate_operator, '_', ' '):
                label = label.lower().replace(symbol, '')
        parts.append(label)
    if len(parts) > 1:  # if result is still weird user should pass explicit values
        if parts[0] in translate_operator.values():
            parts = parts[1:]  # e.g. experiment=abrupt4xco2, stop=None-20
        if parts[-1] in translate_operator.values():
            parts = parts[:-1]  # e.g. experiment=abrupt4xco2-picontrol, stop=20-None
    sep = '' if mode == 'path' else ' '
    return sep.join(parts)


def get_labels(
    *kws_process, mode=None, short=False, restrict=None, strict=None,
    heading=False, identical=False, skip_name=False, **kwargs,  # noqa: E501
):
    """
    Convert reduction operators into human-readable labels.

    Parameters
    ----------
    *kws_process : list of tuple of dict
        The reduction keyword arguments.
    mode : {'long', 'short', 'path'}, optional
        The label mode to return.
    short : bool, optional
        Whether to use short names instead of long.
    restrict : str or sequence, optional
        The keys used to optionally restrict the label.
    strict : str or sequence, optional
        As with `restrict` but also ensure labels are non-none.
    heading : bool, optional
        Whether to format the labels as headings.
    identical : bool, optional
        Whether to keep identical reduce operations across list.
    skip_name : bool, optional
        Whether to skip names and `spatial` in the label.
    **kwargs
        Passed to `get_label` and `_split_label`.

    Returns
    -------
    labels : list
        The unique or identical label(s).
    """
    # Initial stuff
    # NOTE: For grid labels use intersection of identifiers with same number of
    # arguments. Common to have e.g. x vs. y and then just plot x or y as a
    # reference but the former is the relevant information for labels.
    from observed.arrays import mask_region
    from .process import get_result
    from .reduce import reduce_facets, _reduce_data, _reduce_datas
    invert = spatial = False  # {{{
    mode = mode if mode else 'short' if short else 'long'
    strict = strict or ()
    restrict = restrict or ()
    keys_keep = strict + restrict
    kws_infer = []
    for ikws_process in kws_process:
        if not isinstance(ikws_process, list):
            ikws_process = [ikws_process]
        length, ikws_infer = 0, []
        for jkws_process in ikws_process:
            if not isinstance(jkws_process, tuple):
                jkws_process = (jkws_process,)
            if len(jkws_process) > 2:
                raise ValueError('Invalid input arguments. Must be length 1 or 2.')
            if not all(isinstance(kw, dict) for kw in jkws_process):
                raise ValueError('Invalid input arguments. Must be dictionary.')
            length = max(length, len(jkws_process))
            ikws_infer.append(jkws_process)
        ikws_infer = [kws for kws in ikws_infer if len(kws) == length]
        spatial = spatial or any(kw.get('spatial') for kws in ikws_infer for kw in kws)
        invert = invert or any(kw.get('invert') for kws in ikws_infer for kw in kws)
        kws_infer.append(ikws_infer)

    # Reduce labels dicationaries to drop or select redundancies across the list
    # NOTE: Also drop scalar 'identical' specifications of e.g. 'annual', 'slope',
    # and 'eraint' default selections for feedback variants and climate averages?
    if not spatial:  # standard label order  # {{{
        order_back = ['name', 'method']
        order_read = list(ORDER_READABLE)
    else:  # adjusted label order
        order_back = ['name', 'area', 'volume', 'spatial', 'method']
        order_read = [key for key in ORDER_READABLE if key not in order_back] + order_back  # noqa: E501
    reduces = (get_result, mask_region, reduce_facets, _reduce_data, _reduce_datas)
    sorter = lambda key: order_read.index(key) if key in order_read else len(order_read)
    dataset = kwargs.pop('dataset', None)  # required for 'name' labels
    kws_index = []
    for index in range(2):  # regression pair index
        kw_labels, kws_labels = {}, []
        for ikws_infer in kws_infer:  # iterate over subplots
            kw_label, kws_label = {}, []
            ikws_infer = [kws[index] if index < len(kws) else {} for kws in ikws_infer]
            for ikw_infer in ikws_infer:  # iterate inside subplots
                ikw_label, ikw_infer = {}, _group_parts(ikw_infer)
                for key, value in ikw_infer.items():
                    is_reduce = bool(_pop_kwargs({key: value}, *reduces))
                    if skip_name and key in ('name', 'spatial', 'method'):
                        continue
                    if keys_keep and key not in keys_keep:  # string or tuple
                        continue
                    if is_reduce and key not in keys_keep:  # method explicit only
                        continue
                    kws = {key: ikw_label.get(key) for key in ('period', 'startstop')}
                    label = get_label(key, value, mode=mode, dataset=dataset)
                    if key in kws and label in kws.values():  # avoid duplicates
                        label = None
                    elif not label and key in strict and isinstance(value, str):
                        label = value or None  # possibly force-apply
                    ikw_label[key] = label
                kws_label.append(ikw_label)
            for key in sorted((key for kw in kws_label for key in kw), key=sorter):
                values = tuple(kw.get(key, '') for kw in kws_label)
                kw_label[key] = _merge_labels(*values, identical=True)
            kws_labels.append(kw_label)
        for key in sorted((key for kw in kws_labels for key in kw), key=sorter):
            values = tuple(kw.get(key, '') for kw in kws_labels)
            kw_labels[key] = _merge_labels(*values, identical=identical)
        if not identical:  # dictionary for each separate label
            kw_labels = [{key: kw_labels[key][i] for key in kw_labels} for i in range(len(kws_labels))]  # noqa: E501
        if not kw_labels:
            continue
        kws_index.append(kw_labels)

    # Combine label regression pairs
    # NOTE: This optionally assigns labels that are identical across the pair to
    # the front or the back of the combined 'this vs. that' label.
    kws_index = list(zip(*kws_index)) if not identical else [kws_index]  # {{{
    averages = set(kw.get('area') or '' for kws in kws_index for kw in kws)
    regions = averages - {'', 'local', 'global'}
    kwargs.setdefault('fontsize', pplt.rc.fontlarge if heading else pplt.rc.fontsize)
    labels = []
    for ikws_index in kws_index:  # iterate regression pairs
        # Allocate label components
        keys = set(key for kw in ikws_index for key in kw)
        front, left, right, back = [], [], [], []
        if not invert:  # place dependent variable *first*
            ikws_index = ikws_index[::-1]
        for key in sorted(keys, key=sorter):
            seen = set()  # ignore e.g. 'early early' due to 'period' translation
            values = list(kw.get(key) for kw in ikws_index)
            values = [val for val in values if val not in seen and not seen.add(val)]
            if key == 'area' and len(regions) == len(averages) == 1:
                pass
            elif len(set(values)) > 1:  # one unset or both set to different
                if values[0]:  # e.g. local vs. global feedback
                    left.append(values[0])
                if values[1]:
                    right.append(values[1])
            elif any(values):  # non-empty and non-None
                if key in order_back:  # e.g. 'abrupt vs. picontrol *feedback*'
                    back.append(values[0])
                else:  # e.g. '*cmip6* abrupt vs. picontrol'
                    front.append(values[0])
        # Combine and adjust labels
        remove = lambda lab, key: lab[::-1].replace(key[::-1], '', 1)[::-1]
        seen, control, abrupt = set(), 'pre-industrial', r'abrupt 4$\times$CO$_2$'
        left = seen.clear() or [item for item in left if item not in seen and not seen.add(item)]  # noqa: E501
        right = seen.clear() or [item for item in right if item not in seen and not seen.add(item)]  # noqa: E501
        front = seen.clear() or [item for item in front if item not in seen and not seen.add(item)]  # noqa: E501
        back = seen.clear() or [item for item in back if item not in seen and not seen.add(item)]  # noqa: E501
        left, right = ' '.join(filter(None, left)), ' '.join(filter(None, right))
        center = ' vs. '.join(filter(None, (left, right)))
        label = ' '.join(filter(None, (*front, center, *back)))
        if (sub := f'{abrupt} minus {control}') not in label:
            pass  # difference between experiments
        elif label[-8:] == 'feedback' or label in (sub, f'early {sub}', f'late {sub}'):
            pass  # special exceptions in case 'feedback' is missing
        else:
            label = remove(f'{label.replace(sub, abrupt)} response', f'{abrupt} ')
        if (sub := 'CMIP5 plus CMIP6') in label:
            label = label.replace(sub, 'CMIP')
        if (sub := 'temperature response') in label:
            label = label.replace(sub, 'warming')
        if (sub := rf'{abrupt} 2$\times$CO$_2$') in label:  # drop '2xCO2' scaling
            label = label.replace(sub, abrupt, 1)
        if label.count(sub := 'boreal') == 2:  # e.g. 'boreal winter - boreal summer'
            label = remove(label, f'{sub} ')
        if heading and 'anomaly' in label:  # convert plural
            label = label.replace('anomaly', 'anomalies')
        if control in label and 'surface warming' in label:  # convert 'ts'
            label = label.replace('warming', 'temperature')
        if identical and label[-8:] == 'feedback':  # change end to 'feedbacks'
            label = f'{label}s'
        label = _split_label(label.strip(), **kwargs)
        label = get_heading(label) if heading else label
        labels.append(label)
    return labels[0] if identical else labels


def parse_spec(dataset, spec, **kwargs):
    """
    Parse the variable name specification.

    Parameters
    ----------
    dataset : `xarray.Dataset`
        The dataset.
    bootstrap : bool, optional
        Optional alternative for specifying the control experiment period.
    spec : sequence, str, or dict
        The variable specification. Can be a ``(name, kwargs)``, a naked ``name``,
        or a naked ``kwargs``. The name can be omitted or specified inside `kwargs`
        with the `name` dictionary, and it will be specified when intersected with
        other specifications (i.e. the row-column framework; see `parse_specs`).
    **kwargs
        Additional keyword arguments, used as defaults for the unset keys
        in the variable specifications.

    Returns
    -------
    spec : dict
        The keyword arguments used to reduce the data variable with `process_data`.
        Includes `reduce_facets` and `reduce_general` instructions.
    collection : namedtuple of dict
        A named tuple containing keyword arguments for different plotting-related
        commands. The tuple fields are as follows:

          * ``figure``: Passed to `Figure` when the figure is instantiated.
          * ``gridspec``: Passed to `GridSpec` when the gridfspec is instantiated.
          * ``axes``: Passed to `.format` for cartesian or geographic formatting.
          * ``colorbar``: Passed to `.colorbar` for scalar mappable outputs.
          * ``legend``: Passed to `.legend` for other artist outputs.
          * ``command``: Passed to the plotting command (the default field).
          * ``attrs``: Added to `.attrs` for use in resulting plot labels.
          * ``other``: Custom keyword arguments for plotting options.
    """
    # Initial stuff
    # NOTE: This only works for optional keyword arguments with explicit defaults
    # and only for methods that have not been obfuscated (see _format_signature).
    from .general import _format_axes, _format_bars, _format_scatter  # {{{
    from .general import _merge_commands, _setup_command, _setup_defaults
    from .process import get_result, process_constraint
    from .reduce import reduce_facets, _reduce_data, _reduce_datas
    formats = tuple(pplt.Axes._format_signatures.values())  # {{{
    others = (_format_axes, _format_bars, _format_scatter)
    others = (*others, _merge_commands, _setup_command, _setup_defaults)
    grids = [k + s for k in ('', 'ax', 'ref', 'fig') for s in ('', 'num', 'width', 'height', 'aspect')]  # noqa: E501
    props = ('c', 'lw', 'color', 'linewidth', 'facecolor', 'edgecolor', 'a', 'alpha')
    subs = [k + s for k in ('span', 'share', 'align') for s in ('', 'x', 'y')]
    if spec is None:
        name, kw = None, {}
    elif isinstance(spec, str):
        name, kw = spec, {}
    elif isinstance(spec, dict):
        name, kw = None, spec
    else:  # length-2 iterable
        name, kw = spec
    kw = {**kwargs, **kw}  # prefer spec arguments
    name = name or kw.pop('name', None)  # see below

    # Get keyword arguments
    # WARNING: Critical to always parse facet and version levels since figures.py will
    # auto apply these coordinates even if not present e.g. for bootstrap datasets. Then
    # have process.py ignore them when version is not present.
    values = getattr(dataset, 'period', [])  # {{{
    period = '24yr' if '24yr' in values else '23yr' if '23yr' in values else '20yr'
    source = kw.get('source', None)  # TODO: remove kludge?
    institute = kw.get('institute', None)
    bootstrap = kw.pop('bootstrap', None)  # TODO: remove kludge?
    experiment = kw['experiment'] or 'abrupt4xco2' if 'experiment' in kw else None
    kw_process = _pop_kwargs(kw, dataset, get_result, reduce_facets, _reduce_data, _reduce_datas)  # noqa: E501
    kw_attrs = _pop_kwargs(kw, 'short_name', 'long_name', 'standard_name', 'units')
    kw_grid = _pop_kwargs(kw, pplt.GridSpec._update_params)  # overlaps kw_figure
    kw_figure = _pop_kwargs(kw, *subs, *grids, pplt.Figure._format_signature)
    kw_axes = _pop_kwargs(kw, *formats, pplt.Figure._parse_proj)
    kw_other = _pop_kwargs(kw, *others, process_constraint)
    kw_command = _pop_kwargs(kw, 'cmap', 'cycle', 'extend', *props)
    kw_config = _pop_kwargs(kw, tuple(_rc_nodots))  # overlaps kw_command (see above)
    kw_guide = _pop_kwargs(kw, pplt.Axes._add_legend, pplt.Axes._add_colorbar)
    kw_legend = {**kw_guide, **_pop_kwargs(kw, 'legend', pplt.Axes._add_legend)}
    kw_colorbar = {**kw_guide, **_pop_kwargs(kw, 'colorbar', pplt.Axes._add_colorbar)}
    kw_axes.update(kw_config)  # configuration settings
    kw_figure.update(kw_config)  # configuration settings
    kw_command.update(kw)  # unknown kwargs passed to command

    # Repair variable spes
    # NOTE: For subsequent processing we put the variables being combined (usually one)
    # inside process 'name' key. This helps when merging variable specifications
    # between row and column specs and between tuple-style specs (see parse_specs).
    if source == 'merged' and name in (None, 'tstd', 'tdev', 'tpat', 'tabs'):  # {{{
        kw_process.pop('source', None)  # merged scalar spatial data
    if institute == 'wgt':
        kw_process.pop('institute', None)
    if institute == 'wgt':
        kw_other['weight'] = kw_process['weight'] = True
    if name is not None:
        kw_process['name'] = name
    if bootstrap is not None and experiment == 'picontrol':
        kw_process['period'] = period if bootstrap else 'full'  # overwrite control
    elif bootstrap is not None and experiment == 'abrupt4xco2':
        kw_process['period'] = kw_process.get('period', 'full')  # avoid using control
    if 'ocean' in kw_axes:  # ocean mask
        kw_process['ocean'] = kw_axes.pop('ocean')
    if 'mask' in kw_command:  # mask region
        kw_process['mask'] = kw_command.pop('mask')
    if 'width' in kw_figure:  # bar widths
        kw_command['width'] = kw_figure.pop('width')
    if 'colorbar' in kw_colorbar:  # colorbar location, or use 'loc' for both
        kw_colorbar['loc'] = kw_colorbar.pop('colorbar')
    if 'legend' in kw_legend:  # legend location, or use 'loc' for both
        kw_legend['loc'] = kw_legend.pop('legend')
    fields = ('figure', 'gridspec', 'axes', 'command', 'other', 'attrs', 'colorbar', 'legend')  # noqa: E501
    collection = collections.namedtuple('kwargs', fields)
    kw_collection = collection(
        kw_figure, kw_grid, kw_axes, kw_command,
        kw_other, kw_attrs, kw_colorbar, kw_legend
    )
    return kw_process, kw_collection


def parse_specs(
    dataset, rowspecs=None, colspecs=None, rows=None, cols=None, autocmap=None, **kwargs
):
    """
    Parse variable and project specifications and auto-determine row and column
    labels based on the unique names and/or keywords in the spec lists.

    Parameters
    ----------
    dataset : xarray.Dataset
        The source dataset.
    rowspecs : list of name, tuple, dict, or list thereof
        The variable specification(s) per subplot slot.
    colspecs : list of name, tuple, dict, or list thereof
        The variable specification(s) per subplot slot.
    rows, cols : str or sequence, optional
        The `get_label` keys used in row and column labels.
    autocmap : bool or 2-tuple, optional
        Whether to automatically select a non-diverging or diverging colormap.
    **kwargs
        Additional options shared across all specs.

    Returns
    -------
    kws_process : list of list of [12]-tuple of dict
        The keyword arguments for `reduce_facets` and `process_data`.
    kws_collection : list of list of namedtuple
        The keyword arguments for various plotting tasks.
    figlabel : str
        The default figure title center.
    pathlabel : str
        The default figure path center.
    gridlabels : list of list of str
        The default row and column labels.
    """
    # Parse variable specs per gridspec row or column and per subplot
    # NOTE: The two arrays required for two-argument methods can be indicated with
    # either 2-tuple dictionaries in spec lists, conflicting row and column names
    # and coordiantes, or 2-tuple values in single-dictionary specifier lists.
    abcloc = refwidth = refscale = None
    kw_shared = dict(dataset=dataset, scale=kwargs.get('scale'))  # noqa: E501
    kws_process, kws_collection, gridlabels = [], [], []
    for idx, (ispecs, ikeys) in enumerate(((rowspecs, rows), (colspecs, cols))):
        # Generate variable specs
        # NOTE: This permits sharing keywords across each group with trailing dicts
        # in either the primary gridspec list or any of the subplot sub-lists.
        ikws_process, ikws_collection = [], []  # {{{
        if not isinstance(ispecs, list):
            ispecs = [ispecs]
        for jspecs in ispecs:  # specs per figure
            jkws_process, jkws_collection = [], []
            if not isinstance(jspecs, list):
                jspecs = [jspecs]
            for sspecs in jspecs:  # specs per subplot
                # Translate into name and dictionary
                skws_pair, skws_process, skws_collection = [{}, {}], [], []
                if sspecs is None:
                    sspecs = (None,)  # possibly construct from keyword args
                elif isinstance(sspecs, (str, dict)):
                    sspecs = (sspecs,)
                elif len(sspecs) != 2:
                    raise ValueError(f'Invalid variable specifier {sspecs}.')
                elif type(sspecs[0]) != type(sspecs[1]):  # noqa: E721  # (str, dict)
                    sspecs = (sspecs,)
                else:
                    sspecs = tuple(sspecs)
                # Iterate over correlation pairs
                for spec in sspecs:
                    kw_process, kw_collection = parse_spec(dataset, spec, **kwargs)
                    if value := kw_collection.axes.get('abcloc', None):
                        abcloc = value
                    if value := kw_collection.figure.get('refwidth', None):
                        refwidth = value  # for scaling grid labels
                    if not any(kw_process.get(key) for key in ('lon', 'lat', 'area')):
                        refscale = 0.6 if idx == 0 else 1.0  # i.e. longitude-latitude
                    for key, value in tuple(kw_process.items()):
                        if isinstance(value, tuple) and len(value) == 2:
                            skws_pair[0][key], skws_pair[1][key] = kw_process.pop(key)
                    skws_process.append(kw_process)
                    skws_collection.append(kw_collection)
                if any(skws_pair):
                    if len(skws_process) == 1:
                        skws_process = [skws_process[0].copy(), skws_process[0].copy()]
                    for kw_process, kw_pair in zip(skws_process, skws_pair):
                        kw_process.update(kw_pair)
                jkws_process.append(tuple(skws_process))  # denotes correlation-pair
                jkws_collection.append(tuple(skws_collection))
            ikws_process.append(jkws_process)
            ikws_collection.append(jkws_collection)

        # Infer grid labels
        # NOTE: This automatically infers linebreaks based on input values and
        # accounts for space taken by a-b-c labels if possible.
        abcwidth = pplt.units(1 * pplt.rc.fontsize, 'pt', 'in')  # {{{
        abcwidth = abcwidth if abcloc in ('l', 'c', 'r', None) else 0
        refwidth = pplt.units(refwidth or pplt.rc['subplots.refwidth'], 'in')
        refwidth -= abcwidth if len(rowspecs) < 2 or len(colspecs) < 2 else 0
        grdlabels = get_labels(
            *ikws_process,
            identical=False,
            heading=True,
            restrict=ikeys,  # optionally restrict keywords
            refwidth=(refscale or 1) * refwidth,  # account for a-b-c space
            **kw_shared,
        )
        gridlabels.append(grdlabels)
        kws_process.append(ikws_process)
        kws_collection.append(ikws_collection)

    # Combine row and column specifications for plotting and file naming
    # WARNING: Critical to make copies of dictionaries or create new ones
    # here since itertools product repeats the same spec multiple times.
    kws_rowcol = [  # {{{
        [
            list(zip(jkws_process, jkws_collection))
            for jkws_process, jkws_collection in zip(ikws_process, ikws_collection)
        ]
        for ikws_process, ikws_collection in zip(kws_process, kws_collection)
    ]  # }}}
    kws_grid, kws_process, kws_collection = {}, [], []
    for (row, ikws_row), (col, ikws_col) in itertools.product(*map(enumerate, kws_rowcol)):  # noqa: E501
        # Generate variable specifiers
        # NOTE: Several plotted values per subplot can be indicated in either the
        # row or column list, and the specs from the other list are repeated below.
        ikws_row, ikws_col = _expand_lists(ikws_row, ikws_col)  # {{{
        ikws_process, ikws_collection = [], []
        for idx, (jkws_row, jkws_col) in enumerate(zip(ikws_row, ikws_col)):
            # Combine row and column keywords
            rkws_process, rkws_collection = jkws_row
            ckws_process, ckws_collection = jkws_col
            collection = type((rkws_collection or ckws_collection)[0])
            kws = []
            for field in collection._fields:
                kw = {}  # NOTE: previously applied default values here
                for ikws in (rkws_collection, ckws_collection):
                    for ikw in ikws:  # correlation pairs
                        for key, value in getattr(ikw, field).items():
                            kw.setdefault(key, value)  # prefer row entries
                kws.append(kw)
            kw_collection = collection(*kws)
            # Intersect processing keywords
            rkws_process = [kw.copy() for kw in rkws_process]  # iterate pairs
            ckws_process = [kw.copy() for kw in ckws_process]  # iterate pairs
            if len(rkws_process) not in (1, 2) or len(ckws_process) not in (1, 2):
                raise ValueError(f'Invalid combination {rkws_process} and {ckws_process}.')  # noqa: E501
            if len(rkws_process) == 1 and len(ckws_process) == 1:
                kw_process = (*rkws_process, *ckws_process)
            elif len(rkws_process) == 2 and len(ckws_process) == 2:
                kw_process = rkws_process
                for rkw, ckw in zip(rkws_process, ckws_process):
                    for key, value in ckw.items():
                        rkw.setdefault(key, value)
            else:  # one is scalar the other vector
                kw_process = (ckws_process, rkws_process)[len(rkws_process) == 2]
                kw_reference = (ckws_process, rkws_process)[len(rkws_process) != 2]
                for key, value in kw_reference[0].items():  # noqa: E501
                    for kw in kw_process:
                        kw.setdefault(key, value)
            # Combine pairs and possibly apply autocmap
            if len(kw_process) == 2:
                from .reduce import reduce_facets
                for kw1, kw2 in (kw_process, kw_process[::-1]):
                    for key, value in kw2.items():
                        kw1.setdefault(key, value)
                kws_check = [kw.copy() for kw in kw_process]
                for kw in kws_check:
                    _pop_kwargs(kw, reduce_facets)
                if kws_check[0] == kws_check[1]:
                    kw_process = kw_process[:1]
            if autocmap:  # use different colormaps for project and other anomalies
                index = max(
                    0 if not isinstance(value, str) or '-' not in value
                    else 1 if key != 'project' else 2
                    for kw in kw_process for key, value in kw.items()
                )
                cmaps = ('Fire', 'NegPos', 'NegPos') if autocmap is True else autocmap
                kw_collection.command['cmap'] = cmaps[index]
                kw_collection.command.setdefault('robust', 98 - 2 * index)
                kw_collection.command.setdefault('diverging', index > 0)
            kw_grid = kws_grid.setdefault(idx, {})
            kw_grid[row, col] = tuple(kw_process)
            ikws_process.append(tuple(kw_process))
            ikws_collection.append(kw_collection)

        # Infer legend and axes prefixes
        # TODO: Optionaly use either 'short' or 'skip_name'? Often want single label
        # for contours but multiple longer labels for line plots.
        ikws_pair = [ikws_process[0][:1], ikws_process[0][-1:]]  # {{{
        dist_pair = len(ikws_process[0]) == 2 and ikws_process[0][-1].get('area')
        kw_split = dict(refwidth=np.inf, identical=False, heading=False, short=False)
        prefixes_axes = get_labels(*ikws_pair, skip_name=True, **kw_split, **kw_shared)
        prefixes_legend = get_labels(*ikws_process, **kw_split, **kw_shared)
        for axis, prefix in zip('xy', prefixes_axes):
            if prefix and dist_pair:
                ikws_collection[-1].attrs.setdefault(f'{axis}label_prefix', prefix)
        for pspec, prefix in zip(ikws_collection, prefixes_legend):
            if prefix:  # add prefix to this
                pspec.attrs.setdefault('short_prefix', prefix)
        kws_process.append(ikws_process)
        kws_collection.append(ikws_collection)

    # Infer figure label and grid labels
    # NOTE: Here only return tuple gridlabels if more than one are present
    # TODO: Copy below algorithm used to determine column count to general_plot()
    specs0, specs1 = [], []  # {{{
    strict = [name for idx in dataset.indexes.values() for name in idx.names]
    strict.extend(('area', 'name', *dataset.dims, *dataset.coords))
    ncols, nrows = len(colspecs), len(rowspecs)  # figure dimensions
    shift = abcloc in ('l', 'c', 'r', None)
    ncols = ncols if ncols > 1 else nrows if nrows > 1 else 3
    figwidth = ncols * refwidth + 0.3 * refwidth * (ncols - 1)  # approximate width
    kw_figure = dict(refwidth=figwidth, heading=True, identical=True, short=False)
    kw_split = dict(fontsize=pplt.rc.fontlarge, refwidth=refwidth, shift=shift, scale=kw_shared['scale'])  # noqa: E501
    kw_grid = dict(mode='path', refwidth=np.inf)
    figlabel = get_labels(*kws_process, **kw_figure, **kw_shared)
    pathlabel = get_path(dataset, *(spec for ikws in kws_process for spec in ikws))
    if len(rowspecs) == 1 and len(colspecs) == 1:
        gridlabels = None
    elif len(rowspecs) > 1 and len(colspecs) > 1:
        gridlabels = tuple(gridlabels)  # NOTE: tuple critical for general_plot
    elif len(rowspecs) > 1:
        gridlabels = [_split_label(label, **kw_split) for label in gridlabels[0]]
    else:
        gridlabels = [_split_label(label, **kw_split) for label in gridlabels[1]]
    for kws in kws_grid.values():
        args = [tuple(_group_parts(kw) for kw in ikws) for ikws in kws.values()]
        label0 = get_labels(*args, identical=True, **kw_grid, **kw_shared)
        labels = get_labels(*args, identical=False, **kw_grid, **kw_shared)
        specs0.append(label0)
        specs1.append(dict(zip(kws, labels)))
    idxs = sorted(set(key for kws in specs1 for key in kws))
    specs0 = [f'({idx + 1}) {spec!r}' for idx, spec in enumerate(specs0)]
    specs1 = [' '.join(f'{spec!r}' for kws in specs1 if (spec := kws.get(idx))) for idx in idxs]  # noqa: E501
    specs1 = [f'({row + 1}, {col + 1}) {spec}' for (row, col), spec in zip(idxs, specs1)]  # noqa: E501
    print('Specs:', *specs0, *specs1)
    return kws_process, kws_collection, figlabel, pathlabel, gridlabels
