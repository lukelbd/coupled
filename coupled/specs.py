#!/usr/bin/env python3
"""
Helper functions for parsing plot specifications.
"""
import collections
import inspect
import itertools
import re

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
    'season',
    'month',
    'time',  # space and time
    'plev',
    'area',
    'volume',
    'lat',
    'lon',
    'spatial',  # always at the end
)
ORDER_READABLE = (
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
)

# General long and short labels
# NOTE: These are used for figure path generation and for figure axes
# and attributes. See below for details.
GENERAL_LABELS = {
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
    ('institute', 'avg'): None,  # or 'institute-average'
    ('institute', 'flagship'): None,  # or 'flagship-only'
    ('institute', None): None,  # or 'individual-model'
    ('experiment', 'picontrol'): 'control',
    ('experiment', 'abrupt4xco2'): r'4$\times$CO$_2$',
    ('source', 'eraint'): 'Davis et al.',
    ('source', 'zelinka'): 'Zelinka et al.',
    ('style', 'slope'): 'annual',
    ('style', 'annual'): 'annual',
    ('style', 'monthly'): 'monthly',
    ('style', 'ratio'): 'ratio-style',
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
    ('region', 'globe'): 'global-$T$',
    ('region', 'point'): 'local-$T$',
    ('region', 'latitude'): 'zonal-$T$',
    ('region', 'hemisphere'): 'hemispheric-$T$',
    ('spatial', 'slope'): 'spatial regression',  # NOTE: also see _apply_double
    ('spatial', 'proj'): 'spatial projection',
    ('spatial', 'corr'): 'spatial correlation',
    ('spatial', 'cov'): 'spatial covariance',
    ('spatial', 'rsq'): 'spatial variance explained',
    ('plev', 'int'): None,
    ('plev', 'avg'): 'column',
    ('area', None): 'local',  # only used with identical=False
    ('area', 'avg'): 'global',  # only used with identical=False
    ('area', 'trop'): 'tropical',  # only used with identical=False
    ('area', 'tpac'): 'tropical Pacific',
    ('area', 'wpac'): 'West Pacific',
    ('area', 'epac'): 'East Pacific',
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
    ('startstop', (0, 150)): 'full',
    ('startstop', (1, 150)): 'full1',
    ('startstop', (2, 150)): 'full2',
    ('startstop', (0, 20)): 'early',
    ('startstop', (1, 20)): 'early1',
    ('startstop', (2, 20)): 'early2',
    ('startstop', (0, 50)): 'early50',
    ('startstop', (20, 150)): 'late',
    ('startstop', (100, 150)): 'late50',
    ('region', 'point'): 'loc',
    ('region', 'latitude'): 'lat',
    ('region', 'hemisphere'): 'hemi',
    ('region', 'globe'): 'globe',
    ('region', 'apoint'): 'apt',
    ('region', 'alatitude'): 'alat',
    ('region', 'ahemisphere'): 'ahemi',
    ('region', 'aglobe'): 'aglobe',
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
    for key in ('period', 'season')
    for value, label in PERIOD_LONGS.items()
}
PERIOD_SHORTS = {
    (key, value): label
    for key in ('period', 'season')
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
TRANSLATE_PATHS = {
    **GENERAL_PATHS,
    **INSTITUTE_PATHS,
}
TRANSLATE_SHORTS = {
    **GENERAL_LABELS,
    **INSTITUTE_SHORTS,
    **PERIOD_SHORTS,
}
TRANSLATE_LONGS = {
    **GENERAL_LABELS,
    **INSTITUTE_LONGS,
    **PERIOD_LONGS,
}


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
    keys = []
    keys_levels = (*FACETS_LEVELS, *VERSION_LEVELS, 'statistic')
    keys_levels += ('area', 'volume', 'remove', 'detrend', 'correct')
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
        raise ValueError(f'Incompatible mixed lengths {lengths} for values\n{values}.')
    length = length.pop() if length else None
    for i, items in enumerate(args):
        if length and len(items) == 1:
            args[i] = length * items
    return args


def _expand_parts(value):
    """
    Expand reduction keyword arguments.

    Parameters
    ----------
    value : object
        The reduction value.
    """
    # NOTE: This translates e.g. startstop=('20-0', '150-20') to the anomaly
    # pairs [('20', '150'), ('-', '-'), ('0', '20')] for better processing.
    parts, values = [], value if isinstance(value, tuple) else (value,)
    for value in values:
        iparts = []  # support 'startstop' anomaly tuples e.g. startstop=(20-0, 150-20)
        for part in (REGEX_SPLIT.split(value) if isinstance(value, str) else (value,)):
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
    # early vs. unperturbed' regression coefficients, in which case reduce_general()
    # will translate denominator to 'full unperturbed minus full unperturbed' i.e.
    # zero, and case where we are doing e.g. spatial version of this regression,
    # where reduce_general() gets regression coefficients before the subtraction
    # operation so there is no need to worry about denominator. Solution is to
    # keep same number of operators (so numerator and denominator have same number
    # and thus can be combined) but replace with just '+' i.e. a dummy average.
    kwargs = kwargs.copy()
    if 'stop' in kwargs:
        kwargs.setdefault('start', None)
    if 'start' in kwargs:
        kwargs.setdefault('stop', None)
    if 'start' in kwargs and 'stop' in kwargs:
        start, stop = kwargs.pop('start'), kwargs.pop('stop')
        if kwargs.get('experiment') == 'picontrol':
            num1 = sum(map(start.count, '+-')) if isinstance(start, str) else 0
            num2 = sum(map(stop.count, '+-')) if isinstance(stop, str) else 0
            if keep_operators:
                start = '+'.join(itertools.repeat('0', num1 + 1))
                stop = '+'.join(itertools.repeat('150', num2 + 1))
            else:
                start = None
                stop = None
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
    strings = (
        r'2$\times$CO$_2$', r'4$\times$CO$_2$',  # scaling terms
        'TOA', 'top-of-aatmosphere', 'surface', 'atmospheric',  # boundary terms
        'feedback', 'forcing', 'spatial', 'pattern',  # feedback terms
        'flux', 'energy', 'transport', 'convergence',  # transport terms
    )
    strings = [s for s in strings if all(s in label for label in labels)]
    pattern = '|'.join(r'\s*' + re.escape(s) for s in strings)
    regex = re.compile(f'({pattern})')
    ident = all(label == labels[0] for label in labels)
    if identical:
        if ident:  # note always true if labels is singleton
            labels = labels[0]
        else:
            labels = [''.join(regex.findall(labels[0])).strip() for label in labels]
            labels = labels[0] if all(label == labels[0] for label in labels) else ''
    else:
        if ident:  # note always true if labels is singleton
            labels = [''] * len(labels)
        else:
            labels = [regex.sub('', label).strip() for label in labels]
    return labels


def _split_label(label, refwidth=None, fontsize=None, offset=None, nmax=None):
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
    offset : bool, optional
        Whether to offset the final line.
    nmax : int, optional
        The maximum number of breaks to use.
    """
    # Adjust label for math text
    # TODO: Search for existing matplotlib auto-font-wrapping utilities. Suspect
    # could use e.g. tight layout and bounding box logic.
    label = label or ''  # default string
    label = label.replace('\n', ' ') + ' '  # remove previous breaks
    imasks, imaths = [], []  # space within math
    ispace = np.array([i for i, c in enumerate(label) if c == ' '])
    icheck = ispace.astype(float)
    for m in re.finditer(r'\$[^$]+\$', label):  # skip math between
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
    # Select line break positions
    # NOTE: This counts characters toward fixed em-width and optionally offsets
    # to prevent centered axes title from overlapping left-aligned a-b-c label
    for mask1, mask2, adjust in imasks:
        icheck[mask1] = np.nan  # no breaks here
        icheck[mask2] -= adjust
    size_ = fontsize or pplt.rc['font.size']
    scale = pplt.utils._fontsize_to_pt(size_) / pplt.rc['font.size']
    width = refwidth or pplt.rc['subplots.refwidth']
    width = pplt.units(width, 'in', 'em')
    width -= EM_PER_ABC * scale if offset else 0
    splits = CHAR_PER_EM * width * np.arange(1, 20) / scale
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
        if parts:
            labels.append(parts)  # sort these so newer orders overwrite
    result = '_'.join('-'.join(sorted(parts)) for parts in labels)
    return result


def get_label(key, value, mode=None, dataset=None, perturbed=True):
    """
    Return an arbitrary label type based on the dataset.

    Parameters
    ----------
    key : str
        The reduce coordinate.
    value : str or float
        The reduce selection.
    mode : {'long', 'short', 'path'}, optional
        The label mode used to translate selections.
    dataset : xarray.Dataset, optional
        The source dataset. Required for `name` labels.
    perturbed : bool, optional : str, optional
        Whether to use ``'unperturbed'`` and ``'perturbed'`` labels.

    Returns
    -------
    label : str
        The final label.
    """
    # NOTE: This function is used for axis label prefixes, legend entry prefixes, row
    # and column labels, and figure and path titles. It is never used for the 'method'
    # key because this is added to the data short and long names during application,
    # which are subsequently used for axis labels, legend entries, and colorbar labels
    from .feedbacks import ALIAS_VARIABLES, VARIABLE_ALIASES
    from .process import get_result  # avoid recursive import
    mode = mode or 'long'
    if mode not in ('path', 'short', 'long'):
        raise ValueError(f'Invalid label mode {mode!r}.')
    operator_to_label = {'+': 'plus', '-': 'minus', '*': 'times', '/': 'over'}
    dataset = dataset or xr.Dataset({})
    parts = (value,) if key in ('model', 'institute') else _expand_parts(value)
    labels = []
    for part in parts:
        if key == 'name':  # retrieve without calculating using get_result(attr=attr)
            if part and '|' in part:
                *_, part = part.split('|')  # numerator in pattern regression
            part = ALIAS_VARIABLES.get(part, part)
            if part and part in operator_to_label:
                label = part if mode == 'path' else operator_to_label[part]
            elif mode == 'path':
                label = VARIABLE_ALIASES.get(part, part)
            elif mode == 'short':
                label = get_result(dataset, part, 'short_name')
            else:
                label = get_result(dataset, part, 'long_name')
        elif part is None or isinstance(part, (str, tuple)):  # can have 'None' labels
            if part and part[0] == 'a' and part != 'ann' and key in ('region', 'period'):  # noqa: E501
                part = part if mode == 'path' else part[1:]  # abrupt-only label
            if part == (None, None):
                label = None
            elif part and part[0] in operator_to_label:  # e.g. (-, -) startstop tuples
                label = part[0] if mode == 'path' else operator_to_label[part[0]]
            elif mode == 'path':  # e.g. 'startstop'
                label = TRANSLATE_PATHS.get((key, part), part)
            elif mode == 'short':
                label = TRANSLATE_SHORTS.get((key, part), part)
            else:  # e.g. 'startstop'
                label = TRANSLATE_LONGS.get((key, part), part)
            if perturbed and mode != 'path' and part == 'picontrol':
                label = 'unperturbed'
            if perturbed and mode != 'path' and part == 'abrupt4xco2':
                label = 'perturbed'
            if isinstance(label, tuple) and any(_ is None for _ in label):
                label = ''
            elif isinstance(label, tuple):  # i.e. 'startstop' without shorthand
                label = '-'.join(format(lab, 's' if isinstance(lab, str) else '04.0f') for lab in label)  # noqa: E501
        else:
            unit = get_result(dataset, key, 'units')
            if not isinstance(part, ureg.Quantity):
                part = ureg.Quantity(part, unit)
            part = part.to(unit)
            if part.units == ureg.parse_units('degE') and part > 180 * ureg.deg:
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
            for symbol in (deg, *operator_to_label, '_', ' '):
                label = label.lower().replace(symbol, '')
        labels.append(label)
    if len(labels) > 1:  # if result is still weird user should pass explicit values
        if labels[0] in operator_to_label.values():
            labels = labels[1:]  # e.g. experiment=abrupt4xco2, stop=None-20
        if labels[-1] in operator_to_label.values():
            labels = labels[:-1]  # e.g. experiment=abrupt4xco2-picontrol, stop=20-None
    sep = '' if mode == 'path' else ' '
    label = sep.join(labels)
    return label


def get_labels(
    *kws_process, short=False, heading=False, identical=False, skip_name=False, dataset=None, **kwargs,  # noqa: E501
):
    """
    Convert reduction operators into human-readable labels.

    Parameters
    ----------
    *kws_process : list of tuple of dict
        The reduction keyword arguments.
    short : bool, optional
        Whether to use short names instead of long.
    heading : bool, optional
        Whether to format the labels as headings.
    identical : bool, optional
        Whether to keep identical reduce operations across list.
    skip_name : bool, optional
        Whether to skip names and `spatial` in the label.
    dataset : xarray.Dataset
        The source dataset. Required for `name` labels.
    **kwargs
        Passed to `_split_label`.

    Returns
    -------
    labels : list
        The unique or identical label(s).
    """
    # Initial stuff
    # NOTE: For grid labels use intersection of identifiers with same number of
    # arguments. Common to have e.g. x vs. y and then just plot x or y as a
    # reference but the former is the relevant information for labels.
    from .reduce import reduce_facets
    from .process import get_result
    kws_infer = []
    spatial = invert = False
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
        kws_infer.append(ikws_infer)
        spatial = spatial or any(kw.get('spatial') for kws in ikws_infer for kw in kws)
        invert = invert or any(kw.get('invert') for kws in ikws_infer for kw in kws)

    # Reduce labels dicationaries to drop or select redundancies across the list
    # TODO: Also drop scalar 'identical' specifications of e.g. 'annual', 'slope',
    # and 'eraint' default selections for feedback variants and climate averages?
    if not spatial:
        order_back = ('name',)
        order_read = list(ORDER_READABLE)
    else:
        order_back = ['name', 'area', 'volume', 'spatial']
        order_read = [key for key in ORDER_READABLE if key not in order_back] + order_back  # noqa: E501
    sorter = lambda key: order_read.index(key) if key in order_read else len(order_read)
    kw_get = {'dataset': dataset, 'mode': 'short' if short else 'long'}
    kws_label = []
    for n in range(2):  # index in possible correlation pair
        ikws_label = []
        for ikws_infer in kws_infer:  # iterate over subplots
            nkws_label = []
            nkws_infer = [kws[n] if n < len(kws) else {} for kws in ikws_infer]
            for kw_infer in nkws_infer:  # iterate over arguments inside subplot
                kw_label, kw_infer = {}, _group_parts(kw_infer)
                for key, value in kw_infer.items():  # get individual label
                    if _pop_kwargs({key: value}, reduce_facets, get_result):
                        continue
                    if skip_name and key in ('name', 'spatial'):
                        continue
                    kw_label[key] = get_label(key, value, **kw_get)
                nkws_label.append(kw_label)
            kw_label = {}  # merge labels for stuff inside subplot
            for key in sorted((key for kw in nkws_label for key in kw), key=sorter):
                values = tuple(kw.get(key, '') for kw in nkws_label)
                kw_label[key] = _merge_labels(*values, identical=True)
            ikws_label.append(kw_label)
        kw_label = {}  # merge labels across subplots
        for key in sorted((key for kw in ikws_label for key in kw), key=sorter):
            values = tuple(kw.get(key, '') for kw in ikws_label)
            kw_label[key] = _merge_labels(*values, identical=identical)
        if not identical:  # dictionary for each separate label
            kw_label = [{key: kw_label[key][i] for key in kw_label} for i in range(len(ikws_label))]  # noqa: E501
        if not kw_label:  # WARNING: critical or else zip below creates empty list
            continue
        kws_label.append(kw_label)

    # Combine pairs of labels
    # NOTE: This optionally assigns labels that are identical across the pair to
    # the front or the back of the combined 'this vs. that' label.
    kws_label = [kws_label] if identical else list(zip(*kws_label))
    kwargs.setdefault('fontsize', pplt.rc.fontlarge if heading else pplt.rc.fontsize)
    found = set(kw.get('area', None) or '' for kws in kws_label for kw in kws)
    skip = set(item for item in found if not item or item in ('local', 'global'))
    labels = []
    for kws_pair in kws_label:
        # Allocate label components
        keys = set(key for kw in kws_pair for key in kw)
        front, left, right, back = [], [], [], []
        if not invert:
            kws_pair = kws_pair[::-1]  # place dependent variable *first*
        for key in sorted(keys, key=sorter):
            items = list(kw.get(key) for kw in kws_pair)
            if key == 'area' and len(skip) == len(found) == 1:
                pass
            elif len(set(items)) > 1:  # one unset or both set to different
                if items[0]:  # e.g. local vs. global feedback
                    left.append(items[0])
                if items[1]:
                    right.append(items[1])
            elif any(items):  # non-empty and non-None
                if key in order_back:  # e.g. 'abrupt vs. picontrol *feedback*'
                    back.append(items[0])
                else:  # e.g. '*cmip6* abrupt vs. picontrol'
                    front.append(items[0])
        # Combine and adjust labels
        remove = lambda lab, key: lab[::-1].replace(key[::-1], '', 1)[::-1]
        control, abrupt = 'pre-industrial', r'abrupt 4$\times$CO$_2$'
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
    # NOTE: For subsequent processing we put the variables being combined (usually one)
    # inside process 'name' key. This helps when merging variable specifications
    # between row and column specs and between tuple-style specs (see parse_specs).
    # WARNING: Critical to always parse facet and version levels since figures.py will
    # auto apply these coordinates even if not present e.g. for bootstrap datasets. Then
    # have process.py ignore them when version is not present.
    from .general import _merge_dists, _init_command, _props_command, _setup_bars, _setup_scatter  # noqa: E501
    from .process import get_result, process_constraint
    from .reduce import reduce_facets
    keys_figure = [k + s for k in ('', 'ax', 'ref', 'fig') for s in ('', 'num', 'width', 'height', 'aspect')]  # noqa: E501
    keys_tight = [k + s for k in ('span', 'share', 'align') for s in ('', 'x', 'y')]
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
    settings = ('c', 'lw', 'color', 'linewidth', 'color', 'facecolor', 'edgecolor')
    institute = kw.get('institute', None)
    kw.update({'weight': True, 'institute': None} if institute == 'wgt' else {})
    signatures = tuple(pplt.Axes._format_signatures.values())
    kw_process = _pop_kwargs(kw, dataset, get_result, reduce_facets)
    kw_attrs = _pop_kwargs(kw, 'short_name', 'long_name', 'standard_name', 'units')
    kw_grid = _pop_kwargs(kw, pplt.GridSpec._update_params)  # overlaps kw_figure
    kw_command = _pop_kwargs(kw, 'cmap', 'cycle', 'extend', *settings)
    kw_figure = _pop_kwargs(kw, *keys_tight, *keys_figure, pplt.Figure._format_signature)  # noqa: E501
    kw_axes = _pop_kwargs(kw, *signatures, pplt.Figure._parse_proj)
    kw_config = _pop_kwargs(kw, tuple(_rc_nodots))  # overlaps kw_command (see above)
    kw_other = _pop_kwargs(kw, _merge_dists, _init_command, _props_command, _setup_bars, _setup_scatter, process_constraint)  # noqa: E501
    kw_guide = _pop_kwargs(kw, pplt.Axes._add_legend, pplt.Axes._add_colorbar)
    kw_legend = {**kw_guide, **_pop_kwargs(kw, 'legend', pplt.Axes._add_legend)}
    kw_colorbar = {**kw_guide, **_pop_kwargs(kw, 'colorbar', pplt.Axes._add_colorbar)}
    kw_axes.update(kw_config)  # configuration settings
    kw_figure.update(kw_config)  # configuration settings
    kw_command.update(kw)  # unknown kwargs passed to command
    if name is not None:
        kw_process['name'] = name
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


def parse_specs(dataset, rowspecs=None, colspecs=None, autocmap=None, **kwargs):
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
    # and coordiantes, or 2-tuple values in single-dictionary spec lists.
    # NOTE: This permits sharing keywords across each group with trailing dictoinaries
    # in either the primary gridspec list or any of the subplot sub-lists. Currently
    # the 'figures.py' utilities do this automatically but could be useful in future.
    # ic(*rowspecs, *colspecs)  # uncomment for debugging
    from .reduce import reduce_facets
    refwidth = refscale = None
    kws_process, kws_collection, gridlabels = [], [], []
    for i, ispecs in enumerate((rowspecs, colspecs)):
        ikws_process, ikws_collection = [], []
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
                    if value := kw_collection.figure.get('refwidth', None):
                        refwidth = value  # for scaling grid labels
                    if not any(kw_process.get(key) for key in ('lon', 'lat', 'area')):
                        refscale = 0.6 if i == 0 else 1.0  # i.e. longitude-latitude
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
        abcwidth = pplt.units(1 * pplt.rc.fontsize, 'pt', 'in')
        refwidth = pplt.units(refwidth or pplt.rc['subplots.refwidth'], 'in')
        refwidth -= abcwidth if len(rowspecs) < 2 or len(colspecs) < 2 else 0
        grdlabels = get_labels(
            *ikws_process,
            identical=False,
            heading=True,
            refwidth=(refscale or 1) * refwidth,  # account for a-b-c space
            dataset=dataset,
        )
        gridlabels.append(grdlabels)
        kws_process.append(ikws_process)
        kws_collection.append(ikws_collection)

    # Combine row and column specifications for plotting and file naming
    # NOTE: Several plotted values per subplot can be indicated in either the
    # row or column list, and the specs from the other list are repeated below.
    # WARNING: Critical to make copies of dictionaries or create new ones
    # here since itertools product repeats the same spec multiple times.
    kws_rowcol = [
        [
            list(zip(jkws_process, jkws_collection))
            for jkws_process, jkws_collection in zip(ikws_process, ikws_collection)
        ]
        for ikws_process, ikws_collection in zip(kws_process, kws_collection)
    ]
    kws_process, kws_collection = [], []
    for ikws_row, ikws_col in itertools.product(*kws_rowcol):
        # Initial stuff
        ikws_row, ikws_col = _expand_lists(ikws_row, ikws_col)
        ikws_process, ikws_collection = [], []
        for jkws_row, jkws_col in zip(ikws_row, ikws_col):  # subplot entries
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
            ikws_process.append(tuple(kw_process))
            ikws_collection.append(kw_collection)

        # Infer legend and axes prefixes
        # TODO: Optionaly use either 'short' or 'skip_name'? Often want single label
        # for contours but multiple longer labels for line plots.
        kw = dict(refwidth=np.inf, identical=False, heading=False, short=False)
        ikws_pair = [ikws_process[0][:1], ikws_process[0][-1:]]
        dist_pair = len(ikws_process[0]) == 2 and ikws_process[0][-1].get('area')
        prefixes_axes = get_labels(*ikws_pair, skip_name=True, dataset=dataset, **kw)
        prefixes_legend = get_labels(*ikws_process, dataset=dataset, **kw)
        for axis, prefix in zip('xy', prefixes_axes):
            if prefix and dist_pair:
                ikws_collection[-1].attrs.setdefault(f'{axis}label_prefix', prefix)
        for pspec, prefix in zip(ikws_collection, prefixes_legend):
            if prefix:  # add prefix to this
                pspec.attrs.setdefault('short_prefix', prefix)
        kws_process.append(ikws_process)
        kws_collection.append(ikws_collection)

    # Infer figure label and grid labels
    # TODO: Combine column count determination here with general_plot()?
    ncols = len(colspecs) if len(colspecs) > 1 else len(rowspecs) if len(rowspecs) > 1 else 3  # noqa: E501
    figwidth = ncols * refwidth + 0.3 * refwidth * (ncols - 1)
    kw_label = dict(refwidth=figwidth, heading=True, identical=True, short=False,)
    figlabel = get_labels(*kws_process, dataset=dataset, **kw_label)
    pathspecs = [dspec for ikws_process in kws_process for dspec in ikws_process]
    pathlabel = get_path(dataset, *pathspecs)
    kw_split = dict(offset=True, refwidth=refwidth, fontsize=pplt.rc.fontlarge)
    if len(rowspecs) == 1 and len(colspecs) == 1:
        gridlabels = None
    elif len(rowspecs) > 1 and len(colspecs) > 1:
        gridlabels = tuple(gridlabels)  # NOTE: tuple critical for general_plot
    elif len(rowspecs) > 1:
        gridlabels = [_split_label(label, **kw_split) for label in gridlabels[0]]
    else:
        gridlabels = [_split_label(label, **kw_split) for label in gridlabels[1]]
    return kws_process, kws_collection, figlabel, pathlabel, gridlabels
