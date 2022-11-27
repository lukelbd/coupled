#!/usr/bin/env python3
"""
Internal helper functions for figure templates.
"""
import collections
import itertools
import re

import climopy as climo  # noqa: F401
import numpy as np
import proplot as pplt
from climopy import ureg, vreg  # noqa: F401
from icecream import ic  # noqa: F401

from .results import FEEDBACK_TRANSLATIONS

__all__ = ['get_spec', 'parse_specs']

# Threshold for label wrapping
WRAP_PADDING = 1.5
# WRAP_PADDING = 1.4
# WRAP_PADDING = 1.2

# Method keywords
# NOTE: These are used to filter out keywords in varoius places
KEYS_METHOD = (
    'method', 'spatial', 'std', 'pctile', 'invert',
)

# Float detection
REGEX_FLOAT = re.compile(  # allow exponential notation
    r'\A([-+]?[0-9._]+(?:[eE][-+]?[0-9_]+)?)\Z'
)
REGEX_SPLIT = re.compile(  # ignore e.g. leading positive and negative signs
    r'(?<=[^+*/-])([+*/-])(?=[^+*/-])'
)

# Reduce presets
# See (WPG and ENSO): https://doi.org/10.1175/JCLI-D-12-00344.1
# See (WPG and ENSO): https://doi.org/10.1038/s41598-021-99738-3
# See (tropical ratio): https://doi.org/10.1175/JCLI-D-18-0843.1
# See (feedback regions): https://doi.org/10.1175/JCLI-D-17-0087.1
# https://climatedataguide.ucar.edu/climate-data/nino-sst-indices-nino-12-3-34-4-oni-and-tni
AREA_REGIONS = {
    'trop': {'lat_lim': (-30, 30), 'lon_lim': (0, 360)},
    'pool': {'lat_lim': (-30, 30), 'lon_lim': (50, 200)},
    'wlam': {'lat_lim': (-15, 15), 'lon_lim': (150, 170)},
    'elam': {'lat_lim': (-30, 0), 'lon_lim': (260, 280)},
    'nina': {'lat_lim': (0, 10), 'lon_lim': (130, 150)},
    'nino': {'lat_lim': (-5, 5), 'lon_lim': (190, 240)},
    'nino3': {'lat_lim': (-5, 5), 'lon_lim': (210, 270)},
    'nino4': {'lat_lim': (-5, 5), 'lon_lim': (160, 210)},
}

# Prefixes used to detect and segregate keyword arguments
DETECT_FIG = [
    'fig', 'ref', 'space', 'share', 'span', 'align',
]
DETECT_GRIDSPEC = [
    'left', 'right', 'bottom', 'top', 'space', 'ratio', 'group', 'equal',
]
DETECT_GRIDSPEC = [
    f'{s}{prefix}' for prefix in DETECT_GRIDSPEC for s in ('w', 'h', '')
]
DETECT_AXES = [
    'x', 'y', 'lon', 'lat', 'abc', 'title', 'proj', 'land', 'coast', 'rc', 'margin',
]
DETECT_OTHER = [
    'horiz', 'pcolor', 'offset', 'cycle', 'shade', 'oneone', 'linefit', 'annotate',
]
DETECT_ATTRS = [
    'short', 'long', 'standard', 'units',
]
DETECT_COLORBAR = [
    'extend', 'locator', 'formatter', 'tick', 'minor', 'length', 'shrink',
]
DETECT_LEGEND = [
    'ncol', 'order', 'frame', 'handle', 'border', 'column',
]

# Argument sorting constants
# NOTE: Use logical top-down order for file naming and reduction instruction order
# and more complex bottom-up human-readable order for automatic label generation.
ORDER_LOGICAL = (
    'name',
    'facets',  # source facets index
    'project',
    'institute',  # e.g. 'flagship'
    'model',
    'experiment',
    'ensemble',
    'version',  # feedback version index
    'source',
    'statistic',
    'region',
    'startstop',
    'start',
    'stop',
    'period',
    'volume',
    'plev',  # space and time
    'area',
    'lat',
    'lon',
)
ORDER_READABLE = (
    'facets',  # source facets index
    'project',
    'institute',
    'model',
    'lon',  # space and time
    'lat',
    'area',
    'ensemble',
    'experiment',
    'plev',
    'volume',
    'period',
    'startstop',
    'start',
    'stop',
    'region',
    'statistic',
    'source',
    'version',  # feedback version index
    'name',
)

# General translations
TRANSLATE_PATHS = {
    ('lat', 'absmin'): 'min',
    ('lat', 'absmax'): 'max',
    ('lon', 'int'): None,
    ('lon', 'avg'): 'avg',  # always report
    ('area', 'avg'): 'avg',  # always report
    ('region', 'point'): 'point',
    ('region', 'globe'): 'globe',
    ('region', 'latitude'): 'zonal',
    ('region', 'hemisphere'): 'hemi',
    ('institute', 'avg'): 'inst',
    ('institute', 'flagship'): 'flag',
    ('institute', None): 'model',
    ('experiment', 'control'): 'pictl',
    ('experiment', 'response'): '4xco2',
    ('experiment', 'picontrol'): 'pictl',
    ('experiment', 'abrupt4xco2'): '4xco2',
    ('startstop', (0, 150)): 'full',
    ('startstop', (0, 50)): 'hist',  # NOTE: used as 'historical' analogue in lit
    ('startstop', (0, 20)): 'early',
    ('startstop', (20, 150)): 'late',
}
TRANSLATE_LABELS = {
    ('lat', 'absmin'): 'minimum',
    ('lat', 'absmax'): 'maximum',
    ('lon', 'int'): None,
    ('lon', 'avg'): None,
    ('area', None): 'unaveraged',  # NOTE: only used with identical=False
    ('area', 'avg'): 'global-average',  # NOTE: only used with identical=False
    ('area', 'trop'): 'tropical-average',
    ('area', 'pool'): 'warm pool',
    ('area', 'wlam'): 'warm pool',
    ('area', 'elam'): 'cold tongue',
    ('area', 'nina'): 'West Pacific',
    ('area', 'nino'): 'East Pacific',
    ('area', 'nino3'): 'East Pacific',
    ('area', 'nino4'): 'East Pacific',
    ('source', 'eraint'): 'Davis et al.',
    ('source', 'zelinka'): 'Zelinka et al.',
    ('region', 'globe'): 'global-$T$',
    ('region', 'point'): 'local-$T$',
    ('region', 'latitude'): 'zonal-$T$',
    ('region', 'hemisphere'): 'hemispheric-$T$',
    ('institute', 'avg'): 'institute',
    ('institute', 'flagship'): 'flagship',
    ('institute', None): 'model',
    ('project', 'cmip'): 'CMIP',
    ('project', 'cmip5'): 'CMIP5',
    ('project', 'cmip6'): 'CMIP6',
    ('project', 'cmip56'): 'matching CMIP5',
    ('project', 'cmip65'): 'matching CMIP6',
    ('project', 'cmip55'): 'non-matching CMIP5',
    ('project', 'cmip66'): 'non-matching CMIP6',
    ('experiment', 'control'): 'pre-industrial',
    ('experiment', 'response'): r'abrupt 4$\times$CO$_2$',
    ('experiment', 'picontrol'): 'pre-industrial',
    ('experiment', 'abrupt4xco2'): r'abrupt 4$\times$CO$_2$',
    ('startstop', (0, 150)): 'full',
    ('startstop', (0, 50)): '50-year',
    ('startstop', (0, 20)): 'early',
    ('startstop', (20, 150)): 'late',
}

# Time translations
TRANSLATE_LONGS = {
    ('period', 'ann'): 'annual',
    ('period', 'djf'): 'boreal winter',
    ('period', 'mam'): 'boreal spring',
    ('period', 'jja'): 'boreal summer',
    ('period', 'son'): 'boreal autumn',
    ('period', 'jan'): 'January',
    ('period', 'feb'): 'February',
    ('period', 'mar'): 'March',
    ('period', 'apr'): 'April',
    ('period', 'may'): 'May',
    ('period', 'jun'): 'June',
    ('period', 'jul'): 'July',
    ('period', 'aug'): 'August',
    ('period', 'sep'): 'September',
    ('period', 'oct'): 'October',
    ('period', 'nov'): 'November',
    ('period', 'dec'): 'December',
    **TRANSLATE_LABELS
}
TRANSLATE_SHORTS = {
    ('period', 'ann'): 'annual',
    ('period', 'djf'): 'DJF',
    ('period', 'mam'): 'MAM',
    ('period', 'jja'): 'JJA',
    ('period', 'son'): 'SON',
    ('period', 'jan'): 'Jan',
    ('period', 'feb'): 'Feb',
    ('period', 'mar'): 'Mar',
    ('period', 'apr'): 'Apr',
    ('period', 'may'): 'May',
    ('period', 'jun'): 'Jun',
    ('period', 'jul'): 'Jul',
    ('period', 'aug'): 'Aug',
    ('period', 'sep'): 'Sep',
    ('period', 'oct'): 'Oct',
    ('period', 'nov'): 'Nov',
    ('period', 'dec'): 'Dec',
    **TRANSLATE_LABELS
}


def _fix_parts(kwargs):
    """
    Fix reduction keyword args with overrides.

    Parameters
    ----------
    kwargs : dict
        The reduction keyword args.
    """
    kwargs = kwargs.copy()
    if 'stop' in kwargs:
        kwargs.setdefault('start', 0)
    if 'start' in kwargs:
        kwargs.setdefault('stop', 150)
    if 'start' in kwargs and 'stop' in kwargs:
        kwargs['startstop'] = (kwargs.pop('start'), kwargs.pop('stop'))
    return kwargs


def _iter_parts(value):
    """
    Iterate over parts of value split by operation.

    Parameters
    ----------
    value : object
        A reduction value.
    """
    # NOTE: This translates e.g. startstop=('20-0', '150-20') to the anomaly
    # pairs [('20', '150'), ('-', '-'), ('0', '20')] for better processing.
    parts, values = [], value if isinstance(value, tuple) else (value,)
    for value in values:
        iparts = []  # support 'startstop' anomaly tuples e.g. startstop=(20-0, 150-20)
        for part in (REGEX_SPLIT.split(value) if isinstance(value, str) else (value,)):
            if not isinstance(part, str):
                pass
            elif part.isdecimal():
                part = int(part)
            elif REGEX_FLOAT.match(part):
                part = float(part)
            iparts.append(part)
        parts.append(iparts)
    for part in zip(*parts):
        if len(part) == 1:
            part, = part
        if isinstance(part, tuple) and part[0] in ('+', '-', '/', '*'):
            part = part[0]
        yield part


def _get_label(dataset, key, value, mode=None):
    """
    Return an arbitrary label type based on the dataset.

    Parameters
    ----------
    dataset : xarray.Dataset
        The source dataset.
    key : str
        The reduce coordinate.
    value : str or float
        The reduce selection.
    mode : {'path', 'short', 'long'}, optional
        The label mode. Affects various translations.

    Returns
    -------
    label : str
        The final label.
    """
    mode = mode or 'path'
    if mode not in ('path', 'short', 'long'):
        raise ValueError(f'Invalid label mode {mode!r}.')
    alias_to_name = {alias: name for alias, (name, _) in FEEDBACK_TRANSLATIONS.items()}
    name_to_alias = {name: alias for alias, name in alias_to_name.items()}  # keep last
    labels = []
    for part in _iter_parts(value):
        if key == 'name':
            part = alias_to_name.get(part, part)
            if mode == 'path':
                label = name_to_alias.get(part, part)
            elif mode == 'short':
                label = dataset[part].short_name
            else:
                label = dataset[part].long_name
        elif part is None or isinstance(part, (str, tuple)):  # can have 'None' labels
            operators = {'+': 'plus', '-': 'minus', '*': 'times', '/': 'over'}
            if part and part[0] in operators:  # fixes e.g. (-, -) from startstop tuples
                label = part[0] if mode == 'path' else operators[part[0]]
            elif mode == 'path':
                label = TRANSLATE_PATHS.get((key, part), part)
            elif mode == 'short':
                label = TRANSLATE_SHORTS.get((key, part), part)
            else:
                label = TRANSLATE_LONGS.get((key, part), part)
            if isinstance(label, tuple):  # i.e. 'startstop' without shorthand
                label = '-'.join(format(lab, 's' if isinstance(lab, str) else '04.0f') for lab in label)  # noqa: E501
        else:
            unit = dataset[key].climo.units
            if not isinstance(part, ureg.Quantity):
                part = ureg.Quantity(part, unit)
            part = part.to(unit)
            if mode == 'path':
                label = f'{part:~.0f}'
            else:
                label = f'${part:~L.0f}$'
        if label is None:  # e.g. skip 'avg' label
            continue
        if mode == 'path':  # extra processing
            label = label.replace('\N{DEGREE SIGN}', '')
            label = label.replace('+', '')
            label = label.replace('-', '')
            label = label.replace('/', '')
            label = label.replace('*', '')
            label = label.replace('_', '')
            label = label.replace(' ', '')
            label = label.lower()
        labels.append(label)
    result = '' if mode == 'path' else ' '
    return result.join(labels)


def _infer_path(dataset, *kws_process):
    """
    Convert reduction operators into path suitable for saving.

    Parameters
    ----------
    dataset : xarray.Dataset
        The source dataset.
    *kws_process : dict
        The `get_data` keywords.

    Returns
    -------
    path : str
        The parts joined with underscores and dashes.
    """
    # NOTE: This omits keywords that correspond to default values but always
    # includes experiment because default is ambiguous between variables.
    kws_process = [(kw,) if isinstance(kw, dict) else tuple(kw) for kw in kws_process]
    kws_process = [kw.copy() for kws in kws_process for kw in kws]
    labels = []
    defaults = {'project': 'cmip', 'ensemble': 'flagship', 'period': 'ann'}
    defaults.update({'source': 'eraint', 'statistic': 'slope', 'region': 'globe'})
    kws_process = list(map(_fix_parts, kws_process))
    for key in ORDER_LOGICAL:
        seen, parts = set(), []
        values = [kw[key] for kw in kws_process if key in kw]
        values = [value for value in values if value not in seen and not seen.add(value)]  # noqa: E501
        if len(values) == 1 and values[0] == defaults.get(key, None):
            continue
        for value in values:  # across all subplots and tuples
            label = _get_label(dataset, key, value, mode='path')
            if not label:
                continue
            if label not in parts:  # e.g. 'avg' to be ignored
                parts.append(label)
        if parts:
            labels.append(parts)  # do not sort these
    result = '_'.join('-'.join(parts) for parts in labels)
    return result


def _infer_labels(
    dataset, *kws_process, identical=False, long_names=False, title_case=False, **kwargs
):
    """
    Convert reduction operators into human-readable labels.

    Parameters
    ----------
    dataset : xarray.Dataset
        The source dataset.
    *kws_process : tuple of dict
        The reduction keyword arguments.
    identical : bool, optional
        Whether to keep identical reduce operations across list.
    long_names : bool, optional
        Whether to use long reduce instructions.
    title_case : bool, optional
        Whether to return labels in title case.
    **kwargs
        Passed to `_wrap_label`.

    Returns
    -------
    labels : list
        The unique or identical label(s).
    """
    # Reduce label dicationaries to drop or select redundancies across the list
    # TODO: Also drop scalar 'identical' specifications of e.g. 'annual', 'slope',
    # and 'eraint' default selections for feedback variants and climate averages?
    kws_labels = []
    kws_process = [(kw,) if isinstance(kw, dict) else tuple(kw) for kw in kws_process]
    if any(len(tup) > 2 for tup in kws_process):
        raise ValueError('Expected lists of dictionaries or 2-tuples.')
    order_read = list(ORDER_READABLE)
    sorter = lambda key: order_read.index(key) if key in order_read else len(order_read)
    mode = 'long' if long_names else 'short'
    for i in range(2):  # indices in correlation pair
        kws_label = []
        kws_part = [tup[i] for tup in kws_process if i < len(tup)]
        for kw_part in kws_part:
            kw = {}
            kw_part = _fix_parts(kw_part)
            for key in sorted(kw_part, key=sorter):
                value = kw_part[key]
                if key in KEYS_METHOD:
                    continue
                label = _get_label(dataset, key, value, mode=mode)
                kw[key] = label
            kws_label.append(kw)
        keys = sorted((key for kw in kws_label for key in kw), key=sorter)
        kws_merged = {}
        for key in keys:
            values = tuple(kw.get(key, '') for kw in kws_label)
            kws_merged[key] = _merge_labels(*values, identical=identical)
        if not identical:
            kws_merged = [
                {key: kws_merged[key][i] for key in kws_merged}
                for i in range(len(kws_label))
            ]
        if not kws_merged:  # WARNING: critical or else zip below creates empty list
            continue
        kws_labels.append(kws_merged)
    # Combine pairs of labels
    # NOTE: This optionally assigns labels that are identical across the pair to
    # the front or the back of the combined 'this vs. that' label.
    labels = []
    kws_labels = [kws_labels] if identical else list(zip(*kws_labels))
    for i, ikws_labels in enumerate(kws_labels):
        keys = sorted((key for kw in ikws_labels for key in kw), key=sorter)
        ikws_front, ikws_back = {}, {}
        ikws_pairs = ikws_labels[::-1]  # place dependent variable *first*
        for key in keys:
            items = tuple(kw.get(key, None) for kw in ikws_pairs)
            if all(item == items[0] for item in items):  # also non-constraints
                if not items[0] or items[0] in ('averaged', 'unaveraged'):
                    pass
                elif key == 'name':  # e.g. abrupt vs. picontrol *feedback*
                    ikws_back[key] = items[0]
                else:  # e.g. *cmip5* abrupt vs. picontrol
                    ikws_front[key] = items[0]
                for kw in ikws_pairs:
                    kw.pop(key, None)
        front = ikws_front.values()
        back = ikws_back.values()
        center = [' '.join(filter(None, kws.values())) for kws in ikws_pairs]
        center = ' vs. '.join(filter(None, center))
        label = ' '.join(filter(None, (*front, center, *back)))
        label = _wrap_label(label.strip(), **kwargs)
        if title_case and label[:1].islower():
            label = label[:1].upper() + label[1:]
        labels.append(label)
    labels = labels[0] if identical else labels
    return labels


def _merge_labels(*labels, identical=False):
    """
    Helper function to merge labels.

    Parameters
    ----------
    *labels : str
        The labels to be merged.
    identical : bool, optional
        Whether to keep only identical or non-identical portions.
    """
    # NOTE: For typical feedback breakdown this returns e.g. 'feedback' when identical
    # is true and ('net', 'shortwave', 'longwave') when identical is false. Also returns
    # empty string when identical is true but all labels are different, and empty
    # strings when identical is false but all labels (more than one) are the same.
    strings = (
        'feedback',
        'forcing',
        r'2$\times$CO$_2$',  # leads forcing strings
        r'4$\times$CO$_2$',
        'energy',
        'transport',
        'convergence',
    )
    strings = [s for s in strings if all(s in label for label in labels)]
    pattern = '|'.join(r'\s*' + re.escape(s) for s in strings)
    regex = re.compile(f'({pattern})')
    ident = all(label == labels[0] for label in labels)
    if identical:
        if not ident:  # note never true if labels is singleton
            labels = [''.join(regex.findall(labels[0])).strip() for label in labels]
            ident = all(label == labels[0] for label in labels)
        labels = labels[0] if ident else ''
    else:
        if ident:  # note always true if labels is singleton
            labels = [''] * len(labels)
        else:
            labels = [regex.sub('', label).strip() for label in labels]
    return labels


def _wrap_label(label, fontsize=None, refwidth=None, nmax=None):
    """
    Replace spaces with newlines to accommodate a subplot or figure label.

    Parameters
    ----------
    label : str
        The input label.
    refwidth : unit-spec, optional
        The reference maximum width.
    nmax : int, optional
        Optional maximum number of breaks to use.

    Returns
    -------
    label : str
        The label with inserted newlines.
    """
    # NOTE: This adds extra padding to allow labels to extend into border. Also
    # roughly adjust contribution from math, generally for e.g. 2$\times$CO$_2$.
    label = label or ''
    label = label.replace('\n', ' ')  # remove previous wrapping just in case
    label = label + ' '  # end with dummy space for threshold loop below
    idxs_spaces = np.array([i for i, c in enumerate(label) if c == ' '])
    idxs_adjust = idxs_spaces.astype(float)
    idxs_span = [m.span() for m in re.finditer(r'\$[^$]+\$', label)]
    idxs_span = np.array(idxs_span, dtype=float)
    for (i, j) in idxs_span:
        adjust = 2 + 0.8 * (j - i - 1)  # number of characters inside '$$'
        idxs_adjust[(i <= idxs_adjust) & (idxs_adjust < j)] = np.nan
        idxs_adjust[idxs_adjust >= j] -= adjust
        idxs_span -= adjust  # iteration respects modified data
    basesize = pplt.rc['font.size']
    fontscale = pplt.utils._fontsize_to_pt(fontsize or basesize) / basesize
    refwidth = pplt.units(refwidth or pplt.rc['subplots.refwidth'], 'in', 'em')
    thresholds = WRAP_PADDING * refwidth * np.arange(1, 10) / fontscale
    count, chars = 0, list(label)  # convert string to list
    for thresh in thresholds:
        idxs, = np.where(idxs_adjust <= thresh)
        if not np.any(idxs_adjust > thresh):
            continue
        if not idxs.size:  # including empty idxs_adjust
            continue
        if nmax and count >= nmax:
            continue
        count += 1  # add to newline count
        idx = idxs[-1]
        thresholds -= thresh - idxs_adjust[idx]  # closer to threshold by this
        chars[idxs_spaces[idx]] = '\n'
    return ''.join(chars[:-1])


def get_spec(dataset, spec, **kwargs):
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
    kw_process : dict
        The indexers used to reduce the data variable with `.reduce`. This is
        parsed specially compared to other keywords, and its keys are restricted
        to ``'name'`` and any coordinate or multi-index names.
    kw_collection : namedtuple of dict
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
    # inside kw_process 'name' key. This helps when merging variable specifications
    # between row and column specs and between tuple-style specs (see parse_specs).
    if spec is None:
        name, kw = None, {}
    elif isinstance(spec, str):
        name, kw = spec, {}
    elif isinstance(spec, dict):
        name, kw = None, spec
    else:  # length-2 iterable
        name, kw = spec
    kw = {**kwargs, **kw}  # copy
    alt = kw.pop('name', None)
    name = name or alt  # see below
    kw_figure, kw_gridspec, kw_axes = {}, {}, {}
    kw_command, kw_other, kw_attrs = {}, {}, {}
    kw_colorbar, kw_legend, kw_process = {}, {}, {}
    detect_process = list(dataset.sizes)
    detect_process.extend(name for idx in dataset.indexes.values() for name in idx.names)  # noqa: E501
    detect_process.extend(('area', 'volume', 'institute', *KEYS_METHOD))
    for key, value in kw.items():  # NOTE: sorting performed in _parse_labels
        if key in detect_process:
            kw_process[key] = value  # e.g. for averaging
        elif any(key.startswith(prefix) for prefix in DETECT_FIG):
            kw_figure[key] = value
        elif any(key.startswith(prefix) for prefix in DETECT_GRIDSPEC):
            kw_gridspec[key] = value
        elif any(key.startswith(prefix) for prefix in DETECT_AXES):
            kw_axes[key] = value
        elif any(key.startswith(prefix) for prefix in DETECT_OTHER):
            kw_other[key] = value
        elif any(key.startswith(prefix) for prefix in DETECT_ATTRS):
            kw_attrs[key] = value
        elif any(key.startswith(prefix) for prefix in DETECT_COLORBAR):
            kw_colorbar[key] = value
        elif any(key.startswith(prefix) for prefix in DETECT_LEGEND):
            kw_legend[key] = value
        else:  # arbitrary plotting keywords
            kw_command[key] = value
    if isinstance(name, str):  # NOTE: here name of None always ignored
        kw_process['name'] = name
    if 'label' in kw:  # NOTE: overrides for both legend and colorbar
        kw_colorbar['label'] = kw_legend['label'] = kw.pop('label')
    fields = ('figure', 'gridspec', 'axes', 'command', 'other', 'attrs', 'colorbar', 'legend')  # noqa: E501
    collection = collections.namedtuple('kwargs', fields)
    kw_process.update({key: kwargs.pop(key) for key in KEYS_METHOD if key in kwargs})
    kw_collection = collection(
        kw_figure, kw_gridspec, kw_axes, kw_command,
        kw_other, kw_attrs, kw_colorbar, kw_legend
    )
    return kw_process, kw_collection


def parse_specs(dataset, rowspecs=None, colspecs=None, **kwargs):
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
    **kwargs
        Additional options shared across all specs.

    Returns
    -------
    kws_process : list of list of [12]-tuple of dict
        The keyword arguments for `reduce` and `method`.
    kws_collection : list of list of namedtuple
        The keyword arguments for various plotting tasks.
    figlabel : str, optional
        The default figure title center.
    pathlabel : list of list of str
        The default figure path center.
    gridlabels : list of list of str
        The default row and column labels.
    """
    # Parse variable specs per gridspec row or column and per subplot
    # NOTE: This permits sharing keywords across each group with trailing dicts
    # in either the primary gridspec list or any of the subplot sub-lists.
    # NOTE: The two arrays required for two-argument methods can be indicated with
    # either 2-tuples in spec lists or conflicting row and column names or reductions.
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
            for kspecs in jspecs:  # specs per subplot
                kkws_process, kkws_collection = [], []
                if kspecs is None:
                    kspecs = (None,)  # possibly construct from keyword args
                elif isinstance(kspecs, (str, dict)):
                    kspecs = (kspecs,)
                elif len(kspecs) != 2:
                    raise ValueError(f'Invalid variable specs {kspecs}.')
                elif type(kspecs[0]) != type(kspecs[1]):  # noqa: E721  # (str, dict)
                    kspecs = (kspecs,)
                else:
                    kspecs = tuple(kspecs)
                for spec in kspecs:  # specs per correlation pair
                    kw_process, kw_collection = get_spec(dataset, spec, **kwargs)
                    if value := kw_collection.figure.get('refwidth', None):
                        refwidth = value
                    if not any(kw_process.get(key) for key in ('lon', 'lat', 'area')):
                        refscale = 0.7 if i == 0 else 1.0  # i.e. longitude-latitude
                    kkws_process.append(kw_process)
                    kkws_collection.append(kw_collection)
                jkws_process.append(tuple(kkws_process))  # denotes correlation-pair
                jkws_collection.append(tuple(kkws_collection))
            ikws_process.append(jkws_process)
            ikws_collection.append(jkws_collection)
        ncols = len(colspecs) if len(colspecs) > 1 else len(rowspecs) if len(rowspecs) > 1 else 1  # noqa: E501
        refwidth = refwidth or pplt.rc['subplots.refwidth']
        refwidth = (refscale or 1.0) * pplt.units(refwidth, 'in')
        figwidth = 1.5 * ncols * refwidth  # larger value
        abcwidth = 2 * pplt.units(pplt.utils._fontsize_to_pt(pplt.rc.fontlarge), 'pt', 'in')  # noqa: E501
        refwidth -= abcwidth if len(rowspecs) < 2 or len(colspecs) < 2 else 0
        zerospecs = [jkws_process[0] if jkws_process else {} for jkws_process in ikws_process]  # noqa: E501
        grdlabels = _infer_labels(
            dataset,
            *zerospecs,
            identical=False,
            long_names=True,
            title_case=True,
            fontsize=pplt.rc.fontlarge,
            refwidth=refwidth,  # account for a-b-c space
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
    kw_infer = dict(refwidth=np.inf, identical=False, long_names=True, title_case=False)
    for ikws_row, ikws_col in itertools.product(*kws_rowcol):
        if len(ikws_row) == 1:
            ikws_row = list(ikws_row) * len(ikws_col)
        if len(ikws_col) == 1:
            ikws_col = list(ikws_col) * len(ikws_row)
        if len(ikws_row) != len(ikws_col):
            raise ValueError(
                'Incompatible per-subplot spec count.'
                + f'\nRow specs ({len(ikws_row)}): \n' + '\n'.join(map(repr, ikws_row))
                + f'\nColumn specs ({len(ikws_col)}): \n' + '\n'.join(map(repr, ikws_col))  # noqa: E501
            )
        ikws_process, ikws_collection = [], []
        for jkws_row, jkws_col in zip(ikws_row, ikws_col):  # subplot entries
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
            rkws_process = tuple(kw.copy() for kw in rkws_process)  # NOTE: copy needed
            ckws_process = tuple(kw.copy() for kw in ckws_process)
            for ikws, jkws in ((rkws_process, ckws_process), (ckws_process, rkws_process)):  # noqa: E501
                for key in ikws[0]:
                    if key not in ikws[-1]:
                        continue
                    if ikws[0][key] == ikws[-1][key]:
                        for kw in jkws:  # apply identical value to other pair
                            kw.setdefault(key, ikws[0][key])
            kw_process = {}  # filter unique specifications
            for ikw_process in (*rkws_process, *ckws_process):  # possible correlations
                keys = sorted(key for key in ikw_process if key not in KEYS_METHOD)
                keyval = tuple((key, ikw_process[key]) for key in keys)
                if tuple(keyval) in kw_process:
                    continue
                others = tuple(other for other in kw_process if set(keyval) < set(other))  # noqa: E501
                if others:
                    continue
                others = tuple(other for other in kw_process if set(other) < set(keyval))  # noqa: E501
                for other in others:  # prefer more selection keywords
                    kw_process.pop(other)
                kw_process[tuple(keyval)] = ikw_process
            kw_process = tuple(kw_process.values())  # possible correlation tuple
            if len(kw_process) > 2:
                raise RuntimeError(
                    'Expected 1-2 specs for combining with get_data but got '
                    f'{len(kw_process)} specs: ' + '\n'.join(map(repr, kw_process))
                )
            ikws_process.append(kw_process)
            ikws_collection.append(kw_collection)
        ikws_pair = [ikws_process[0][:1], ikws_process[0][1:]]
        ax_prefixes = _infer_labels(dataset, *ikws_pair, **kw_infer)
        leg_prefixes = _infer_labels(dataset, *ikws_process, **kw_infer)
        for axis, prefix in zip('xy', ax_prefixes):
            if prefix:  # apply for last item in subplot
                ikws_collection[-1].attrs.setdefault(f'{axis}label_prefix', prefix)
        for pspec, prefix in zip(ikws_collection, leg_prefixes):
            if prefix:  # add prefix to this
                pspec.attrs.setdefault('short_prefix', prefix)
        kws_process.append(ikws_process)
        kws_collection.append(ikws_collection)
    subspecs = [dspec for ikws_process in kws_process for dspec in ikws_process]
    figlabel = _infer_labels(
        dataset,
        *subspecs,
        identical=True,
        long_names=True,
        title_case=True,
        fontsize=pplt.rc.fontlarge,
        refwidth=figwidth,
    )
    pathlabel = _infer_path(dataset, *subspecs)
    return kws_process, kws_collection, figlabel, pathlabel, gridlabels
