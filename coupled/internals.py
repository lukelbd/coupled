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
# NOTE: Play with this to prevent annoying line breaks
# WRAP_SCALE = 1.2
# WRAP_SCALE = 1.4
WRAP_SCALE = 1.8

# Keywords for inter-model reduction methods
KEYS_METHOD = (  # skip 'spatial' because used in apply_reduce
    'method', 'std', 'pctile', 'invert',
)

# Regexes for float and operator detection
REGEX_FLOAT = re.compile(  # allow exponential notation
    r'\A([-+]?[0-9._]+(?:[eE][-+]?[0-9_]+)?)\Z'
)
REGEX_SPLIT = re.compile(  # ignore e.g. leading positive and negative signs
    r'(?<=[^+*/-])([+*/-])(?=[^+*/-])'
)

# Prefixes used to detect and segregate keyword arguments
# NOTE: See templates.py documentation for various 'other' arguments.
DETECT_FIG = (
    'fig', 'ref', 'space', 'share', 'span', 'align', 'inner', 'outer', 'panel', 'tight',
)
DETECT_GRIDSPEC = (
    'left', 'right', 'bottom', 'top', 'space', 'ratio', 'group', 'equal', 'pad',
)
DETECT_GRIDSPEC = [
    f'{s}{prefix}' for prefix in DETECT_GRIDSPEC for s in ('w', 'h', '')
]
DETECT_AXES = (
    'x', 'y', 'lon', 'lat', 'abc', 'title', 'proj', 'land', 'coast', 'rc', 'margin',
)
DETECT_OTHER = (
    'horiz', 'pcolor', 'cycle', 'multi', 'one', 'line', 'offset', 'annot', 'corr', 'inter',  # noqa: E501
)
DETECT_ATTRS = (
    'short', 'long', 'standard', 'units',
)
DETECT_GUIDE = (
    'loc', 'label', 'align',
)
DETECT_COLORBAR = (
    'colorbar', 'locator', 'formatter', 'tick', 'minor', 'extend', 'length', 'shrink',
)
DETECT_LEGEND = (
    'legend', 'order', 'frame', 'handle', 'border', 'column',
)

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
    'month',  # space and time
    'season',
    'period',  # NOTE: this is outdated
    'plev',
    'area',
    'volume',
    'lat',
    'lon',
    'spatial',  # always at the end
)
ORDER_READABLE = (
    'facets',  # source facets index
    'project',
    'institute',
    'model',
    'lon',  # space and time
    'lat',
    'area',
    'volume',
    'plev',
    'period',  # NOTE: this is outdated
    'season',
    'month',
    'startstop',
    'start',
    'stop',
    'ensemble',  # remaining facets
    'experiment',
    'region',  # feedback version index
    'statistic',
    'source',
    'version',
    'name',
    'suffix',
    'spatial',  # always at the end
)

# General translations
TRANSLATE_PATHS = {
    ('lat', 'absmin'): 'min',
    ('lat', 'absmax'): 'max',
    ('lon', 'int'): None,
    ('lon', 'avg'): 'avg',  # always report
    ('area', 'avg'): 'avg',  # always report
    ('region', 'point'): 'pt',
    ('region', 'latitude'): 'lat',
    ('region', 'hemisphere'): 'hemi',
    ('region', 'apoint'): 'apt',
    ('region', 'alatitude'): 'alat',
    ('region', 'ahemisphere'): 'ahemi',
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
    ('area', None): 'local',  # NOTE: only used with identical=False
    ('area', 'avg'): 'global',  # NOTE: only used with identical=False
    ('area', 'trop'): 'tropical',
    # ('area', None): 'unaveraged',  # NOTE: only used with identical=False
    # ('area', 'avg'): 'global-average',  # NOTE: only used with identical=False
    # ('area', 'trop'): 'tropical-average',
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
    ('spatial', 'slope'): 'spatial regression',  # NOTE: also see _apply_double
    ('spatial', 'proj'): 'spatial projection',
    ('spatial', 'corr'): 'spatial correlation',
    ('spatial', 'cov'): 'spatial covariance',
    ('spatial', 'rsq'): 'spatial variance explained',
    ('source', 'eraint'): 'custom',
    ('source', 'zelinka'): 'Zelinka',
    # ('source', 'eraint'): 'Davis et al.',
    # ('source', 'zelinka'): 'Zelinka et al.',
    ('region', 'globe'): 'global-$T$',
    ('region', 'point'): 'local-$T$',
    ('region', 'latitude'): 'zonal-$T$',
    ('region', 'hemisphere'): 'hemispheric-$T$',
    ('institute', 'avg'): None,
    ('institute', 'flagship'): None,
    ('institute', None): None,
    # ('institute', 'avg'): 'institute',
    # ('institute', 'flagship'): 'flagship-only',
    # ('institute', None): 'model',
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
    ('experiment', 'control'): 'control',  # or unperturbed and perturbed
    ('experiment', 'response'): r'4$\times$CO$_2$',
    ('experiment', 'picontrol'): 'control',
    ('experiment', 'abrupt4xco2'): r'4$\times$CO$_2$',
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


def _group_parts(kwargs, keep_operators=False):
    """
    Group reduction keyword arguments.

    Parameters
    ----------
    kwargs : dict
        The reduction keyword args.
    keep_operators : bool, optional
        Whether to keep pre-industrial start and stop operators.
    """
    # WARNING: Need process_data() to accomodate situations where e.g. generating
    # several 'scatter plots' that get concatenated into a bar plot of regression
    # coefficients of 'late minus early vs. unperturbed', in which case apply_reduce()
    # will translate denominator to 'full unperturbed minus full unperturbed' i.e.
    # zero, and case where we are doing e.g. spatial version of this regression,
    # where apply_reduce() gets regression coefficients before the subtraction
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
            n1 = sum(map(start.count, '+-')) if isinstance(start, str) else 0
            n2 = sum(map(stop.count, '+-')) if isinstance(stop, str) else 0
            if keep_operators:
                start = '+'.join(itertools.repeat('0', n1 + 1))
                stop = '+'.join(itertools.repeat('150', n2 + 1))
            else:
                start = None
                stop = None
        kwargs['startstop'] = (start, stop)
    return kwargs


def _ungroup_parts(value):
    """
    Ungroup reduction argument.

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


def _to_lists(*args, equal=True):
    """
    Ensure input argument lengths match.

    Parameters
    ----------
    *args
        The lengths of objects.
    equal : bool, optional
        Whether to enforce equal lengths.
    """
    args = list(args)  # modifable
    if len(args) == 2 and len(args[0]) == 1 and len(args[1]) != 1:
        args[0] = args[0] * len(args[1])
    if len(args) == 2 and len(args[1]) == 1 and len(args[0]) != 1:
        args[1] = args[1] * len(args[0])
    if equal and len(set(lengths := list(map(len, args)))) > 1:
        vals = '\n'.join(map(repr, args))
        raise ValueError(f'Incompatible mixed lengths {lengths} for values:\n{vals}.')
    return args


def _capitalize_label(label, prefix=None, suffix=None):
    """
    Cast the input label to title case.

    Parameters
    ----------
    label : str
        The display label.
    prefix, suffix : optional
        The label prefix and suffix.
    """
    if prefix and not label[:2].isupper():
        label = label[:1].lower() + label[1:]
    if prefix:
        prefix = prefix[:1].upper() + prefix[1:]
    else:
        label = label[:1].upper() + label[1:]
    parts = (prefix, label, suffix)
    label = ' '.join(filter(None, parts))
    return label


def _combine_labels(*labels, identical=False):
    """
    Helper function to combine labels.

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
        'spatial',
        'pattern',
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


def _fit_label(label, fontsize=None, refwidth=None, nmax=None):
    """
    Fit label into given width by replacing spaces with newlines.

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
    # NOTE: This adds extra padding to allow labels to extend into border.
    # NOTE: Adjust contribution from math, e.g. 2$\times$CO$_2$, since this
    # consists of mahy characters that do not contribute to actual space.
    # WARNING: Play with 'adjust' for math
    label = label or ''
    label = label.replace('\n', ' ')  # remove previous wrapping just in case
    label = label + ' '  # end with dummy space for threshold loop below
    idxs_space = np.array([i for i, c in enumerate(label) if c == ' '])
    idxs_check = idxs_space.astype(float)
    for m in re.finditer(r'\$[^$]+\$', label):
        i, j = m.span()
        tex = label[i + 1:j - 1]  # text inside '$$'
        tex = re.sub(r'\\[A-Za-z]+\{([^}]*)\}', r'\1', tex)  # replace e.g. \mathrm{}
        tex = re.sub(r'[_^]\{?[0-9A-Za-z+-]*\}?', '#', tex)  # replace exponents
        tex = re.sub(r'\s*\\[.,:;]\s*', '', tex)  # replace space between terms
        tex = re.sub(r'\\[A-Za-z]+', 'x', tex)  # replace tex symbol
        idxs_check[(idxs_check >= i) & (idxs_check < j)] = np.nan  # no breaks here
        idxs_check[idxs_check >= j] -= j - i - 1 - len(tex)  # size used for wrap
    basesize = pplt.rc['font.size']
    fontscale = pplt.utils._fontsize_to_pt(fontsize or basesize) / basesize
    refwidth = pplt.units(refwidth or pplt.rc['subplots.refwidth'], 'in', 'em')
    refscale = WRAP_SCALE / fontscale  # font scaling
    thresholds = refscale * refwidth * np.arange(1, 20)
    seen, chars, count = set(), list(label), 0  # convert string to list
    for thresh in thresholds:
        idxs, = np.where(idxs_check <= thresh)
        if not np.any(idxs_check > thresh):  # no spaces or end-of-string over this
            continue
        if not idxs.size:  # including empty idxs_check
            continue
        if nmax and count >= nmax:
            continue
        if idxs[-1] not in seen:  # avoid infinite loop and jump to next threshold
            seen.add(idx := idxs[-1])
            thresholds -= thresh - idxs_check[idx]  # closer to threshold by this
            chars[idxs_space[idx]] = '\n'
            count += 1  # update newline count
    return ''.join(chars[:-1])


def _get_label(dataset, key, value, mode=None, name=None):
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
        The label mode used to translate selections.
    name : str, optional
        The name used to determine experiment translations.

    Returns
    -------
    label : str
        The final label.
    """
    # NOTE: This function is used for axis label prefixes, legend entry prefixes, row
    # and column labels, and figure and path titles. It is never used for the 'method'
    # key because this is added to the data array short and long names during
    # application, which are subsequently used for axis labels, legend entries, and
    # colorbar labels. Note the method is also manually appended to the figure path.
    mode = mode or 'path'
    if mode not in ('path', 'short', 'long'):
        raise ValueError(f'Invalid label mode {mode!r}.')
    operator_to_label = {'+': 'plus', '-': 'minus', '*': 'times', '/': 'over'}
    alias_to_name = {alias: name for alias, (name, _) in FEEDBACK_TRANSLATIONS.items()}
    name_to_alias = {name: alias for alias, name in alias_to_name.items()}  # keep last
    name = alias_to_name.get(name, name)
    version = 'version' in getattr(dataset.get(name, None), 'coords', {})
    version = True  # TODO: remove
    parts, labels = _ungroup_parts(value), []
    for part in parts:
        if key == 'name':
            if part and '|' in part:  # typically numerator in pattern regression
                *_, part = part.split('|')
            part = alias_to_name.get(part, part)
            if part and part in operator_to_label:
                label = part if mode == 'path' else operator_to_label[part]
            elif mode == 'path':
                label = name_to_alias.get(part, part)
            elif mode == 'short':
                label = dataset[part].short_name
            else:
                label = dataset[part].long_name
        elif part is None or isinstance(part, (str, tuple)):  # can have 'None' labels
            if part and part[0] == 'a' and part != 'ann' and key in ('region', 'period'):  # noqa: E501
                part = part if mode == 'path' else part[1:]  # abrupt-only label
            if part == (None, None):
                label = None
            elif part and part[0] in operator_to_label:  # e.g. (-, -) startstop tuples
                label = part[0] if mode == 'path' else operator_to_label[part[0]]
            elif mode == 'path':
                label = TRANSLATE_PATHS.get((key, part), part)
            elif mode == 'short':
                label = TRANSLATE_SHORTS.get((key, part), part)
            else:
                label = TRANSLATE_LONGS.get((key, part), part)
            if version and mode != 'path' and part == 'picontrol':
                label = 'unperturbed'
            if version and mode != 'path' and part == 'abrupt4xco2':
                label = 'perturbed'
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
                label = f'{part:~P.0f}'.replace(' ', r'$\,$')
        if label is None:  # e.g. skip 'avg' label
            continue
        if mode == 'path':  # extra processing
            for symbol in ('\N{DEGREE SIGN}', *operator_to_label, '_', ' '):
                label = label.lower().replace(symbol, '')
        labels.append(label)
    sep = '' if mode == 'path' else ' '
    if len(labels) > 1:  # if result is still weird user should pass explicit values
        if labels[0] in operator_to_label.values():
            labels = labels[1:]  # e.g. experiment=abrupt4xco2, stop=None-20
        if labels[-1] in operator_to_label.values():
            labels = labels[:-1]  # e.g. experiment=abrupt4xco2-picontrol, stop=20-None
    return sep.join(labels)


def _infer_labels(
    dataset, *kws_process, identical=False,
    title_case=False, long_names=False, skip_names=False, **kwargs,
):
    """
    Convert reduction operators into human-readable labels.

    Parameters
    ----------
    dataset : xarray.Dataset
        The source dataset.
    *kws_process : list of tuple of dict
        The reduction keyword arguments.
    identical : bool, optional
        Whether to keep identical reduce operations across list.
    title_case : bool, optional
        Whether to return labels in title case.
    long_names : bool, optional
        Whether to use long reduce instructions.
    skip_names : bool, optional
        Whether to skip names and `spatial` in the label.
    **kwargs
        Passed to `_fit_label`.

    Returns
    -------
    labels : list
        The unique or identical label(s).
    """
    # Initial stuff
    # NOTE: For grid labels use intersection of identifiers with same number of
    # arguments. Common to have e.g. x vs. y and then just plot x or y as a
    # reference but the former is the relevant information for labels.
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

    # Reduce label dicationaries to drop or select redundancies across the list
    # TODO: Also drop scalar 'identical' specifications of e.g. 'annual', 'slope',
    # and 'eraint' default selections for feedback variants and climate averages?
    if not spatial:
        order_back = ('name',)
        order_read = list(ORDER_READABLE)
    else:
        order_back = ['name', 'area', 'volume', 'spatial']
        order_read = [key for key in ORDER_READABLE if key not in order_back] + order_back  # noqa: E501
    sorter = lambda key: order_read.index(key) if key in order_read else len(order_read)
    kws_label = []
    for n in range(2):  # index in possible correlation pair
        ikws_label = []
        for ikws_infer in kws_infer:  # iterate over subplots
            nkws_label = []
            nkws_infer = [kws[n] if n < len(kws) else {} for kws in ikws_infer]
            for kw_infer in nkws_infer:  # iterate over arguments inside subplot
                mode = 'long' if long_names else 'short'
                name = kw_infer.get('name', None)
                kw_label, kw_infer = {}, _group_parts(kw_infer)
                for key, value in kw_infer.items():  # get individual label
                    if key in KEYS_METHOD or skip_names and key in ('name', 'spatial'):
                        continue
                    kw_label[key] = _get_label(dataset, key, value, mode=mode, name=name)  # noqa: E501
                nkws_label.append(kw_label)
            kw_label = {}  # merge labels for stuff inside subplot
            for key in sorted((key for kw in nkws_label for key in kw), key=sorter):
                values = tuple(kw.get(key, '') for kw in nkws_label)
                kw_label[key] = _combine_labels(*values, identical=True)
            ikws_label.append(kw_label)
        kw_label = {}  # merge labels across subplots
        for key in sorted((key for kw in ikws_label for key in kw), key=sorter):
            values = tuple(kw.get(key, '') for kw in ikws_label)
            kw_label[key] = _combine_labels(*values, identical=identical)
        if not identical:  # dictionary for each separate label
            kw_label = [{key: kw_label[key][i] for key in kw_label} for i in range(len(ikws_label))]  # noqa: E501
        if not kw_label:  # WARNING: critical or else zip below creates empty list
            continue
        kws_label.append(kw_label)

    # Combine pairs of labels
    # NOTE: This optionally assigns labels that are identical across the pair to
    # the front or the back of the combined 'this vs. that' label.
    items_skip = ('local', 'global')  # WARNING: update if area labels change
    kws_label = [kws_label] if identical else list(zip(*kws_label))
    labels = []
    for i, kws_pair in enumerate(kws_label):
        # Allocate label components
        keys = set(key for kw in kws_pair for key in kw)
        front, left, right, back = [], [], [], []
        if not invert:
            kws_pair = kws_pair[::-1]  # place dependent variable *first*
        for key in sorted(keys, key=sorter):
            items = list(filter(None, (kw.get(key) for kw in kws_pair)))
            if len(set(items)) > 1:
                both = items[0] in items_skip and items[1] in items_skip
                if both or items[0] not in items_skip:
                    left.append(items[0])  # *both* e.g. local vs. global feedback
                if both or items[1] not in items_skip:
                    right.append(items[1])
            elif items:
                if items[0] in items_skip:
                    pass  # not a special region e.g. 'West Pacific'
                elif key in order_back:
                    back.append(items[0])  # e.g. abrupt vs. picontrol *feedback*
                else:
                    front.append(items[0])  # e.g. *cmip5* abrupt vs. picontrol
        # Combine and adjust labels
        reverse_drop = lambda lab, key: lab[::-1].replace(key[::-1], '', 1)[::-1]
        left, right = ' '.join(filter(None, left)), ' '.join(filter(None, right))
        center = ' vs. '.join(filter(None, (left, right)))
        label = ' '.join(filter(None, (*front, center, *back)))
        abrupt = r'abrupt 4$\times$CO$_2$'
        control = 'pre-industrial'
        change = f'{abrupt} minus {control}'
        warming = 'temperature response'
        scaling = rf'{abrupt} 2$\times$CO$_2$'
        if change in label:
            if label[-8:] == 'feedback':
                pass  # do not replace with 'response'
            elif label in (change, f'early {change}', f'late {change}'):
                pass  # special exceptions in case 'feedback' is missing
            else:
                label = reverse_drop(f'{label.replace(change, abrupt)} response', f'{abrupt} ')  # noqa: E501
        if warming in label:
            label = label.replace(warming, 'warming')
        if identical and label[-8:] == 'feedback':  # change end to 'feedbacks'
            label = f'{label}s'
        if scaling in label:  # drop '2xCO2' scaling
            label = label.replace(scaling, abrupt, 1)
        if label.count('boreal') == 2:  # drop e.g. 'boreal winter minus boreal summer'
            label = reverse_drop(label, 'boreal ')
        if control in label and 'surface warming' in label:  # convert 'ts'
            label = label.replace('warming', 'temperature')
        label = _fit_label(label.strip(), **kwargs)
        if title_case:
            label = _capitalize_label(label)
        labels.append(label)

    return labels[0] if identical else labels


def _infer_path(dataset, *kws_process):
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
    # NOTE: This omits keywords that correspond to default values but always
    # includes experiment because default is ambiguous between variables.
    kws_process = [(kw,) if isinstance(kw, dict) else tuple(kw) for kw in kws_process]
    kws_process = [kw.copy() for kws in kws_process for kw in kws]
    labels = []
    defaults = {'project': 'cmip', 'ensemble': 'flagship', 'period': 'ann'}
    defaults.update({'source': 'eraint', 'statistic': 'slope', 'region': 'globe'})
    kws_process = list(map(_group_parts, kws_process))
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
            labels.append(parts)  # sort these so newer orders overwrite
    result = '_'.join('-'.join(sorted(parts)) for parts in labels)
    return result


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
    detect_process.extend(('area', 'volume', 'spatial', 'institute', *KEYS_METHOD))
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
        elif any(key.startswith(prefix) for prefix in DETECT_GUIDE):
            kw_colorbar[key] = kw_legend[key] = value  # shared keywords
        else:  # arbitrary plotting keywords
            kw_command[key] = value
    if isinstance(name, str):  # note name of None always ignored
        kw_process['name'] = name
    if 'extend' in kw_colorbar:  # special case, required by contour
        kw_command['extend'] = kw_colorbar['extend']
    if 'colorbar' in kw_colorbar:  # colorbar location, or use 'loc' for both
        kw_colorbar['loc'] = kw_colorbar.pop('colorbar')
    if 'legend' in kw_legend:  # legend location, or use 'loc' for both
        kw_legend['loc'] = kw_legend.pop('legend')
    fields = ('figure', 'gridspec', 'axes', 'command', 'other', 'attrs', 'colorbar', 'legend')  # noqa: E501
    collection = collections.namedtuple('kwargs', fields)
    kw_collection = collection(
        kw_figure, kw_gridspec, kw_axes, kw_command,
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
        The keyword arguments for `reduce` and `method`.
    kws_collection : list of list of namedtuple
        The keyword arguments for various plotting tasks.
    figlabel : str
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
    # ic('Row specs:', *rowspecs, sep='\n')  # uncomment for debugging
    # ic('Column specs:', *colspecs, sep='\n')
    for i, ispecs in enumerate((rowspecs, colspecs)):
        # Collect specs per subplot
        ikws_process, ikws_collection = [], []
        if not isinstance(ispecs, list):
            ispecs = [ispecs]
        for jspecs in ispecs:  # specs per figure
            jkws_process, jkws_collection = [], []
            if not isinstance(jspecs, list):
                jspecs = [jspecs]
            for kspecs in jspecs:  # specs per subplot
                # Translate into name and dictionary
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
                # Iterate over correlation pairs
                for spec in kspecs:
                    kw_process, kw_collection = get_spec(dataset, spec, **kwargs)
                    if value := kw_collection.figure.get('refwidth', None):
                        refwidth = value  # for scaling grid labels
                    if not any(kw_process.get(key) for key in ('lon', 'lat', 'area')):
                        refscale = 0.6 if i == 0 else 1.0  # i.e. longitude-latitude
                    kkws_process.append(kw_process)
                    kkws_collection.append(kw_collection)
                jkws_process.append(tuple(kkws_process))  # denotes correlation-pair
                jkws_collection.append(tuple(kkws_collection))
            ikws_process.append(jkws_process)
            ikws_collection.append(jkws_collection)
        # Infer grid labels
        abcwidth = pplt.units(2 * pplt.rc.fontsize, 'pt', 'in')
        refwidth = pplt.units(refwidth or pplt.rc['subplots.refwidth'], 'in')
        refwidth -= abcwidth if len(rowspecs) < 2 or len(colspecs) < 2 else 0
        grdlabels = _infer_labels(
            dataset,
            *ikws_process,
            identical=False,
            long_names=True,
            title_case=True,
            fontsize=pplt.rc.fontlarge,
            refwidth=(refscale or 1) * refwidth,  # account for a-b-c space
        )
        gridlabels.append(grdlabels)
        kws_process.append(ikws_process)
        kws_collection.append(ikws_collection)

    # Combine row and column specifications for plotting and file naming
    # NOTE: Several plotted values per subplot can be indicated in either the
    # row or column list, and the specs from the other list are repeated below.
    # WARNING: Critical to make copies of dictionaries or create new ones
    # here since itertools product repeats the same spec multiple times.
    autocmap = ('Fire', 'NegPos', 'NegPos') if autocmap is True else autocmap
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
        ikws_row, ikws_col = _to_lists(ikws_row, ikws_col)
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
            rkws_process = [kw.copy() for kw in rkws_process]  # NOTE: copy needed
            ckws_process = [kw.copy() for kw in ckws_process]
            if len(rkws_process) == 1 and len(ckws_process) == 1:
                kw_process = (*rkws_process, *ckws_process)
            elif len(rkws_process) == 2 and len(ckws_process) == 2:
                kw_process = rkws_process
                for rkw, ckw in zip(rkws_process, ckws_process):
                    for key, value in ckw.items():
                        rkw.setdefault(key, value)
            elif len(rkws_process) in (1, 2) and len(ckws_process) in (1, 2):
                if len(rkws_process) == 1:
                    ref, kw_process = rkws_process[0], ckws_process
                else:
                    ref, kw_process = ckws_process[0], rkws_process
                for key, value in ref.items():
                    for kw in kw_process:
                        kw.setdefault(key, value)
            else:
                raise ValueError(f'Impossible combination {rkws_process} and {ckws_process}.')  # noqa: E501

            # Merge pairs and possibly apply autocmap
            if len(kw_process) == 2:
                keys1, keys2 = map(set, kw_process)
                for kw1, kw2 in (kw_process, kw_process[::-1]):
                    for key, value in kw2.items():
                        kw1.setdefault(key, value)
                if kw_process[0] == kw_process[1]:
                    kw_process = kw_process[:1]
            if autocmap:  # use different colormaps for project and other anomalies
                idx = max(
                    0 if not isinstance(value, str) or '-' not in value
                    else 1 if key != 'project' else 2
                    for kw in kw_process for key, value in kw.items()
                )
                kw_collection.command['cmap'] = autocmap[idx]
                kw_collection.command.setdefault('robust', 98 - 2 * idx)
                kw_collection.command.setdefault('symmetric', idx > 0)
            ikws_process.append(tuple(kw_process))
            ikws_collection.append(kw_collection)

        # Infer legend and axes prefixes
        kw = dict(refwidth=np.inf, identical=False, skip_names=True, title_case=False)
        ikws_pair = [  # process arguments for e.g. scatter plot
            ikws_process[0][:1],
            ikws_process[0][1:] or ikws_process[0][:1]
        ]
        prefixes_axes = _infer_labels(dataset, *ikws_pair, **kw)
        prefixes_legend = _infer_labels(dataset, *ikws_process, **kw)
        for axis, prefix in zip('xy', prefixes_axes):
            if prefix:  # apply for last item in subplot
                ikws_collection[-1].attrs.setdefault(f'{axis}label_prefix', prefix)
        for pspec, prefix in zip(ikws_collection, prefixes_legend):
            if prefix:  # add prefix to this
                pspec.attrs.setdefault('short_prefix', prefix)
        kws_process.append(ikws_process)
        kws_collection.append(ikws_collection)

    # Infer figure label and grid labels
    ncols = len(colspecs) if len(colspecs) > 1 else len(rowspecs) if len(rowspecs) > 1 else 3  # noqa: E501
    figwidth = ncols * refwidth + 0.3 * refwidth * (ncols - 1)
    figlabel = _infer_labels(
        dataset,
        *kws_process,
        refwidth=figwidth,
        identical=True,
        long_names=True,
        title_case=True,
        fontsize=pplt.rc.fontlarge
    )
    pathspecs = [dspec for ikws_process in kws_process for dspec in ikws_process]
    pathlabel = _infer_path(dataset, *pathspecs)
    fontwidth = pplt.utils._fontsize_to_pt(pplt.rc.fontlarge)  # a-b-c label adjustment
    axeswidth = refwidth - 3 * pplt.units(fontwidth, 'pt', 'in')
    kw_fit = dict(fontsize=fontwidth, refwidth=axeswidth)
    if len(rowspecs) == 1 and len(colspecs) == 1:
        gridlabels = None
    elif len(rowspecs) > 1 and len(colspecs) > 1:
        gridlabels = tuple(gridlabels)  # NOTE: tuple critical for create_plot
    elif len(rowspecs) > 1:
        gridlabels = [_fit_label(label, **kw_fit) for label in gridlabels[0]]
    else:
        gridlabels = [_fit_label(label, **kw_fit) for label in gridlabels[1]]
    return kws_process, kws_collection, figlabel, pathlabel, gridlabels
