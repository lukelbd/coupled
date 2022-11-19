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

__all__ = ['parse_spec', 'parse_specs']

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
    'region',
    'statistic',
    'source',
    'version',  # feedback version index
    'name',
)

# General translations
TRANSLATE_PATHS = {
    ('area', 'avg'): 'avg',
    # ('area', 'avg'): None,  # TODO: consider changing back
    ('lon', 'avg'): 'avg',
    # ('lon', 'avg'): None,  # TODO: consider changing back
    ('lon', 'int'): None,
    ('lat', 'absmin'): 'min',
    ('lat', 'absmax'): 'max',
    ('region', 'point'): 'point',
    ('region', 'globe'): 'globe',
    ('region', 'latitude'): 'zonal',
    ('region', 'hemisphere'): 'hemi',
    ('institute', 'avg'): 'inst',
    ('institute', 'flagship'): 'memb',
    ('experiment', 'control'): 'pictl',
    ('experiment', 'response'): '4xco2',
    ('experiment', 'picontrol'): 'pictl',
    ('experiment', 'abrupt4xco2'): '4xco2',
    ('ensemble', 'flagship'): 'flag',
}
TRANSLATE_LABELS = {  # default is title-case of input
    ('area', 'avg'): 'global-average',
    # ('area', 'avg'): None,  # TODO: consider changing back
    ('area', 'trop'): 'tropical-average',
    ('area', 'pool'): 'warm pool',
    ('area', 'wlam'): 'warm pool',
    ('area', 'elam'): 'cold tongue',
    ('area', 'nina'): 'West Pacific',
    ('area', 'nino'): 'East Pacific',
    ('area', 'nino3'): 'East Pacific',
    ('area', 'nino4'): 'East Pacific',
    ('lon', 'avg'): 'zonal-average',
    # ('lon', 'avg'): None,  # TODO: consider changing back
    ('lon', 'int'): None,
    ('lat', 'absmin'): 'minimum',
    ('lat', 'absmax'): 'maximum',
    ('source', 'eraint'): 'internal',
    ('source', 'zelinka'): 'Zelinka',
    ('region', 'globe'): 'globally-normalized',
    ('region', 'point'): 'locally-normalized',
    ('region', 'latitude'): 'zonally-normalized',
    ('region', 'hemisphere'): 'hemisphere-normalized',
    ('institute', 'avg'): 'institute-average',
    ('institute', 'flagship'): 'institute-flagship',
    ('project', 'cmip'): 'CMIP',
    ('project', 'cmip5'): 'CMIP5',
    ('project', 'cmip6'): 'CMIP6',
    ('project', 'cmip56'): 'matching CMIP5',
    ('project', 'cmip65'): 'matching CMIP6',
    ('project', 'cmip55'): 'non-matching CMIP5',
    ('project', 'cmip66'): 'non-matching CMIP6',
    ('project', 'cmip5665'): 'matching CMIP',
    ('project', 'cmip6556'): 'matching CMIP',
    ('experiment', 'control'): 'pre-industrial',
    ('experiment', 'response'): r'abrupt 4$\times$CO$_2$',
    ('experiment', 'picontrol'): 'pre-industrial',
    ('experiment', 'abrupt4xco2'): r'abrupt 4$\times$CO$_2$',
    ('ensemble', 'flagship'): 'flagship-ensemble',
}

# Time translations
TRANSLATE_LONGS = {  # default is title-case of input
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
    parts = REGEX_SPLIT.split(value) if isinstance(value, str) else (value,)
    labels = []
    for part in parts:
        if part is None:
            continue
        if isinstance(part, str) and REGEX_FLOAT.match(part):  # e.g. 850-250hPa
            part = float(part)
        if key == 'name':
            part = alias_to_name.get(part, part)
            if mode == 'path':
                label = name_to_alias.get(part, part)
            elif mode == 'short':
                label = dataset[part].short_name
            else:
                label = dataset[part].long_name
        elif isinstance(part, str):
            operators = {'+': 'plus', '-': 'minus', '*': 'times', '/': 'over'}
            if part in '+-*/':  # TODO: support other operations?
                label = part if mode == 'path' else operators[part]
            elif mode == 'path':
                label = TRANSLATE_PATHS.get((key, part), part)
            elif mode == 'short':
                label = TRANSLATE_SHORTS.get((key, part), part)
            else:
                label = TRANSLATE_LONGS.get((key, part), part)
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


def _infer_path(dataset, *kws_red):
    """
    Convert reduction operators into path suitable for saving.

    Parameters
    ----------
    dataset : xarray.Dataset
        The source dataset.
    *kws_red : dict
        The `get_data` keywords.

    Returns
    -------
    path : str
        The parts joined with underscores and dashes.
    """
    # NOTE: This omits keywords that correspond to default values but always
    # includes experiment because default is ambiguous between variables.
    kws_red = [(kw,) if isinstance(kw, dict) else tuple(kw) for kw in kws_red]
    labels = []
    defaults = {'project': 'cmip', 'ensemble': 'flagship', 'period': 'ann'}
    defaults.update({'source': 'eraint', 'statistic': 'slope', 'region': 'globe'})
    for key in ORDER_LOGICAL:
        seen = set()
        parts = []
        values = [kw[key] for kws in kws_red for kw in kws if key in kw]
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
    dataset, *dicts, identical=False, long_names=False, title_case=False, **kwargs
):
    """
    Convert reduction operators into human-readable labels.

    Parameters
    ----------
    dataset : xarray.Dataset
        The source dataset.
    *dicts : tuple of dict
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
    mode = 'long' if long_names else 'short'
    order = list(ORDER_READABLE)
    sorter = lambda key: order.index(key) if key in order else len(order)
    dicts = [(kw,) if isinstance(kw, dict) else tuple(kw) for kw in dicts]
    labels = []
    for i in range(2):  # indices in correlation pair
        kws = []
        ikws = [tup[i] for tup in dicts if i < len(tup)]
        for ikw in ikws:
            kw = {}
            ignore = ('method', 'std', 'pctile', 'invert')
            for key in sorted(ikw, key=sorter):
                value = ikw[key]
                if key in ignore:
                    continue
                label = _get_label(dataset, key, value, mode=mode)
                kw[key] = label
            kws.append(kw)
        keys = sorted((key for kw in kws for key in kw), key=sorter)
        labs = {}
        for key in keys:
            values = tuple(kw.get(key, '') for kw in kws)
            labs[key] = _merge_labels(*values, identical=identical)
        if not identical:
            labs = [{key: labs[key][i] for key in labs} for i in range(len(kws))]
        if not labs:  # WARNING: critical or else zip below creates empty list
            continue
        labels.append(labs)

    # Combine pairs of labels
    # NOTE: This optionally assigns labels that are identical across the pair to
    # the front or the back of the combined 'this vs. that' label.
    result = []
    labels = [[labs] for labs in labels] if identical else list(zip(*labels))
    keys = sorted((key for kw in kws for key in kw), key=sorter)
    for i, labs in enumerate(labels):
        for key in keys:
            if len(labs) != 2:
                pass
            elif any(kw.get(key) != labs[0].get(key, object()) for kw in labs):
                pass
            elif key == 'name':
                del labs[0][key]  # e.g. abrupt vs. picontrol *feedback*
            else:
                del labs[1][key]  # e.g. *cmip5* abrupt vs. picontrol
        label = []
        for kw in labs:
            lab = ' '.join(lab for lab in kw.values() if lab)
            if not lab:
                continue
            label.append(lab)
        label = ' vs. '.join(label)
        label = _wrap_label(label, **kwargs)
        if title_case and label[:1].islower():
            label = label[:1].upper() + label[1:]
        result.append(label)
    result = result[0] if identical else result
    return result


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


def _wrap_label(label, refwidth=None, refscale=None, nmax=None):
    """
    Replace spaces with newlines to accommodate a subplot or figure label.

    Parameters
    ----------
    label : str
        The input label.
    refwidth : unit-spec, optional
        The reference maximum width.
    refscale : float, optional
        Additional scale on the width.
    nmax : int, optional
        Optional maximum number of breaks to use.

    Returns
    -------
    label : str
        The label with inserted newlines.
    """
    label = label or ''
    label = label.replace('\n', ' ')  # remove previous wrapping
    idxs = np.array([i for i, c in enumerate(label) if c == ' '])
    adjs = idxs.astype(float)
    for m in re.finditer(r'\$[^$]+\$', label):  # ignore all latex-math components
        i, j = m.span()
        adjs[(i <= idxs) & (idxs <= j)] = 0  # ignore by making lower than thresh
        adjs[idxs > j] -= 0.9 * (j - i)  # try to account for actual space
    refscale = refscale or 1
    refwidth = refwidth or pplt.rc['subplots.refwidth']
    threshs = refscale * 1.3 * pplt.units(refwidth, 'in', 'em') * np.arange(1, 10)
    label = list(label)  # convert string to list
    count = 0
    for thresh in threshs:
        if not any(adjs > thresh):  # including empty adjs
            continue
        if nmax and count >= nmax:
            continue
        idx = np.argmin(np.abs(adjs - thresh))
        count += 1  # add to newline count
        threshs += (adjs[idx] - thresh)  # adjust next thresholds
        label[idxs[idx]] = '\n'
    label = ''.join(label)
    return label


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
    kw_dat : dict
        The indexers used to reduce the data variable with `.reduce`. This is
        parsed specially compared to other keywords, and its keys are restricted
        to ``'name'`` and any coordinate or multi-index names.
    kw_plt : namedtuple of dict
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
    # inside the 'name' key in kw_dat. This helps when merging variable specifications
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
    kw_fig, kw_grd, kw_axs = {}, {}, {}
    kw_cmd, kw_oth, kw_att = {}, {}, {}
    kw_cba, kw_leg, kw_dat = {}, {}, {}
    dat_detect = list(dataset.sizes)
    dat_detect.extend(name for idx in dataset.indexes.values() for name in idx.names)
    dat_detect.extend(('area', 'volume', 'institute', 'method', 'invert', 'pctile', 'std'))  # noqa: E501
    fig_detect = ['fig', 'ref', 'space', 'share', 'span', 'align']
    grd_detect = ['space', 'ratio', 'group', 'equal', 'left', 'right', 'bottom', 'top']
    grd_detect = [f'{prefix}{key}' for key in grd_detect for prefix in ('w', 'h', '')]
    axs_detect = ['x', 'y', 'lon', 'lat', 'abc', 'title', 'proj', 'land', 'coast']
    oth_detect = ['horiz', 'pcolor', 'offset', 'cycle', 'shade', 'oneone', 'linefit', 'annotate']  # noqa: E501
    att_detect = ['short', 'long', 'standard', 'xlabel_', 'ylabel_']
    cba_detect = ['extend', 'tick', 'locator', 'formatter', 'minor', 'label', 'length', 'shrink']  # noqa: E501
    leg_detect = ['ncol', 'order', 'frame', 'handle', 'border', 'column']
    for key, value in kw.items():  # NOTE: sorting performed in _parse_labels
        if key in dat_detect:
            kw_dat[key] = value  # e.g. for averaging
        elif any(key.startswith(prefix) for prefix in fig_detect):
            kw_fig[key] = value
        elif any(key.startswith(prefix) for prefix in grd_detect):
            kw_grd[key] = value
        elif any(key.startswith(prefix) for prefix in axs_detect):
            kw_axs[key] = value
        elif any(key.startswith(prefix) for prefix in oth_detect):
            kw_oth[key] = value
        elif any(key.startswith(prefix) for prefix in att_detect):
            kw_att[key] = value
        elif any(key.startswith(prefix) for prefix in cba_detect):
            kw_cba[key] = value
        elif any(key.startswith(prefix) for prefix in leg_detect):
            kw_leg[key] = value
        else:  # arbitrary plotting keywords
            kw_cmd[key] = value
    if isinstance(name, str):  # NOTE: here name of None always ignored
        kw_dat['name'] = name  # always place last for gridspec labels
    keys = ('method', 'std', 'pctile', 'invert')
    fields = ('figure', 'gridspec', 'axes', 'command', 'other', 'attrs', 'colorbar', 'legend')  # noqa: E501
    tuple_ = collections.namedtuple('kwargs', fields)
    kw_dat.update({key: kwargs.pop(key) for key in keys if key in kwargs})
    kw_plt = tuple_(kw_fig, kw_grd, kw_axs, kw_cmd, kw_oth, kw_att, kw_cba, kw_leg)
    return kw_dat, kw_plt


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
    dataspecs : list of list of tuple of dict
        The reduction keyword argument specifications.
    plotspecs : list of list of kwargs
        The keyword arguments used for plotting.
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
    dataspecs, plotspecs, gridlabels = [], [], []
    for n, inspecs in enumerate((rowspecs, colspecs)):
        datspecs, pltspecs = [], []
        if not isinstance(inspecs, list):
            inspecs = [inspecs]
        for ispecs in inspecs:  # specs per figure
            dspecs, pspecs = [], []
            if not isinstance(ispecs, list):
                ispecs = [ispecs]
            for ispec in ispecs:  # specs per subplot
                dspec, pspec = [], []
                if ispec is None:
                    ispec = (None,)  # possibly construct from keyword args
                elif isinstance(ispec, (str, dict)):
                    ispec = (ispec,)
                elif len(ispec) != 2:
                    raise ValueError(f'Invalid variable specs {ispec}.')
                elif type(ispec[0]) != type(ispec[1]):  # noqa: E721  # i.e. (str, dict)
                    ispec = (ispec,)
                else:
                    ispec = tuple(ispec)
                for spec in ispec:  # specs per correlation pair
                    kw_dat, kw_plt = parse_spec(dataset, spec, **kwargs)
                    if value := kw_plt.figure.get('refwidth', None):
                        refwidth = value
                    if not any(kw_dat.get(key, None) for key in ('lon', 'lat', 'area')):
                        refscale = 0.7 if n == 0 else None
                    dspec.append(kw_dat)
                    pspec.append(kw_plt)
                dspecs.append(tuple(dspec))  # tuple to identify as correlation-pair
                pspecs.append(tuple(pspec))
            datspecs.append(dspecs)
            pltspecs.append(pspecs)
        ncols = len(colspecs) if len(colspecs) > 1 else len(rowspecs) if len(rowspecs) > 1 else 4  # noqa: E501
        refwidth = refwidth or pplt.rc['subplots.refwidth']
        figwidth = 2 * ncols * refwidth  # larger value
        zerospecs = [dspecs[0] if dspecs else {} for dspecs in datspecs]
        grdlabels = _infer_labels(
            dataset,
            *zerospecs,
            identical=False,
            long_names=True,
            title_case=True,
            refwidth=refwidth,
            refscale=refscale,
        )
        gridlabels.append(grdlabels)
        dataspecs.append(datspecs)
        plotspecs.append(pltspecs)

    # Combine row and column specifications for plotting and file naming
    # NOTE: Several plotted values per subplot can be indicated in either the
    # row or column list, and the specs from the other list are repeated below.
    # WARNING: Critical to make copies of dictionaries or create new ones
    # here since itertools product repeats the same spec multiple times.
    bothspecs = [
        [list(zip(dspecs, pspecs)) for dspecs, pspecs in zip(datspecs, pltspecs)]
        for datspecs, pltspecs in zip(dataspecs, plotspecs)
    ]
    kw_infer = dict(identical=False, long_names=True, title_case=False)
    dataspecs, plotspecs = [], []
    for (i, rspecs), (j, cspecs) in itertools.product(*map(enumerate, bothspecs)):
        if len(rspecs) == 1:
            rspecs = list(rspecs) * len(cspecs)
        if len(cspecs) == 1:
            cspecs = list(cspecs) * len(rspecs)
        if len(rspecs) != len(cspecs):
            raise ValueError(
                'Incompatible per-subplot spec count.'
                + f'\nRow specs ({len(rspecs)}): \n' + '\n'.join(map(repr, rspecs))
                + f'\nColumn specs ({len(cspecs)}): \n' + '\n'.join(map(repr, cspecs))
            )
        dspecs, pspecs = [], []
        for k, (rspec, cspec) in enumerate(zip(rspecs, cspecs)):  # subplot entries
            rkws_dat, rkws_plt, ckws_dat, ckws_plt = *rspec, *cspec
            kwargs = type((rkws_plt or ckws_plt)[0])
            kws = []
            for field in kwargs._fields:
                kw = {}  # NOTE: previously applied default values here
                for ikws_plt in (rkws_plt, ckws_plt):
                    for ikw_plt in ikws_plt:  # correlation pairs
                        for key, value in getattr(ikw_plt, field).items():
                            kw.setdefault(key, value)  # prefer row entries
                kws.append(kw)
            kw_plt = kwargs(*kws)
            rkws_dat = tuple(kw.copy() for kw in rkws_dat)  # NOTE: copy is critical
            ckws_dat = tuple(kw.copy() for kw in ckws_dat)
            for ikws_dat, jkws_dat in ((rkws_dat, ckws_dat), (ckws_dat, rkws_dat)):
                for key in ikws_dat[0]:
                    if key not in ikws_dat[-1]:
                        continue
                    if ikws_dat[0][key] == ikws_dat[-1][key]:  # identical across pair
                        for kw in jkws_dat:  # apply to other pair
                            kw.setdefault(key, ikws_dat[0][key])
            kws_dat = {}  # filter unique specifications
            ignore = ('method', 'std', 'pctile', 'invert')
            for kw_dat in (*rkws_dat, *ckws_dat):  # correlations for rows and columns
                keys = sorted(key for key in kw_dat if key not in ignore)
                keyval = tuple((key, kw_dat[key]) for key in keys)
                if tuple(keyval) in kws_dat:
                    continue
                others = tuple(other for other in kws_dat if set(keyval) < set(other))
                if others:
                    continue
                others = tuple(other for other in kws_dat if set(other) < set(keyval))
                for other in others:  # prefer more selection keywords
                    kws_dat.pop(other)
                kws_dat[tuple(keyval)] = kw_dat
            kws_dat = tuple(kws_dat.values())  # possible correlation tuple
            if len(kws_dat) > 2:
                raise RuntimeError(
                    'Expected 1-2 specs for combining with get_data but got '
                    f'{len(kws_dat)} specs: ' + '\n'.join(map(repr, kws_dat))
                )
            dspecs.append(kws_dat)
            pspecs.append(kw_plt)
        specs1 = tuple(kws[:1] for kws in dspecs)
        specs2 = tuple(kws[1:] for kws in dspecs)
        labels = _infer_labels(dataset, *dspecs, refwidth=figwidth, **kw_infer)
        xlabels = _infer_labels(dataset, *specs1, refwidth=refwidth, **kw_infer)
        ylabels = _infer_labels(dataset, *specs2, refwidth=refwidth, **kw_infer)
        ylabels = ylabels or [None] * len(xlabels)
        for pspec, label, xlabel, ylabel in zip(pspecs, labels, xlabels, ylabels):
            if label:  # use identical label as fallback
                pspec.attrs.setdefault('short_prefix', label)
            if xlabel:
                pspec.attrs.setdefault('xlabel_prefix', xlabel)
            if ylabel:
                pspec.attrs.setdefault('ylabel_prefix', ylabel)
        dataspecs.append(dspecs)
        plotspecs.append(pspecs)
    subspecs = [dspec for dspecs in dataspecs for dspec in dspecs]
    figlabel = _infer_labels(
        dataset,
        *subspecs,
        identical=True,
        long_names=True,
        title_case=True,
        refwidth=figwidth,
        refscale=1.0,
    )
    pathlabel = _infer_path(dataset, *subspecs)
    return dataspecs, plotspecs, figlabel, pathlabel, gridlabels
