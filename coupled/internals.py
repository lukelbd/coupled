#!/usr/bin/env python3
"""
Internal helper functions for figure templates.
"""
import itertools
import re

import climopy as climo  # noqa: F401
import numpy as np
import pandas as pd
import proplot as pplt
import xarray as xr
from climopy import ureg, vreg  # noqa: F401
from icecream import ic  # noqa: F401

from .output import FEEDBACK_TRANSLATIONS
from cmip_data.internals import MODELS_INSTITUTES, INSTITUTES_ABBREVS

__all__ = [
    'get_data',
    'get_breakdown',
]

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

# Abbreviation constants
TRANSLATE_ABBREVS = {
    ('area', 'avg'): 'avg',
    # ('area', 'avg'): None,  # TODO: consider changing back
    ('lon', 'avg'): None,
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
    ('lon', 'avg'): None,
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
TRANSLATE_LONGS = {  # default is title-case of input
    ('period', 'ann'): 'annual',
    ('period', 'djf'): 'boreal winter',
    ('period', 'mam'): 'boreal spring',
    ('period', 'jja'): 'boreal summer',
    ('period', 'son'): 'boreal autumn',
    **TRANSLATE_LABELS
}
TRANSLATE_SHORTS = {
    ('period', 'ann'): 'annual',
    ('period', 'djf'): 'DJF',
    ('period', 'mam'): 'MAM',
    ('period', 'jja'): 'JJA',
    ('period', 'son'): 'SON',
    **TRANSLATE_LABELS
}


def _infer_newlines(string, refwidth=None, scale=None):
    """
    Replace spaces with line breaks to accommodate a subplot or figure width.

    Parameters
    ----------
    string : str
        The input string.
    refwidth : unit-spec, optional
        The reference maximum width.
    scale : float, optional
        Additional scale on the width.
    """
    string = string or ''
    idxs = np.array([i for i, c in enumerate(string) if c == ' '])
    adjs = idxs.copy()
    for m in re.finditer(r'\$[^$]+\$', string):  # ignore non-math texts
        i, j = m.span()
        adjs[(i <= idxs) & (idxs <= j)] = 0  # i.e. ignore since lower than thresh
        adjs[idxs > j] -= j - i
    scale = scale or 1.1
    refwidth = refwidth or pplt.rc['subplots.refwidth']
    threshs = scale * pplt.units(refwidth, 'in', 'em') * np.arange(1, 10)
    chars = list(string)  # string to list
    for thresh in threshs:
        mask = adjs > thresh
        if not mask.any():
            continue
        chars[np.min(idxs[mask])] = '\n'
    return ''.join(chars)


def _infer_string(dataset, key, value, mode='abbrv'):
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
    mode : {'abbrv', 'short', 'long'}, optional
        The label type.
    """
    if mode not in ('abbrv', 'short', 'long'):
        raise ValueError(f'Invalid label mode {mode!r}.')
    alias_to_name = {alias: name for alias, (name, _) in FEEDBACK_TRANSLATIONS.items()}
    name_to_alias = {name: alias for alias, name in alias_to_name.items()}  # keep last
    labels = []
    parts = REGEX_SPLIT.split(value) if isinstance(value, str) else (value,)
    for part in parts:
        if part is None:
            continue
        if isinstance(part, str) and REGEX_FLOAT.match(part):  # e.g. 850-250hPa
            part = float(part)
        if key == 'name':
            part = alias_to_name.get(part, part)
            if mode == 'abbrv':
                label = name_to_alias.get(part, part)
            elif mode == 'short':
                label = dataset[part].short_name
            else:
                label = dataset[part].long_name
        elif isinstance(part, str):
            operators = {'+': 'plus', '-': 'minus', '*': 'times', '/': 'over'}
            if part in '+-*/':  # TODO: support other operations?
                label = part if mode == 'abbrv' else operators[part]
            elif mode == 'abbrv':
                label = TRANSLATE_ABBREVS.get((key, part), part)
            elif mode == 'short':
                label = TRANSLATE_SHORTS.get((key, part), part)
            else:
                label = TRANSLATE_LONGS.get((key, part), part)
        else:
            unit = dataset[key].climo.units
            if not isinstance(part, ureg.Quantity):
                part = ureg.Quantity(part, unit)
            part = part.to(unit)
            if mode == 'abbrv':
                label = f'{part:~.0f}'
            else:
                label = f'${part:~L.0f}$'
        if label is None:  # e.g. skip 'avg' label
            continue
        if mode == 'abbrv':  # extra processing
            label = label.replace('\N{DEGREE SIGN}', '')
            label = label.replace('+', '')
            label = label.replace('-', '')
            label = label.replace('/', '')
            label = label.replace('*', '')
            label = label.replace('_', '')
            label = label.replace(' ', '')
            label = label.lower()
        labels.append(label)
    string = '' if mode == 'abbrv' else ' '
    return string.join(labels)


def _infer_abbrevs(
    dataset,
    *kwargs,
    identical=False
):
    """
    Convert reudction operators into

    Parameters
    ----------
    dataset : xarray.Dataset
        The source dataset.
    *kwargs : dict
        The `get_data` keywords.
    """
    keys_method = ('method', 'std', 'pctile', 'invert')  # special method keywords
    kwargs = [(kw,) if isinstance(kw, dict) else tuple(kw) for kw in kwargs]
    order = list(ORDER_LOGICAL)
    sorter = lambda key: order.index(key) if key in order else len(order)
    seen = set()
    keys = sorted((key for kws in kwargs for kw in kws for key in kw), key=sorter)
    keys = [key for key in keys if key not in seen and not seen.add(key)]
    labels = []
    for key in keys:
        if key in keys_method:
            continue
        values = [kw[key] for kws in kwargs for kw in kws if key in kw]
        ident = all(value == values[0] for value in values)
        if len(values) == 1:
            if not identical:
                continue
        else:
            if ident != identical:
                continue
        for value in values:
            label = _infer_string(dataset, key, value, mode='abbrv')
            if not label:
                continue
            if label not in labels:  # e.g. 'avg' to be ignored
                labels.append(label)
    return labels


def _infer_labels(
    dataset,
    *kwargs,
    keeppairs=True,
    identical=False,
    title_case=False,
    long_names=False,
    skip_area=False,
    skip_name=False,
    refwidth=None,
):
    """
    Convert reduction operators into human-readable labels.

    Parameters
    ----------
    dataset : xarray.Dataset
        The source dataset.
    *kwargs : tuple of dict
        The reduction keyword arguments.
    keeppairs : bool, optional
        Whether to keep identical reduce operations in 'vs.' pair.
    identical : bool, optional
        Whether to keep identical reduce operations across list.
    title_case : bool or str, optional
        Whether to capitlize the first letter of each label.
    long_names : bool, optional
        Whether to use long reduce instructions.
    skip_area : bool, optional
        Whether to skip the area average indication.
    skip_name : bool, optional
        Whether to skip the 'name' key for e.g. prefix usage.
    refwidth : float, optional
        Passed to `_infer_newlines`.
    """
    # Reduce the label dicationaries to account for redundancies across the list,
    # either dropping them (useful for gridspec labels and legend labels) or
    # dropping everything but the redundancies (useful for figure titles).
    # TODO: Also drop scalar 'identical' specifications of e.g. 'annual', 'slope',
    # and 'eraint' default selections for feedback variants and climate averages?
    modifiers = (
        'feedback',
        'forcing',
        r'2$\times$CO$_2$',  # leads forcing strings
        r'4$\times$CO$_2$',
        'energy transport',
        'energy convergence',
    )
    mode = 'long' if long_names else 'short'
    order = list(ORDER_READABLE)
    sorter = lambda key: order.index(key) if key in order else len(order)
    replace = lambda s, r: s.replace(f' {r}', '').replace(f'{r} ', '').replace('effective', 'net')  # noqa: E501
    keys_method = ('method', 'std', 'pctile', 'invert')  # special method keywords
    kwargs = [(kw,) if isinstance(kw, dict) else tuple(kw) for kw in kwargs]
    labels = []
    for i in range(2):  # indices in correlation pair
        # Infer labels from keywords
        kws = []
        ikws = [tup[i] for tup in kwargs if i < len(tup)]
        for ikw in ikws:
            kw = {}
            for key in sorted(ikw, key=sorter):
                if key in keys_method:
                    continue
                value = ikw[key]
                if key == 'name' and skip_name:
                    continue
                if key == 'area' and value == 'avg' and skip_area:
                    continue
                label = _infer_string(dataset, key, value, mode=mode)
                kw[key] = label
            kws.append(kw)
        # Combine labels across keyword list
        kw = {}
        keys = sorted((key for kw in kws for key in kw), key=sorter)
        for key in keys:
            labs = tuple(kw.get(key, '') for kw in kws)  # label per key
            for modifier in modifiers:
                if not identical and all(modifier in lab for lab in labs):
                    labs = [replace(lab, modifier) for lab in labs]
            # if len(labs) > 1  # TODO: restrict to non-scalar?
            if len(labs) == 1 or identical == all(lab == labs[0] for lab in labs):
                kw[key] = labs[0] if identical else labs
        if identical:
            kws = [kw]
        else:
            kws = [{key: kw[key][i] for key in kw} for i in range(len(kws))]
        if kws:  # WARNING: critical or else zip below creates empty list
            labels.append(kws)
    # Combine pairs of labels
    keys = sorted((key for kw in kws for key in kw), key=sorter)
    pairs = list(zip(*labels))
    labels = []
    for i, pair in enumerate(pairs):
        for key in keys:
            if len(pair) != 2:
                pass
            elif any(kw.get(key) != pair[0].get(key, object()) for kw in pair):
                pass
            elif not keeppairs:  # e.g. internal vs. zelinka feedback
                del pair[0][key], pair[1][key]
            elif key == 'name':
                del pair[0][key]  # e.g. abrupt vs. picontrol feedback
            else:
                del pair[1][key]  # e.g. cmip5 abrupt vs. picontrol
        label = ' vs. '.join(
            ' '.join(lab for lab in kw.values() if lab) for kw in pair if kw
        )
        label = _infer_newlines(label, refwidth=refwidth)
        if title_case is True or title_case == 'first' and i == 0:
            if label[:1].islower():
                label = label[:1].upper() + label[1:]
        labels.append(label or '')
    return labels


def _parse_project(data, project):
    """
    Return plot labels and facet filter for the project indicator.

    Parameters
    ----------
    data : xarray.Dataset or xarray.DataArray
        The data. Must contain a ``'facets'`` coordinate.
    project : str
        The selection. Values should start with ``'cmip'``. No integer ending indicates
        all cmip5 and cmip6 models, ``5`` (``6``) indicates just cmip5 (cmip6) models,
        ``56`` (``65``) indicates cmip5 (cmip6) models filtered to those from the same
        institutes as cmip6 (cmip5), and ``55`` (``66``) indicates institutes found
        only in cmip5 (cmip6). Note two-digit integers can be combined, e.g. ``5665``.

    Returns
    -------
    callable
        A `facets` filter function.
    """
    # WARNING: Critical to assign name to filter so that _parse_specs can detect
    # differences between row and column specs at given subplot entry.
    project = project.lower()
    name_to_inst = MODELS_INSTITUTES.copy()
    name_to_inst.update(  # support facets with institutes names instead of models
        {
            (proj, abbrv): inst
            for inst, abbrv in INSTITUTES_ABBREVS.items()
            for proj in ('CMIP5', 'CMIP6')
        }
    )
    if not project.startswith('cmip'):
        raise ValueError(f'Invalid project indicator {project}. Must contain cmip.')
    _, num = project.split('cmip')
    imax = max(1, len(num))
    if imax not in (1, 2, 4):
        raise ValueError(f'Invalid project indicator {project}. 1/2/4 numbers allowed.')
    funcs = []  # permit e.g. cmip6556 or inst6556
    for i in range(0, imax, 2):
        n = num[i:i + 2]
        if not n:
            func = lambda key: True  # noqa: U100
        elif n in ('5', '6'):
            func = lambda key: key[0][-1] == n
        elif n in ('65', '66', '56', '55'):
            b = True if len(set(n)) == 2 else False
            o = '6' if n[0] == '5' else '5'
            func = lambda key, boo=b, num=n, opp=o: (
                num[0] == key[0][-1]
                and boo == any(
                    name_to_inst.get((key[0], key[1]), object())
                    == name_to_inst.get((other[0], other[1]), object())
                    for other in data.facets.values if opp == other[0][-1]
                )
            )
        else:
            raise ValueError(f'Invalid project number {n!r}.')
        funcs.append(func)
    func = lambda key: any(func(key) for func in funcs)
    func.name = project  # WARNING: critical for get_data() detection of 'all_projs'
    return func


def _parse_institute(data, institute):
    """
    Return plot labels and facet filter for the institute indicator.

    Parameters
    ----------
    data : xarray.Dataset or xarray.DataArray
        The data. Must contain a ``'facets'`` coordinate.
    institute : str
        The selection. Can be ``'avg'`` to perform an institute-wise `groupby` average
        and replace the model ids in the multi-index with curated institute ids, the
        name of an institute to select only its associated models, or the special
        key ``'flagship'`` to select "flagship" models from unique institutes (i.e.
        the final models in the `cmip_data.internals` dictionary).

    Returns
    -------
    callable or xarray.DataArray
        A `facets` filter function or `groupby` array.
    """
    # NOTE: Averages across a given institution can be accomplished using the default
    # method='avg' along with e.g. institute='GFDL'. The special institute='avg' is
    # supported for special weighted facet-averages and bar or scatter plots.
    inst_to_abbrv = INSTITUTES_ABBREVS.copy()  # see also _parse_constraints
    model_to_inst = MODELS_INSTITUTES.copy()
    if institute == 'avg':
        insts = [
            model_to_inst.get((key[0], key[1]), 'U')
            for key in data.facets.values
        ]
        facets = [
            (key[0], inst_to_abbrv.get(inst, inst), key[2], key[3])
            for inst, key in zip(insts, data.facets.values)
        ]
        filt = xr.DataArray(  # WARNING: critical fro groupby() to name this 'facets'
            pd.MultiIndex.from_tuples(facets, names=data.indexes['facets'].names),
            attrs=data.facets.attrs,
            dims='facets',
            name='facets',
        )
    elif institute == 'flagship':  # flagship from either cmip5 or cmip6 series
        inst_to_model = {
            (proj, inst): model for (proj, model), inst in model_to_inst.items()
        }
        filt = lambda key: (
            key[1]
            == inst_to_model.get((key[0], model_to_inst.get((key[0], key[1]))))
        )
        filt.name = institute  # unnecessary but why not
    else:
        abbrv_to_inst = {
            abbrv: inst for inst, abbrv in inst_to_abbrv.items()
        }
        filt = lambda key: (
            abbrv_to_inst.get(institute, institute)
            == model_to_inst.get((key[0], key[1]))
        )
        filt.name = institute  # unnecessary but why not
    return filt


def _parts_composite(data0, data1, pctile=None):  # noqa: E301
    """
    Return low and high composite components of `data1` based on `data0`.

    Parameters
    ----------
    data0 : xarray.DataArray
        The data used to build the composite.
    data1 : xarray.DataArray
        The data being composited.
    pctile : float, optional
        The percentile threshold.

    Returns
    -------
    data_lo, data_hi : xarray.DataArray
        The composite components.
    """
    thresh = 33 if pctile is None else pctile
    data0, data1 = xr.broadcast(data0, data1)
    comp_lo = np.nanpercentile(data0, thresh)
    comp_hi = np.nanpercentile(data0, 100 - thresh)
    mask_lo, = np.where(data0 <= comp_lo)
    mask_hi, = np.where(data0 >= comp_hi)
    data_hi = data1.isel(facets=mask_hi)
    data_lo = data1.isel(facets=mask_lo)
    with np.errstate(all='ignore'):
        data_hi = data_hi.mean('facets', keep_attrs=True)
        data_lo = data_lo.mean('facets', keep_attrs=True)
    data_hi = data_hi.climo.quantify()
    data_lo = data_lo.climo.quantify()
    return data_lo, data_hi


def _parts_covariance(data0, data1, both=True):
    """
    Return covariance and standard deviations of `data0` and optionally `data1`.

    Parameters
    ----------
    data0 : xarray.DataArray
        The first data. Standard deviation is always returned.
    data1 : xarray.DataArray
        The second data. Standard deviation is optionally returned.
    both : bool, optional
        Whether to also return standard deviation of `data0`.

    Returns
    -------
    covar, std0, std1 : xarray.DataArray
        The covariance and standard deviation components.
    """
    # NOTE: Currently masked arrays are used in climopy 'covar' and might also have
    # overhead from metadata stripping stuff and permuting. So go manual here.
    data0, data1 = xr.broadcast(data0, data1)
    data0 = data0.climo.quantify()
    data1 = data1.climo.quantify()
    skipna = True
    with np.errstate(all='ignore'):
        mean0 = data0.mean(dim='facets', skipna=skipna)
        mean1 = data1.mean(dim='facets', skipna=skipna)
        anom0 = data0 - mean0
        anom1 = data1 - mean1
        covar = (anom0 * anom1).sum(dim='facets', skipna=skipna)
        std0 = (anom0 ** 2).sum(dim='facets', skipna=skipna)
        std0 = np.sqrt(std0)
        if both:
            std1 = (anom1 ** 2).sum(dim='facets', skipna=skipna)
            std1 = np.sqrt(std1)
    return (covar, std0, std1) if both else (covar, std0)


def _apply_reduce(data, attrs=None, **kwargs):
    """
    Carry out arbitrary reduction of the given dataset variables.

    Parameters
    ----------
    data : xarray.DataArray
        The dataset.
    attrs : dict, optional
        The optional attribute overrides.
    **kwargs
        The reduction selections. Requires a `name`.

    Returns
    -------
    data : xarray.DataArray
        The data array.
    """
    # Apply special reductions
    # TODO: Might consider adding 'institute' multi-index so e.g. grouby('institute')
    # is possible... then again would not reduce complexity because would now need to
    # load files in special non-alphabetical order to enable selecting 'flaghip' models
    # with e.g. something like .sel(institute=-1)... so don't bother for now.
    project = kwargs.pop('project', None)
    if project is not None:  # see _parse_project
        if callable(project):
            facets = list(filter(project, data.facets.values))
            data = data.sel(facets=facets)
        else:
            raise RuntimeError(f'Unsupported project {project!r}.')
    institute = kwargs.pop('institute', None)
    if institute is not None:  # ignore dummy None
        if callable(institute):  # facets filter function
            facets = list(filter(institute, data.facets.values))
            data = data.sel(facets=facets)
        elif isinstance(institute, xr.DataArray):  # groupby multi-index data array
            if project:  # filter works with both models and institutes
                bools = list(map(project, institute.facets.values))
                institute = institute[{'facets': bools}]
            data = data.groupby(institute).mean(skipna=False, keep_attrs=True)
            facets = data.indexes['facets']  # xarray bug causes dropped level names
            facets.names = institute.indexes['facets'].names
            facets = xr.DataArray(
                facets,
                name='facets',
                dims='facets',
                attrs=institute.facets.attrs  # WARNING: avoid overwriting input kwarg
            )
            data = data.assign_coords(facets=facets)
        else:
            raise RuntimeError(f'Unsupported institute {institute!r}.')

    # Apply defaults and iterate over options
    # NOTE: This silently skips dummy selections (e.g. area=None) that may be required
    # to prevent _parse_specs from merging e.g. average and non-average selections.
    # WARNING: Sometimes multi-index reductions can eliminate previously valid coords,
    # so critical to iterate one-by-one and validate selections each time.
    name = data.name
    attrs = attrs or {}
    attrs = attrs.copy()  # WARNING: critical
    defaults = {'period': 'ann', 'ensemble': 'flagship', 'experiment': 'picontrol'}
    versions = {'source': 'eraint', 'statistic': 'slope', 'region': 'globe'}
    if 'version' in data.coords:
        defaults.update({'experiment': 'abrupt4xco2', **versions})
    for key, value in defaults.items():
        kwargs.setdefault(key, value)
    for key, value in data.attrs.items():
        attrs.setdefault(key, value)
    order = list(ORDER_LOGICAL)
    sorter = lambda key: order.index(key) if key in order else len(order)
    for key in sorted(kwargs, key=sorter):
        value = kwargs[key]
        options = list(data.sizes)
        options.extend(name for idx in data.indexes.values() for name in idx.names)
        options.extend(('area', 'volume'))
        if value is None:
            continue
        if key not in options:
            continue
        if key == 'volume':  # auto-skip if coordinates not available
            if not data.sizes.keys() & {'lon', 'lat', 'plev'}:
                continue
        if key == 'area':  # auto-skip if coordinates not available
            region = AREA_REGIONS.get(value, None)
            if not data.sizes.keys() & {'lon', 'lat'}:
                continue
            elif region is not None:
                data, value = data.climo.truncate(region), 'avg'
            elif value != 'avg':
                raise ValueError(f'Unknown averaging region {value!r}.')
        data = data.climo.reduce(**{key: value})
        data = data.squeeze()
    data.name = name
    data.attrs.update(attrs)
    return data


def _apply_method(
    *datas, method=None, std=None, pctile=None, invert=False, verbose=False
):
    """
    Reduce along the facets coordinate using an arbitrary method.

    Parameters
    ----------
    *datas : xarray.DataArray
        The data array(s).
    method : str, optional
        The method. ``dist``, ``dstd``, and ``dpctile`` retain the facets dimension
        before plotting. ``avg``, ``std``, and ``pctile`` reduce the facets dimension
        for a single input argument. ``corr``, ``diff``, ``proj``, and ``slope``
        reduce the facets dimension for two input arguments. See below for dtails.
    pctile : float or sequence, optional
        The percentile thresholds for related methods. The default is ``33``
        for `diff`, ``25`` for `pctile`, and `90` for `dpctile`.
    std : float or sequence, optional
        The standard deviation multiple for related methods. The default is
        ``1`` for `std` and ``3`` for `dstd`.
    invert : bool, optional
        Whether to invert the direction of composites, projections, and regressions so
        that the first variable is the predictor instead of the second variable.

    Returns
    -------
    args : tuple
        The output plotting arrays.
    method : str
        The resulting method used.
    kwargs : dict
        The plotting and `_infer_command` keyword arguments.
    """
    # Combine one array along facets dimension
    # NOTE: Here `pctile` is shared between inter-model percentile differences and
    # composites of a second array based on values in the first array.
    # NOTE: Currently proplot will automatically apply xarray tuple multi-index
    # coordinates to bar plot then error out so apply numpy array coords for now.
    defaults = {}
    datas = tuple(data.copy() for data in datas)
    ndims = tuple(data.ndim for data in datas)
    args = name = long = short = None
    if invert:
        datas = datas[::-1]
    if max(ndims) == 1:
        method = method or 'dist'  # only possibility
    elif len(datas) == 1:
        method = method or 'avg'
    else:
        method = method or 'rsq'
    if len(datas) == 1:
        data, = datas
        name = data.name
        if method == 'avg':
            data = data.mean('facets', skipna=True, keep_attrs=True)
        elif method == 'med':
            data = data.median('facets', skipna=True, keep_attrs=True)
        elif method == 'std':
            std = 1.0 if std is None else std
            with xr.set_options(keep_attrs=True):  # note name is already kept
                data = std * data.std('facets', skipna=True)
            short = f'{data.short_name} standard deviation'
            long = f'{data.long_name} standard deviation'
        elif method == 'pctile':
            pctile = 25.0 if pctile is None else pctile
            with xr.set_options(keep_attrs=True):  # note name is already kept
                hi = data.quantile(1 - 0.01 * pctile, dim='facets', skipna=True)
                lo = data.quantile(0.01 * pctile, dim='facets', skipna=True)
                data = hi - lo
            short = f'{data.short_name} percentile range'
            long = f'{data.long_name} percentile range'
        elif method == 'dstd':
            assert max(ndims) == 2
            shade = 3.0 if std is None else 3.0
            defaults.update({'means': True, 'shadestds': shade})
        elif method == 'dpctile':
            assert max(ndims) == 2
            shade = 90.0 if pctile is None else pctile
            defaults.update({'medians': True, 'shadepctiles': shade})
        elif method == 'dist':  # horizontal lines
            assert max(ndims) == 1
            data = data[~data.isnull()]
            args = (data,)
        else:
            raise ValueError(f'Invalid single-variable method {method}.')

    # Combine two arrays along facets dimension
    # NOTE: The idea for 'diff' reductions is to build the feedback-based composite
    # difference defined ``data[feedback > 100 - pctile] - data[feedback < pctile]``.
    elif len(datas) == 2:
        data0, data1 = datas
        name = f'{data0.name}-{data1.name}'
        if method == 'rsq':  # correlation coefficient
            cov, std0, std1 = _parts_covariance(*datas, both=True)
            data = (cov / (std0 * std1)) ** 2
            data = data.climo.to_units('percent').climo.dequantify()
            short = f'{data1.short_name} variance explained'
            long = f'{data0.long_name}-{data1.long_name} variance explained'
        elif method == 'corr':  # correlation coefficient
            cov, std0, std1 = _parts_covariance(*datas, both=True)
            data = cov / (std0 * std1)
            data = data.climo.to_units('dimensionless').climo.dequantify()
            short = f'{data1.short_name} correlation'
            long = f'{data0.long_name}-{data1.long_name} correlation coefficient'  # noqa: E501
        elif method == 'proj':  # projection onto x
            cov, std = _parts_covariance(*datas, both=False)
            data = cov / std
            data = data.climo.dequantify()
            short = f'{data1.short_name} projection'
            long = f'{data1.long_name} vs. {data0.long_name}'
        elif method == 'slope':  # regression coefficient
            cov, std = _parts_covariance(*datas, both=False)
            data = cov / std ** 2
            data = data.climo.dequantify()
            short = f'{data1.short_name} regression coefficient'
            long = f'{data1.long_name} vs. {data0.long_name} regression coefficient'
        elif method == 'diff':  # composite difference along first arrays
            data_lo, data_hi = _parts_composite(*datas)
            data = data_hi - data_lo
            data = data.climo.dequantify()
            short = f'{data1.short_name} composite difference'
            long = f'{data0.long_name}-composite {data1.long_name} difference'  # noqa: E501
        elif method == 'dist':  # scatter points
            assert max(ndims) == 1
            data0, data1 = data0[~data0.isnull()], data1[~data1.isnull()]
            data0, data1 = xr.align(data0, data1)  # intersection-style broadcast
            args = (data0, data1)
            name = None
        else:
            raise ValueError(f'Invalid double-variable method {method}')
    else:
        raise ValueError(f'Unexpected argument count {len(datas)}.')

    # Standardize the result and print information
    # NOTE: This modifies
    args = args or (data,)
    args = list(args)
    if name:
        args[-1].name = name
    if long:
        args[-1].attrs['long_name'] = long
    if short:
        args[-1].attrs['short_name'] = short
    keys = ('facets', 'time', 'plev', 'lat', 'lon')  # sorting order
    args = [arg.transpose(..., *(key for key in keys if key in arg.sizes)) for arg in args]  # noqa: E501
    if verbose:
        masks = [(~arg.isnull()).any(arg.sizes.keys() - {'facets'}) for arg in args]
        valid = invalid = ''
        if len(masks) == 2:  # show individual and their intersection
            mask = masks[0] & masks[1]
            valid, invalid = f' ({np.sum(mask).item()})', f' ({np.sum(~mask).item()})'
        for mask, data in zip(masks, args[len(args) - len(masks):]):
            min_, max_, mean = data.min().item(), data.mean().item(), data.max().item()
            print(format(f'{data.name} {method}:', ' <20s'), end=' ')
            print(f'min {min_: <+7.2f} max {max_: <+7.2f} mean {mean: <+7.2f}', end=' ')
            print(f'valid {np.sum(mask).item()}{valid}', end=' ')
            print(f'invalid {np.sum(~mask).item()}{invalid}', end='\n')
    return args, method, defaults


def get_data(dataset, *kws_dat, attrs=None):
    """
    Combine the data based on input reduce dictionaries.

    Parameters
    ----------
    dataset : xarray.Dataset
        The source dataset.
    *kws_dat : dict
        The reduction keyword dictionaries.
    attrs : dict, optional
        The attribute dictionaries.

    Returns
    -------
    args : tuple
        The output plotting arrays.
    method : str
        The resulting method used.
    kwargs : dict
        The plotting and `_infer_command` keyword arguments.
    """
    # Group added/subtracted reduce instructions into separate dictionaries
    # NOTE: Initial kw_red values are formatted as (('[+-]', value), ...) to
    # permit arbitrary combinations of names and indexers (see _parse_specs).
    alias_to_name = {
        alias: name for alias, (name, _) in FEEDBACK_TRANSLATIONS.items()
    }
    if 'control' in dataset.experiment:
        alias_to_name.update({'picontrol': 'control', 'abrupt4xco2': 'response'})
    else:
        alias_to_name.update({'control': 'picontrol', 'response': 'abrupt4xco2'})
    kws_method = []  # each item represents a method argument
    keys_method = ('method', 'std', 'pctile', 'invert')  # special method keywords
    for kw_dat in kws_dat:
        scale = 1
        kw_reduce = {}
        kw_method = {key: kw_dat.pop(key) for key in keys_method if key in kw_dat}
        for key, value in kw_dat.items():
            sels = ['+']
            parts = REGEX_SPLIT.split(value) if isinstance(value, str) else (value,)
            for i, part in enumerate(parts):
                if isinstance(part, str) and REGEX_FLOAT.match(part):  # e.g. 850-250hPa
                    part = float(part)
                if part in (None, '+', '-', '*', '/'):  # dummy coordinate or operator
                    sel = part
                elif key == 'project':
                    sel = _parse_project(dataset, part)
                elif key == 'institute':
                    sel = _parse_institute(dataset, part)
                elif isinstance(part, str):
                    sel = alias_to_name.get(part, part)
                else:
                    unit = dataset[key].climo.units
                    if not isinstance(part, ureg.Quantity):
                        part = ureg.Quantity(part, unit)
                    sel = part.to(unit)
                sels.append(sel)
            signs, values = sels[0::2], sels[1::2]
            scale *= sum(sign == '+' for sign in signs)
            kw_reduce[key] = tuple(zip(signs, values))
        kws_reduce = []
        for values in itertools.product(*kw_reduce.values()):
            signs, values = zip(*values)
            sign = -1 if signs.count('-') % 2 else +1
            kw = dict(zip(kw_reduce, values))
            kw.update(kw_method)
            kws_reduce.append((sign, kw))
        kws_method.append((scale, kws_reduce))

    # Reduce along facets dimension and carry out operation
    # TODO: Add other possible reduction methods, e.g. covariance
    # or regressions instead of normalized correlation.
    # NOTE: Here 'skip' prevents subtracting or averaging identical selections
    # from identical selections (i.e. all-zero array generation).
    scales, kws_method = zip(*kws_method)
    if len(set(scales)) > 1:
        raise RuntimeError(f'Mixed reduction scalings {scales}.')
    all_membs = all_projs = True
    kws_method = list(kws_method)
    skip = None
    if len(kws_method) == 2 and len(kws_method[0]) == 1 and len(kws_method[1]) != 1:
        skip = 0
        kws_method[0] = kws_method[0] * len(kws_method[1])
    if len(kws_method) == 2 and len(kws_method[1]) == 1 and len(kws_method[0]) != 1:
        skip = 1
        kws_method[1] = kws_method[1] * len(kws_method[0])
    kws_persum = zip(*kws_method)
    kwargs = {}
    datas_persum = []  # each item part of a summation
    methods_persum = set()
    for kws_reduce in kws_persum:
        kw_method = {}
        keys = ('std', 'pctile', 'invert', 'method')
        datas = []
        signs, kws_reduce = zip(*kws_reduce)
        if skip is not None:  # ignore scalar indices
            signs = (signs[1 - skip],)
        if len(set(signs)) > 1:
            raise RuntimeError(f'Mixed reduction signs {signs}.')
        for kw in kws_reduce:  # two for e.g. 'corr', one for e.g. 'avg'
            kw_reduce = kw.copy()
            for key in tuple(kw_reduce):
                if key in keys:
                    kw_method.setdefault(key, kw_reduce.pop(key))
            institute = kw_reduce.get('institute', None)
            project = getattr(kw_reduce.get('project'), 'name', '')
            all_membs = all_membs and institute is None
            all_projs = all_projs and len(project) not in (5, 6)
            data = dataset[kw_reduce.pop('name')]
            data = _apply_reduce(data, attrs=attrs, **kw_reduce)
            datas.append(data)
        datas, method, default = _apply_method(*datas, **kw_method)
        for key, value in default.items():
            kwargs.setdefault(key, value)
        datas_persum.append((signs[0], datas))  # plotting command arguments
        methods_persum.add(method)
        if len(methods_persum) > 1:
            raise RuntimeError(f'Mixed reduction methods {methods_persum}.')

    # Combine arrays specified with reduction '+' and '-' keywords
    # NOTE: The default inferred prefixes associated with reductions are passed
    # here with 'attrs' so only need to possibly modify the suffix.
    # NOTE: The additions below are scaled as *averages* so e.g. project='cmip5+cmip6'
    # gives the average across cmip5 and cmip6 inter-model averages.
    # NOTE: The operations below use a dummy 'cmip' project before acting to support
    # specific case of e.g. project='cmip6-cmip5' with institute='avg'.
    def _replace_project(data, project=None):  # temporarily replace project
        project = project or 'CMIP'
        projects = data.project.values if hasattr(data, 'project') else np.array([])
        original = projects[0] if len(projects) > 0 else None
        if len(projects) > 1 and all(proj == original for proj in projects):
            facets = [(project, *key[1:]) for key in data.facets.values]
            facets = xr.DataArray(
                pd.MultiIndex.from_tuples(facets, names=data.indexes['facets'].names),
                attrs=data.facets.attrs,
                dims='facets',
                name='facets',
            )
            data = data.assign_coords(facets=facets)
        return data, original
    args = []
    signs, datas_persum = zip(*datas_persum)
    kwargs.update({'projects': all_projs, 'members': all_membs})
    for s, datas in enumerate(zip(*datas_persum)):
        if s == skip and method == 'dist':
            data = datas[0]
        else:
            datas, projs = zip(*map(_replace_project, datas))
            datas = xr.align(*datas)
            if 'facets' in datas[0].coords and not datas[0].facets.size:
                raise RuntimeError(
                    'Empty facets dimension. This is most likely due to an '
                    'operation across projects without an institute average.'
                )
            with xr.set_options(keep_attrs=True):
                data = sum(sign * sdata for sign, sdata in zip(signs, datas))
                data = data / scales[0]
            proj = 'CMIP6' if len(set(projs)) > 1 else projs[0]
            data, _ = _replace_project(data, proj)
            if any(sign == -1 for sign in signs):
                data.attrs['long_suffix'] = 'anomaly'
                data.attrs['short_suffix'] = 'anomaly'
        args.append(data)
    args = xr.align(*args)  # re-align after summation
    return args, method, kwargs


def get_breakdown(
    component=None,
    breakdown=None,
    feedbacks=True,
    adjusts=False,
    forcing=False,
    sensitivity=False,
    maxcols=None,
):
    """
    Return the feedback, forcing, and sensitivity parameter specs sensibly
    organized depending on the number of columns in the plot.

    Parameters
    ----------
    component : str, optional
        The individual component to use.
    breakdown : str, default: 'all'
        The breakdown preset to use.
    feedbacks, adjusts, forcing, sensitivity : bool, optional
        Whether to include various components.
    maxcols : int, default: 4
        The maximum number of columns (influences order of the specs).

    Returns
    -------
    specs : list of str
        The variables.
    maxcols : int
        The possibly adjusted number of columns.
    gridskip : list of int
        The gridspec entries to skip.
    """
    # Interpret keyword arguments from suffixes
    # NOTE: Idea is to automatically remove feedbacks, filter out all-none
    # rows and columns at the end, and then infer the 'gridskip' from them.
    original = maxcols = maxcols or 4
    if breakdown is None:
        component = component or 'net'
    if component is None:
        breakdown = breakdown or 'all'
    if breakdown and '_lam' in breakdown:
        breakdown, *_ = breakdown.split('_lam')
        feedbacks = True
    if breakdown and '_adj' in breakdown:
        breakdown, *_ = breakdown.split('_erf')
        adjusts, forcing = True, False  # effective forcing *and* rapid adjustments
    if breakdown and '_erf' in breakdown:
        breakdown, *_ = breakdown.split('_erf')
        forcing, adjusts = True, False  # effective forcing *without* rapid adjustments
    if breakdown and '_ecs' in breakdown:
        breakdown, *_ = breakdown.split('_ecs')
        sensitivity = True
    if not component and not feedbacks and not forcing and not sensitivity:
        raise RuntimeError

    # Three variable breakdowns
    # NOTE: Options include 'wav', 'atm', 'alb', 'res', 'all', 'atm_wav', 'alb_wav'
    # with the 'wav' suffixes including longwave and shortwave cloud components
    # instead of a total cloud feedback. Strings denote successively adding atmospheric,
    # albedo, residual, and remaining temperature/humidity feedbacks with options.
    def _get_array(cols):
        names = np.array([[None] * cols] * 25)
        iflat = names.flat
        return names, iflat
    if component is not None:
        names, iflat = _get_array(maxcols)
        iflat[0] = component
        gridskip = None
    elif breakdown in ('net', 'wav', 'atm'):  # shortwave longwave
        if breakdown == 'net':  # net lw/sw
            lams = ['net', 'sw', 'lw']
            erfs = ['erf', 'rsnt_erf', 'rlnt_erf']
        elif breakdown == 'wav':  # cloud lw/sw
            lams = ['net', 'swcld', 'lwcld']
            erfs = ['erf', 'cl_rsnt_erf', 'cl_rlnt_erf']
        elif breakdown == 'atm':  # net cloud, atmosphere
            lams = ['net', 'cld', 'atm']
            erfs = ['erf', 'cl_rfnt_erf', 'atm_rfnt_erf']
        else:
            raise RuntimeError
        if maxcols == 2:
            names, iflat = _get_array(maxcols)
            if sensitivity:
                iflat[0] = 'ecs'
            if feedbacks and adjusts:
                names[1:4, 0] = lams
                names[1:4, 1] = erfs
            elif feedbacks and forcing:
                names[1, :] = lams[:1] + erfs[:1]
                names[2, :] = lams[1:]
            elif feedbacks:
                iflat[1] = lams[0]
                names[1, :] = lams[1:]
            elif adjusts:
                iflat[1] = erfs[0]
                names[1, :] = erfs[1:]
        else:
            offset = 0
            maxcols = 1 if maxcols == 1 else 3
            names, iflat = _get_array(maxcols)
            if sensitivity:
                iflat[offset] = 'ecs'
            if forcing:
                iflat[offset + 1] = 'erf'
            if feedbacks:
                idx = 2 * maxcols
                iflat[idx:idx + 3] = lams
            if adjusts:
                idx = 3 * maxcols
                iflat[idx:idx + 3] = erfs

    # Four variable breakdowns
    elif breakdown in ('cld_wav', 'atm_wav', 'atm_res', 'alb'):
        if 'cld' in breakdown:  # net cloud, cloud lw/sw
            lams = ['net', 'cld', 'swcld', 'lwcld']
            erfs = ['erf', 'cl_rfnt_erf', 'cl_rsnt_erf', 'cl_rlnt_erf']
        elif 'res' in breakdown:  # net cloud, residual, atmosphere
            lams = ['net', 'cld', 'resid', 'atm']
            erfs = ['erf', 'cl_rfnt_erf', 'resid_rfnt_erf', 'atm_rfnt_erf']  # noqa: E501
        elif 'atm' in breakdown:  # cloud lw/sw, atmosphere
            lams = ['net', 'swcld', 'lwcld', 'atm']
            erfs = ['erf', 'cl_rsnt_erf', 'cl_rlnt_erf', 'atm_rfnt_erf']
        elif breakdown == 'alb':  # net cloud, atmosphere, albedo
            lams = ['net', 'cld', 'alb', 'atm']
            erfs = ['erf', 'cl_rfnt_erf', 'alb_rfnt_erf', 'atm_rfnt_erf']
        else:
            raise RuntimeError
        if maxcols == 2:
            names, iflat = _get_array(maxcols)
            if sensitivity:
                iflat[0] = 'ecs'
            if feedbacks and adjusts:
                names[1:5, 0] = lams
                names[1:5, 1] = erfs
            elif feedbacks:
                iflat[1] = 'erf' if forcing else None
                names[1, :] = lams[::3]
                names[2, :] = lams[1:3]
            elif adjusts:
                names[1, :] = erfs[::3]
                names[2, :] = erfs[1:3]
        else:
            offset = 0 if maxcols == 1 else 1
            maxcols = 1 if maxcols == 1 else 3
            names, iflat = _get_array(maxcols)
            if sensitivity:
                iflat[-offset % 3] = 'ecs'
            if forcing or adjusts:
                iflat[2 - offset] = 'erf'
            if feedbacks:
                idx = 3
                iflat[1 - offset] = lams[0]
                iflat[idx:idx + 3] = lams[1:]
            if adjusts:
                idx = 3 + 3
                iflat[idx:idx + 3] = erfs[1:]

    # Five variable breakdowns
    # NOTE: Currently this is the only breakdown with both clouds
    elif breakdown in ('atm_cld', 'alb_wav', 'res', 'res_wav'):
        if 'atm' in breakdown:  # net cloud, cloud lw/sw, atmosphere
            lams = ['net', 'cld', 'swcld', 'lwcld', 'atm']
            erfs = ['erf', 'cl_rfnt_erf', 'cl_rsnt_erf', 'cl_rlnt_erf', 'atm_rfnt_erf']
        elif 'alb' in breakdown:  # cloud lw/sw, atmosphere, albedo
            lams = ['net', 'swcld', 'lwcld', 'alb', 'atm']
            erfs = ['erf', 'cl_rsnt_erf', 'cl_rlnt_erf', 'alb_rfnt_erf', 'atm_rfnt_erf']
        elif breakdown == 'res':  # net cloud, atmosphere, albedo, residual
            lams = ['net', 'cld', 'alb', 'resid', 'atm']
            erfs = ['erf', 'cl_rfnt_erf', 'alb_rfnt_erf', 'resid_rfnt_erf', 'atm_rfnt_erf']  # noqa: E501
        elif 'res' in breakdown:  # cloud lw/sw, atmosphere, residual
            lams = ['net', 'swcld', 'lwcld', 'resid', 'atm']
            erfs = ['erf', 'cs_rlnt_erf', 'cl_rlnt_erf', 'resid_rfnt_erf', 'atm_rfnt_erf']  # noqa: E501
        else:
            raise RuntimeError
        if maxcols == 2:
            names, iflat = _get_array(maxcols)
            if sensitivity:
                iflat[0] = 'ecs'
            if feedbacks and adjusts:
                names[1:5, 0] = lams
                names[1:5, 1] = erfs  # noqa: E501
            elif feedbacks and forcing:
                names[1, :] = lams[:1] + erfs[:1]
                names[2, :] = lams[1:3]
                names[3, :] = lams[3:5]
            elif feedbacks:
                names[0, 1] = lams[0]
                names[1, :] = lams[1:3]
                names[2, :] = lams[3:5]
            elif adjusts:
                names[0, 1] = erfs[0]
                names[1, :] = erfs[1:3]
                names[2, :] = erfs[3:5]
        else:
            offset = 0 if maxcols == 1 else 1
            maxcols = 1 if maxcols == 1 else 4  # disallow 3 columns
            names, iflat = _get_array(maxcols)
            if sensitivity:
                iflat[-offset % 3] = 'ecs'  # either before or after net and erf
            if forcing or adjusts:
                iflat[2 - offset] = erfs[0]
            if feedbacks:
                idx = 4  # could leave empty single-column row
                iflat[1 - offset] = lams[0]
                iflat[idx:idx + 4] = lams[1:]
            if adjusts:
                idx = 4 + 4
                iflat[idx:idx + 4] = erfs[1:]

    # Full breakdown
    elif breakdown == 'all':
        lams = ['net', 'cld', 'swcld', 'lwcld', 'atm', 'alb', 'resid']
        erfs = ['erf', 'cl_rfnt_erf', 'cl_rsnt_erf', 'cl_rlnt_erf', 'atm_rfnt_erf', 'alb_rfnt_erf', 'resid_rfnt_erf']  # noqa: E501
        hums = ['wv', 'rh', 'lr', 'lr*', 'pl', 'pl*']
        if maxcols == 2:
            names, iflat = _get_array(maxcols)
            if sensitivity:
                iflat[0] = 'ecs'
            if feedbacks and adjusts:
                names[1:8, 0] = lams
                names[1:8, 1] = erfs
                iflat[16:22] = hums
            elif feedbacks:
                iflat[0] = lams[0]
                iflat[1] = 'erf' if forcing else None
                names[2, :] = lams[1:3]
                names[3, :] = lams[3:5]
            elif adjusts:
                iflat[0] = erfs[0]
                names[1, :] = erfs[1:3]
                names[2, :] = erfs[3:5]
        else:
            offset = 0 if maxcols == 1 else 1
            maxcols = 1 if maxcols == 1 else 4  # disallow 3 columns
            names, iflat = _get_array(maxcols)
            if sensitivity:
                iflat[-offset % 3] = 'ecs'  # either before or after net and erf
            if forcing or adjusts:
                iflat[2 - offset] = erfs[0]
            if feedbacks:
                idx = 4  # could leave empty single-column row
                iflat[1 - offset] = lams[0]
                iflat[idx:idx + 4] = lams[1:5]
                if maxcols == 1:
                    idx = 4 + 4
                    iflat[idx:idx + 8] = lams[5:7] + hums[:]
                else:
                    idx = 4 + 4
                    iflat[idx:idx + 4] = lams[5:6] + hums[0::2]
                    idx = 4 + 2 * 4
                    iflat[idx:idx + 4] = lams[6:7] + hums[1::2]
            if adjusts:
                idx = 4 + 3 * 4
                iflat[idx:idx + 6] = erfs[1:7]
    else:
        raise RuntimeError(f'Invalid breakdown key {breakdown!r}.')

    # Remove all-none segments and determine gridskip
    # NOTE: This also automatically flattens the arangement if there are
    # fewer than names than the originally requested maximum column count.
    idx, = np.where(np.any(names != None, axis=0))  # noqa: E711
    names = np.take(names, idx, axis=1)
    idx, = np.where(np.any(names != None, axis=1))  # noqa: E711
    names = np.take(names, idx, axis=0)
    idxs = np.where(names == None)  # noqa: E711
    gridskip = np.ravel_multi_index(idxs, names.shape)
    names = names.ravel().tolist()
    names = [spec for spec in names if spec is not None]
    if len(names) <= original and maxcols != 1:  # then keep it simple!
        maxcols = len(names)
        gridskip = np.array([])
    return names, maxcols, gridskip
