#!/usr/bin/env python3
"""
Internal helper functions for figure templates.
"""
import functools
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
    'apply_composite',
    'apply_variance',
    'apply_reduce',
    'apply_method',
    'names_breakdown',
]

# Feedback variable names and abbreviations
_seen = set()
FEEDBACKS_NAMES = {
    key: name
    for key, (name, _) in FEEDBACK_TRANSLATIONS.items()
}
FEEDBACKS_ABBREVS = {
    name: key for key, name in FEEDBACKS_NAMES.items()
    if name not in _seen and not _seen.add(name)  # translate using first entry
}
del _seen

# Short names associated with standard units
# NOTE: This is partly based on CLIMATE_UNITS from output.py. Remove
# this and use registered variable short names instead.
VARIABLES_SHORTS = {
    'K': 'temperature',
    'hPa': 'pressure',
    'dam': 'surface height',
    'mm': 'liquid depth',
    'mm day^-1': 'accumulation',
    'm s^-1': 'wind speed',
    'Pa': 'wind stress',  # default tau units
    'g kg^-1': 'concentration',
    'W m^-2': 'flux',
    'PW': 'transport'
}

# Reduce defaults, abbreviations, and labels
REDUCE_DEFAULTS = {
    'period': 'ann',
    'experiment': 'picontrol',
    'ensemble': 'flagship',
    'source': 'eraint',
    'statistic': 'slope',
    'region': 'globe',
}
REDUCE_ABBREVS = {
    ('area', 'avg'): 'avg',
    ('lon', 'avg'): None,
    ('lon', 'int'): None,
    ('lat', 'absmin'): 'min',
    ('lat', 'absmax'): 'max',
    ('region', 'point'): 'point',
    ('region', 'globe'): 'globe',
    ('region', 'latitude'): 'zonal',
    ('region', 'hemisphere'): 'hemi',
    ('ensemble', 'flagship'): 'flag',
    ('experiment', 'control'): 'pictl',
    ('experiment', 'response'): '4xco2',
    ('experiment', 'picontrol'): 'pictl',
    ('experiment', 'abrupt4xco2'): '4xco2',
}
REDUCE_LABELS = {  # default is title-case of input
    ('area', 'avg'): 'average',
    ('lon', 'avg'): None,
    ('lon', 'int'): None,
    ('period', 'ann'): 'annual',
    ('period', 'djf'): 'DJF',
    ('period', 'mam'): 'MAM',
    ('period', 'jja'): 'JJA',
    ('period', 'son'): 'SON',
    ('lat', 'absmin'): 'minimum',
    ('lat', 'absmax'): 'maximum',
    ('region', 'point'): 'local',
    ('region', 'globe'): 'global',
    ('region', 'latitude'): 'zonal',
    ('region', 'hemisphere'): 'hemisphere',
    ('experiment', 'control'): 'pre-industrial',
    ('experiment', 'response'): r'abrupt 4$\times$CO$_2$',
    ('experiment', 'picontrol'): 'pre-industrial',
    ('experiment', 'abrupt4xco2'): r'abrupt 4$\times$CO$_2$',
}

# Argument sorting constants
ORDER_ABBREVS = (
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
    'plev',  # space and time
    'area',
    'lat',
    'lon',
)
ORDER_LABELS = (
    'lon',  # space and time
    'lat',
    'area',
    'plev',
    'period',
    'version',  # feedback version index
    'source',
    'statistic',
    'region',
    'facets',  # source facets index
    'institute',
    'project',
    'model',
    'ensemble',
    'experiment',
)


def _parse_institute(dataset, institute):
    """
    Return plot labels and facet filter for the institute indicator.

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset. Must contain a ``'facets'`` coordinate.
    institute : str
        The selection. Can be ``'avg'`` to perform an institute-wise `groupby` average
        and replace the model ids in the multi-index with curated institute ids, the
        name of an institute to select only its associated models, or the special
        key ``'flagship'`` to select "flagship" models from unique institutes (i.e.
        the final models in the `cmip_data.internals` dictionary). Note `_parse_project`
        also contains institute-parsing logic but use separate functions for clarity.
    """
    # NOTE: Averages across a given institution can be accomplished using the default
    # method='avg' along with e.g. institute='GFDL'. The special institute='avg' is
    # supported for special weighted facet-averages and bar or scatter plots.
    inst_to_abbrv = INSTITUTES_ABBREVS.copy()
    abbrv_to_inst = {
        abbrv: inst for inst, abbrv in inst_to_abbrv.items()
    }
    model_to_inst = MODELS_INSTITUTES.copy()  # see also _parse_constraints
    inst_to_model = {  # overwrite to final 'flagship' models in dictionary
        (proj, inst): model for (proj, model), inst in model_to_inst.items()
    }
    if institute == 'avg':
        insts = [
            model_to_inst.get((key[0], key[1]), 'U')
            for key in dataset.facets.values
        ]
        facets = [
            (key[0], inst_to_abbrv.get(inst, inst), key[2], key[3])
            for inst, key in zip(insts, dataset.facets.values)
        ]
        filter = pd.MultiIndex.from_tuples(facets, names=dataset.indexes['facets'].names)  # noqa: E501
        filter = xr.DataArray(filter, dims='facets', attrs=dataset.facets.attrs)
        label = 'institute-average'
        abbrv = filter.name = 'inst'
    elif institute == 'flagship':  # flagship from either cmip5 or cmip6 series
        filter = lambda key: (
            key[1] == inst_to_model.get((key[0], model_to_inst.get((key[0], key[1]))))
        )
        label = 'institute-flagship'
        abbrv = filter.__name__ = 'flag'
    else:
        filter = lambda key: (
            abbrv_to_inst.get(institute, institute) == model_to_inst.get((key[0], key[1]))  # noqa: E501
        )
        label = inst_to_abbrv.get(institute, institute)
        abbrv = filter.__name__ = inst_to_abbrv.get(institute, institute)
    if callable(filter) and not any(filter(facet) for facet in dataset.facets.values):
        raise ValueError(f'Invalid institute identifier {institute!r}. No matches.')
    return abbrv, label, filter


def _parse_project(dataset, project):
    """
    Return plot labels and facet filter for the project indicator.

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset. Must contain a ``'facets'`` coordinate.
    project : str
        The selection. Values should start with ``'cmip'``. No integer ending indicates
        all cmip5 and cmip6 models, ``5`` (``6``) indicates just cmip5 (cmip6) models,
        ``56`` (``65``) indicates cmip5 (cmip6) models filtered to those from the same
        institutes as cmip6 (cmip5), and ``55`` (``66``) indicates institutes found
        only in cmip5 (cmip6). Note two-digit integers can be combined, e.g. ``5665``.

    Returns
    -------
    abbrv : str
        The file name abbreviation.
    label : str
        The column or row label string.
    filter : callable
        Function for filtering ``facets`` coordinates.
    """
    # WARNING: Critical to assign name to filter so that _parse_specs can detect
    # differences between row and column specs at given subplot entry.
    project = project.lower()
    name_to_inst = {  # support facets with institutes names instead of models
        (proj, abbrv): inst
        for inst, abbrv in INSTITUTES_ABBREVS.items()
        for proj in ('CMIP5', 'CMIP6')
    }
    name_to_inst.update(MODELS_INSTITUTES)
    if not project.startswith('cmip'):
        raise ValueError(f'Invalid project indicator {project}. Must contain cmip.')
    _, num = project.split('cmip')
    imax = max(1, len(num))
    labs, nums, filts = [], [], []  # permit e.g. cmip6556 or inst6556
    for i in range(0, imax, 2):
        n = num[i:i + 2]
        if not n:
            lab = ''
            filt = lambda key: True  # noqa: U100
        elif n in ('5', '6'):
            lab = ''
            filt = lambda key: key[0][-1] == n
        elif n in ('65', '66', '56', '55'):
            b = True if len(set(n)) == 2 else False
            o = '6' if n[0] == '5' else '5'
            lab = '' if b and n[0] == '5' else ('non-matching ', 'matching ')[b]
            filt = lambda key, boo=b, num=n, opp=o: (
                num[0] == key[0][-1] and boo == any(
                    name_to_inst.get((key[0], key[1]), object())
                    == name_to_inst.get((other[0], other[1]), object())
                    for other in dataset.facets.values if opp == other[0][-1]
                )
            )
        else:
            raise ValueError(f'Invalid project number {n!r}.')
        nums.append('' if n[:1] in nums else n[:1])
        labs.append('' if lab in labs else lab)
        filts.append(filt)
    num = '' if set(nums) == {'5', '6'} else ''.join(nums)
    label = ''.join(labs) + f'CMIP{num}'
    filter = lambda key: any(filt(key) for filt in filts)
    abbrv = filter.__name__ = project  # required in _parse_specs
    return abbrv, label, filter


def _parse_reduce(dataset, **kwargs):
    """
    Standardize the indexers and translate into labels suitable for figures.

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset.
    **kwargs
        The coordinate indexers or `name` specification. Numeric indexers are assumed
        to have the units of the corresponding coordinate, and string indexers can
        have arithmetic operators ``+-`` to perform operations between multiple
        selections or variables (e.g. seasonal differences relative to annual
        average or one project relative to another). Note the special key `project`
        is handled by `_parse_project` to permit institution matching.

    Returns
    -------
    abbrvs : dict
        Abbreviations for the coordinate reductions. This is destined
        for use in file names.
    labels : dict
        Labels for the coordinate reductions. This is destined for
        use in row and column figure labels.
    reduce : dict
        The reduction selections. Contains either scalar sanitized selections
        or two-tuples of selections for operations (plus an `operator` key).
    """
    # NOTE: Here a dummy coordinate None might be used to differentiate a variable
    # spec with reduce dimensions versus a variable spec with unreduced dimensions
    # before merging in _parse_specs (e.g. global average vs. local feedbacks).
    abbrvs, labels, kw_red = {}, {}, {}
    keys = ('method', 'std', 'pctile', 'invert')
    kw_red.update({key: kwargs.pop(key) for key in keys if key in kwargs})
    for key, value in kwargs.items():
        abvs, labs, sels = [], [], []
        parts = re.split('([+-])', value) if isinstance(value, str) else (value,)
        for i, part in enumerate(parts):
            if isinstance(part, str):  # string coordinate
                if part in '+-':
                    sel = part
                    abbrv = part
                    label = 'plus' if part == '+' else 'minus'
                elif key == 'name':
                    sel = FEEDBACKS_NAMES.get(part, part)
                    abbrv = FEEDBACKS_ABBREVS.get(sel, sel)
                    label = dataset[sel].long_name
                elif key in ('project', 'institute'):
                    if key == 'project':
                        abbrv, label, sel = _parse_project(dataset, part)
                    else:
                        abbrv, label, sel = _parse_institute(dataset, part)
                else:
                    if 'control' in dataset.experiment:
                        opts = {'picontrol': 'control', 'abrupt4xco2': 'respone'}
                    else:
                        opts = {'control': 'picontrol', 'response': 'abrupt4xco2'}
                    sel = opts.get(part, part)
                    abbrv = REDUCE_ABBREVS.get((key, sel), sel)
                    label = REDUCE_LABELS.get((key, sel), sel)
            else:  # numeric or dummy coordinate
                if part is None:
                    sel = part
                    abbrv = label = None
                else:
                    coords = dataset[key]
                    unit = coords.climo.units
                    if not isinstance(part, ureg.Quantity):
                        part = ureg.Quantity(part, unit)
                    sel = part.to(unit)
                    abbrv = f'{sel:~.0f}'.replace('/', 'p').replace(' ', '')
                    label = f'${sel:~L}$'
            sels.append(sel)
            if abbrv is not None:  # e.g. operator or 'avg'
                abvs.append(abbrv)
            if label is not None:  # e.g. operator or 'avg'
                labs.append(label)
        sels = ('+', *sels)  # note sels is never empty (see above notes)
        sels = tuple(zip(sels[0::2], sels[1::2]))  # tuple of (([+-], value), ...)
        kw_red[key] = sels
        if abvs:
            abbrvs[key] = ''.join(abvs)
        if labs:
            labels[key] = ' '.join(labs)
    return abbrvs, labels, kw_red


def _parse_labels(specs, refwidth=None):
    """
    Return suitable grid label and file name strings given the input specs.

    Parameters
    ----------
    specs : sequence
        Sequence of sequence of variable specs in length-1 or length-2 form
        respresenting either individual reductions or correlations.
    refwidth : float, optional
        The reference width used to scale the axes. This is used to auto-wrap
        lengthy row and column labels.

    Returns
    -------
    gridspecs : list of list of str
        Strings suitable for the figure labels. Returned as a length-2 list of
        descriptive row and column labels.
    filespecs : list of list of str
        Strings suitable for the default file name. Returned as a length-4 list of
        method indicators, shared indicators, row indicators, and column indicators.
    """
    # Reduce the label dicationaries to account for redundancies across the list,
    # either dropping them (useful for gridspec labels and legend labels) or
    # dropping everything but the redundancies (useful for figure titles).
    # TODO: Also drop scalar 'identical' specifications of e.g. 'annual', 'slope',
    # and 'eraint' default selections for feedback variants and climate averages?
    def _reduce_labels(kws, identical=False):
        seen = set()
        keys = [
            key for kw in kws for key in kw
            if key not in seen and not seen.add(key)
        ]
        labels = {}
        modifiers = (
            'feedback',
            'forcing',
            r'2$\times$CO$_2$',  # leads forcing strings
            r'4$\times$CO$_2$',
            'energy transport',
            'energy convergence',
        )
        for key in keys:
            labs = tuple(kw.get(key, None) for kw in kws)  # label per key
            for modifier in modifiers:
                if key == 'name' and all(modifier in lab for lab in labs):
                    if not identical:
                        labs = [modifier] * len(labs)
                    else:
                        labs = [
                            lab
                            .replace(f' {modifier}', '')
                            .replace(f'{modifier} ', '')
                            .replace('effective', 'net')  # similar to net feedback
                            for lab in labs
                        ]
            ident = all(lab == labs[0] for lab in labs)
            if ident == identical:  # preserve identical or unique labels
                labels[key] = labs[0] if identical else labs
        if identical:
            result = [labels]
        else:
            result = [{key: labels[key][i] for key in keys} for i in range(len(kws))]
        return keys, result

    # Convert list of strings associated with e.g. gridspec rows and columns or plotted
    # elements in a single subplot into succinct identical or non-identical labels.
    def _combine_labels(kws, identical=False):
        keys = []
        labels = []
        for i in range(2):  # indices in the list
            ikws = [pair[i] for pair in kws if i < len(pair)]
            ikeys, labs = _reduce_labels(ikws, identical=identical)
            if keys:
                keys.extend(key for key in ikeys if key not in keys)
            if labs:  # i.e. not an empty list due to all-identical labels
                labels.append(labs)
        pairs = list(zip(*labels))
        labels = []
        for pair in pairs:
            for key in keys:
                if len(pair) < 2:
                    continue
                if all(kw.get(key) == pair[0].get(key) for kw in pair):
                    del labels[1][key]  # e.g. cmip5 abrupt vs. picontrol
            seen = set()
            labs = [' '.join(filter(None, kw.values())) for kw in pair]
            labs = [lab for lab in labs if lab not in seen and not seen.add(lab)]
            labels.append(' vs. '.join(labs))  # possibly singleton
        return labels

    # # Consolidate figure grid and file name labels
    # # NOTE: Here correlation pairs are handled with 'vs.' indicators in the row
    # # or column labels (also components are kept only if distinct across slots).
    # seen = set()
    # filespecs = [  # avoid e.g. cmip5-cmip5 segments
    #     spec.replace('_', '') for spec in sorted(filter(None, abbrvs))
    #     if spec and spec not in seen and not seen.add(spec)
    # ]
    # seen = set()
    # gridspecs = [  # avoid e.g. 'cmip5 vs. cmip5' labels
    #     ' vs. '.join(filter(None,
    #         (spec for spec in specs if spec not in seen and not seen.add(spec))  # noqa: E128, E501
    #     )) for specs in zip(*gridspecs) if not seen.clear()  # refresh each time
    # ]
    # Get component pieces
    # TODO: Apply these labels to the legend handles and figure title automatically.
    # Can pass to the main function.
    # NOTE: This selects just the first variable specification in each subplot. Others
    # are assumed to be secondary and not worth encoding in labels or file name.
    abbrvs, labels, *_ = zip(
        *(
            zip(*ispecs[0])  # specs per correlation pair for first spec in the subplot
            for ispecs  # specs per subplot
            in specs  # specs per figure
        )
    )
    func = lambda key, order=None: order.index(key) if key in order else len(order)
    keys = [key for abvs in abbrvs for kw in abvs for key in kw]
    keys = sorted(keys, key=functools.partial(func, order=list(ORDER_ABBREVS)))
    abbrvs = tuple(
        kw[key] for abvs in abbrvs for kw in abvs for key in keys if key in kw
    )
    # npairs = max(len(kws) for kws in abbrvs)
    # filespecs = []
    # for i in range(nspecs):  # indices in the list
    #     fspecs = [kws[i] for kws in abbrvs if i < len(kws)]
    #     pass
    npairs = max(len(kws) for kws in labels)
    gridspecs = []
    for i in range(npairs):  # indices in the list
        gspecs = [kws[i] for kws in labels if i < len(kws)]
        if len(set(tuple(kw.items()) for kw in gspecs)) <= 1:
            continue
        seen = set()
        keys = [key for kw in gspecs for key in kw if key not in seen and not seen.add(key)]  # noqa: E501
        keys = sorted(keys, key=functools.partial(func, order=list(ORDER_LABELS)))
        gspecs = [tuple(kw.get(key, None) for kw in gspecs) for key in keys]
        gspecs = [tup for tup in gspecs if any(spec != tup[0] for spec in tup)]
        gspecs = [' '.join(filter(None, tup)) for tup in zip(*gspecs)]
        upper = lambda s: s and ((s[0].upper() if s[0].islower() else s[0]) + s[1:])
        parts = ('feedback', 'forcing', 'energy', 'transport', 'convergence')
        for part in parts:
            if all(f' {part}' in spec for spec in gspecs):
                gspecs = [spec.replace(f' {part}', '') for spec in gspecs]
        gridspecs.append(list(map(upper, gspecs)) or [''] * len(specs))

    # Consolidate figure grid and file name labels
    # NOTE: Here correlation pairs are handled with 'vs.' indicators in the row
    # or column labels (also components are kept only if distinct across slots).
    seen = set()
    filespecs = [  # avoid e.g. cmip5-cmip5 segments
        spec.replace('_', '') for spec in abbrvs  # already sorted
        if spec and spec not in seen and not seen.add(spec)
    ]
    seen = set()
    gridspecs = [  # avoid e.g. 'cmip5 vs. cmip5' labels
        ' vs. '.join(filter(None,
            (spec for spec in specs if spec not in seen and not seen.add(spec))  # noqa: E128, E501
        )) for specs in zip(*gridspecs) if not seen.clear()  # refresh each time
    ]

    # Automatically insert spaces in row and column labels
    # NOTE: This also ignores math text delimited by dollar signs
    outspecs = []
    for gridspec in gridspecs:
        idxs = np.array([i for i, c in enumerate(gridspec) if c == ' '])
        adjs = idxs.copy()
        for m in re.finditer(r'\$[^$]+\$', gridspec):  # ignore non-math texts
            i, j = m.span()
            adjs[(i <= idxs) & (idxs <= j)] = 0  # i.e. ignore since lower than thresh
            adjs[idxs > j] -= j - i
        refwidth = refwidth or pplt.rc['subplots.refwidth']
        threshs = 1.2 * pplt.units(refwidth, 'in', 'em') * np.arange(1, 10)
        chars = list(gridspec)  # string to list
        for thresh in threshs:
            mask = adjs > thresh
            if not mask.any():
                continue
            chars[np.min(idxs[mask])] = '\n'
        outspecs.append(''.join(chars))
    return filespecs, outspecs


def apply_composite(data0, data1, pctile=None):  # noqa: E301
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


def apply_variance(data0, data1, both=True, skipna=True):
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
    skipna : bool, optional
        Whether to skip nan-values in the summation.
    """
    # NOTE: Currently masked arrays are used in climopy 'covar' and might also have
    # overhead from metadata stripping stuff and permuting. So go manual here.
    data0, data1 = xr.broadcast(data0, data1)
    data0 = data0.climo.quantify()
    data1 = data1.climo.quantify()
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


def apply_reduce(dataset, attrs=None, **kwargs):
    """
    Carry out arbitrary reduction of the given dataset variables.

    Parameters
    ----------
    dataset : xarray.Dataset
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
    # Get data array and apply special reductions
    # TODO: Might consider adding 'institute' multi-index so e.g. grouby('institute')
    # is possible... then again would not reduce complexity because would now need to
    # load files in special non-alphabetical order to enable selecting 'flaghip' models
    # with e.g. something like .sel(institute=-1)... so don't bother for now.
    name = kwargs.pop('name', None)
    if name is None:
        raise ValueError('Variable name missing from reduction specification.')
    data = dataset[name]
    project = kwargs.pop('project', None)
    institute = kwargs.pop('institute', None)
    if project is not None:  # see _parse_projects
        if callable(project):
            facets = list(filter(project, data.facets.values))
            data = data.sel(facets=facets)
        else:
            raise RuntimeError(f'Unsupported project {project!r}.')
    if institute is not None:  # ignore dummy None
        if callable(institute):  # facets filter function
            facets = list(filter(institute, data.facets.values))
            data = data.sel(facets=facets)
        elif isinstance(institute, xr.DataArray):  # groupby multi-index data array
            institute.name = 'facets'
            if project:  # filter works with both models and institutes
                bools = list(map(project, institute.facets.values))
                institute = institute[{'facets': bools}]
            names = institute.indexes['facets'].names
            attrs = institute.facets.attrs.copy()
            data = data.groupby(institute)  # should result in multi-index coordinates
            data = data.mean(skipna=False, keep_attrs=True)
            index = data.indexes['facets']  # xarray bug causes dropped level names
            index.names = names
            index = xr.DataArray(index, name='facets', dims='facets', attrs=attrs)
            data = data.assign_coords(facets=index)
        else:
            raise RuntimeError(f'Unsupported institute {institute!r}.')

    # Apply defaults and iterate over options
    # NOTE: Sometimes multi-index reductions can eliminate previously valid
    # coordinates, so iterate one-by-one and validate selections each time.
    # NOTE: This silently skips dummy selections (e.g. area=None) that may be
    # required to prevent _parse_bulk from merging variable specs that differ
    # only in that one contains a selection and the other doesn't (generally
    # when constraining local feedbacks vs. global feedbacks).
    for key, value in REDUCE_DEFAULTS.items():
        kwargs.setdefault(key, value)
    attrs = attrs or {}
    for key, value in data.attrs.items():
        attrs.setdefault(key, value)
    for key, value in kwargs.items():
        if value is None:
            continue
        data = data.squeeze()
        options = [*data.sizes, 'area', 'volume', 'institute']
        options.extend(name for idx in data.indexes.values() for name in idx.names)
        if key in options:
            data = data.climo.reduce(**{key: value})
    data = data.squeeze()
    data.name = name
    data.attrs.update(attrs)
    return data


def apply_method(
    *datas, method=None, std=None, pctile=None, invert=False, verbose=False
):
    """
    Reduce along the facets coordinate using an arbitrary method.

    Parameters
    ----------
    *datas : xarray.DataArray
        The data array(s).
    method : str, optional
        The user-declared reduction method. The methods ``dist``, ``dstd``, and
        ``dpctile`` retain the facets dimension before plotting. The methods
        ``avg``, ``std``, and ``pctile`` reduce the facets dimension for a single
        input argument. The methods ``corr``, ``diff``, ``proj``, and ``slope``.
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
    datas : tuple
        The data array(s). Two are returned for `hist2d` plots.
    """
    # Infer short names from the unit strings (standardized by open_climate) for use
    # with future colorbar and legend labels (long names are used for grid labels).
    # TODO: Update climopy and implement functionality with new cf variables.
    datas = tuple(data.copy() for data in datas)
    ndims = tuple(data.ndim for data in datas)
    for data in datas:
        for array in (data, *data.coords.values()):
            if 'units' not in array.attrs or 'long_name' not in array.attrs:
                continue
            if 'short_name' in array.attrs:
                continue
            long = array.long_name
            units = array.units
            if 'feedback' in long:
                short = 'feedback'
            elif 'forcing' in long:
                short = 'forcing'
            else:
                short = VARIABLES_SHORTS.get(units, long)
            array.attrs['short_name'] = short

    # Combine one array along facets dimension
    # NOTE: Here `pctile` is shared between inter-model percentile differences and
    # composites of a second array based on values in the first array.
    # NOTE: Currently proplot will automatically apply xarray tuple multi-index
    # coordinates to bar plot then error out so apply numpy array coords for now.
    if invert:
        datas = datas[::-1]
    if max(ndims) == 1:
        method = 'dist'  # only possibility
    elif len(datas) == 1:
        method = method or 'avg'
    else:
        method = method or 'rsq'
    if len(datas) == 1:
        if method == 'dist':  # horizontal lines
            data, = datas
            data = data[~data.isnull()]
            data = (np.arange(data.size), data.sortby(data, ascending=False))
            assert max(ndims) == 1
        elif method == 'dstd':
            data, = datas
            kw = {'means': True, 'shadestds': 3.0 if std is None else 3.0}
            data = (data, kw)
            assert max(ndims) == 2
        elif method == 'dpctile':
            data, = datas
            kw = {'medians': True, 'shadepctiles': 90.0 if pctile is None else pctile}
            data = (data, kw)
            assert max(ndims) == 2
        elif method == 'avg':
            data = datas[0].mean('facets', skipna=True, keep_attrs=True)
            data.name = datas[0].name
            data.attrs['units'] = datas[0].units
        elif method == 'std':
            std = 1.0 if std is None else 1.0
            data = std * datas[0].std('facets', skipna=True)
            data.name = datas[0].name
            data.attrs['short_name'] = f'{datas[0].short_name} standard deviation'
            data.attrs['long_name'] = f'{datas[0].long_name} standard deviation'
            data.attrs['units'] = datas[0].units
        elif method == 'pctile':
            pctile = 25.0 if pctile is None else pctile
            lo_vals = np.nanpercentile(datas[0], pctile)
            hi_vals = np.nanpercentile(datas[0], 100 - pctile)
            lo_vals = hi_vals.mean('facets') - lo_vals.mean('facets')
            data = hi_vals - lo_vals
            data.attrs['short_name'] = f'{datas[0].short_name} percentile range'
            data.attrs['long_name'] = f'{datas[0].long_name} percentile range'
            data.attrs['units'] = datas[0].units
        else:
            raise ValueError(f'Invalid single-variable method {method}.')

    # Combine two arrays along facets dimension
    # NOTE: The idea for 'diff' reductions is to build the feedback-based composite
    # difference defined ``data[feedback > 100 - pctile] - data[feedback < pctile]``.
    elif len(datas) == 2:
        if method == 'dist':  # scatter points
            datas = xr.broadcast(*datas)
            mask = ~datas[0].isnull() & ~datas[1].isnull()
            data = (datas[0][mask], datas[1][mask])
            assert max(ndims) == 1
        elif method == 'rsq':  # correlation coefficient
            cov, std0, std1 = apply_variance(*datas, both=True)
            data = (cov / (std0 * std1)) ** 2
            data = data.climo.to_units('percent').climo.dequantify()
            short_name = 'variance explained'
            long_name = f'{datas[0].long_name}-{datas[1].long_name} variance explained'
            data.name = f'{datas[0].name}-{datas[1].name}'
            data.attrs['short_name'] = short_name
            data.attrs['long_name'] = long_name
        elif method == 'corr':  # correlation coefficient
            cov, std0, std1 = apply_variance(*datas, both=True)
            data = cov / (std0 * std1)
            data = data.climo.to_units('dimensionless').climo.dequantify()
            short_name = 'correlation coefficient'
            long_name = f'{datas[0].long_name}-{datas[1].long_name} correlation coefficient'  # noqa: E501
            data.name = f'{datas[0].name}-{datas[1].name}'
            data.attrs['short_name'] = short_name
            data.attrs['long_name'] = long_name
        elif method == 'proj':  # projection onto x
            cov, std = apply_variance(*datas, both=False)
            data = cov / std
            data = data.climo.dequantify()
            short_name = f'{datas[1].short_name} projection'
            long_name = f'{datas[1].long_name} vs. {datas[0].long_name}'
            data.name = f'{datas[0].name}-{datas[1].name}'
            data.attrs['short_name'] = short_name
            data.attrs['long_name'] = long_name
        elif method == 'slope':  # regression coefficient
            cov, std = apply_variance(*datas, both=False)
            data = cov / std ** 2
            data = data.climo.dequantify()
            short_name = f'{datas[1].short_name} regression'
            long_name = f'{datas[1].long_name} vs. {datas[0].long_name}'
            data.name = f'{datas[0].name}-{datas[1].name}'
            data.attrs['short_name'] = short_name
            data.attrs['long_name'] = long_name
        elif method == 'diff':  # composite difference along first arrays
            data_lo, data_hi = apply_composite(*datas)
            data = data_hi - data_lo
            data = data.climo.dequantify()
            short_name = f'{datas[1].short_name} difference'
            long_name = f'{datas[0].long_name}-composite {datas[1].long_name} difference'  # noqa: E501
            data.name = f'{datas[0].name}-{datas[1].name}'
            data.attrs['short_name'] = short_name
            data.attrs['long_name'] = long_name
        else:
            raise ValueError(f'Invalid double-variable method {method}')
    else:
        raise ValueError(f'Unexpected argument count {len(datas)}.')

    # Print information and standardize result. Shows both the
    # available models and the intersection once they are combined.
    # print('input!', data, 'result!', *datas, sep='\n')
    order = ('facets', 'time', 'plev', 'lat', 'lon')
    result = tuple(
        part.transpose(..., *(key for key in order if key in part.sizes))
        if isinstance(part, xr.DataArray) else part
        for part in (data if isinstance(data, tuple) else (data,))
    )
    if verbose:
        masks = [
            (~data.isnull()).any(data.sizes.keys() - {'facets'})
            for data in datas if isinstance(data, xr.DataArray)
        ]
        valid = invalid = ''
        if len(masks) == 2:
            mask = masks[0] & masks[1]
            valid, invalid = f' ({np.sum(mask).item()})', f' ({np.sum(~mask).item()})'
        for mask, data in zip(masks, datas[len(datas) - len(masks):]):
            min_, max_, mean = data.min().item(), data.mean().item(), data.max().item()
            print(format(f'{data.name} {method}:', ' <20s'), end=' ')
            print(f'min {min_: <+7.2f} max {max_: <+7.2f} mean {mean: <+7.2f}', end=' ')
            print(f'valid {np.sum(mask).item()}{valid}', end=' ')
            print(f'invalid {np.sum(~mask).item()}{invalid}', end='\n')
    return result, method


def names_breakdown(
    breakdown='net',
    feedbacks=True,
    adjust=False,
    forcing=False,
    sensitivity=False,
    maxcols=4,
):
    """
    Return the feedback, forcing, and sensitivity parameter specs sensibly
    organized depending on the number of columns in the plot.

    Parameters
    ----------
    breakdown : str
        The breakdown format.
    feedbacks, adjust, forcing, sensitivity : bool, optional
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
    if '_lam' in breakdown:
        breakdown, *_ = breakdown.split('_lam')
        feedbacks = True
    if '_adj' in breakdown:
        breakdown, *_ = breakdown.split('_erf')
        adjust, forcing = True, False  # effective forcing *and* rapid adjustments
    if '_erf' in breakdown:
        breakdown, *_ = breakdown.split('_erf')
        forcing, adjust = True, False  # effective forcing *without* rapid adjustments
    if '_ecs' in breakdown:
        breakdown, *_ = breakdown.split('_ecs')
        sensitivity = True
    if not feedbacks and not forcing and not sensitivity:
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
    if breakdown in ('atm', 'wav'):  # shortwave longwave
        if breakdown == 'atm':
            lams = ['net', 'cld', 'atm']
            erfs = ['erf', 'cl_rfnt_erf', 'atm_rfnt_erf']
        else:
            lams = ['net', 'sw', 'lw']
            erfs = ['erf', 'rsnt_erf', 'rlnt_erf']
        if maxcols == 2:
            names, iflat = _get_array(maxcols)
            if sensitivity:
                iflat[0] = 'ecs'
            if feedbacks and adjust:
                names[1:4, 0] = lams
                names[1:4, 1] = erfs
            elif feedbacks and forcing:
                names[1, :] = lams[:1] + erfs[:1]
                names[2, :] = lams[1:]
            elif feedbacks:
                iflat[1] = lams[0]
                names[1, :] = lams[1:]
            elif adjust:
                iflat[1] = erfs[0]
                names[1, :] = erfs[1:]
        else:
            offset = 0
            maxcols = 1 if maxcols == 1 else 3
            names, iflat = _get_array(maxcols)
            if sensitivity:
                iflat[offset] = 'ecs'
            if forcing:  # adjust 'erf' handled below
                iflat[offset + 1] = 'erf'
            if feedbacks:
                idx = 2 * maxcols
                iflat[idx:idx + 3] = lams
            if adjust:
                idx = 3 * maxcols
                iflat[idx:idx + 3] = erfs

    # Four variable breakdowns
    elif breakdown in ('alb', 'atm_wav'):
        if breakdown == 'alb':
            lams = ['net', 'cld', 'alb', 'atm']
            erfs = ['erf', 'cl_rfnt_erf', 'alb_rfnt_erf', 'atm_rfnt_erf']
        else:
            lams = ['net', 'swcld', 'lwcld', 'atm']
            erfs = ['erf', 'cl_rsnt_erf', 'cl_rlnt_erf', 'atm_rfnt_erf']
        if maxcols == 2:
            names, iflat = _get_array(maxcols)
            if sensitivity:
                iflat[0] = 'ecs'
            if feedbacks and adjust:
                names[1:5, 0] = lams
                names[1:5, 1] = erfs
            elif feedbacks:
                iflat[1] = 'erf' if forcing else None
                names[1, :] = lams[::3]
                names[2, :] = lams[1:3]
            elif adjust:
                names[1, :] = erfs[::3]
                names[2, :] = erfs[1:3]
        else:
            offset = 0 if maxcols == 1 else 1
            maxcols = 1 if maxcols == 1 else 3
            names, iflat = _get_array(maxcols)
            if sensitivity:
                iflat[-offset % 3] = 'ecs'
            if forcing or adjust:
                iflat[2 - offset] = 'erf'
            if feedbacks:
                idx = 3
                iflat[1 - offset] = lams[0]
                iflat[idx:idx + 3] = lams[1:]
            if adjust:
                idx = 3 + 3
                iflat[idx:idx + 3] = erfs[1:]

    # Five variable breakdowns
    # NOTE: Currently this is the only breakdown with both clouds
    elif breakdown in ('cld', 'res', 'alb_wav'):
        if breakdown == 'cld':
            lams = ['net', 'cld', 'swcld', 'lwcld', 'atm']
            erfs = ['erf', 'cl_rfnt_erf', 'cl_rsnt_erf', 'cl_rlnt_erf', 'atm_rfnt_erf']
        elif breakdown == 'res':
            lams = ['net', 'cld', 'alb', 'resid', 'atm']
            erfs = ['erf', 'cl_rfnt_erf', 'alb_rfnt_erf', 'atm_rfnt_erf', 'resid_rfnt_erf']  # noqa: E501
        else:
            lams = ['net', 'swcld', 'lwcld', 'alb', 'atm']
            erfs = ['erf', 'cl_rsnt_erf', 'cl_rlnt_erf', 'alb_rfnt_erf', 'atm_rfnt_erf']
        if maxcols == 2:
            names, iflat = _get_array(maxcols)
            if sensitivity:
                iflat[0] = 'ecs'
            if feedbacks and adjust:
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
            elif adjust:
                names[0, 1] = erfs[0]
                names[1, :] = erfs[1:3]
                names[2, :] = erfs[3:5]
        else:
            offset = 0 if maxcols == 1 else 1
            maxcols = 1 if maxcols == 1 else 4  # disallow 3 columns
            names, iflat = _get_array(maxcols)
            if sensitivity:
                iflat[-offset % 3] = 'ecs'  # either before or after net and erf
            if forcing or adjust:
                iflat[2 - offset] = erfs[0]
            if feedbacks:
                idx = 4  # could leave empty single-column row
                iflat[1 - offset] = lams[0]
                iflat[idx:idx + 4] = lams[1:]
            if adjust:
                idx = 4 + 4
                iflat[idx:idx + 4] = erfs[1:]

    # Full breakdown
    elif breakdown == 'all':
        lams = ['net', 'atm', 'cld', 'swcld', 'lwcld', 'alb', 'resid']
        erfs = ['erf', 'cl_rfnt_erf', 'cl_rsnt_erf', 'cl_rlnt_erf', 'atm_rfnt_erf', 'alb_rfnt_erf', 'resid_rfnt_erf']  # noqa: E501
        hums = ['wv', 'rh', 'lr', 'lr*', 'pl', 'pl*']
        if maxcols == 2:
            names, iflat = _get_array(maxcols)
            if sensitivity:
                iflat[0] = 'ecs'
            if feedbacks and adjust:
                names[1:8, 0] = lams
                names[1:8, 1] = erfs  # noqa: E501
                iflat[16:22] = hums
            elif feedbacks:
                iflat[0] = lams[0]
                iflat[1] = 'erf' if forcing else None
                names[2, :] = lams[1:3]
                names[3, :] = lams[3:5]
            elif adjust:
                iflat[0] = erfs[0]
                names[1, :] = erfs[1:3]
                names[2, :] = erfs[3:5]
        else:
            offset = 0 if maxcols == 1 else 1
            maxcols = 1 if maxcols == 1 else 4  # disallow 3 columns
            names, iflat = _get_array(maxcols)
            if sensitivity:
                iflat[-offset % 3] = 'ecs'  # either before or after net and erf
            if forcing or adjust:
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
            if adjust:
                idx = 4 + 3 * 4
                iflat[idx:idx + 6] = erfs[1:7]
    else:
        iflat[0] = breakdown
        gridskip = None

    # Remove all-none segments and determine gridskip
    idx, = np.where(np.any(names != None, axis=0))  # noqa: E711
    names = np.take(names, idx, axis=1)
    idx, = np.where(np.any(names != None, axis=1))  # noqa: E711
    names = np.take(names, idx, axis=0)
    idxs = np.where(names == None)  # noqa: E711
    gridskip = np.ravel_multi_index(idxs, names.shape)
    names = names.ravel().tolist()
    names = [spec for spec in names if spec is not None]
    return names, maxcols, gridskip
