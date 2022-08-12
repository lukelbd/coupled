#!/usr/bin/env python3
"""
Utilities for plotting coupled model output.
"""
import collections
import itertools
import re
from pathlib import Path

import numpy as np
import xarray as xr
from icecream import ic  # noqa: F401

import climopy as climo
import proplot as pplt
from climopy import decode_units, format_units, ureg, vreg  # noqa: F401
from .output import FEEDBACK_DEFINITIONS, MODELS_INSTITUTIONS


# Variable constants
_seen = set()
NAMES_SHORT2DATA = {
    key: name
    for key, (name, _) in FEEDBACK_DEFINITIONS.items()
}
NAMES_DATA2SHORT = {
    name: key for key, name in NAMES_SHORT2DATA.items()
    if name not in _seen and not _seen.add(name)  # translate with first entry
}
del _seen

# Keyword argument constants
# NOTE: For sake of grammar we place 'ensemble' before 'experiment' here
KWARGS_ALL = collections.namedtuple(
    'kwargs',
    ('figure', 'gridspec', 'axes', 'colorbar', 'legend', 'plot', 'attrs')
)
KWARGS_ORDER = (
    'lon',  # space and time
    'lat',
    'area',
    'plev',
    'period',
    'version',  # feedback version index
    'source',
    'statistic',
    'region',
    'facets',  # cmip facets index
    'project',
    'model',
    'ensemble',
    'experiment',
)
KWARGS_DEFAULT = {
    'geo': {
        'coast': True,
        'lonlines': 30,
        'latlines': 30,
        'refwidth': 2.3,
    },
    'lat': {
        'xlabel': 'latitude',
        'xformatter': 'deg',
        'xlocator': 30,
        'xscale': 'sine',
        'xlim': (-89, 89),
        'refwidth': 1.5,
    },
    'plev': {
        'ylocator': 200,
        'yreverse': True,
        'ylabel': 'pressure (hPa)',
        'refwidth': 1.5,
    },
}

# Reduce constants
REDUCE_ABBRVS = {
    'point': 'local',
    'globe': 'globe',
    'latitude': 'zone',
    'hemisphere': 'hemi',
}
REDUCE_LABELS = {  # default is title-case of input
    'ann': 'annual',
    'djf': 'DJF',
    'mam': 'MAM',
    'jja': 'JJA',
    'son': 'SON',
    'avg': 'average',
    'int': 'integral',
    'absmin': 'minimum',
    'absmax': 'maximum',
    'point': 'local',
    'globe': 'global',
    'latitude': 'zonal',
    'hemisphere': 'hemisphere',
}

# Unit constants
# NOTE: This is partly based on CONVERSIONS_STANDARD from output.py. Need to
# eventually forget this and use registered variables instead.
UNITS_LABELS = {
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


def _apply_reduce(dataset, names, attrs=None, **kwargs):
    """
    Carry out arbitrary reduction of the given dataset variables.

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset.
    names : str or sequence of str
        The names being summed.
    attrs : dict, optional
        The optional attribute overrides.
    **kwargs
        The reduction selections.
    """
    # WARNING: Sometimes multi-index reductions can eliminate previously valid
    # coordinates, so iterate one-by-one and validate selections each time.
    # NOTE: This silently skips dummy selections (e.g. area=None) that may be
    # required to prevent _parse_bulk from merging variable specs that differ
    # only in that one contains a selection and the other doesn't (generally
    # when constraining local feedbacks vs. global feedbacks).
    defaults = {
        'period': 'ann',
        'experiment': 'control',
        'ensemble': 'flagship',
        'source': 'eraint',
        'statistic': 'regression',
        'region': 'globe',
    }
    names = (names,) if isinstance(names, str) else tuple(names)
    for key, value in defaults.items():
        kwargs.setdefault(key, value)
    with xr.set_options(keep_attrs=True):  # e.g. pl+lr+hus
        data = sum(dataset[name] for name in names)
    if facets := kwargs.pop('facets', None):  # see _parse_projects
        data = data.sel(facets=list(filter(facets, data.facets.values)))
    attrs = attrs or {}
    for key, value in data.attrs.items():
        attrs.setdefault(key, value)
    for key, value in kwargs.items():
        if value is None:
            continue
        data = data.squeeze()
        options = [*data.sizes, 'area', 'volume']
        options.extend(name for idx in data.indexes.values() for name in idx.names)
        if key in options:
            data = data.climo.reduce(**{key: value})
    data = data.squeeze()
    data.name = '+'.join(names)
    data.attrs.update(attrs)
    return data


def _apply_method(
    *datas,
    method=None,
    method1='avg',
    method2='corr',
    pctile=33.3,
    verbose=False,
    std=1.0
):
    """
    Reduce along the facets coordinate using an arbitrary method.

    Parameters
    ----------
    *datas : xarray.DataArray
        The data array(s).
    method : str, optional
        The user-declared reduction method.
    method1 : {'avg', 'std', 'pctile'}, optional
        The method for reducing the facets dimension on individual non-scalar
        variables. The default is an average. Can also pass `method` as a spec.
    method2 : {'corr', 'diff'}, optional
        The method for reducing the facets dimension on paired non-scalar
        variables. The default is an average. Can also pass `method` as a spec.
    std : float or sequence, optional
        The standard deviation multiple for `std` reductions. The default
        of ``1`` simply shows the standard deviation.
    pctile : float or sequence, optional
        The percentile thresholds for `pctile` and `diff` reductions. The lower
        and upper bounds are computed as ``(pctile, 100 - pctile)``.

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
                short = UNITS_LABELS.get(units, long)
            array.attrs['short_name'] = short

    # Combine 1 array along facets dimension
    # NOTE: Here `pctile` is shared between inter-model percentile differences and
    # composites of a second array based on values in the first array.
    if max(ndims) == 1:
        method = 'dist'
    elif len(datas) == 1:
        method = method or method1
    else:
        method = method or method2
    if len(datas) == 1:
        if method == 'dist':  # horizontal lines
            data, = datas
            data = data[~data.isnull()]
            data = (data.sortby(data, ascending=False),)
        elif method == 'avg':
            data = datas[0].mean('facets', skipna=True)
            data.name = datas[0].name
            data.attrs['short_name'] = f'average {datas[0].short_name}'
            data.attrs['long_name'] = f'average {datas[0].long_name}'
            data.attrs['units'] = datas[0].units
        elif method == 'std':
            data = std * datas[0].std('facets', skipna=True)
            data.name = datas[0].name
            data.attrs['short_name'] = f'{datas[0].short_name} standard deviation'
            data.attrs['long_name'] = f'{datas[0].long_name} standard deviation'
            data.attrs['units'] = datas[0].units
        elif method == 'pct':
            lo_vals = np.nanpercentile(datas[0], pctile)
            hi_vals = np.nanpercentile(datas[0], 100 - pctile)
            lo_vals = hi_vals.mean('facets') - lo_vals.mean('facets')
            data = hi_vals - lo_vals
            data.attrs['short_name'] = f'{datas[0].short_name} percentile range'
            data.attrs['long_name'] = f'{datas[0].long_name} percentile range'
            data.attrs['units'] = datas[0].units
        else:
            raise ValueError(f'Invalid single-variable method {method}.')

    # Combine 2 arrays along facets dimension
    # NOTE: The idea for 'diff' reductions is to build the feedback-based composite
    # difference defined ``data[feedback > 100 - pctile] - data[feedback < pctile]``.
    elif len(datas) == 2:
        if method == 'dist':  # scatter points
            datas = xr.broadcast(*datas)
            mask = ~datas[0].isnull() & ~datas[1].isnull()
            data = (datas[0][mask], datas[1][mask])
        elif method == 'corr':  # correlation coefficient
            datas = xr.broadcast(*datas)
            _, data = climo.corr(*datas, dim='facets')  # updates units
            data = data.isel(lag=0)
            short_name = 'correlation'
            long_name = f'{datas[0].long_name}-{datas[1].long_name} correlation'
            data.name = f'{datas[0].name}-{datas[1].name}'
            data.attrs['short_name'] = short_name
            data.attrs['long_name'] = long_name
            data.attrs['units'] = ''
        elif method == 'diff':  # composite difference along first arrays
            datas = xr.broadcast(*datas)
            lo_comp = np.nanpercentile(datas[0], pctile)
            hi_comp = np.nanpercentile(datas[0], 100 - pctile)
            lo_mask, = np.where(datas[0] <= lo_comp)
            hi_mask, = np.where(datas[0] >= hi_comp)
            hi_data = datas[1].isel(facets=hi_mask).mean('facets')
            lo_data = datas[1].isel(facets=lo_mask).mean('facets')
            data = hi_data - lo_data
            short_name = f'{datas[1].short_name} difference'
            long_name = f'{datas[0].long_name}-composite {datas[1].long_name} difference'  # noqa: E501
            data.name = f'{datas[0].name}-{datas[1].name}'
            data.attrs['short_name'] = short_name
            data.attrs['long_name'] = long_name
            data.attrs['units'] = datas[1].units
        else:
            raise ValueError(f'Invalid double-variable method {method}')
    else:
        raise ValueError(f'Unexpected argument count {len(datas)}.')

    # Print information and standardize result. Shows both the available models
    # and the intersection once theya re combined.
    # print('input!', data, 'result!', *datas, sep='\n')
    order = ('time', 'plev', 'lat', 'lon')
    result = tuple(
        part.transpose(..., *(key for key in order if key in part.sizes))
        for part in (data if isinstance(data, tuple) else (data,))
    )
    if verbose:
        masks = [(~data.isnull()).any(data.sizes.keys() - {'facets'}) for data in datas]
        valid = invalid = ''
        if len(datas) == 2:
            mask = masks[0] & masks[1]
            valid, invalid = f' ({np.sum(mask).item()})', f' ({np.sum(~mask).item()})'
        for mask, data in zip(masks, datas):
            min_, max_, mean = data.min().item(), data.mean().item(), data.max().item()
            print(format(f'{data.name} {method}:', ' <20s'), end=' ')
            print(f'min {min_: <+7.2f} max {max_: <+7.2f} mean {mean: <+7.2f}', end=' ')
            print(f'valid {np.sum(mask).item()}{valid}', end=' ')
            print(f'invalid {np.sum(~mask).item()}{invalid}', end='\n')
    return result, method


def _parse_descrips(specs):
    """
    Return suitable grid label and file name descriptions given the input specs.

    Parameters
    ----------
    specs : sequence
        Sequence of sequence of variable specs in length-1 or length-2 form
        respresenting either individual reductions or correlations.
    """
    # Get component pieces
    # NOTE: Unlike grid labels for this we simply merge disparate names
    # or reduce instructions across the correlation pairs if necessary.
    shorts, longs, abbrvs, labels, *_ = zip(
        *(zip(*ispecs[0]) for ispecs in specs)
    )
    shorts = tuple(key for short in shorts for key in short)
    abbrvs = tuple(abv for abvs in abbrvs for opts in abvs for abv in opts.values())
    gspecs = []
    nspecs = max(len(labs) for labs in labels)
    for i in range(nspecs):  # indices in the list
        labs = [labs[i] for labs in labels if i < len(labs)]
        lngs = [lngs[i] for lngs in longs if i < len(lngs)]
        gspec = []
        if len(set(tuple(kw.items()) for kw in labs)) > 1:
            seen = set()
            keys = [key for kw in labs for key in kw if key not in seen and not seen.add(key)]  # noqa: E501
            labs = [tuple(kw.get(key, None) for kw in labs) for key in keys]
            labs = [tup for tup in labs if any(lab != tup[0] for lab in tup)]
            labs = [' '.join(filter(None, tup)) for tup in zip(*labs)]
            gspec.append(labs)
        if len(set(lngs)) > 1:  # unique variables
            for replace in (' feedback', ' forcing'):
                if all(replace in long for long in lngs):
                    lngs = [long.replace(replace, '') for long in lngs]
            gspec.append(lngs)
        gspec = [
            spec.capitalize().replace('planck', 'Planck')
            .replace('Toa', 'TOA').replace('Toa', 'TOA')
            .replace('Cmip', 'CMIP').replace('Cmip', 'CMIP')
            for specs in zip(*gspec) if (spec := ' '.join(filter(None, specs)))
        ]
        gspecs.append(gspec or [''] * len(specs))

    # Consolidate figure grid and file name labels
    # NOTE: Here correlation pairs are handled with 'vs. ' indicators in the row
    # or column labels (also components are kept only if distinct across slots).
    seen = set()
    fspecs = [  # avoid e.g. cmip5-cmip5 segments
        spec.replace('_', '') for spec in (
            *sorted(filter(None, shorts)), *sorted(filter(None, abbrvs))
        ) if spec and spec not in seen and not seen.add(spec)
    ]
    seen = set()
    gspecs = [  # avoid e.g. 'cmip5 vs. cmip5' labels
        ' vs. '.join(filter(None,
            (spec for spec in specs if spec not in seen and not seen.add(spec))  # noqa: E128, E501
        )) for specs in zip(*gspecs) if not seen.clear()  # refresh each time
    ]
    return fspecs, gspecs


def _parse_project(dataset, project):
    """
    Return plot labels and facet tuples for the project indicator.

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset. Must contain a ``'facets'`` coordinates.
    project : str
        The selection. Values can optionally start with ``'cmip'`` and must end with
        integers indicating the facet values. No integer indicates all cmip5 and cmip6
        models, ``5`` (``6``) indicates just cmip5 (cmip6) models, ``56`` (``65``)
        indicates cmip5 (cmip6) models filtered to those from the same institutions
        as cmip6 (cmip5), and ``55`` (``66``) indicates models from institutions found
        only in cmip5 (cmip6). A common combination might be ``5``, ``65``, ``66``.

    Returns
    -------
    abbrv : str
        The file name abbreviation.
    label : str
        The column or row label string.
    filter : callable
        Function for filtering ``facets`` coordinates.
    """
    # WARNING: Critical that 'facets' selection is list because accessor reduce method
    # will pass tuples to '.get()' for interpolation onto variable-derived locations.
    # WARNING: Critical to assign name to filter so that _parse_bulk can detect
    # differences between row and column specs at given subplot entry.
    s1, s2 = object(), object()  # sentinels
    _, num = project.lower().split('cmip')
    if not num:
        filter = lambda key: True  # noqa: U100
        label = 'CMIP'
        abbrv = 'cmip'
    elif num in ('5', '6'):
        filter = lambda key: key[0][-1] == num
        label = f'CMIP{num}'
        abbrv = f'cmip{num}'
    elif num in ('65', '66', '56', '55'):
        idx = len(set(num)) - 1  # zero if only one unique integer
        opp = '5' if num[0] == '6' else '6'  # opposite project number
        filter = lambda key: (
            key[0][-1] == num[0] and idx == any(
                MODELS_INSTITUTIONS.get(key[1], s1)
                == MODELS_INSTITUTIONS.get(other[1], s2)
                for other in dataset.facets.values  # iterate over keys
                if other[0][-1] == opp
            )
        )
        label = ('other', 'matched')[idx] + f' CMIP{num[0]}'
        abbrv = filter.__name__ = f'cmip{num}'  # required in _parse_bulk
    else:
        raise ValueError(f'Invalid project {num!r}.')
    return abbrv, label, filter


def _parse_reduce(dataset, **indexers):
    """
    Standardize the indexers and translate into labels suitable for figures.

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset.
    **indexers
        The indexers. Numeric are assumed to be the units of the array and
        string indexers can have arithmetic operators ``+-`` to perform
        operations between multiple selections (e.g. seasonal differences
        relative to annual average or one project relative to another). Note
        the special key `project` is handled by `_parse_project`.

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
    # before merging in _parse_bulk (e.g. global average vs. local feedbacks).
    # NOTE: Here names are handled separately and only additions are supported; this
    # is so that e.g. specifications like `name=pl+lr+wv` with `project=cmip6-cmip5`
    # are possible (currently have no way to combine different sort of operators or
    # more than one operation along a given selection). Should revisit this.
    operators = set()
    names = indexers.pop('names', None)
    method = indexers.pop('method', None)
    abbrvs, labels, kw_red = {}, {}, {}
    for dim, coord in indexers.items():  # WARNING: critical to preserve order
        abvs, labs, sels = [], [], []
        parts = re.split('([+-])', coord) if isinstance(coord, str) else (coord,)
        if len(parts) not in (1, 3):  # maximum one operator
            raise ValueError(f'Unexpected coordinate format {coord!r}.')
        for i, part in enumerate(parts):
            if isinstance(part, str):  # string coordinate
                if part in '+-':
                    sel, abbrv, label = part, part, 'plus' if part == '+' else 'minus'
                    operators.add('add' if part == '+' else 'sub')
                    if len(operators) > 1:
                        raise ValueError(f'Conflicting selection {operators=}.')
                elif dim != 'project':
                    sel = part
                    abbrv = REDUCE_ABBRVS.get(part, part.lower())
                    label = REDUCE_LABELS.get(part, part)
                else:
                    abbrv, label, sel = _parse_project(dataset, part)
            else:  # numeric or dummy coordinate
                if part is None:
                    sel = part
                    abbrv = label = None
                else:
                    coords = dataset[dim]
                    unit = coords.climo.units
                    if not isinstance(part, ureg.Quantity):
                        part = ureg.Quantity(part, unit)
                    sel = part.to(unit)
                    abbrv = f'{sel:~.0f}'.replace('/', 'p').replace(' ', '')
                    label = f'${sel:~L}$'
            if abbrv:
                abvs.append(abbrv)
            if label:
                labs.append(label)
            if not isinstance(sel, str) or sel not in '+-':
                sels.append(sel)
        dim = 'facets' if dim == 'project' else dim  # NOTE: critical to wait until here
        abbrvs[dim] = ''.join(abvs)  # include the operator
        labels[dim] = ' '.join(labs)
        kw_red[dim] = sels[0] if len(sels) == 1 else tuple(sels)
    if names:  # critical this is tuple for membership in _parse_bulk set
        kw_red['names'] = tuple(names)
    if method:  # delay parsing until _apply_method following _apply_reduce
        kw_red['method'] = method
    if operators:  # critical to only assign if exists for _parse_bulk tests
        kw_red['operator'] = operators.pop()
    return abbrvs, labels, kw_red


def _parse_item(dataset, spec, **kwargs):
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
        other specifications (i.e. the row-column framework; see `_parse_bulk`).
    **kwargs
        Additional keyword arguments, used as defaults for the unset keys
        in the variable specifications.

    Returns
    -------
    short : str
        The short name. Summation of multiple variables (e.g. feedback terms)
        can be indicated with e.g. ``pl+lr+wv``. Shorthand aliases are translated.
        This is destined for use in file names.
    long : str
        The long name. Names are combined for summation of multiple variables
        using e.g. ``Planck + lapse rate + water vapor``. This is destined for
        use in figure row or column labels.
    kw_red : dict
        The indexers used to reduce the data variable with `.reduce`. This
        is parsed specially compared to other keywords, and contains ``'names'``
        and ``'operator'`` keys for variable selection and indexer operations.
    kw_all : namedtuple of dict
        A named tuple containing keyword arguments for different plotting-related
        commands. The tuple fields are as follows:

          * ``figure``: Passed to `Figure` when the figure is instantiated.
          * ``gridspec``: Passed to `GridSpec` when the gridfspec is instantiated.
          * ``axes``: Passed to `.format` for cartesian or geographic formatting.
          * ``colorbar``: Passed to `.colorbar` for scalar mappable outputs.
          * ``legend``: Passed to `.legend` for other artist outputs.
          * ``plot``: Passed to the plotting command (the default field).
          * ``attrs``: Added to `.attrs` for use in resulting plot labels.

        These keywords are applied at different points throughout the `plot_bulk`
        function. The figure and gridspec ones are only passed on instantiation.
    """
    # NOTE: For convenience during subsequent processing we put the variables being
    # added (usually just one) inside the 'names' key in kw_red (here `short` is
    # shortened relative to actual dataset names and intended for file names only).
    # This helps when merging variable specifications between row and column
    # specifications and between tuple-style specifications (see _parse_bulk).
    options = [*dataset.sizes, 'area', 'volume', 'method', 'operator']
    options.extend(name for idx in dataset.indexes.values() for name in idx.names)
    if spec is None:
        name, kw = None, {}
    elif isinstance(spec, str):
        name, kw = spec, {}
    elif isinstance(spec, dict):
        name, kw = None, spec
    else:  # 2-tuple required
        name, kw = spec
    kw = {**kwargs, **kw}
    if name:  # remove (this is a correlation indicator)
        kw.pop('name', None)
    else:
        name = name or kw.pop('name', None)  # e.g. global correlation
    short = long = None
    kw_red, kw_att = {}, {}
    kw_fig, kw_grd, kw_axs = {}, {}, {}
    kw_plt, kw_bar, kw_leg = {}, {}, {}
    grd_keys = ('space', 'ratio', 'group', 'equal', 'left', 'right', 'bottom', 'top')
    if isinstance(name, str):  # TODO: climopy get() add methods instead?
        kw_red['names'] = names = [NAMES_SHORT2DATA.get(n, n) for n in name.split('+')]
        short = '+'.join(NAMES_DATA2SHORT.get(n, n) for n in names)
        long = ' + '.join(dataset[n].long_name for n in names)
    att_detect = ('short', 'long', 'standard')
    fig_detect = ('fig', 'ref', 'space', 'share', 'align')
    grd_detect = tuple(s + key for key in grd_keys for s in ('w', 'h', ''))
    axs_detect = ('x', 'y', 'lon', 'lat', 'abc', 'title', 'coast')
    bar_detect = ('extend', 'tick', 'locator', 'formatter', 'minor', 'label')
    leg_detect = ('ncol', 'col', 'row', 'order', 'frame', 'handle', 'border', 'column')
    order = list(KWARGS_ORDER)
    sort = lambda key: order.index(key) if key in order else len(order)
    for key in sorted(kw, key=sort):
        value = kw[key]  # sort for name and label standardization
        if key in options:
            kw_red[key] = value  # e.g. for averaging
        elif any(key.startswith(prefix) for prefix in att_detect):
            kw_att[key] = value
        elif any(key.startswith(prefix) for prefix in fig_detect):
            kw_fig[key] = value
        elif any(key.startswith(prefix) for prefix in grd_detect):
            kw_grd[key] = value
        elif any(key.startswith(prefix) for prefix in axs_detect):
            kw_axs[key] = value
        elif any(key.startswith(prefix) for prefix in bar_detect):
            kw_bar[key] = value
        elif any(key.startswith(prefix) for prefix in leg_detect):
            kw_leg[key] = value
        else:  # arbitrary plotting keywords
            kw_plt[key] = value
    kw_all = KWARGS_ALL(kw_fig, kw_grd, kw_axs, kw_bar, kw_leg, kw_plt, kw_att)
    return short, long, kw_red, kw_all


def _parse_bulk(dataset, rowspecs, colspecs, **kwargs):
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
    gridspecs : list of list of str
        Strings suitable for the figure labels. Returned as a length-2 list of
        descriptive row and column labels.
    filespecs : list of list of str
        Strings suitable for the default file name. Returned as a length-4 list of
        method indicators, shared indicators, row indicators, and column indicators.
    plotspecs : list of list of tuple
        Specifications for plotting in subplots. Returned as list totaling the number
        of axes with length-3 sublists containing the reduction method, a length-1
        or length-2 list of coordinate reduction keyword arguments, and a named tuple
        storing the shared plotting-related keyword arguments.
    """
    # Parse variable specs per gridspec row or column and per subplot, and generate
    # abbrviated figure labels and file names based on the first entries.
    # NOTE: This permits sharing keywords across each group with trailing dicts
    # in either the primary gridspec list or any of the subplot sub-lists.
    # NOTE: For specifications intended to be correlated with one another they
    # should be supplied in a 2-tuple. Otherwise only lists can be used.
    filespecs, gridspecs, allspecs = [], [], []
    for inspecs in (rowspecs, colspecs):
        outspecs = []  # specs containing general information
        if not isinstance(inspecs, list):
            inspecs = [inspecs]
        for ispecs in inspecs:  # specs per subplot
            ospecs = []
            if not isinstance(ispecs, list):
                ispecs = [ispecs]
            for ispec in ispecs:  # specs per correlation pair
                ospec = []
                if isinstance(ispec, (str, dict)):
                    ispec = (ispec,)
                elif len(ispec) != 2:
                    raise ValueError(f'Invalid variable specs {ispec}.')
                elif type(ispec[0]) != type(ispec[1]):  # noqa: E721  # i.e. (str, dict)
                    ispec = (ispec,)
                else:
                    ispec = tuple(ispec)
                for spec in ispec:
                    short, long, kw_red, kw_all = _parse_item(dataset, spec, **kwargs)
                    abrvs, labs, kw_red = _parse_reduce(dataset, **kw_red)
                    ospec.append((short, long, abrvs, labs, kw_red, kw_all))
                ospecs.append(ospec)  # correlation pairs
            outspecs.append(ospecs)  # subplot entries
        fspecs, gspecs = _parse_descrips(outspecs)
        allspecs.append(outspecs)
        filespecs.append(fspecs)
        gridspecs.append(gspecs)

    # Combine row and column specifications for plotting and file naming
    # NOTE: Multiple plotted values per subplot can be indicated in either the row
    # or column list, and the specs from the other list are repeated below.
    # NOTE: Multiple data arrays to be combined with `method2` can be indicated with
    # either 2-tuples in spec lists or conflicting row and column names or reductions.
    # WARNING: Critical to make copies of dictionaries or create new ones
    # here since itertools product repeats the same spec multiple times.
    nrows, ncols = map(len, allspecs)
    plotspecs = []
    for (i, rspecs), (j, cspecs) in itertools.product(*map(enumerate, allspecs)):
        if len(rspecs) == 1:
            rspecs = list(rspecs) * len(cspecs)
        if len(cspecs) == 1:
            cspecs = list(cspecs) * len(rspecs)
        if len(rspecs) != len(cspecs):
            raise ValueError(
                f'Incompatible per-subplot spec count: {len(rspecs)} from '
                f'the row specs, {len(cspecs)} from the column specs.'
            )
        pspecs = []
        for k, (rspec, cspec) in enumerate(zip(rspecs, cspecs)):  # subplot entries
            _, _, _, _, rkws_red, rkws_all = zip(*rspec)
            _, _, _, _, ckws_red, ckws_all = zip(*cspec)
            kws = []
            for field in KWARGS_ALL._fields:
                kw = {}  # NOTE: previously applied default values here
                for ikws_all in (rkws_all, ckws_all):
                    for ikw_all in ikws_all:  # correlation pairs
                        for key, value in getattr(ikw_all, field).items():
                            kw.setdefault(key, value)  # prefer row entries
                kws.append(kw)
            kw_all = KWARGS_ALL(*kws)
            rkws_red = tuple(kw.copy() for kw in rkws_red)
            ckws_red = tuple(kw.copy() for kw in ckws_red)
            for ikws_red, jkws_red in ((rkws_red, ckws_red), (ckws_red, rkws_red)):
                for key in ikws_red[0]:
                    if key in ikws_red[-1] and ikws_red[0][key] == ikws_red[-1][key]:
                        for kw in jkws_red:  # possible correlation pairs
                            kw.setdefault(key, ikws_red[0][key])
            kws_red = {  # filter unique specifications
                tuple(
                    (key, getattr(kw_red[key], '__name__', kw_red[key]))
                    for key in sorted(kw_red)
                ): kw_red for kw_red in (*rkws_red, *ckws_red)
            }
            kws_red = tuple(  # filter partial specifications
                kw_red for keyval, kw_red in kws_red.items()
                if not any(set(keyval) < set(other) for other in kws_red)
            )
            pspecs.append((kws_red, kw_all))
        plotspecs.append(pspecs)

    # Return the specifications
    both = [spec for spec in filespecs[0] if spec in filespecs[1]]
    rows = [spec for spec in filespecs[0] if spec not in both]
    cols = [spec for spec in filespecs[1] if spec not in both]
    filespecs = [both, rows, cols]
    return gridspecs, filespecs, plotspecs


def plot_bulk(
    dataset,
    rowspecs,
    colspecs,
    maxcols=3,
    argskip=None,
    figtitle=None,
    gridskip=None,
    rowlabels=None,
    collabels=None,
    hcolorbar='right',
    vcolorbar='bottom',
    dcolorbar='bottom',
    hlegend='bottom',
    vlegend='bottom',
    dlegend='bottom',
    pcolor=False,
    cycle=None,
    proj='eqearth',
    proj_kw=None,
    save=None,
    **kwargs
):
    """
    Plot any combination of variables across rows and columns.

    Parameters
    ----------
    dataset : xarray.Dataset
        A dataset generated by `open_bulk`.
    rowspecs, colspecs : list of 2-tuple
        Tuples containing the ``(name, kwargs)`` passed to ``ClimoAccessor.get``
        used to generate data in rows and columns. See `_parse_bulk` for details.
    figtitle, rowlabels, collabels : optional
        The figure settings. The labels are determined automatically from
        the specs but can be overridden in a pinch.
    maxcols : int, optional
        The maximum number of columns. Used only if one of the row or variable
        specs is scalar (in which case there are no row or column labels).
    argskip : int or sequence, optional
        The axes indices to omit from auto color scaling in each group of axes
        that shares the same colorbar. Can be used to let columns saturate.
    gridskip : int of sequence, optional
        The gridspec slots to skip. Can be useful for uneven grids when we
        wish to leave earlier slots empty.
    hcolorbar, hlegend : {'right', 'left', 'bottom', 'top'}
        The location for colorbars or legends annotating horizontal rows.
        Automatically placed along the relevant axes.
    vcolorbar, vlegend : {'bottom', 'top', 'right', 'left'}
        The location for colorbars or legends annotating vertical columns,
        Automatically placed along the relevant axes.
    dcolorbar : {'bottom', 'right', 'top', 'left'}
        The default location for colorbars or legends annotating a mix of
        axes. Placed with the figure colorbar or legend method.
    cycle : cycle-spec, optional
        The color cycle used for line plots. Default is to iterate
        over open-color colors.
    pcolor : bool, optional
        Whether to use `pcolormesh` for the default first two-dimensional plot
        instead of `contourf`. Former is useful for noisy data.
    proj : str, optional
        The cartopy projection for longitude-latitude type plots. Default
        is the area-weighted projection ``'eqearth'``.
    proj_kw : dict, optional
        The cartopy projection keyword arguments for longitude-latitude
        type plots. Default is ``{'lon_0': 180}``.
    save : path-like, optional
        The save folder base location. Stored inside a `figures` subfolder.
    **kw_specs
        Passed to `_parse_bulk`.
    **kw_method
        Passed to `_apply_method`.

    Notes
    -----
    The data resulting from each ``ClimoAccessor.get`` operation must be less
    than 3D. 2D data will be plotted with `pcolor`, then darker contours, then
    lighter contours; 1D data will be plotted with `line`, then on an alternate
    axes (so far just two axes are allowed); and 0D data will omit the average
    or correlation step and plot each model with `scatter` (if both variables
    are defined) or `hlines` (if only one model is defined).
    """
    # Initital stuff and figure out geometry
    # TODO: Support e.g. passing 2D arrays to line plotting methods with built-in
    # shadestd, shadepctile, etc. methods instead of using map. See _apply_method.
    kws_annotate = {'fontsize': 'small', 'textcoords': 'offset points'}
    kws_default = {'nozero': True, 'linewidth': pplt.rc.metawidth, 'robust': 96}
    kws_contour = (
        {'color': 'gray8', **kws_default},
        {'color': 'gray3', 'linestyle': ':', **kws_default}
    )
    kws_method = {
        key: kwargs.pop(key) for key in tuple(kwargs)
        if any(key.startswith(s) for s in ('std', 'pctile'))
    }
    gridspecs, filespecs, plotspecs = _parse_bulk(
        dataset, rowspecs, colspecs,
        **kwargs
    )
    nrows, ncols = map(len, gridspecs)
    cycle = pplt.Cycle(cycle or ['blue7', 'red7', 'yellow', 'gray7'])
    colors = pplt.get_colors(cycle)
    titles = (None,) * nrows * ncols
    if nrows == 1 or ncols == 1:
        naxes = max(nrows, ncols)
        ncols = min(naxes, maxcols)
        nrows = 1 + (naxes - 1) // ncols
        titles = max(gridspecs, key=lambda labels: len(labels))
        gridspecs = (None, None)

    # Iterate over axes and plots
    # NOTE: Critical to disable 'grouping' so that e.g. colorbars or legends that
    # extend into other panel slots are not considered in the tight layout algorithm.
    fig = gs = None  # delay instantiation
    proj = pplt.Proj(proj, **(proj_kw or {'lon_0': 180}))
    argskip = np.atleast_1d(argskip or ())
    gridskip = np.atleast_1d(gridskip or ())
    methods = []
    commands = {}
    iterator = zip(titles, plotspecs)
    print('Getting data...')
    for i in range(nrows * ncols):
        if i in gridskip:
            continue
        ax = None  # restart the axes
        aunits = set()
        asizes = set()
        try:
            title, pspecs = next(iterator)
        except StopIteration:
            continue
        for j, (kws_red, kw_all) in enumerate(pspecs):
            # Group reduction selections associated with each side of an operation
            # NOTE: See `_parse_bulk` for more information.
            okws_red = []  # correlation pairs or singleton
            for kw_red in kws_red:
                method = kw_red.pop('method', None)
                operator = kw_red.pop('operator', None)
                if not operator:
                    okws_red.append((kw_red,))
                else:
                    ikw_red, jkw_red = {}, {}
                    for dim, sel in kw_red.items():
                        if isinstance(sel, tuple) and len(sel) == 2 and dim != 'names':
                            ikw_red[dim], jkw_red[dim] = sel
                        else:
                            ikw_red[dim] = jkw_red[dim] = sel
                    okws_red.append((ikw_red, jkw_red))
            if len(okws_red[0]) != len(okws_red[-1]):  # e.g. different operators
                raise RuntimeError(f'Inconsistent variable specs {okws_red}.')

            # Reduce along facets dimension and carry out operation
            # TODO: Add other possible reduction methods, e.g. covariance
            # or average addition or subtraction of variables.
            odatas = []  # each side of operation pair
            for kws_red in zip(*okws_red):
                datas = []
                for kw_red in kws_red:
                    attrs = kw_all.attrs
                    names = kw_red.pop('names', None)  # names for summation
                    if not names:
                        raise ValueError(f'No names found in variable spec {kw_red}.')
                    data = _apply_reduce(dataset, names=names, attrs=attrs, **kw_red)
                    datas.append(data)
                datas, method = _apply_method(*datas, method=method, **kws_method)
                odatas.append(datas)  # possible (x, y) pairs
            if operator is not None and len(odatas) == 2:
                with xr.set_options(keep_attrs=True):
                    datas = [getattr(d, f'__{operator}__')(o) for d, o in zip(*odatas)]
            elif len(odatas) == 1:
                datas, = odatas
            else:
                raise RuntimeError('Operator missing for selection difference.')

            # Instantiate and setup the figure, gridspec, axes
            # NOTE: Here creation is delayed so we can pass arbitrary loose keyword
            # arguments for relevant objects and parse them in _parse_item.
            units = odatas[0][0].attrs['units']
            sizes = odatas[0][0].sizes.keys() - {'facets', 'version', 'period'}
            sharex = True if 'lat' in sizes else 'labels'
            sharey = True if 'plev' in sizes else 'labels'
            kw_fig = {'sharex': sharex, 'sharey': sharey, 'span': False}
            kw_axs = {'title': title}  # possibly none
            if sizes == {'lon', 'lat'}:
                kw_axs.update(KWARGS_DEFAULT['geo'])
                projection = proj  # already-instantiated projection
            else:
                kw_axs.update(KWARGS_DEFAULT['lat'] if 'lat' in sizes else {})
                kw_axs.update(KWARGS_DEFAULT['plev'] if 'plev' in sizes else {})
                projection = 'cartesian'
            nunits = len(aunits)
            aunits.add(units)
            asizes.add(tuple(sorted(sizes)))
            kw_fig.update(refwidth=kw_axs.pop('refwidth', None))
            for key, value in kw_fig.items():
                kw_all.figure.setdefault(key, value)
            for key, value in kw_axs.items():
                kw_all.axes.setdefault(key, value)
            if fig is None:
                fig = pplt.figure(**kw_all.figure)
            if gs is None:
                gs = pplt.GridSpec(nrows, ncols, **kw_all.gridspec)
            if ax is None:
                ax = jax = fig.add_subplot(gs[i], projection=projection, **kw_all.axes)
            if len(asizes) > 1:
                raise ValueError(f'Conflicting plot types with spatial coordinates {asizes}.')  # noqa: E501
            if hasattr(ax, 'alty') != (projection == 'cartesian'):
                raise ValueError(f'Invalid projection for spatial coordinates {sizes}.')

            # Carry out the operation and queue the plotting command
            # NOTE: This automatically generates alternate axes depending on the
            # units of the datas. Currently works only for 1D plots.
            if operator:
                operator = f'__{operator}__'
            if len(odatas) == 1:
                datas, = odatas
            elif operator and len(odatas) == 2:
                with xr.set_options(keep_attrs=True):
                    datas = [getattr(d, operator)(o) for d, o in zip(*odatas)]
            else:
                raise RuntimeError('Operator missing for selection difference.')
            if len(sizes) == 0:
                command = 'hlines' if len(datas) == 1 else 'scatter'
            elif len(sizes) == 1:
                command = 'linex' if 'plev' in sizes else 'line'
                color = colors[j % len(colors)]
                attr = 'altx' if 'plev' in sizes else 'alty'
                jax = ax if nunits == len(aunits) or not nunits else getattr(ax, attr)(color=color, **kw_all.axes)  # noqa: E501
                kw_all.plot.setdefault('color', color)
            elif len(sizes) == 2:
                command = 'pcolormesh' if pcolor else 'contourf'
                kw_all.plot.setdefault('robust', 98)
                if j > 0:
                    command = 'contour'
                    for key, value in kws_contour[j - 1].items():
                        kw_all.plot.setdefault(key, value)
                    kw_all.plot.setdefault('labels', True)
            else:
                raise ValueError(f'Invalid dimension count {len(sizes)} and sizes {sizes}.')  # noqa: E501
            name = '-'.join(data.name for data in datas)
            datas = tuple(datas)
            values = commands.setdefault((name, method, command), [])
            values.append((jax, datas, kw_all))
            if method not in methods:
                methods.append(method)

    # Carry out the plotting commands
    # NOTE: Axes are always added top-to-bottom and left-to-right so leverage
    # this fact below when selecting axes for legends and colorbars.
    print('Plotting data...')
    labels = set()  # figure labels
    kws_leg = {}  # figure legend
    for k, (key, values) in enumerate(commands.items()):
        # Get plotting arguments
        # NOTE: Here 'argskip' is used to skip arguments with vastly different
        # ranges when generating levels, accounting for variable specifications
        # that are unchanged with respect to the gridspec row.
        # print(format(f'{name} {method}: ', ' <20s'), end=' ')
        # print(f'{command} ({len(values)})')
        name, method, command = key
        axs, args, kwargs = zip(*values)
        kw_bar, kw_leg, kw_plt = {}, {}, {}
        for kw in kwargs:
            kw_bar.update(kw.colorbar)
            kw_leg.update(kw.legend)
            kw_plt.update(kw.plot)
        if command in ('pcolormesh', 'contourf'):
            guide = 'colorbar'
            hori, vert, default, kw_guide = hcolorbar, vcolorbar, dcolorbar, kw_bar
            label = re.sub(r' \(', '\n(', args[0][-1].climo.short_label)
            kw_guide.setdefault('label', label)
            kw_guide.setdefault('extendsize', 1.2 + 0.6 * (ax._name != 'cartesian'))
        else:
            guide = 'legend'
            hori, vert, default, kw_guide = hlegend, vlegend, dlegend, kw_leg
            label = args[0][-1].climo.short_label  # avoid duplicate label entries
            kw_guide.setdefault('ncols', 1)
            kw_guide.setdefault('frame', False)
        if command not in ('pcolormesh', 'contourf', 'contour'):
            keys = []
            keys.extend(('vmin', 'vmax', 'levels', 'locator'))
            keys.extend(('cmap', 'norm', 'symmetric', 'diverging', 'robust'))
            kw_plt = {key: val for key, val in kw_plt.items() if key not in keys}
            kw_plt.setdefault('label', label if label not in labels else None)
        else:
            kw_plt['locator'] = 0.1 if label == 'correlation' else None
            levels, *_, kw_plt = axs[0]._parse_level_vals(
                *(args[0][-1].coords[dim] for dim in args[0][-1].dims),
                *(a for l, arg in enumerate(args) for a in arg if l % ncols not in argskip),  # noqa: E501
                norm_kw={}, **kw_plt
            )
            kw_plt['levels'] = levels
            kw_plt.setdefault('extend', 'both')
            if guide == 'colorbar':
                nbins = 7
                locator = pplt.DiscreteLocator(levels, nbins=nbins)
                minorlocator = pplt.DiscreteLocator(levels, nbins=nbins, minor=True)
                kw_guide.setdefault('locator', locator)
                kw_guide.setdefault('minorlocator', minorlocator)  # scaled internally

        # Call plotting commands and add manual annotations
        # NOTE: Had already tried hist and hist2d plots for distributions but they
        # were information-poor and ugly (esp. comparing different feedback sources).
        # NOTE: Using set_in_layout False significantly improves appearance since
        # generally don't mind overlapping with tick labels for scatter plots and
        # improves draw time since tight bounding box calculation is expensive.
        for l, (ax, arg) in enumerate(zip(axs, args)):
            obj = getattr(ax, command)(*arg, **kw_plt)
            if label in labels and 'line' in command:
                obj[0].set_label('_no_label')
                obj[-1].set_label('_no_label')
            if label not in labels and command == 'contour' and obj.collections:
                labels.add(label)
                obj.collections[-1].set_label(label)  # try to pick +ve contour
            if ax._name == 'cartesian':
                x = arg[0]  # default is to use same coordinates
                if command not in ('linex', 'hlines'):
                    x = arg[0].coords[arg[0].dims[-1]]  # e.g. contour should be y by x
                y = arg[-1]
                if command not in ('line', 'scatter'):
                    y = arg[-1].coords[arg[-1].dims[0]]
                xlabel = x.climo.short_label if 'units' in x.attrs else None
                if not ax.get_xlabel():
                    ax.set_xlabel(xlabel)
                ylabel = y.climo.short_label if 'units' in y.attrs else None
                if not ax.get_ylabel():
                    ax.set_ylabel(ylabel)
            if command == 'hlines':
                for i, a in enumerate(arg[0]):  # iterate over scalar arrays
                    kw = {'ha': 'left' if a > 0 else 'right', **kws_annotate}
                    off = 5 if a > 0 else -5
                    tup = a.facets.item()  # multi-index is destroyed
                    model = tup[1] if 'CMIP' in tup[0] else tup[0]
                    res = ax.annotate(model, (a.item(), i), (off, 0), **kw)
                    res.set_in_layout(False)
                xlim, ylim = ax.get_xlim()
                width, _ = ax._get_size_inches()
                diff = (pplt.rc.fontsize / 72) * (max(xlim) - min(xlim)) / width,
                xmin = xlim[0] - 5 * diff * np.any(arg[0] < 0)
                xmax = xlim[1] - 5 * diff * np.any(arg[0] > 0)
                ymin, ymax = ylim[0] - 0.5 * diff, ylim[1] + 0.5 * diff
                xlim = (xmin, xmax) if ax.get_autoscalex_on() else None
                ylim = (ymin, ymax) if ax.get_autoscaley_on() else None
                ax.format(xlim=xlim, ylim=ylim)  # skip if overridden by user
            if command == 'scatter':
                for x, y in zip(*arg):  # iterate over scalar arrays
                    kw = {'ha': 'left', 'va': 'top', **kws_annotate}
                    tup = x.facets.item()  # multi-index is destroyed
                    model = tup[1] if 'CMIP' in tup[0] else tup[0]
                    identical = x.name == y.name and x.units == y.units
                    res = ax.annotate(model, (x.item(), y.item()), (5, -5), **kw)
                    res.set_in_layout(False)
                xlim, ylim = ax.get_xlim(), ax.get_ylim()
                width, _ = ax._get_size_inches()
                diff = (pplt.rc.fontsize / 72) * (max(xlim) - min(xlim)) / width
                xmax = xlim[1] + 5 * diff if ax.get_autoscalex_on() else None
                ymin = ylim[0] - 1 * diff if ax.get_autoscaley_on() else None
                ax.format(xmax=xmax, ymin=ymin)  # skip if overridden by user
                lim = (*ax.get_xlim(), *ax.get_ylim())
                lim = (min(lim), max(lim)) if identical else None
                ax.format(xlim=lim, ylim=lim)
                dat = lim if identical else np.nan
                ax.plot(dat, dat, color='k', ls='--', lw=1.5 * pplt.rc.metawidth)

        # Generate colorbars and legends
        rows = set(ax._get_topmost_axes()._range_subplotspec('y') for ax in axs)
        rows = rows.pop() if len(rows) == 1 else None
        cols = set(ax._get_topmost_axes()._range_subplotspec('x') for ax in axs)
        cols = cols.pop() if len(cols) == 1 else None
        if rows is None and cols is None:
            src = fig
            loc = default
        elif rows is not None:  # prefer single row
            if hori[0] in 'lr':
                src = axs[0] if hori[0] == 'l' else axs[-1]
            else:
                src = axs[len(axs) // 2]  # TODO: support even-numbered axes
            loc = hori
        else:  # single column
            if vert[0] in 'tb':
                src = axs[0] if vert[0] == 't' else axs[-1]
            else:
                src = axs[len(axs) // 2]  # TODO: support even-numbered axes
            loc = vert
        if guide == 'colorbar':
            obj = src.colorbar(obj, loc=loc, **kw_guide)
        else:
            kws_leg.update(**kw_guide)
    if labels:  # TODO: support axes legends or auto-reverting to figure legends
        obj = fig.legend(loc=default, **kws_leg)

    # Format the axes and optionally save
    # NOTE: This permits overwriting individual labels with non-none entries
    kw = {}
    custom = {'rowlabels': rowlabels, 'collabels': collabels}
    default = dict(zip(('rowlabels', 'collabels'), gridspecs))
    for (key, clabels), (_, dlabels) in zip(custom.items(), default.items()):
        nlabels = nrows if key == 'rowlabels' else ncols
        clabels = clabels or [None] * nlabels
        dlabels = dlabels or [None] * nlabels
        if len(dlabels) != nlabels or len(clabels) != nlabels:
            raise RuntimeError(f'Expected {nlabels} labels but got {len(dlabels)} and {len(clabels)}.')  # noqa: E501
        kw[key] = labels = [clab or dlab for clab, dlab in zip(clabels, dlabels)]
    fig.format(figtitle=figtitle, **kw)
    if save:
        path = Path(save).expanduser()
        if '.pdf' not in path.name:
            name = '-'.join(methods) + '_'
            name += '_'.join('-'.join(specs) for specs in filespecs if specs)
            path.mkdir(exist_ok=True)
            path = path / 'figures' / f'{name}.pdf'
        print(f'Saving {path.parent}/{path.name}...')
        fig.save(path)
    return fig, fig.subplotgrid
