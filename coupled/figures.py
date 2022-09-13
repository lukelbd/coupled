#!/usr/bin/env python3
"""
Utilities for plotting coupled model output.
"""
import collections
import itertools
import re
import warnings
from pathlib import Path

import numpy as np
import xarray as xr
from icecream import ic  # noqa: F401

import climopy as climo
import proplot as pplt
from climopy import decode_units, format_units, ureg, var, vreg  # noqa: F401
from .output import FEEDBACK_TRANSLATIONS, MODELS_INSTITUTIONS


# Variable constants
_seen = set()
NAMES_LONG = {
    key: name
    for key, (name, _) in FEEDBACK_TRANSLATIONS.items()
}
NAMES_SHORT = {
    name: key for key, name in NAMES_LONG.items()
    if name not in _seen and not _seen.add(name)  # translate with first entry
}
del _seen

# Keyword argument constants
# NOTE: For sake of grammar we place 'ensemble' before 'experiment' here
KWARGS_ALL = collections.namedtuple(
    'kwargs', (
        'figure',
        'gridspec',
        'axes',
        'colorbar',
        'legend',
        'plot',
        'attrs'
    )
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
        'xscale': 'linear',
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

# Label constants
# NOTE: Unit constants are partly based on CONVERSIONS_STANDARD from output.py. Need
# to eventually forget this and use registered variable short names instead.
REDUCE_ABBRVS = {
    'avg': None,
    'int': None,
    'absmin': 'min',
    'absmax': 'max',
    'point': 'point',
    'globe': 'globe',
    'latitude': 'zonal',
    'hemisphere': 'hemi',
}
REDUCE_LABELS = {  # default is title-case of input
    '+': 'plus',
    '-': 'minus',
    'avg': None,
    'int': None,
    'ann': 'annual',
    'djf': 'DJF',
    'mam': 'MAM',
    'jja': 'JJA',
    'son': 'SON',
    'absmin': 'minimum',
    'absmax': 'maximum',
    'point': 'local',
    'globe': 'global',
    'latitude': 'zonal',
    'hemisphere': 'hemisphere',
    'control': 'pre-industrial',
    'response': r'abrupt 4$\times$CO$_2$',
    'picontrol': 'pre-industrial',
    'abrupt4xco2': r'abrupt 4$\times$CO$_2$',
}
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


def _apply_reduce(dataset, attrs=None, **kwargs):
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
    # WARNING: Sometimes multi-index reductions can eliminate previously valid
    # coordinates, so iterate one-by-one and validate selections each time.
    # NOTE: This silently skips dummy selections (e.g. area=None) that may be
    # required to prevent _parse_bulk from merging variable specs that differ
    # only in that one contains a selection and the other doesn't (generally
    # when constraining local feedbacks vs. global feedbacks).
    defaults = {
        'period': 'ann',
        'experiment': 'picontrol',
        'ensemble': 'flagship',
        'source': 'eraint',
        'statistic': 'slope',
        'region': 'globe',
    }
    for key, value in defaults.items():
        kwargs.setdefault(key, value)
    name = kwargs.pop('name', None)
    if name is None:
        raise ValueError('Variable name missing from reduction specification.')
    data = dataset[name]
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
                short = UNITS_LABELS.get(units, long)
            array.attrs['short_name'] = short

    # Combine one array along facets dimension
    # NOTE: Here `pctile` is shared between inter-model percentile differences and
    # composites of a second array based on values in the first array.
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
            data = (np.arange(data.size), data.sortby(data, ascending=False),)
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
            datas = xr.broadcast(*datas)
            _, cov = climo.covar(*datas, dim='facets')  # updates units
            cov = cov.climo.quantify().isel(lag=0)
            var0 = datas[0].climo.quantify().var(dim='facets')
            var1 = datas[1].climo.quantify().var(dim='facets')
            data = cov ** 2 / (var0 * var1)
            data = data.climo.to_units('percent').climo.dequantify()
            short_name = 'variance explained'
            long_name = f'{datas[0].long_name}-{datas[1].long_name} variance explained'
            data.name = f'{datas[0].name}-{datas[1].name}'
            data.attrs['short_name'] = short_name
            data.attrs['long_name'] = long_name
        elif method == 'corr':  # correlation coefficient
            datas = xr.broadcast(*datas)
            _, cov = climo.covar(*datas, dim='facets')  # updates units
            cov = cov.climo.quantify().isel(lag=0)
            std0 = datas[0].climo.quantify().std(dim='facets')
            std1 = datas[1].climo.quantify().std(dim='facets')
            data = cov / (std0 * std1)
            data = data.climo.to_units('dimensionless').climo.dequantify()
            short_name = 'correlation coefficient'
            long_name = f'{datas[0].long_name}-{datas[1].long_name} correlation coefficient'  # noqa: E501
            data.name = f'{datas[0].name}-{datas[1].name}'
            data.attrs['short_name'] = short_name
            data.attrs['long_name'] = long_name
        elif method == 'diff':  # composite difference along first arrays
            pctile = 33 if pctile is None else pctile
            datas = xr.broadcast(*datas)
            lo_comp = np.nanpercentile(datas[0], pctile)
            hi_comp = np.nanpercentile(datas[0], 100 - pctile)
            lo_mask, = np.where(datas[0] <= lo_comp)
            hi_mask, = np.where(datas[0] >= hi_comp)
            hi_data = datas[1].isel(facets=hi_mask).mean('facets')
            lo_data = datas[1].isel(facets=lo_mask).mean('facets')
            data = hi_data.climo.quantify() - lo_data.climo.quantify()
            data = data.climo.dequantify()
            short_name = f'{datas[1].short_name} difference'
            long_name = f'{datas[0].long_name}-composite {datas[1].long_name} difference'  # noqa: E501
            data.name = f'{datas[0].name}-{datas[1].name}'
            data.attrs['short_name'] = short_name
            data.attrs['long_name'] = long_name
        elif method == 'proj':  # projection onto x
            datas = xr.broadcast(*datas)
            _, cov = climo.covar(*datas, dim='facets')  # updates units
            cov = cov.climo.quantify().isel(lag=0)
            std = datas[0].climo.quantify().std(dim='facets')
            data = cov / std
            data = data.climo.dequantify()
            short_name = f'{datas[1].short_name} projection'
            long_name = f'{datas[1].long_name} vs. {datas[0].long_name}'
            data.name = f'{datas[0].name}-{datas[1].name}'
            data.attrs['short_name'] = short_name
            data.attrs['long_name'] = long_name
        elif method == 'slope':  # regression coefficient
            datas = xr.broadcast(*datas)
            _, cov = climo.covar(*datas, dim='facets')  # updates units
            cov = cov.climo.quantify().isel(lag=0)
            std = datas[0].climo.quantify().std(dim='facets')
            data = cov / std ** 2
            data = data.climo.dequantify()
            short_name = f'{datas[1].short_name} regression'
            long_name = f'{datas[1].long_name} vs. {datas[0].long_name}'
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


def _parse_dicts(dataset, spec, **kwargs):
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
    kw_red : dict
        The indexers used to reduce the data variable with `.reduce`. This is
        parsed specially compared to other keywords, and its keys are restricted
        to ``'name'`` and any coordinate or multi-index names.
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
    # NOTE: For subsequent processing we put the variables being combined (usually
    # just one) inside the 'name' key in kw_red (here `short` is shortened relative
    # to actual dataset names and intended for file names only). This helps when
    # merging variable specifications between row and column specifications and
    # between tuple-style specifications (see _parse_bulk).
    options = [*dataset.sizes, 'area', 'volume', 'method', 'std', 'pctile', 'invert']
    options.extend(name for idx in dataset.indexes.values() for name in idx.names)
    if spec is None:
        name, kw = None, {}
    elif isinstance(spec, str):
        name, kw = spec, {}
    elif isinstance(spec, dict):
        name, kw = None, spec
    else:  # length-2 iterable
        name, kw = spec
    alt = kw.pop('name', None)
    name = name or alt  # see below
    kw = {**kwargs, **kw}
    kw_red, kw_att = {}, {}
    kw_fig, kw_grd, kw_axs = {}, {}, {}
    kw_plt, kw_bar, kw_leg = {}, {}, {}
    keys = ('space', 'ratio', 'group', 'equal', 'left', 'right', 'bottom', 'top')
    att_detect = ('short', 'long', 'standard')
    fig_detect = ('fig', 'ref', 'space', 'share', 'align')
    grd_detect = tuple(s + key for key in keys for s in ('w', 'h', ''))
    axs_detect = ('x', 'y', 'lon', 'lat', 'abc', 'title', 'coast')
    bar_detect = ('extend', 'tick', 'locator', 'formatter', 'minor', 'label')
    leg_detect = ('ncol', 'order', 'frame', 'handle', 'border', 'column')
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
    if isinstance(name, str):
        kw_red['name'] = name  # always place last for gridspec labels
    return kw_red, kw_all


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
    elif num in ('5', '6'):
        filter = lambda key: key[0][-1] == num
        label = f'CMIP{num}'
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
    else:
        raise ValueError(f'Invalid project {num!r}.')
    abbrv = filter.__name__ = f'cmip{num}'  # required in _parse_bulk
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
    # before merging in _parse_bulk (e.g. global average vs. local feedbacks).
    abbrvs, labels, kw_red = {}, {}, {}
    keys = ('method', 'std', 'pctile', 'invert')
    kw_red.update({key: kwargs.pop(key) for key in keys if key in kwargs})
    for key, value in kwargs.items():  # WARNING: critical to preserve order
        abvs, labs, sels = [], [], []
        parts = re.split('([+-])', value) if isinstance(value, str) else (value,)
        for i, part in enumerate(parts):
            if isinstance(part, str):  # string coordinate
                if key == 'project' and part not in '+-':
                    abbrv, label, sel = _parse_project(dataset, part)
                    if i == len(parts) - 1:  # WARNING: critical to wait until end
                        key = 'facets'
                elif key == 'name' and part not in '+-':
                    sel = NAMES_LONG.get(part, part)
                    abbrv = NAMES_SHORT.get(sel, sel)
                    label = dataset[sel].long_name
                else:
                    if 'control' in dataset.experiment:
                        opts = {'picontrol': 'control', 'abrupt4xco2': 'respone'}
                    else:
                        opts = {'control': 'picontrol', 'response': 'abrupt4xco2'}
                    sel = opts.get(part, part)
                    abbrv = REDUCE_ABBRVS.get(sel, sel)
                    label = REDUCE_LABELS.get(sel, sel)
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
            if abbrv is not None:  # e.g. operator or 'avg'
                abvs.append(abbrv)
            if label is not None:  # e.g. operator or 'avg'
                labs.append(label)
            if sel is not None:
                sels.append(sel)
        sels = ('+', *sels) if sels else ()
        abbrvs[key] = ''.join(abvs)
        labels[key] = ' '.join(labs)
        kw_red[key] = tuple(zip(sels[0::2], sels[1::2]))  # (([+-], value), ...) tuple
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
    # Get component pieces
    # NOTE: This selects just the first variable specification in each subplot. Others
    # are assumed to be secondary and not worth encoding in labels or file name.
    abbrvs, labels, *_ = zip(*(zip(*ispecs[0]) for ispecs in specs))
    abbrvs = tuple(abv for abvs in abbrvs for opts in abvs for abv in opts.values())
    npairs = max(len(kws) for kws in abbrvs)
    filespecs = []
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
        gspecs = [tuple(kw.get(key, None) for kw in gspecs) for key in keys]
        gspecs = [tup for tup in gspecs if any(spec != tup[0] for spec in tup)]
        gspecs = [' '.join(filter(None, tup)) for tup in zip(*gspecs)]
        upper = lambda s: (s[0].upper() if s[0].islower() else s[0]) + s[1:]
        parts = ('feedback', 'forcing', 'energy', 'transport', 'convergence')
        for part in parts:
            if all(f' {part}' in spec for spec in gspecs):
                gspecs = [spec.replace(f' {part}', '') for spec in gspecs]
        gridspecs.append(list(map(upper, gspecs)) or [''] * len(specs))

    # Consolidate figure grid and file name labels
    # NOTE: Here correlation pairs are handled with 'vs. ' indicators in the row
    # or column labels (also components are kept only if distinct across slots).
    seen = set()
    filespecs = [  # avoid e.g. cmip5-cmip5 segments
        spec.replace('_', '') for spec in sorted(filter(None, abbrvs))
        if spec and spec not in seen and not seen.add(spec)
    ]
    seen = set()
    gridspecs = [  # avoid e.g. 'cmip5 vs. cmip5' labels
        ' vs. '.join(filter(None,
            (spec for spec in specs if spec not in seen and not seen.add(spec))  # noqa: E128, E501
        )) for specs in zip(*gridspecs) if not seen.clear()  # refresh each time
    ]
    outspecs = []
    for gridspec in gridspecs:
        refwidth = refwidth or pplt.rc['subplots.refwidth']
        threshs = pplt.units(refwidth, 'in', 'em') * np.arange(1, 10)
        chars = list(gridspec)  # string to list
        idxs = np.array([i for i, c in enumerate(chars) if c == ' '])
        for thresh in threshs:
            if np.any(idxs > thresh):
                chars[np.min(idxs[idxs > thresh])] = '\n'
        outspecs.append(''.join(chars))
    return filespecs, outspecs


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
    filespecs : list of list of str
        Strings suitable for the figure file name. See `_parse_labels`.
    gridspecs : list of list of str
        Strings suitable for the figure grid labels. See `_parse_labels`.
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
    filespecs, gridspecs, plotspecs = [], [], []
    for inspecs in (rowspecs, colspecs):
        outspecs = []  # specs containing general information
        if not isinstance(inspecs, list):
            inspecs = [inspecs]
        for ispecs in inspecs:  # specs per figure
            ospecs = []
            if not isinstance(ispecs, list):
                ispecs = [ispecs]
            for ispec in ispecs:  # specs per subplot
                ospec = []
                if isinstance(ispec, (str, dict)):
                    ispec = (ispec,)
                elif len(ispec) != 2:
                    raise ValueError(f'Invalid variable specs {ispec}.')
                elif type(ispec[0]) != type(ispec[1]):  # noqa: E721  # i.e. (str, dict)
                    ispec = (ispec,)
                else:
                    ispec = tuple(ispec)
                for spec in ispec:  # specs per correlation pair
                    kw_red, kw_all = _parse_dicts(dataset, spec, **kwargs)
                    abbrvs, labels, kw_red = _parse_reduce(dataset, **kw_red)
                    ospec.append((abbrvs, labels, kw_red, kw_all))
                ospecs.append(ospec)
            outspecs.append(ospecs)
        fspecs, gspecs = _parse_labels(outspecs, refwidth=kwargs.get('refwidth'))
        plotspecs.append(outspecs)
        filespecs.append(fspecs)
        gridspecs.append(gspecs)

    # Combine row and column specifications for plotting and file naming
    # NOTE: Multiple plotted values per subplot can be indicated in either the row
    # or column list, and the specs from the other list are repeated below.
    # NOTE: Multiple data arrays to be combined with `method2` can be indicated with
    # either 2-tuples in spec lists or conflicting row and column names or reductions.
    # WARNING: Critical to make copies of dictionaries or create new ones
    # here since itertools product repeats the same spec multiple times.
    iter_ = itertools.product(*map(enumerate, plotspecs))
    nrows, ncols = map(len, plotspecs)
    plotspecs = []
    for (i, rspecs), (j, cspecs) in iter_:
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
            _, _, rkws_red, rkws_all = zip(*rspec)
            _, _, ckws_red, ckws_all = zip(*cspec)
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
            kws_red = {}  # filter unique specifications
            for kw_red in (*rkws_red, *ckws_red):
                keyval = []
                for key in sorted(kw_red):
                    value = kw_red[key]
                    if isinstance(value, tuple) and all(isinstance(val, tuple) for val in value):  # noqa: E501
                        value = tuple(tuple(getattr(v, '__name__', v) for v in val) for val in value)  # noqa: E501
                    if key not in ('method',):  # cannot 'intersect' methods
                        keyval.append((key, value))
                keyval = tuple(keyval)
                if tuple(keyval) in kws_red:
                    continue
                others = tuple(other for other in kws_red if set(keyval) < set(other))
                if others:
                    continue
                others = tuple(other for other in kws_red if set(other) < set(keyval))
                for other in others:
                    kws_red.pop(other)
                kws_red[tuple(keyval)] = kw_red
            kws_red = tuple(kws_red.values())
            pspecs.append((kws_red, kw_all))
        plotspecs.append(pspecs)

    # Return the specifications
    both = [spec for spec in filespecs[0] if spec in filespecs[1]]
    rows = [spec for spec in filespecs[0] if spec not in both]
    cols = [spec for spec in filespecs[1] if spec not in both]
    filespecs = [both, rows, cols]
    return filespecs, gridspecs, plotspecs


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
    standardize=False,
    annotate=False,
    linefit=False,
    oneone=False,
    pcolor=False,
    cycle=None,
    proj='kav7',
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
        The location for colorbars or legends annotating vertical columns.
        Automatically placed along the relevant axes.
    dcolorbar : {'bottom', 'right', 'top', 'left'}
        The default location for colorbars or legends annotating a mix of
        axes. Placed with the figure colorbar or legend method.
    standardize : bool, optional
        Whether to standardize axis limits to span the same range for all
        plotted content with the same units.
    annotate : bool, optional
        Whether to annotate scatter plots and `hlines` or `vlines`
        plots with model names.
    linefit : bool, optional
        Whether to draw best-fit lines for scatter plots of two arbitrary
        variables. Uses `climo.linefit`.
    oneone : bool, optional
        Whether to draw dashed one-one lines for scatter plots of two variables
        with the same units.
    pcolor : bool, optional
        Whether to use `pcolormesh` for the default first two-dimensional plot
        instead of `contourf`. Former is useful for noisy data.
    cycle : cycle-spec, optional
        The color cycle used for line plots. Default is to iterate
        over open-color colors.
    proj : str, optional
        The cartopy projection for longitude-latitude type plots. Default
        is the shape-preserving projection ``'kav7'``.
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
    filespecs, gridspecs, plotspecs = _parse_bulk(dataset, rowspecs, colspecs, **kwargs)
    nrows, ncols = map(len, gridspecs)
    nrows, ncols = max(nrows, 1), max(ncols, 1)
    titles = (None,) * nrows * ncols
    if nrows == 1 or ncols == 1:
        naxes = max(nrows, ncols)
        ncols = min(naxes, maxcols)
        nrows = 1 + (naxes - 1) // ncols
        titles = max(gridspecs, key=lambda labels: len(labels))
        gridspecs = (None, None)
    cycle = pplt.Cycle(cycle or ['blue7', 'red7', 'yellow7', 'gray7'])
    colors = pplt.get_colors(cycle)
    kw_annotate = {'fontsize': 0.5 * pplt.rc.fontsize, 'textcoords': 'offset points'}
    kw_contour = {'robust': 96, 'nozero': True, 'linewidth': pplt.rc.metawidth}
    kw_contourf = {'levels': 20, 'extend': 'both'}
    kws_contour = []
    kws_contour.append({'color': 'gray8', 'linestyle': None, **kw_contour})
    kws_contour.append({'color': 'gray3', 'linestyle': ':', **kw_contour})
    kw_hlines = {'negpos': True, 'linewidth': 1.5 * pplt.rc.metawidth}
    kw_line = {'linestyle': '-', 'linewidth': 1.5 * pplt.rc.metawidth}
    kw_scatter = {'color': 'gray7', 'linewidth': 1.5 * pplt.rc.metawidth}
    kw_scatter.update({'marker': 'x', 'markersize': 0.1 * pplt.rc.fontsize ** 2})

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
            # Group added/subtracted reduce instructions into separate dictionaries
            # NOTE: Initial kw_red values are formatted as (('[+-]', value), ...) to
            # permit arbitrary combinations of names and indexers (see _parse_bulk).
            kws_method = []  # each item represents a method argument
            for kw in kws_red:
                kw_extra, kw_reduce, scale = {}, {}, 1
                for key, value in kw.items():
                    if isinstance(value, tuple) and isinstance(value[0], tuple):
                        kw, count = kw_reduce, sum(pair.count('+') for pair in value)
                    else:
                        kw, count = kw_extra, 1  # e.g. a 'std' or 'pctile' keyword
                    kw[key] = value
                    scale *= count
                kws_persum = []
                for values in itertools.product(*kw_reduce.values()):
                    signs, values = zip(*values)
                    sign = -1 if signs.count('-') % 2 else +1
                    kw = dict(zip(kw_reduce, values))
                    kw.update(kw_extra)
                    kws_persum.append((sign, kw))
                kws_method.append((scale, kws_persum))

            # Reduce along facets dimension and carry out operation
            # TODO: Add other possible reduction methods, e.g. covariance
            # or regressions instead of normalized correlation.
            scales, kws_method = zip(*kws_method)
            if len(set(scales)) > 1:
                raise RuntimeError(f'Mixed reduction scalings {scales}.')
            kws_persum = zip(*kws_method)
            datas_persum = []  # each item represents part of a summation
            methods_persum = set()
            for kws_reduce in kws_persum:
                kw_method = {}
                keys = ('std', 'pctile', 'invert', 'method')
                datas = []
                signs, kws_reduce = zip(*kws_reduce)
                if len(set(signs)) > 1:
                    raise RuntimeError(f'Mixed reduction signs {signs}.')
                for kw_reduce in kws_reduce:  # two for e.g. 'corr', one for e.g. 'avg'
                    for key in tuple(kw_reduce):
                        if key in keys:
                            kw_method.setdefault(key, kw_reduce.pop(key))
                    attrs = kw_all.attrs.copy()
                    data = _apply_reduce(dataset, attrs=attrs, **kw_reduce)
                    datas.append(data)
                datas, method = _apply_method(*datas, **kw_method)
                if isinstance(datas[-1], dict):
                    *datas, kw = datas
                    for key, val in kw.items():
                        kw_all.plot.setdefault(key, val)
                datas_persum.append((signs[0], datas))  # plotting command arguments
                methods_persum.add(method)
                if len(methods_persum) > 1:
                    raise RuntimeError(f'Mixed reduction methods {methods_persum}.')
            args = []
            signs, datas_persum = zip(*datas_persum)
            for datas in zip(*datas_persum):
                with xr.set_options(keep_attrs=True):
                    arg = sum(sign * data for sign, data in zip(signs, datas))
                    arg = arg / scales[0]
                if isinstance(arg, xr.DataArray):
                    names = (data.name for data in datas if hasattr(data, 'name'))
                    arg.name = '-'.join(names)
                args.append(arg)

            # Instantiate and setup the figure, gridspec, axes
            # NOTE: Here creation is delayed so we can pass arbitrary loose keyword
            # arguments for relevant objects and parse them in _parse_dicts.
            sizes = args[-1].sizes.keys() - {'facets', 'version', 'period'}
            asizes.add(tuple(sorted(sizes)))
            sharex = True if 'lat' in sizes or 'plev' in sizes else 'labels'
            sharey = True if 'lat' in sizes or 'plev' in sizes else 'labels'
            kw_fig = {'sharex': sharex, 'sharey': sharey, 'span': False}
            kw_axs = {'title': title}  # possibly none
            dims = ('geo',) if sizes == {'lon', 'lat'} else sizes & {'lat', 'plev'}
            projection = proj if sizes == {'lon', 'lat'} else 'cartesian'
            kw_def = {key: val for dim in dims for key, val in KWARGS_DEFAULT[dim].items()}  # noqa: E501
            kw_fig.update(refwidth=kw_def.pop('refwidth', None))
            kw_axs.update(kw_def)
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

            # Apply default plotting command arguments
            # NOTE: This automatically generates alternate axes depending on the
            # units of the datas. Currently works only for 1D plots.
            nunits = len(aunits)
            units = args[-1].attrs['units']  # avoid numpy coordinates
            aunits.add(units)
            if len(sizes) == 0:
                if isinstance(args[0], xr.DataArray):
                    command = 'scatter'
                    for key, value in kw_scatter.items():
                        kw_all.plot.setdefault(key, value)
                else:
                    command = 'hlines'
                    ax.format(ylocator='null')
                    for key, value in kw_hlines.items():
                        kw_all.plot.setdefault(key, value)
            elif len(sizes) == 1:
                if 'plev' in sizes:
                    command = 'linex'
                    color = colors[j % len(colors)]
                    jax = ax
                    if nunits and nunits != len(aunits):
                        jax = ax.altx(color=color)  # noqa: E501
                    else:
                        jax = ax
                    kw_all.plot.setdefault('color', color)
                    for key, value in kw_line.items():
                        kw_all.plot.setdefault(key, value)
                else:
                    command = 'line'
                    color = colors[j % len(colors)]
                    jax = ax
                    if nunits and nunits != len(aunits):
                        jax = ax.alty(color=color)
                    else:
                        jax = ax
                    kw_all.plot.setdefault('color', color)
            elif len(sizes) == 2:
                if 'hatches' in kw_all.plot:
                    command = 'contourf'
                elif j == 0:
                    command = 'pcolormesh' if pcolor else 'contourf'
                    kw_all.plot.setdefault('robust', 98)
                    for key, value in kw_contourf.items():
                        kw_all.plot.setdefault(key, value)
                else:
                    command = 'contour'
                    kw_all.plot.setdefault('labels', True)
                    for key, value in kws_contour[j - 1].items():
                        kw_all.plot.setdefault(key, value)
            else:
                raise ValueError(f'Invalid dimension count {len(sizes)} and sizes {sizes}.')  # noqa: E501

            # Queue the plotting command
            # NOTE: This will automatically allocate separate colorbars for
            # variables with different declared level-restricting arguments.
            args = tuple(args)
            name = '-'.join(arg.name for arg in args if isinstance(arg, xr.DataArray))
            cmap = kw_all.plot.get('cmap', None)
            color = kw_all.plot.get('color', None)
            key = (name, method, command, cmap, color)
            values = commands.setdefault(key, [])
            values.append((jax, args, kw_all))
            if method not in methods:
                methods.append(method)

    # Carry out the plotting commands
    # NOTE: Axes are always added top-to-bottom and left-to-right so leverage
    # this fact below when selecting axes for legends and colorbars.
    print('Plotting data...')
    axs_objs = {}
    axs_units = {}  # axes grouped by units
    for k, (key, values) in enumerate(commands.items()):
        # Get plotting arguments
        # NOTE: Here 'argskip' is isued to skip arguments with vastly different
        # ranges when generating levels that annotate multiple different subplots.
        name, method, command, cmap, color = key
        axs, args, kw_all = zip(*values)
        kw_bar, kw_leg, kw_plt = {}, {}, {}
        for kw in kw_all:
            kw_bar.update(kw.colorbar)
            kw_leg.update(kw.legend)
            kw_plt.update(kw.plot)
        if command in ('pcolormesh', 'contourf', 'contour'):
            xy = (args[0][-1].coords[dim] for dim in args[0][-1].dims)
            zs = (a for l, arg in enumerate(args) for a in arg if l % ncols not in argskip)  # noqa: E501
            kw_add = {key: kw_plt[key] for key in ('extend',) if key in kw_plt}
            levels, *_, kw_plt = axs[0]._parse_level_vals(*xy, *zs, norm_kw={}, **kw_plt)  # noqa: E501
            kw_plt.update({**kw_add, 'levels': levels})
            kw_plt.pop('robust', None)
        if command in ('pcolormesh', 'contourf') and 'hatches' not in kw_plt:
            guide, kw_guide = 'colorbar', kw_bar
            label = args[0][-1].climo.short_label
            label = re.sub(r' \(', '\n(', label)
            locator = pplt.DiscreteLocator(levels, nbins=7)
            minorlocator = pplt.DiscreteLocator(levels, nbins=7, minor=True)
            kw_guide.setdefault('locator', locator)
            kw_guide.setdefault('minorlocator', minorlocator)  # scaled internally
            kw_guide.setdefault('extendsize', 1.2 + 0.6 * (ax._name != 'cartesian'))
        else:  # TODO: permit short *or* long
            guide, kw_guide = 'legend', kw_leg
            climo = args[0][-1].climo
            label = climo.long_label if command == 'contour' else climo.long_name
            label = None if 'hatches' in kw_plt else label
            keys = ('robust', 'symmetric', 'diverging', 'levels', 'locator', 'extend')
            keys = () if command == 'contour' else keys
            keys += ('cmap', 'norm', 'norm_kw')
            kw_plt = {key: val for key, val in kw_plt.items() if key not in keys}
            kw_guide.setdefault('ncols', 1)
            kw_guide.setdefault('frame', False)

        # Iterate over axes and arguments
        obj = result = None
        for l, (ax, arg) in enumerate(zip(axs, args)):
            # Add plotted content and formatting
            # TODO: Support hist and hist2d plots in addition to hlines and scatter
            # (or just hist since, hist2d usually looks ugly with so little data)
            if ax._name == 'cartesian':  # x-axis formatting
                x = arg[-1] if command == 'hlines' else arg[0]
                if command not in ('linex', 'scatter', 'hlines'):
                    x = arg[-1].coords[arg[-1].dims[-1]]  # e.g. contour() is y by x
                units = getattr(x, 'units', None)
                axes = axs_units.setdefault(('x', units), [])
                axes.append(ax)
                xlabel = x.climo.short_label if 'units' in getattr(x, 'attrs', {}) else None  # noqa: E501
                if not ax.get_xlabel():
                    rows = ax._get_topmost_axes()._range_subplotspec('y')
                    if ax == ax._get_topmost_axes() or max(rows) == nrows - 1:
                        ax.set_xlabel(xlabel)
            if ax._name == 'cartesian':  # y-axis formatting
                y = None if command == 'hlines' else arg[-1]
                if command not in ('line', 'scatter', 'hlines'):
                    y = arg[-1].coords[arg[-1].dims[0]]
                units = getattr(y, 'units', None)
                axes = axs_units.setdefault(('y', units), [])
                axes.append(ax)
                ylabel = y.climo.short_label if 'units' in getattr(y, 'attrs', {}) else None  # noqa: E501
                if not ax.get_ylabel():
                    cols = ax._get_topmost_axes()._range_subplotspec('x')
                    if ax == ax._get_topmost_axes() or max(cols) == ncols - 1:
                        ax.set_ylabel(ylabel)
            with warnings.catch_warnings():  # ignore 'masked to nan'
                warnings.simplefilter('ignore', UserWarning)
                result = getattr(ax, command)(*arg, **kw_plt)
                if isinstance(result, (list, tuple)):  # silent list or tuple
                    obj = result[0][1] if isinstance(result[0], tuple) else result[0]
                elif command == 'contour' and result.collections:
                    obj = result.collections[-1]
                elif command in ('contourf', 'pcolormesh'):
                    obj = result
                else:  # e.g. lines or scatter
                    pass

            # Add other content
            # NOTE: Using set_in_layout False significantly improves appearance since
            # generally don't mind overlapping with tick labels for scatter plots and
            # improves draw time since tight bounding box calculation is expensive.
            if 'line' in command and ax == ax._get_topmost_axes():
                kw = dict(color='k', ls='-', lw=1.5 * pplt.rc.metawidth)
                cmd = 'axhline' if command == 'line' else 'axvline'
                getattr(ax, cmd)(0.0, **kw)
            if 'lines' in command:
                if annotate:
                    xlim, ylim = ax.get_xlim(), ax.get_ylim()
                    width, _ = ax._get_size_inches()
                    diff = (pplt.rc.fontsize / 72) * (max(xlim) - min(xlim)) / width
                    xmin = xlim[0] - 5 * diff * np.any(arg[-1] < 0)
                    xmax = xlim[1] + 5 * diff * np.any(arg[-1] > 0)
                    ymin, ymax = ylim[0] - 0.5 * diff, ylim[1] + 0.5 * diff
                    xlim = (xmin, xmax) if ax.get_autoscalex_on() else None
                    ylim = (ymin, ymax) if ax.get_autoscaley_on() else None
                    ax.format(xlim=xlim, ylim=ylim)  # skip if overridden by user
                    for i, a in enumerate(arg[-1]):  # iterate over scalar arrays
                        ha = 'left' if a > 0 else 'right'
                        kw = {'ha': ha, 'va': 'center', **kw_annotate}
                        off = 2 if a > 0 else -2
                        tup = a.facets.item()  # multi-index is destroyed
                        model = tup[1] if 'CMIP' in tup[0] else tup[0]
                        res = ax.annotate(model, (a.item(), i), (off, 0), **kw)
                        res.set_in_layout(False)
            if command == 'scatter':
                if annotate:
                    xlim, ylim = ax.get_xlim(), ax.get_ylim()
                    width, _ = ax._get_size_inches()
                    diff = (pplt.rc.fontsize / 72) * (max(xlim) - min(xlim)) / width
                    xmax = xlim[1] + 5 * diff if ax.get_autoscalex_on() else None
                    ymin = ylim[0] - 1 * diff if ax.get_autoscaley_on() else None
                    ax.format(xmax=xmax, ymin=ymin)  # skip if overridden by user
                    for x, y in zip(*arg):  # iterate over scalar arrays
                        kw = {'ha': 'left', 'va': 'top', **kw_annotate}
                        tup = x.facets.item()  # multi-index is destroyed
                        model = tup[1] if 'CMIP' in tup[0] else tup[0]
                        res = ax.annotate(model, (x.item(), y.item()), (2, -2), **kw)
                        res.set_in_layout(False)
                if oneone:
                    lim = (*ax.get_xlim(), *ax.get_ylim())
                    lim = (min(lim), max(lim))
                    avg = 0.5 * (lim[0] + lim[1])
                    span = lim[1] - lim[0]
                    ones = (avg - 1e3 * span, avg + 1e3 * span)
                    ax.format(xlim=lim, ylim=lim)  # autoscale disabled
                    ax.plot(ones, ones, ls='--', lw=1.5 * pplt.rc.metawidth, color='k')
                if linefit:  # https://en.wikipedia.org/wiki/Simple_linear_regression
                    idx = np.argsort(arg[0].values)
                    x, y = arg[0].values[idx], arg[1].values[idx]
                    slope, stderr, rsquare, fit, lower, upper = var.linefit(x, y, adjust=False)  # noqa: E501
                    rsquare = ureg.Quantity(rsquare.item(), '').to('percent')
                    ax.format(ultitle=rf'$R^2 = {rsquare:~L.1f}$'.replace('%', r'\%'))
                    ax.plot(x, fit, color='r', ls='-', lw=1.5 * pplt.rc.metawidth)
                    ax.area(x, lower, upper, color='r', alpha=0.5 ** 2, lw=0)

        # Update legend and colorbar queues
        # NOTE: Commands are grouped so that levels can be synchronized between axes
        # and referenced with a single colorbar... but for contour and other legend
        # entries only the unique labels and handle properties matter. So re-group
        # here into objects with unique labels by the rows and columns they span.
        # WARNING: Must record rows and columns here instead of during iteration
        # over legends and colorbars because hidden panels will change index.
        if not obj or not label:
            continue
        if command in ('contourf', 'pcolormesh'):
            key = (name, method, command, cmap, guide, label)
        else:
            key = (command, color, guide, label)
        rows = [n for ax in axs for n in ax._get_topmost_axes()._range_subplotspec('y')]
        cols = [n for ax in axs for n in ax._get_topmost_axes()._range_subplotspec('x')]
        objs = axs_objs.setdefault(key, [])
        objs.append((axs, rows, cols, obj, kw_guide))

    # Add colorbar and legend objects
    # TODO: Should support colorbars spanning multiple columns or rows in the
    # center of the gridspec in addition to figure edges.
    for key, objs in axs_objs.items():
        *_, guide, label = key
        axs, rows, cols, objs, kws_guide = zip(*objs)
        kw_guide = {key: val for kw in kws_guide for key, val in kw.items()}
        if guide == 'colorbar':
            hori, vert, default = hcolorbar, vcolorbar, dcolorbar
        else:
            hori, vert, default = hlegend, vlegend, dlegend
        axs = [ax for iaxs in axs for ax in iaxs]
        rows = set(n for row in rows for n in row)
        cols = set(n for col in cols for n in col)
        if len(rows) != 1 and len(cols) != 1:
            src = fig
            loc = default
        elif len(rows) == 1:  # single row
            loc = hori
            if loc[0] == 'l':
                src = fig if min(cols) == 0 else axs[0]
            elif loc[0] == 'r':
                src = fig if max(cols) == ncols - 1 else axs[-1]
            elif loc[0] == 't':
                src = fig if min(rows) == 0 else axs[len(axs) // 2]
            elif loc[0] == 'b':
                src = fig if max(rows) == nrows - 1 else axs[len(axs) // 2]
            else:
                raise ValueError(f'Invalid location {loc!r}.')
        else:  # single column
            loc = vert
            if loc[0] == 't':
                src = fig if min(rows) == 0 else axs[0]
            elif loc[0] == 'b':
                src = fig if max(rows) == nrows - 1 else axs[-1]
            elif loc[0] == 'l':
                src = fig if min(cols) == 0 else axs[len(axs) // 2]
            elif loc[0] == 'r':
                src = fig if max(cols) == ncols - 1 else axs[len(axs) // 2]
            else:
                raise ValueError(f'Invalid location {loc!r}.')
        if src is not fig:
            pass
        elif loc[0] in 'lr':
            kw_guide['rows'] = (min(rows) + 1, max(rows) + 1)
        else:
            kw_guide['cols'] = (min(cols) + 1, max(cols) + 1)
        if guide == 'colorbar':
            result = src.colorbar(objs[0], label=label, loc=loc, **kw_guide)
        else:
            result = src.legend(objs[0], label, loc=loc, queue=True, **kw_guide)

    # Format the axes and optionally save
    # NOTE: Here default labels are overwritten with non-none 'rowlabels' or
    # 'collabels', and the file name can be overwritten with 'save'.
    kw = {}
    custom = {'rowlabels': rowlabels, 'collabels': collabels}
    default = dict(zip(('rowlabels', 'collabels'), gridspecs))
    for (key, clabels), (_, dlabels) in zip(custom.items(), default.items()):
        nlabels = nrows if key == 'rowlabels' else ncols
        clabels = clabels or [None] * nlabels
        dlabels = dlabels or [None] * nlabels
        if len(dlabels) != nlabels or len(clabels) != nlabels:
            raise RuntimeError(f'Expected {nlabels} labels but got {len(dlabels)} and {len(clabels)}.')  # noqa: E501
        kw[key] = [clab or dlab for clab, dlab in zip(clabels, dlabels)]
    fig.format(figtitle=figtitle, **kw)
    if standardize:
        for (axis, _), axes in axs_units.items():
            lims = [getattr(ax, f'get_{axis}lim')() for ax in axes]
            span = max(abs(lim[1] - lim[0]) for lim in lims)
            avgs = [0.5 * (lim[0] + lim[1]) for lim in lims]
            lims = [(avg - 0.5 * span, avg + 0.5 * span) for avg in avgs]
            for ax, lim in zip(axes, lims):
                getattr(ax, f'set_{axis}lim')(lim)
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
