#!/usr/bin/env python3
"""
Utilities for reducing coupled model data coordinates.
"""
import functools
import re

import cftime
import climopy as climo  # noqa: F401
import numpy as np
import pandas as pd
import xarray as xr
from climopy import var, ureg, vreg  # noqa: F401
from scipy import stats
from icecream import ic  # noqa: F401

from .specs import ORDER_LOGICAL, INSTITUTE_LABELS

__all__ = ['reduce_facets', 'reduce_general']

# Reduction defaults
# NOTE: Default 'style' depends on styles present and may be overwritten, and default
# 'time' can be overwritten by None (see below). See also PATHS_IGNORE in specs.py.
REDUCE_DEFAULTS = {
    'experiment': 'picontrol',  # possibly overwritten
    'ensemble': 'flagship',
    'period': 'ann',
    'time': 'avg',
    'source': 'eraint',
    'style': 'monthly',  # possibly overwritten
    'start': 0,
    'stop': 150,
    'region': 'globe',
    'remove': 'average',
    'detrend': 'xy',
    'correct': 'r',
}

# Truncation region presets
# NOTE: WPG and ENSO regions defined in https://doi.org/10.1175/JCLI-D-12-00344.1 and
# https://doi.org/10.1038/s41598-021-99738-3, tropical ratio and feedback regions in
# https://doi.org/10.1175/JCLI-D-18-0843.1 and https://doi.org/10.1175/JCLI-D-17-0087.1.
# 'wpac': {'lat_lim': (-15, 15), 'lon_lim': (90, 150)},  # based on own feedbacks
# 'epac': {'lat_lim': (-30, 0), 'lon_lim': (260, 290)},
# 'wpac': {'lat_lim': (-15, 15), 'lon_lim': (150, 170)},  # paper based on +4K
# 'epac': {'lat_lim': (-30, 0), 'lon_lim': (260, 280)},
AREA_REGIONS = {
    'so': {'lat_lim': (-60, -30)},  # labeled 'southern ocean'
    'se': {'lat_lim': (-60, -30)},  # labeled 'southern extratropics'
    'ne': {'lat_lim': (30, 60)},
    'sh': {'lat_lim': (-90, 0)},
    'nh': {'lat_lim': (0, 90)},
    'trop': {'lat_lim': (-30, 30)},
    'tpac': {'lat_lim': (-30, 30), 'lon_lim': (120, 280)},
    'pool': {'lat_lim': (-30, 30), 'lon_lim': (50, 200)},
    'wpac': {'lat_lim': (-15, 15), 'lon_lim': (90, 150)},  # slightly reduced bounds
    'epac': {'lat_lim': (-30, 0), 'lon_lim': (260, 290)},
    'nina': {'lat_lim': (0, 10), 'lon_lim': (130, 150)},
    'nino': {'lat_lim': (-5, 5), 'lon_lim': (190, 240)},
    'nino3': {'lat_lim': (-5, 5), 'lon_lim': (210, 270)},
    'nino4': {'lat_lim': (-5, 5), 'lon_lim': (160, 210)},
}


def _get_weights(data, dim=None):
    """
    Return weights sorting by unique index names.

    Parameters
    ----------
    data : xarray.DataArray
        The source data.
    dim : str, optional
        The dimension for weights.

    Returns
    -------
    wgts : xarray.DataArray
        The weights.
    """
    # TODO: Implement institute weightings in 'parse_specs' by adding 'weight=True'
    # to the 'other' kwarg group whenever institute='weight' is passed.
    dim = dim or 'facets'
    coord = data.coords[dim]
    index = data.indexes[dim]
    wgts = xr.ones_like(coord, dtype=float)
    ignore = list({'model', 'ensemble'} & set(index.names))
    groups = coord.reset_index(ignore, drop=True)
    groups = groups.coords[dim]  # values on data array unchanged
    wgts = wgts.groupby(groups) / wgts.groupby(groups).sum()
    return wgts


def _get_regression(data0, data1, weight=False, **kwargs):
    """
    Return regression along facets possibly weighted by model count.

    Parameters
    ----------
    data0 : xarray.DataArray
        The data used to build the composite.
    data1 : xarray.DataArray
        The data being composited.
    weight : bool, optional
        Whether to weight by model count.
    **kwargs
        Passed to `regress_dims`.

    Returns
    -------
    slope, slope_lower, slope_upper : xarray.DataArray
        The regression estimate and uncertainty bounds.
    fit, fit_lower, fit_upper, rss : xarray.DataArray, optional
        The least squares fit and residual root sum of squares.
    rsq, dof : xarray.DataArray
        The variance explained and degrees of freedom.
    """
    # NOTE: This uses N - 2 degrees of freedom for residual variance, has significant
    # effect. See: https://en.wikipedia.org/wiki/Errors_and_residuals#Regressions
    from observed.arrays import regress_dims
    dim = kwargs.setdefault('dim', 'facets')
    kwargs.update(manual=True, correct=False, nobnds=False)
    if not weight:
        wgts = xr.ones_like(data0.coords[dim], dtype=float)
    elif dim == 'facets':
        wgts = _get_weights(data0, dim=dim)
    else:
        raise TypeError(f'Invalid option {weight=} for regression dimension {dim!r}.')
    result = regress_dims(data0, data1, weights=wgts, **kwargs)
    slope, slope_lower, slope_upper, *fits, rsq, dof = result
    if fits:  # i.e. nofit=False
        rss = wgts * (data1 - fits[0]) ** 2
        rss = np.sqrt(rss.sum(dim, skipna=True) / (wgts.sum() - 2))
        fits = (*fits, rss)
    return slope, slope_lower, slope_upper, *fits, rsq, dof


def _get_composite(data0, data1, pctile=None, dim='facets'):
    """
    Return low and high composite components of `data1` based on `data0`.

    Parameters
    ----------
    data0 : xarray.DataArray
        The data used to build the composite.
    data1 : xarray.DataArray
        The data being composited.
    pctile : float, default: 33
        The percentile threshold.

    Returns
    -------
    data_lo, data_hi : xarray.DataArray
        The composite components.
    """
    # NOTE: This can be used to e.g. composite the temperature pattern response on
    # models with high vs. low global average feedbacks. So far underitilized.
    thresh = 33 if pctile is None or pctile is True else pctile
    data0, data1 = xr.broadcast(data0, data1)
    comp_lo = np.nanpercentile(data0, thresh)
    comp_hi = np.nanpercentile(data0, 100 - thresh)
    mask_lo, = np.where(data0 <= comp_lo)
    mask_hi, = np.where(data0 >= comp_hi)
    data_hi = data1.isel({dim: mask_hi})
    data_lo = data1.isel({dim: mask_lo})
    with np.errstate(all='ignore'):
        data_hi = data_hi.mean(dim, keep_attrs=True)
        data_lo = data_lo.mean(dim, keep_attrs=True)
    data_hi = data_hi.climo.quantify()
    data_lo = data_lo.climo.quantify()
    return data_lo, data_hi


def _get_datetime(data, time=None, season=None, month=None):
    """
    Reduce the time coordinate of the data using variety of operators.

    Parameters
    ----------
    data : xarray.DataArray or xarray.Dataset
        The input data.
    time, season, month : optional
        The time coordinate or ``'avg'`` instruction.

    Returns
    -------
    data : xarray.Dataset or xarray.DataArray
        The reduced data.
    """
    if 'time' not in data.dims:
        return data
    if season is not None:
        seasons = data.time.dt.season.str.lower()
        data = data.isel(time=(seasons == season.lower()))
    elif isinstance(month, str):
        months = data.time.dt.strftime('%b').str.lower()
        data = data.isel(time=(months == month.lower()))
    elif month is not None:
        months = data.time.dt.month
        data = data.isel(time=(months == month))
    elif not isinstance(time, str) and time is not None and time is not False:
        time = time  # NOTE: above respects user-input 'None'
        data = data.sel(time=time, method='nearest')
    elif isinstance(time, str) and time != 'avg':
        time = cftime.datetime.strptime(time, '%Y-%m-%d')  # cftime 1.6.2
        data = data.sel(time=time, method='nearest')
    elif isinstance(time, str) and time == 'avg':
        days = data.time.dt.days_in_month.astype(data.dtype)
        with xr.set_options(keep_attrs=True):  # average over entire record
            data = (data * days).sum('time', skipna=False) / days.sum()
    elif time is not None:
        raise ValueError(f'Unknown time reduction method {time!r}.')
    return data


def _get_filters(facets, project=None, institute=None):
    """
    Return plot labels and facet filter for the project indicator.

    Parameters
    ----------
    facets : numpy.ndarray
        The facets. Used to search for models from the same institute in other projects.
    project : str, default: 'cmip'
        The selection. Values should start with ``'cmip'``. No integer ending indicates
        all cmip5 and cmip6 models, ``5`` (``6``) indicates just cmip5 (cmip6) models,
        ``56`` (``65``) indicates cmip5 (cmip6) models filtered to those from the same
        institutes as cmip6 (cmip5), and ``55`` (``66``) indicates institutes found
        only in cmip5 (cmip6). Note two-digit integers can be combined, e.g. ``5665``.
    institute : str, default: None
        The selection. If ``'avg'`` then simply return the averaging function and
        `reduce_facets` will delay application to end of other selections. Otherwise
        this should be a specific institute name or ``'flagship'`` to select the
        curated "flagship" models from each institute, defined as the last models
        in the institute definition lists from the `cmip_data.facets` database.

    Returns
    -------
    project : callable
        A `facets` filter function.
    institute : callable
        A `facets` filter function.
    """
    # WARNING: Critical to assign name to filter so that _parse_specs can detect
    # differences between row and column specs at given subplot entry.
    # NOTE: Averages across a given institution are done using the default method='avg'
    # with e.g. institute='GFDL'. Averages across each institute followed by reductions
    # along result are instead done with institute='avg' (see _institute_average)
    def _institute_other(key, num=None, facets=None):
        num = str(num)
        both = len(set(num)) == 2
        other = '6' if num[0] == '5' else '5'
        exists = any(obj[1] == key[1] for obj in facets if obj[0][-1] == other)
        if num[0] != key[0][-1]:  # wrong project
            return False
        else:  # whether other project exists
            return both == exists
    def _institute_average(data):  # noqa: E306
        index = data.indexes.get('facets', None)
        if index is None:  # missing index
            return data
        if 'model' not in index.names:  # already averaged or selected
            return data
        index = index.droplevel('model')  # preserve institute experiment
        group = xr.DataArray(index, name='facets', dims='facets', attrs=data.facets.attrs)  # noqa: E501
        data = data.groupby(group).mean(skipna=False, keep_attrs=True)
        facets = data.indexes['facets']  # WARNING: xarray bug drops level names
        facets.names = index.names
        facets = xr.DataArray(facets, dims='facets', attrs=group.attrs)
        return data.climo.replace_coords(facets=facets)
    inst_to_model = {facet[-2]: facet[-1] for facet in facets if len(facet) >= 2}
    name_to_inst = {label: key for key, label in INSTITUTE_LABELS.items()}
    if not institute:
        inst = lambda key: True  # noqa: U100
    elif institute == 'avg':
        inst = _institute_average
    elif institute == 'flagship':
        inst = lambda key: len(key) < 2 or key[-1] == inst_to_model.get(key[-2], None)
    elif any(item == institute for pair in name_to_inst.items() for item in pair):  # noqa: E501
        inst = lambda key: key[-2] == name_to_inst.get(institute, institute)
    else:
        raise ValueError(f'Invalid institute name {institute!r}.')
    project = (project or 'cmip').lower()
    number = re.sub(r'\Acmip', '', project)
    digits = max(1, len(number))
    projs = []  # permit e.g. cmip6556 or inst6556
    for idx in range(0, digits, 2):
        num = number[idx:idx + 2]
        if not num:
            proj = lambda key: True  # noqa: U100
        elif num in ('5', '6'):
            proj = lambda key: key[0][-1] == num
        elif num in ('65', '66', '56', '55'):
            proj = functools.partial(_institute_other, num=num, facets=facets)
        else:
            raise ValueError(f'Invalid project indictor {project!r}.')
        projs.append(proj)
    proj = lambda key: any(proj(key) for proj in projs)
    proj.name = project  # WARNING: critical for process_data() detection of 'cmip'
    inst.name = institute  # WARNING: critical for reduce_general() detection of 'avg'
    return proj, inst


def _reduce_data(data, method, dim=None, pctile=None, std=None, preserve=True):
    """
    Reduce individual data array using arbitrary method.

    Parameters
    ----------
    data : xarray.DataArray
        The input data.
    method : str
        The reduction method (see `reduce_facets`).
    dim : str, optional
        The reduction dimension. Default is the first dimension.

    Other Parameters
    ----------------
    pctile : float or sequence, optional
        The percentile range or bounds for related methods. If ``True`` then defaults
        of ``80`` and ``50`` are used for ``avg|med`` and ``pctile``. If two values
        passed e.g. ``(80, 100)`` both `shade` and `fade` keywords are used.
    std : float or sequence, optional
        The standard deviation multiple for related methods. If ``True`` then defaults
        of ``3`` and ``1`` are used for ``avg|med`` and ``std``. If two values
        passed e.g. ``(1, 3)`` both `shade` and `fade` keywords are used.
    preserve : bool, optional
        Whether to apply ``avg|med`` operations immediately or preserve the distribution
        until plotting using ``shade|fade`` keyword arguments. If ``False`` this will
        cause ``process_data()`` to show uncertainty for differences of selections on
        each model e.g. ``experiment='abrupt4xco2-picontrol'`` intead of ensembles.

    Returns
    -------
    result : xarray.DataArray
        The reduced data.
    defaults : dict
        The default attributes and command keyword args.
    """
    # NOTE: Here `pctile` keyword argument is shared between inter-model percentile
    # differences and composites of a second array based on values in the first array.
    # NOTE: Here we put 'variance' on additional DataArray dimension since variance
    # of gaussian random variables is linearly additive, so can allow process_data()
    # to carry out any operations (e.g. project=cmip6-cmip5) and generate 'shadedata'
    # and 'fadedata' arrays from the input arguments. Note resulting percentiles will
    # only be an approximation to some actual Monte Carlo sampled difference of
    # t-distributions. See: https://en.wikipedia.org/wiki/Variance#Propertieswill
    pctile = None if pctile is False else pctile
    pctile = True if method == 'pctile' and pctile is None else pctile
    pctile = (80, 80)[method == 'pctile'] if pctile is True else pctile
    pctile = np.atleast_1d(pctile) if pctile is not None else pctile
    std = None if std is False else std
    std = True if method == 'std' and std is None else std
    std = (1, 1)[method == 'std'] if std is True else std
    std = np.atleast_1d(std) if std is not None else std
    defaults = {}
    short = long = None
    name = data.name
    dim = dim or data.dims[0]
    kwargs = {'dim': dim, 'skipna': False}
    if method == 'dist':  # bars or boxes
        if data.ndim > 1:
            raise ValueError(f'Invalid dimensionality {data.ndim!r} for distribution.')
        result = data[~data.isnull()]
        name = None
    elif method == 'std':
        with xr.set_options(keep_attrs=True):  # note name is already kept
            std = std.item()
            result = std * data.std(**kwargs)
        short = f'{data.short_name} spread'
        long = f'{data.long_name} standard deviation'
    elif method == 'var':
        with xr.set_options(keep_attrs=True):  # note name is already kept
            result = data.var(**kwargs)
        data.attrs['units'] = f'({data.units})^2'
        short = f'{data.short_name} variance'
        long = f'{data.long_name} variance'
    elif method == 'skew':  # see: https://stackoverflow.com/a/71149901/4970632
        with xr.set_options(keep_attrs=True):
            result = data.reduce(func=stats.skew, **kwargs)
        data.attrs['units'] = ''
        short = f'{data.short_name} skewness'
        long = f'{data.long_name} skewness'
    elif method == 'kurt':  # see: https://stackoverflow.com/a/71149901/4970632
        with xr.set_options(keep_attrs=True):
            result = data.reduce(func=stats.kurtosis, **kwargs)
        data.attrs['units'] = ''
        short = f'{data.short_name} kurtosis'
        long = f'{data.long_name} kurtosis'
    elif method == 'pctile':
        with xr.set_options(keep_attrs=True):  # note name is already kept
            nums = (0.5 - 0.005 * pctile.item(), 0.5 + 0.005 * pctile.item())
            result = data.quantile(nums[1], **kwargs) - data.quantile(nums[0], **kwargs)
        short = f'{data.short_name} spread'
        long = f'{data.long_name} percentile range'
    elif method == 'avg' or method == 'med':
        key = 'mean' if method == 'avg' else 'median'
        cmd = getattr(data, key)
        result = cmd(**kwargs, keep_attrs=True)
        descrip = 'mean' if method == 'avg' else 'median'
        short = f'{descrip} {data.short_name}'  # only if no range included
        long = f'{descrip} {data.long_name}'
        if result.ndim == 1 and any(nums is not None for nums in (pctile, std)):
            nums = pctile if pctile is not None else std
            which = 'pctile' if pctile is not None else 'std'
            shade, fade = nums if nums.size == 2 else (nums.item(), None)
            defaults.update({f'shade{which}s': shade, f'fade{which}s': fade})
            defaults.update({'fadealpha': 0.15, 'shadealpha': 0.3})
            if preserve:  # preserve distributions for now
                result = data
                defaults.update({f'{descrip}s': True})
            else:  # reduce and record variance
                result = xr.concat((result, data.var(ddof=1, **kwargs)), dim='sigma')
                defaults['dof'] = data.sizes[dim] - 1
    else:
        raise ValueError(f'Invalid single-variable method {method}.')
    defaults.update({'name': name, 'short_name': short, 'long_name': long})
    return result, defaults


def _reduce_datas(data0, data1, method, dim=None, pctile=None, std=None, invert=None):
    """
    Reduce pair of data arrays using arbitrary method.

    Parameters
    ----------
    data0, data1 : xarray.DataArray
        The input data.
    method : str
        The reduction method (see `reduce_facets`).
    dim : str, optional
        The reduction dimension. Default is the first shared dimension.

    Other Parameters
    ----------------
    pctile : float or sequence, optional
        The percentile range or bounds for ``cov|proj|slope``. If ``True`` then default
        of ``95`` is used. If two values passed then both `shade` and `fade` are used.
    std : float or sequence, optional
        The standard deviation multiple for ``cov|proj|slope``. If ``True`` then default
        of ``3`` is used. If two values passed then both `shade` and `fade` are used.
    invert : bool, optional
        Whether to invert the independent and dependent ``diff|proj|slope`` variables.
        This has no effect on the symmetric operations ``cov|corr|rsq``.

    Returns
    -------
    result : xarray.DataArray
        The reduced data.
    defaults : dict
        The default attributes and command keyword args.
    """
    # NOTE: Normalization and anomalies with respect to global average as in climopy
    # accessor.get() are supported in `reduce_facets`.
    # NOTE: Here we put 'variance' on additional DataArray dimension since variance
    # of gaussian random variables is linearly additive, so can allow process_data()
    # to carry out any operations (e.g. project=cmip6-cmip5) and generate 'shadedata'
    # and 'fadedata' arrays from the input arguments. Note resulting percentiles will
    # only be an approximation to some actual Monte Carlo sampled difference of
    # t-distributions. See: https://en.wikipedia.org/wiki/Variance#Propertieswill
    from observed.arrays import regress_dims
    pctile = None if pctile is False else pctile
    pctile = 80 if pctile is True else pctile
    pctile = np.atleast_1d(pctile) if pctile is not None else pctile
    std = None if std is False else std
    std = 1 if std is True else std
    std = np.atleast_1d(std) if std is not None else std
    data0, data1 = (data1, data0) if invert else (data0, data1)
    defaults = {}
    shared = [dim for dim in data0.dims if dim in data1.dims]
    short = long = None
    name = f'{data0.name}|{data1.name}'  # NOTE: needed for _combine_commands labels
    dim = dim or shared[0]
    if dim == 'area':  # special consideration
        short_prefix, long_prefix = 'spatial ', f'{data1.short_name} spatial '
    else:  # TODO: revisit this and permit customization
        short_prefix, long_prefix = '', f'{data1.short_name} '
    if method == 'dist':  # scatter or bars
        if max(data0.ndim, data1.ndim) > 1:
            raise ValueError(f'Invalid dimensionality {data0.ndim} x {data1.ndim} for distribution.')  # noqa: E501
        result0, result1 = data0[~data0.isnull()], data1[~data1.isnull()]
        result0, result1 = xr.align(result0, result1)  # intersection-style broadcast
        result = (result0, result1)
        name = None
    elif method == 'diff':  # composite difference along first arrays
        result0, result1 = _get_composite(data0, data1, pctile=pctile, dim=dim)
        result = result1 - result0
        result = result.climo.dequantify()
        result.attrs['units'] = data1.units
        short = f'{data1.short_name} composite difference'
        long = f'{data0.long_name}-composite {data1.long_name} difference'
    elif method in ('corr', 'rsq'):
        manual = dim == 'facets'
        kw_regress = dict(stat='corr', nobnds=True, manual=manual)
        result = regress_dims(data0, data1, dim, **kw_regress)
        if method == 'corr':  # correlation coefficient
            result = result.climo.to_units('dimensionless').climo.dequantify()
            result.attrs['units'] = ''
            short = f'{short_prefix}correlation'
            long = f'{long_prefix}correlation coefficient'
        else:  # variance explained
            result = (result ** 2).climo.to_units('percent').climo.dequantify()
            result.attrs['units'] = '%'
            short = f'{short_prefix}variance explained'
            long = f'{long_prefix}variance explained'
    elif method in ('cov', 'proj', 'slope'):
        nobnds = all(nums is None for nums in (pctile, std))
        manual = dim == 'facets'
        kw_regress = dict(stat=method, nobnds=nobnds, manual=manual)
        result, *bnds = regress_dims(data0, data1, dim, **kw_regress)
        if method == 'cov':
            result.attrs['units'] = f'{data1.units} {data0.units}'
            short = f'{short_prefix}covariance'
            long = f'{long_prefix}covariance'
        elif method == 'proj':
            result.attrs['units'] = data1.units
            short = f'{short_prefix}projection'
            long = f'{long_prefix}projection'
        else:
            result.attrs['units'] = f'{data1.units} / ({data0.units})'
            short = f'{short_prefix}regression coefficient'
            long = f'{long_prefix}regression coefficient'
        if not nobnds and result.ndim == 1:
            key = 'pctile' if pctile is not None else 'std'
            nums = pctile if pctile is not None else std
            sigma = 0.5 * (bnds[1] - bnds[0])
            result = xr.concat((result, sigma ** 2), dim='sigma')  # see process_data()
            shade, fade = nums if nums.size == 2 else (nums.item(), None)
            defaults['dof'] = data1.sizes[dim] - 2
            defaults.update({f'shade{key}s': shade, f'fade{key}s': fade})
            defaults.update({'shadealpha': 0.25, 'fadealpha': 0.15})
    else:
        raise ValueError(f'Invalid double-variable method {method}')
    defaults.update({'name': name, 'short_name': short, 'long_name': long})
    return result, defaults


def reduce_facets(
    *datas, method=None, pctile=None, std=None, preserve=None, invert=None, verbose=False,  # noqa: E501
):
    """
    Reduce along the model facets coordinate using an arbitrary method.

    Parameters
    ----------
    *datas : xarray.DataArray
        The data array(s).
    method : str, optional
        The reduction method. Here ``dist`` retains the facets dimension for e.g. bar
        and scatter plots (and their multiples), ``avg|med|std|pctile`` reduce the
        facets dimension for a single input argument, and ``corr|diff|proj|slope``
        reduce the facets dimension for two input arguments. Can also end string
        with ``_anom`` or ``_norm`` to take global anomaly or use normalization.
    pctile, std, preserve : optional
        Passed to `_reduce_data`.
    pctile, std, invert : optional
        Passed to `_reduce_datas`.
    verbose : bool, optional
        Whether to print extra information.

    Returns
    -------
    args : tuple
        The output plotting arrays.
    method : str
        The method used to reduce the data.
    defaults : dict
        The default plotting arguments.
    """
    # Apply single or double reduction methods
    # NOTE: This supports on-the-fly anomalies and normalization. Should eventually
    # move this stuff to climopy (already implemented in accessor getter).
    ndim = max(data.ndim for data in datas)
    datas = tuple(data.copy() for data in datas)  # e.g. for distribution updates
    default = 'dist' if ndim == 1 else 'avg' if len(datas) == 1 else 'slope'
    method, *options = (method or default).split('_')
    anomaly = 'anom' in options
    normalize = 'norm' in options
    kwargs = dict(dim='facets', std=std, pctile=pctile)
    if len(datas) == 1:
        data, defaults = _reduce_data(*datas, method, preserve=preserve, **kwargs)
    elif len(datas) == 2:
        data, defaults = _reduce_datas(*datas, method, invert=invert, **kwargs)
    else:
        raise ValueError(f'Unexpected argument count {len(datas)}.')
    if anomaly:
        if 'lon' not in data.coords and 'lat' not in data.coords:
            raise NotImplementedError('Anomaly methods require spatial coordinates.')
        with xr.set_options(keep_attrs=True):
            data = data - data.climo.average('area')
    if normalize:
        if 'lon' not in data.coords and 'lat' not in data.coords:
            raise NotImplementedError('Normalized methods require spatial coordinates.')
        with xr.set_options(keep_attrs=True):
            data = data / data.climo.average('area')
        if units := data.attrs.get('units', ''):
            data.attrs['units'] = f'{units} / ({units})'

    # Standardize and possibly print information
    # NOTE: Considered re-applying coordinates here but better instead to relegate
    # to process_data so that operators can be retained more easily.
    keys = ('facets', 'time', 'plev', 'lat', 'lon')  # coordinate sorting order
    args = tuple(data) if isinstance(data, tuple) else (data,)
    args = [arg.transpose(..., *(key for key in keys if key in arg.sizes)) for arg in args]  # noqa: E501
    dependent = args[-1]
    if name := defaults.pop('name', None):
        dependent.name = name
    if long := defaults.pop('long_name', None):
        dependent.attrs['long_name'] = long
    if short := defaults.pop('short_name', None):
        dependent.attrs['short_name'] = short
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


def reduce_general(data, attrs=None, **kwargs):
    """
    Carry out arbitrary reduction of the given dataset variables.

    Parameters
    ----------
    data : xarray.Dataset or xarray.DataArray
        The dataset or data array.
    attrs : dict, optional
        The optional attribute overrides.
    **kwargs
        The reduction selections. Requires a `name`.

    Returns
    -------
    data : xarray.DataArray
        The data array.
    """
    # Apply facet reductions and grouped averages
    # TODO: Replace _get_filters and _parse_institute with single function that
    # does generalized project and institute selections (e.g. 'cmip65' is already
    # an institute-based selection so makes sense to put in institute function).
    institute = kwargs.pop('institute', None)  # apply after end
    project = kwargs.pop('project', None)  # apply after end
    index = data.indexes.get('facets', pd.Index([]))
    proj = inst = None  # filter functions
    if 'project' in index.names and 'institute' in index.names:
        facets = data.facets.values
        label = data.facets.attrs.get('long_name') or 'source facets'
        proj, inst = _get_filters(facets, project=project, institute=institute)
        if project is not None:
            facets = list(filter(proj, facets))
            data = data.sel(facets=facets)
        if institute is not None and 'institute' not in label and inst.name != 'avg':
            facets = list(filter(inst, facets))
            data = data.sel(facets=facets)

    # Apply time reductions and grouped averages
    # TODO: Replace 'average_periods' with this and normalize by annual temperature.
    # Also should add generalized no-op instruction for leaving coordinates alone.
    # NOTE: The new climopy cell duration calculation will auto-detect monthly and
    # yearly data, but not yet done, so use explicit days-per-month weights for now.
    season = kwargs.pop('season', None)
    month = kwargs.pop('month', None)
    time = kwargs.get('time', 'avg')  # then get average later on
    if 'time' in data.dims:
        if season is not None:
            seasons = data.time.dt.season.str.lower()
            data = data.isel(time=(seasons == season.lower()))
        elif isinstance(month, str):
            months = data.time.dt.strftime('%b').str.lower()
            data = data.isel(time=(months == month.lower()))
        elif month is not None:
            months = data.time.dt.month
            data = data.isel(time=(months == month))
        elif isinstance(time, str) and time != 'avg':
            time = cftime.datetime.strptime(time, '%Y-%m-%d')  # cftime 1.6.2
            data = data.sel(time=time, method='nearest')
        elif not isinstance(time, str) and time is not None and time is not False:
            time = time  # should already be datetime
            data = data.sel(time=time, method='nearest')

    # Iterate over data arrays
    # NOTE: Delay application of defaults until here so that we only include default
    # selections in automatically-generated labels if user explicitly passed them.
    # WARNING: Sometimes multi-index reductions can eliminate previously valid
    # coords, so critical to iterate one-by-one and validate selections each time.
    order = list(ORDER_LOGICAL)
    sorter = lambda item: order.index(item[0]) if item[0] in order else len(order)
    results = []
    if is_dataset := isinstance(data, xr.Dataset):
        datas = data.data_vars.values()
    else:  # iterate over arrays
        datas = (data,)
    for data in datas:
        # Apply default reduce instructions
        # NOTE: None is automatically replaced with default values (e.g. period=None)
        # except for time=None used as placeholder to prevent time averaging.
        ikwargs, defaults = kwargs.copy(), REDUCE_DEFAULTS.copy()
        version = data.indexes.get('version', pd.Index([]))
        abrupt = (ikwargs.get('experiment', None) or 'abrupt4xco2') == 'abrupt4xco2'
        external = ikwargs.get('source', None) in ('zelinka', 'geoffroy', 'forster')
        experiment = region = period = None  # see below
        if version.size > 1:
            defaults['experiment'] = 'abrupt4xco2'
        if 'period' in version.names:
            defaults['start'], defaults['period'] = 'mar', '23yr'
        elif abrupt or external:
            defaults['style'] = 'annual'
        if 'startstop' in ikwargs:
            ikwargs['start'], ikwargs['stop'] = ikwargs.pop('startstop')
        for dim, value in defaults.items():
            if dim in ikwargs and ikwargs[dim] is None and dim != 'time':
                ikwargs[dim] = value  # typically use 'None' as default placeholder
            else:  # for 'time' use 'None' to bypass operation
                ikwargs.setdefault(dim, value)
        if version.size > 1:
            experiment, region, period = ikwargs['experiment'], ikwargs['region'], ikwargs['period']  # noqa: E501
        if period and 'yr' not in period and experiment == 'picontrol':
            ikwargs['start'], ikwargs['stop'] = 0, 150  # overwrite defaults
        if period and period[0] == 'a' and period != 'ann':
            ikwargs['period'] = period[1:] if experiment == 'abrupt4xco2' else 'ann'
        if region and region[0] == 'a':  # abrupt-only region
            ikwargs['region'] = region[1:] if experiment == 'picontrol' else 'globe'
        if data.name in ('tpat', 'tstd', 'tdev', 'tabs'):  # others undefined
            ikwargs['region'] = 'globe'

        # Iterate over reductions
        # NOTE: This silently skips dummy selections (e.g. area=None) that may be needed
        # to prevent _parse_specs from merging e.g. average and non-average selections.
        result = data
        spatials = ('area', 'start', 'stop', 'experiment')
        for dim, value in sorted(ikwargs.items(), key=sorter):
            sizes = [*result.sizes, 'area', 'volume', 'spatial', 'time', 'month', 'season']  # noqa: E501
            names = [key for idx in result.indexes.values() for key in idx.names]
            quants = result.coords.keys() - set(names) - {'time', 'month', 'season'}
            if value is None or dim not in sizes and dim not in names:
                continue
            if dim in spatials and ikwargs.get('spatial', None) is not None:
                continue
            if dim == 'area' and not result.sizes.keys() & {'lon', 'lat'}:
                continue
            if dim == 'volume' and not result.sizes.keys() & {'lon', 'lat', 'plev'}:
                continue
            if dim == 'area' and value != 'avg':  # average over truncated region
                result, value = result.climo.truncate(AREA_REGIONS[value]), 'avg'
            if dim in quants and not isinstance(value, (str, tuple)):
                value = ureg.Quantity(value, result.coords[dim].climo.units)
            if dim in ('time', 'month', 'season'):  # time selections
                result = _get_datetime(result, **{dim: value})
            else:  # non-time selections
                result = result.climo.reduce({dim: value}, method='interp').squeeze()
            if dim not in result.coords:
                result.coords[dim] = xr.DataArray(value).climo.dequantify()
        result.name = data.name
        result.attrs.update(data.attrs)
        result.attrs.update(attrs or {})
        results.append(result)

    # Re-combine and return
    # NOTE: Here drop then restore project to simplify general_plot() annotations and
    # permit aligning linear operations along institutes from different projects.
    if is_dataset:
        result = xr.Dataset({result.name: result for result in results})
    else:  # singular array
        result = results[0]
    result = inst(result) if inst and inst.name == 'avg' else result
    index = result.indexes.get('facets', pd.Index([]))
    names = set(index.names)  # facet names
    if 'project' in index.names and len(set(result.project.values)) == 1:
        proj, names = result.project.values[0], names - {'project'}
        result = result.reset_index('project', drop=True)
        result.coords['project'] = proj
    if 'facets' in result.dims and len(names) == 1:
        coord = result.coords['facets']
        index = pd.MultiIndex.from_arrays((coord.values,), names=(names.pop(),))
        result = result.assign_coords({'facets': index})
    return result
