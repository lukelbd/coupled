#!/usr/bin/env python3
"""
Utilities for reducing coupled model data coordinates.
"""
import functools
import math
import re
import warnings

import cftime
import climopy as climo  # noqa: F401
import numpy as np
import pandas as pd
import xarray as xr
from climopy import var, ureg, vreg  # noqa: F401
from scipy import stats
from icecream import ic  # noqa: F401

from .specs import ORDER_LOGICAL, INSTITUTE_LABELS, _pop_kwargs

__all__ = ['reduce_time', 'reduce_facets', 'reduce_general']

# Coordinate reduction defaults
# NOTE: Default 'style' depends on styles present and may be overwritten, and default
# 'time' can be overwritten by None (see below). See also PATHS_IGNORE in specs.py.
GENERAL_DEFAULTS = {
    'statistic': 'slope',
    'experiment': 'picontrol',  # possibly overwritten
    'ensemble': 'flagship',
    'time': 'avg',
    'source': 'eraint',
    'style': 'monthly',  # possibly overwritten
    'start': 0,
    'stop': 150,
    'region': 'globe',
    'area': None,
}
RESPONSE_DEFAULTS = {
    'period': 'full',
    'initial': 'jan',
    'style': 'annual',
    'remove': 'climate',
    'detrend': '',
    'correct': 'r',
}
SCALAR_DEFAULTS = {
    'period': 'full',  # includes 150-year control
    'initial': 'mar',
    'remove': 'average',
    'detrend': 'xy',
    'error': 'regression',
    'correct': 'r',
}
OBSERVED_DEFAULTS = {
    'source': 'hadcrut',
    'period': '2000-2024',
    'initial': 'jan',
    'remove': 'average',
    'detrend': 'xy',
    'correct': 'r',
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
    dim = dim or 'facets'  # xarray bug: https://github.com/pydata/xarray/issues/7695
    coord = data.coords[dim]
    index = data.indexes[dim]
    wgts = xr.ones_like(coord, dtype=float)
    wgts = wgts.drop_vars(wgts.coords.keys() & set(index.names))  # xarray bug
    keys = list({'model', 'ensemble'} & set(index.names))
    groups = coord.reset_index(keys, drop=True)
    groups = groups.coords[dim]  # values on data array unchanged
    with warnings.catch_warnings():  # xarray future warning
        warnings.simplefilter('ignore')
        groups = groups.assign_coords(facets=coord)
    groups = groups.drop_vars(groups.coords.keys() & set(index.names))  # xarray bug
    with warnings.catch_warnings():  # xarray future warning
        warnings.simplefilter('ignore')
        grps = wgts.groupby(groups, squeeze=False)
    wgts = grps / grps.sum()
    with warnings.catch_warnings():  # xarray future warning
        warnings.simplefilter('ignore')
        wgts = wgts.assign_coords({dim: coord})
    return wgts


def _get_regression(
    data0, data1, dim=None, weight=False, standard=False, relative=False, **kwargs,
):
    """
    Return regression along facets possibly weighted by model count.

    Parameters
    ----------
    data0 : xarray.DataArray
        The data used to build the composite.
    data1 : xarray.DataArray
        The data being composited.
    dim : str, default: 'facets'
        The regression dimension.
    weight : bool, optional
        Whether to weight the regression by model count.
    standard : bool, optional
        Whether to instead return the standard error.
    relative : bool, optional
        Whether to instead return the relative error.
    **kwargs
        Passed to `regress_dims`.

    Returns
    -------
    slope, stat1, stat2 : xarray.DataArray
        The regression estimate and uncertainty bounds.
    fit, fit1, fit2, rse : xarray.DataArray, optional
        The least squares fit and regression standard error.
    rsq, dof : xarray.DataArray
        The variance explained and degrees of freedom.
    """
    # NOTE: This uses N - 2 degrees of freedom for residual variance, has significant
    # effect. See: https://en.wikipedia.org/wiki/Errors_and_residuals#Regressions
    # NOTE: Here model emergent constraint uncertainty using sigma_b * x + sigma_e
    # where sigma_b = sigma_e / sigma_x is the slope uncertainty, and this is the
    # same as sigma_b * (x + sigma_x). Can compare bars to estimate constraint. Also
    # note regression model has two degrees of freedom, so when modeling constraint
    # should sample slope and intercept uncertainty independently... probably.
    from observed.arrays import regress_dims
    dim = dim or 'facets'
    kwargs.update(manual=True, correct=False, nobnds=False)
    if standard or relative:  # required for residual sum of squares
        kwargs['nofit'] = False
    if not weight:
        wgts = xr.ones_like(data0.coords[dim], dtype=float)
    elif dim == 'facets':
        wgts = _get_weights(data0, dim=dim)
    else:
        raise TypeError(f'Invalid option {weight=} for regression dimension {dim!r}.')
    if isinstance(index := wgts.indexes.get(dim), pd.MultiIndex):
        wgts = wgts.drop_vars(wgts.coords.keys() & set(index.names))
    if isinstance(index := data1.indexes.get(dim), pd.MultiIndex):
        data1 = data1.drop_vars(data1.coords.keys() & set(index.names))
    result = regress_dims(data0, data1, dim=dim, weights=wgts, **kwargs)
    coords = {'facets': data0.coords['facets']} if dim == 'facets' else {}
    stat, stat1, stat2, *fits, rsq, dof = result
    if not standard and not relative:
        units = f'{data1.units} / ({data0.units})'
        stat.attrs['short_name'] = 'regression coefficient'
        stat.attrs['units'] = stat1.attrs['units'] = stat2.attrs['units'] = units
    if fits:  # drop_vars() is xarray bug: https://github.com/pydata/xarray/issues/7695
        fits = [fit.drop_vars(fit.coords.keys() & {'facets'}) for fit in fits]
        rss = (wgts * (data1 - fits[0]) ** 2).sum(dim, skipna=True)  # residual squares
        rse = np.sqrt(rss / (wgts.sum() - 2))  # weighted regression standard error
        fits = [fit.assign_coords(coords) for fit in fits] + [rse]
    if fits and standard:  # propagate constraint uncertainty (see notes)
        xbar = data0.weighted(wgts).mean(dim, skipna=True)
        sxx = np.sqrt(((data0 - xbar) ** 2).weighted(wgts).sum(dim, skipna=True))
        stat, stat1, stat2 = (sxx * _ for _ in (stat, stat1, stat2))
        stat.attrs['units'] = stat1.attrs['units'] = stat2.attrs['units'] = data1.units
        stat.attrs['short_name'] = 'standard error'
    if fits and relative:  # propagate constraint uncertainty (see notes)
        xbar = data0.weighted(wgts).mean(dim, skipna=True)
        sxx = np.sqrt(((data0 - xbar) ** 2).weighted(wgts).sum(dim, skipna=True))
        ybar = data1.weighted(wgts).mean(dim, skipna=True)
        syy = np.sqrt(((data1 - ybar) ** 2).weighted(wgts).sum(dim, skipna=True))
        stat, stat1, stat2 = (sxx * _ / syy for _ in (stat, stat1, stat2))
        stat.attrs['units'] = stat1.attrs['units'] = stat2.attrs['units'] = ''
        stat.attrs['short_name'] = 'relative error'
    return stat, stat1, stat2, *fits, rsq, dof


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
    # NOTE: Averages across a given institution are done using the default method 'avg'
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
        idx = data.indexes.get('facets', None)
        if idx is None:  # missing index
            return data
        if 'model' not in idx.names:  # already averaged or selected
            return data
        idx = idx.droplevel('model')  # preserve institute experiment
        group = xr.DataArray(idx, name='facets', dims='facets', attrs=data.facets.attrs)
        group = group.drop_vars('facets')  # WARNING: recent xarray bug
        result = data.groupby(group).mean(skipna=False, keep_attrs=True)
        coord = result.indexes['facets']
        coord.names = idx.names  # WARNING: older xarray bug
        coord = xr.DataArray(coord, dims='facets', attrs=group.attrs)
        with warnings.catch_warnings():  # xarray future warning
            warnings.simplefilter('ignore')
            return result.climo.replace_coords(facets=coord)
    idx = 1 + any('CMIP' in facet[0] for facet in facets if len(facet) > 0)
    inst_to_model = {tuple(facet[:idx]): facet[idx] for facet in facets if len(facet) > 2}  # noqa: E501
    name_to_inst = {label: key for key, label in INSTITUTE_LABELS.items()}
    if not institute:
        inst = lambda key: True  # noqa: U100
    elif institute == 'avg':
        inst = _institute_average
    elif institute == 'flagship':
        inst = lambda key: len(key) <= 2 or key[idx] == inst_to_model.get(key[:idx], None)  # noqa: E501
    elif any(item == institute for pair in name_to_inst.items() for item in pair):
        inst = lambda key: key[idx - 1] == name_to_inst.get(institute, institute)
    else:
        raise ValueError(f'Invalid institute name {institute!r}.')
    project = (project or 'cmip').lower()
    number = re.sub(r'\Acmip', '', project)
    digits = max(1, len(number))
    projs = []  # permit e.g. cmip6556 or inst6556
    for jdx in range(0, digits, 2):
        num = number[jdx:jdx + 2]
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


def _fill_kwargs(data, **kwargs):
    """
    Return default keyword arguments given the input data.

    Parameters
    ----------
    dataset : xarray.Dataset or xarray.DataArray
        The input dataset.
    **kwargs : optional
        The reduction instructions.

    Returns
    -------
    kwargs : dict
        The keyword arguments with defaults applied.
    """
    # Apply default reduce instructions
    # NOTE: None is automatically replaced with default values (e.g. period=None)
    # except for time=None used as placeholder to prevent time averaging.
    version = data.indexes.get('version', pd.Index([]))
    facets = data.indexes.get('facets', pd.Index([]))
    feedbacks = version.size > 1  # feedback datasets
    coupled = facets.size > 1  # coupled model dataset
    coupled1 = 'experiment' in facets.names  # general coupled model data
    coupled2 = 'experiment' in version.names  # scalar feedback model data
    external = kwargs.get('source', None) in ('zelinka', 'geoffroy', 'forster')
    response = kwargs.get('experiment', None) in (None, 'abrupt4xco2')
    experiment = region = period = None  # see below
    defaults = GENERAL_DEFAULTS.copy()
    if 'initial' in version.names:  # scalar models or observations
        defaults.update(SCALAR_DEFAULTS)
    if (coupled1 or coupled2) and (response or external):  # abrupt response
        defaults.update(RESPONSE_DEFAULTS)
    if not coupled1 and not coupled2:  # scalar observations
        defaults.update(OBSERVED_DEFAULTS)
    if coupled1 and not coupled2 and 'period' in version.names:  # outdated format
        defaults.update(period='ann')  # select coord made by average_periods()
    if coupled and feedbacks and response:
        defaults.update(initial='init', experiment='abrupt4xco2')
    if response and kwargs.get('period') and kwargs.get('startstop'):
        kwargs.pop('startstop', None)  # accidental invalid control period
    if 'startstop' in kwargs:
        kwargs['start'], kwargs['stop'] = kwargs.pop('startstop')
    for dim, value in defaults.items():
        if dim in kwargs and kwargs[dim] is None and dim != 'time':
            kwargs[dim] = value  # typically use 'None' as default placeholder
        else:  # for 'time' use 'None' to bypass operation
            kwargs.setdefault(dim, value)
    if coupled and feedbacks:
        experiment, region, period = map(kwargs.get, ('experiment', 'region', 'period'))
    if (coupled1 or coupled2) and np.unique(data.experiment.values).size == 1:
        experiment, _ = None, kwargs.pop('experiment', None)  # single-experiment data
    if coupled and feedbacks:
        start, stop, period = map(kwargs.get, ('start', 'stop', 'period'))
    if (coupled1 or coupled2) and experiment and 'control' in experiment:
        kwargs['start'], kwargs['stop'] = 0, 150  # overwrite defaults
    if response and coupled1 and period == 'early':
        kwargs['start'], kwargs['stop'] = 0, 20
    if response and coupled1 and period == 'late':
        kwargs['start'], kwargs['stop'] = 20, 150
    if response and coupled2 and (start, stop) == (0, 20):
        kwargs['period'] = 'early'
    if response and coupled2 and (start, stop) == (20, 150):
        kwargs['period'] = 'late'
    if coupled1 and not coupled2 and period and period[0] == 'a' and period != 'ann':
        kwargs['period'] = period[1:] if response else 'ann'  # abrupt-only period
    if data.name in ('tpat', 'tstd', 'tdev', 'tabs'):
        kwargs['region'] = 'globe'  # others undeined
    if coupled and region and region[0] == 'a':
        kwargs['region'] = region[1:] if response else 'globe'  # abrupt-only region
    return kwargs


def _reduce_data(
    data, method, dim=None, weight=None, sample=None, skipna=False, pctile=None, std=None, preserve=None,  # noqa: E501
):
    """
    Reduce individual data array using arbitrary method.

    Parameters
    ----------
    data : xarray.DataArray
        The input data.
    method : str, default: 'avg'
        The reduction method. See `reduce_facets` for details.
    dim : str, optional
        The reduction dimension. Default is the first dimension.
    weight : bool, optional
        Whether to weight by institute counts.
    sample : bool, optional
        Whether to return sample mean instead of population statistics.
    skipna : bool, optional
        Whether to skip null values.

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
        until plotting with ``shade|fade`` keyword arguments. If ``False`` this will
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
    method = method or 'avg'
    default = 95 if method == 'pct' else 80  # default percentile
    pctile = None if pctile is False else pctile  # see below
    pctile = default if pctile is True else pctile
    bnds = np.array(default if pctile is None or np.size(pctile) != 1 else pctile)
    bnds = 0.01 * np.array([50 - 0.5 * bnds.item(), 50 + 0.5 * bnds.item()])
    std = None if std is False else std
    std = True if method == 'std' and std is None else std
    std = (1, 1)[method == 'std'] if std is True else std
    std = np.atleast_1d(std) if std is not None else std
    defaults = {}
    short = long = None
    name = data.name
    dim = dim or data.dims[0]
    idim = dim if isinstance(dim, str) else None
    index = data.indexes.get(idim, None)
    units = f'({data.units})' if data.units else data.units
    kwargs = {'dim': dim, 'skipna': skipna}
    if not weight:
        wgts = xr.ones_like(data.coords[dim], dtype=float)
    elif dim == 'facets':
        wgts = _get_weights(data, dim=dim)
    else:
        raise TypeError(f'Invalid option {weight=} for regression dimension {dim!r}.')
    if method != 'dist' and isinstance(index, pd.MultiIndex):  # xarray weighted error
        data = data.drop_vars(data.coords.keys() & set(index.names))
    if method == 'dist':
        name = None  # bars or boxes
        if data.ndim == 1:
            result = data[~data.isnull()]
        else:
            raise ValueError(f'Invalid dimensionality {data.ndim!r} for distribution.')
    elif method == 'pctile':  # percentile range
        with xr.set_options(keep_attrs=True):  # note name is already kept
            data0 = data.quantile(bnds[0], **kwargs)
            data1 = data.quantile(bnds[1], **kwargs)
            result = data1 - data0
        short = f'{data.short_name} spread'
        long = f'{data.long_name} percentile range'
    elif method == 'pct':  # t-distribution percentile
        with xr.set_options(keep_attrs=True):
            dof = xr.ones_like(data).weighted(wgts).sum(**kwargs)
            loc = data.weighted(wgts).mean(**kwargs)
            scale = data.weighted(wgts).std(**kwargs)
            scale = scale / np.sqrt(dof) if sample else scale
        dist = stats.t(df=dof, loc=loc, scale=scale)
        result = 0.5 * (dist.ppf(bnds[1]) - dist.ppf(bnds[0]))  # i.e. symmetric range
        result = xr.DataArray(result, attrs=scale.attrs, dims=scale.dims, coords=scale.coords)  # noqa: E501
        short = f'{data.short_name} spread'
        long = f'{data.long_name} standard deviation'
    elif method == 'std':  # weithed standard deviation
        with xr.set_options(keep_attrs=True):  # note name is already kept
            dof = xr.ones_like(data).weighted(wgts).sum(**kwargs)
            result = data.weighted(wgts).std(**kwargs)
            result = result / np.sqrt(dof) if sample else result
        short = f'{data.short_name} spread'
        long = f'{data.long_name} standard deviation'
    elif method == 'var':  # weighted variance
        with xr.set_options(keep_attrs=True):  # note name is already kept
            dof = xr.ones_like(data).weighted(wgts).sum(**kwargs)
            result = data.weighted(wgts).var(**kwargs)
            result = result / dof if sample else result
        result.attrs['units'] = f'{units}^2' if units else ''
        short = f'{data.short_name} variance'
        long = f'{data.long_name} variance'
    elif method == 'skew' or method == 'kurt':  # weighted skewness kurtosis
        with xr.set_options(keep_attrs=True):
            func = stats.kurtosis if method == 'kurt' else stats.skew
            result = data.weighted(wgts).reduce(func=func, **kwargs)
        result.attrs['units'] = ''  # see: https://stackoverflow.com/a/71149901/4970632
        short = f'{data.short_name} skewness'
        long = f'{data.long_name} skewness'
    elif method == 'kurt':  # see: https://stackoverflow.com/a/71149901/4970632
        with xr.set_options(keep_attrs=True):
            result = data.weighted(wgts).reduce(func=stats.kurtosis, **kwargs)
        result.attrs['units'] = ''
        short = f'{data.short_name} kurtosis'
        long = f'{data.long_name} kurtosis'
    elif method == 'avg' or method == 'med':  # weighted average median
        args = () if method == 'avg' else (0.5,)
        key = 'mean' if method == 'avg' else 'quantile'
        with xr.set_options(keep_attrs=True):
            result = getattr(data.weighted(wgts), key)(*args, **kwargs)
        descrip = 'mean' if method == 'avg' else 'median'
        short = f'{data.long_name}'  # NOTE: differs from other methods
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


def _reduce_datas(
    data0, data1, method=None, dim=None, weight=None, pctile=None, std=None, invert=None, **kwargs,  # noqa: E501
):
    """
    Reduce pair of data arrays using arbitrary method.

    Parameters
    ----------
    data0, data1 : xarray.DataArray
        The input data.
    method : str, default: 'slope'
        The reduction method. See `reduce_facets` for details.
    dim : str, optional
        The reduction dimension. Default is the first shared dimension.
    weight : bool, optional
        Whether to weight by institute counts.

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
    # NOTE: Normalization and anomalies with respect to global average
    # as in climopy accessor.get() are supported in `reduce_facets`.
    # NOTE: Here we put 'variance' on additional DataArray dimension since variance
    # of gaussian random variables is linearly additive, so can allow process_data()
    # to carry out any operations (e.g. project=cmip6-cmip5) and generate 'shadedata'
    # and 'fadedata' arrays from the input arguments. Note resulting percentiles will
    # only be an approximation to some actual Monte Carlo sampled difference of
    # t-distributions. See: https://en.wikipedia.org/wiki/Variance#Propertieswill
    from observed.arrays import regress_dims
    method = method or 'slope'
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
    if dim == 'facets':
        wgts, manual = _get_weights(data0, dim=dim) if weight else None, True
    else:  # possibly auto weights
        wgts, manual = None, False if weight is None else not weight
    if method == 'dist':  # scatter or bars
        if max(data0.ndim, data1.ndim) > 1:
            raise ValueError(f'Invalid dimensionality {data0.ndim} x {data1.ndim} for distribution.')  # noqa: E501
        if kwargs:
            raise TypeError(f'Unexpected keyword arguments {kwargs}.')
        result0, result1 = data0[~data0.isnull()], data1[~data1.isnull()]
        result0, result1 = xr.align(result0, result1)  # intersection-style broadcast
        result = (result0, result1)
        name = None
        sym = r'$X$'
    elif method == 'diff':  # composite difference along first arrays
        kw_composite = dict(dim=dim, pctile=pctile)
        kw_composite.update(kwargs)
        result0, result1 = _get_composite(data0, data1, **kw_composite)
        result = result1 - result0
        result = result.climo.dequantify()
        result.attrs['units'] = data1.units
        short = f'{data1.short_name} composite difference'
        long = f'{data0.long_name}-composite {data1.long_name} difference'
        sym = r'$\Delta X$'
    elif method in ('corr', 'rsq'):
        nobnds = all(nums is None for nums in (pctile, std))
        kw_regress = dict(stat='corr', nobnds=nobnds, manual=manual, weights=wgts)
        kw_regress.update(kwargs)
        results = regress_dims(data0, data1, dim, **kw_regress)
        result, *bnds = (results,) if isinstance(results, xr.DataArray) else results
        result = ureg.dimensionless * result
        if method == 'corr':  # correlation coefficient
            result = result.climo.to_units('dimensionless').climo.dequantify()
            result.attrs['units'] = ''
            short = f'{short_prefix}correlation'
            long = f'{long_prefix}correlation coefficient'
            sym = r'$\rho_{I,\,F}$'
        else:  # variance explained
            result = (result ** 2).climo.to_units('percent').climo.dequantify()
            result.attrs['units'] = '%'
            short = f'{short_prefix}variance explained'
            long = f'{long_prefix}variance explained'
            sym = r'$r^2_{I,\,F}$'
        if not nobnds and result.ndim == 1:
            key = 'pctile' if pctile is not None else 'std'
            dims = np.atleast_1d(dim).tolist()
            nums = pctile if pctile is not None else std
            sigma = 0.5 * (bnds[1] - bnds[0])
            result = xr.concat((result, sigma ** 2), dim='sigma')  # see process_data()
            shade, fade = nums if nums.size == 2 else (nums.item(), None)
            defaults['dof'] = math.prod(data1.sizes[_] for _ in dims) - 2
            defaults.update({f'shade{key}s': shade, f'fade{key}s': fade})
            defaults.update({'shadealpha': 0.25, 'fadealpha': 0.15})
    elif method in ('cov', 'proj', 'slope'):
        nobnds = all(nums is None for nums in (pctile, std))
        kw_regress = dict(stat=method, nobnds=nobnds, weights=wgts, manual=manual)
        kw_regress.update(kwargs)
        results = regress_dims(data0, data1, dim=dim, **kw_regress)
        result, *bnds = (results,) if isinstance(results, xr.DataArray) else results
        units = f'({data0.units})' if '/' in data0.units else data0.units
        if method == 'cov':
            result.attrs['units'] = f'{data1.units} {data0.units}'
            short = f'{short_prefix}covariance'
            long = f'{long_prefix}covariance'
            sym = r'Cov$(I,\,F)$'  # covariance
        elif method == 'proj':
            result.attrs['units'] = f'{data1.units} / sigma'
            short = f'{short_prefix}projection'
            long = f'{long_prefix}projection'
            sym = r'Proj$(I,\,F)$'  # projection
        else:
            result.attrs['units'] = f'{data1.units} / {units}' if units else data1.units
            short = f'{short_prefix}regression coefficient'
            long = f'{long_prefix}regression coefficient'
            sym = r'$\beta_{I,\,F}$'  # regression
        if not nobnds and result.ndim == 1:
            key = 'pctile' if pctile is not None else 'std'
            dims = np.atleast_1d(dim).tolist()
            nums = pctile if pctile is not None else std
            sigma = 0.5 * (bnds[1] - bnds[0])
            result = xr.concat((result, sigma ** 2), dim='sigma')  # see process_data()
            shade, fade = nums if nums.size == 2 else (nums.item(), None)
            defaults['dof'] = math.prod(data1.sizes[_] for _ in dims) - 2
            defaults.update({f'shade{key}s': shade, f'fade{key}s': fade})
            defaults.update({'shadealpha': 0.25, 'fadealpha': 0.15})
    else:
        raise ValueError(f'Invalid double-variable method {method}')
    kw = {'name': name, 'symbol': sym, 'short_name': short, 'long_name': long}
    defaults.update(kw)
    return result, defaults


def reduce_time(data, time=None, season=None, month=None):
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
    # TODO: Replace this with 'observed/tables' reduce_time(), combining with
    # 'cmip_data.utils' average_periods() in-place utility. See tables.py for details
    # TODO: Replace 'average_periods' with this and normalize by annual temperature.
    # Also should add generalized no-op instruction for leaving coordinates alone.
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


def reduce_facets(
    *datas, method=None, preserve=None, invert=None, verbose=False, **kwargs,
):
    """
    Reduce along the model facets coordinate using an arbitrary method.

    Parameters
    ----------
    *datas : xarray.DataArray
        The data array(s).
    method : str, optional
        The reduction method. Here ``dist`` retains the facets dimension for e.g. bar
        and scatter plots (and their multiples), ``avg|med|std|pct|pctile`` reduce
        the facets dimension for a single input argument, and ``corr|diff|proj|slope``
        reduce the facets dimension for two input arguments. Optionally append ``_anom``
        to subtract the global average, ``_norm`` to normalize by the global average,
        or ``_mean`` to return sample mean uncertainty (valid for ``std``, ``pct``).

    Other Parameters
    ----------------
    verbose : bool, optional
        Whether to print extra information.
    preserve : optional
        Passed to `_reduce_data`.
    invert : optional
        Passed to `_reduce_datas`.
    **kwargs
        Passed to `_reduce_data` and `_reduce_datas`.

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
    kwargs.update(dim='facets')  # {{{
    ndim = max(data.ndim for data in datas)
    datas = tuple(data.copy() for data in datas)  # e.g. for distribution updates
    default = 'dist' if ndim == 1 else 'avg' if len(datas) == 1 else 'slope'
    method, *options = (method or default).split('_')
    mean = 'mean' in options
    anomaly = 'anom' in options
    normalize = 'norm' in options
    if set(options) - {'mean', 'anom', 'norm'}:
        raise ValueError('Invalid method option(s)', repr('_'.join(options)))
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
            base = data.climo.average('area')
            data = data - base
    if normalize:
        if 'lon' not in data.coords and 'lat' not in data.coords:
            raise NotImplementedError('Normalized methods require spatial coordinates.')
        with xr.set_options(keep_attrs=True):
            denom = data.climo.average('area')
            data = data / denom
        if units := data.attrs.get('units', ''):
            denom = f'({units})' if '/' in units else units
            data.attrs['units'] = f'{units} / {denom}'

    # Standardize and possibly print information
    # NOTE: Considered re-applying coordinates here but better instead to relegate
    # to process_data so that operators can be retained more easily.
    keys = ('facets', 'time', 'plev', 'lat', 'lon')  # {{{
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


def reduce_general(data, attrs=None, hemi=None, hemisphere=None, **kwargs):
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
    # NOTE: Delay application of defaults until here so that we only include default
    # selections in automatically-generated labels if user explicitly passed them.
    from observed.arrays import mask_region
    proj = inst = None  # filter functions  # {{{
    index = data.indexes.get('facets', pd.Index([]))
    project = kwargs.pop('project', None)  # apply after end
    institute = kwargs.pop('institute', None)  # apply after end
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

    # Iterate over data reductions
    # WARNING: Sometimes multi-index reductions can eliminate previously valid
    # coords, so critical to iterate one-by-one and validate selections each time.
    # TODO: Restore functionality for 'spatial' reductions.
    # NOTE: This silently skips dummy selections (e.g. area=None) that may be needed
    # to prevent _parse_specs from merging e.g. average and non-average selections.
    hemi = hemi or hemisphere  # {{{
    order = list(ORDER_LOGICAL)
    sorter = lambda item: order.index(item[0]) if item[0] in order else len(order)
    results = []
    kw_mask = _pop_kwargs(kwargs, mask_region)
    if is_dataset := isinstance(data, xr.Dataset):
        datas = data.data_vars.values()
    else:  # iterate over arrays
        datas = (data,)
    for data in datas:
        result = data
        ikwargs = _fill_kwargs(data, **kwargs)  # includes default area=None
        for dim, value in sorted(ikwargs.items(), key=sorter):
            sizes = [*result.sizes, 'area', 'volume', 'spatial', 'time', 'month', 'season']  # noqa: E501
            names = [key for idx in result.indexes.values() for key in idx.names]
            quants = result.coords.keys() - set(names) - {'time', 'month', 'season'}
            if value is None or dim not in sizes and dim not in names:
                continue
            if dim == 'area' and not result.sizes.keys() & {'lon', 'lat'}:
                continue
            if dim == 'volume' and not result.sizes.keys() & {'lon', 'lat', 'plev'}:
                continue
            if dim == 'area' and hemi:  # TODO: allow truncation without averages
                data = data.climo.sel_hemisphere(hemi)
            if dim == 'area' and value != 'avg':  # TODO: revisit this, ensure works
                mask = mask_region(data, value, **kw_mask)
                data = xr.where(mask, data, np.nan)  # unweight region
                data, value = data.climo.add_cell_measures(), 'avg'
            if dim in quants and not isinstance(value, (str, tuple)):
                value = ureg.Quantity(value, result.coords[dim].climo.units)
            if dim in ('time', 'month', 'season'):  # time selections
                result = reduce_time(result, **{dim: value})
            elif dim in ('start', 'stop'):  # TODO: fix climopy bugs
                result = result.sel({dim: value}).squeeze()
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
    if is_dataset:  # {{{
        result = xr.Dataset({result.name: result for result in results})
    else:  # singular array
        result = results[0]
    result = inst(result) if inst and inst.name == 'avg' else result
    index = result.indexes.get('facets', pd.Index([]))
    coord, names = {}, set(index.names)  # facet names
    if 'project' in index.names and len(set(result.project.values)) == 1:
        value = result.project.values[0]
        names = names - {'project'}
        result = result.reset_index('project', drop=True)
        coord['project'] = value
    if 'facets' in result.dims and len(names) == 1:
        values = result.facets.values
        index = pd.MultiIndex.from_arrays((values,), names=(names.pop(),))
        index = xr.DataArray(index, name='facets', dims='facets')
        coord['facets'] = index
    with warnings.catch_warnings():  # xarray future warning
        warnings.simplefilter('ignore')
        result = result.assign_coords(coord)
    return result
