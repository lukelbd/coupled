#!/usr/bin/env python3
"""
Internal helper functions for figure templates.
"""
import itertools

import climopy as climo  # noqa: F401
import numpy as np
import pandas as pd
import xarray as xr
from climopy.var import _get_bounds, linefit
from climopy import ureg, vreg  # noqa: F401
from icecream import ic  # noqa: F401

from .internals import KEYS_METHOD, ORDER_LOGICAL
from .internals import _group_parts, _ungroup_parts, _to_lists
from .results import FACETS_LEVELS, FEEDBACK_TRANSLATIONS, VERSION_LEVELS
from cmip_data.internals import MODELS_INSTITUTES, INSTITUTES_LABELS

__all__ = ['apply_reduce', 'apply_method', 'process_data']

# Reduce defaults
GENERAL_DEFAULTS = {
    'period': 'ann',
    'experiment': 'picontrol',
    'ensemble': 'flagship',
}

# Version defaults
VERSION_DEFAULTS = {
    'source': 'eraint',
    'statistic': 'slope',
    'region': 'globe',
    'start': 0,
    'stop': 150,
}

# Reduce presets
# See (WPG and ENSO): https://doi.org/10.1175/JCLI-D-12-00344.1
# See (WPG and ENSO): https://doi.org/10.1038/s41598-021-99738-3
# See (tropical ratio): https://doi.org/10.1175/JCLI-D-18-0843.1
# See (feedback regions): https://doi.org/10.1175/JCLI-D-17-0087.1
# https://climatedataguide.ucar.edu/climate-data/nino-sst-indices-nino-12-3-34-4-oni-and-tni
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
    # 'wpac': {'lat_lim': (-15, 15), 'lon_lim': (90, 150)},  # based on own feedbacks
    # 'epac': {'lat_lim': (-30, 0), 'lon_lim': (260, 290)},
    # 'wpac': {'lat_lim': (-15, 15), 'lon_lim': (150, 170)},  # paper based on +4K
    # 'epac': {'lat_lim': (-30, 0), 'lon_lim': (260, 280)},
    'nina': {'lat_lim': (0, 10), 'lon_lim': (130, 150)},
    'nino': {'lat_lim': (-5, 5), 'lon_lim': (190, 240)},
    'nino3': {'lat_lim': (-5, 5), 'lon_lim': (210, 270)},
    'nino4': {'lat_lim': (-5, 5), 'lon_lim': (160, 210)},
}


def _components_composite(data0, data1, pctile=None, dim='facets'):
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
    # NOTE: This can be used to e.g. composite the temperature pattern
    # response on models with high vs. low global average feedbacks.
    thresh = 33 if pctile is None else pctile
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


def _components_covariance(data0, data1, both=True, dim='facets'):
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
    # NOTE: For simplicity this gives the *biased* weighted variance estimator (analogue
    # for unweighted data is SS / N) instead of the complex *unbiased* weighted variance
    # estimator (analogue for unweighted data SS / N - 1). See full formula here:
    # https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Reliability_weights
    if dim == 'area':
        dims = ('lon', 'lat')
    elif dim == 'volume':
        dims = ('lon', 'lat', 'plev')
    else:
        dims = (dim,)
    weight = xr.ones_like(data0)
    if 'lon' in dims or 'lat' in dims:
        weight *= data0.coords['cell_width']
    if 'lat' in dims:
        weight *= data0.coords['cell_depth']
    if 'plev' in dims:
        weight *= data0.coords['cell_height']
    data0, data1, weight = xr.broadcast(data0, data1, weight)
    mask0, mask1 = ~np.isnan(data0), ~np.isnan(data1)
    data0, data1 = data0.fillna(0), data1.fillna(0)
    data0, data1 = data0.climo.quantify(), data1.climo.quantify()
    anom0 = data0 - (weight * data0).sum(dims) / (weight * mask0).sum(dims)
    anom1 = data1 - (weight * data1).sum(dims) / (weight * mask1).sum(dims)
    covar = (weight * anom0 * anom1).sum(dims) / (weight * mask0 * mask1).sum(dims)
    std0 = np.sqrt((weight * anom0 ** 2).sum(dims) / (weight * mask0).sum(dims))
    if both:
        std1 = np.sqrt((weight * anom1 ** 2).sum(dims) / (weight * mask1).sum(dims))
    return (covar, std0, std1) if both else (covar, std0)


def _components_corr(data0, data1, dim=None, pctile=None):
    """
    Return components of a correlation evalutation.

    Parameters
    ----------
    data0, data1 : xarray.DataArray
        The coordinates to be compared.
    dim : str, optional
        The dimension for the correlation.
    pctile : float, optional
        The percentile range for the lower and upper uncertainty bounds.

    Returns
    -------
    corr, corr_lower, corr_upper : xarray.DataArray
        The correlation estimates with `pctile` lower and upper bounds.
    rsquare : xarray.DataArray
        The variance explained by the relationship.
    """
    # NOTE: Here use special t-test for correlation uncertainty bounds.
    # See: https://en.wikipedia.org/wiki/Pearson_correlation_coefficient#Standard_error
    dim = dim or data0.dims[0]
    data0, data1 = xr.align(data0, data1)
    pctile = 0.5 * (100 - np.atleast_1d(pctile or 95))
    pctile = np.array([pctile, 100 - pctile])  # e.g. [90, 50] --> [[5, 25], [95, 75]]
    anom0 = data0 - data0.mean(dim)
    anom1 = data1 - data1.mean(dim)
    std0 = np.sqrt((anom0 ** 2).sum(dim))
    std1 = np.sqrt((anom1 ** 2).sum(dim))
    corr = (anom0 * anom1).sum(dim) / (std0 * std1)  # correlation coefficient
    ndim = data0.sizes[dim]
    rsquare = corr ** 2  # variance explained == correlation squared
    sigma = np.sqrt((1 - corr ** 2) / (ndim - 2))  # standard error
    t = corr * np.sqrt((ndim - 2) / (1 - corr ** 2))  # t-statistic
    dt_lower, dt_upper = _get_bounds(sigma, pctile, dof=data0.size - 2)
    dt_lower = xr.DataArray(dt_lower, dims=('pctile', *sigma.dims))
    dt_upper = xr.DataArray(dt_upper, dims=('pctile', *sigma.dims))
    t_lower, t_upper = t + dt_lower, t + dt_upper
    corr_lower = t_lower / np.sqrt(ndim - 2 + t_lower ** 2)
    corr_upper = t_upper / np.sqrt(ndim - 2 + t_upper ** 2)
    return corr, corr_lower, corr_upper, rsquare


def _components_slope(data0, data1, dim=None, adjust=False, pctile=None):
    """
    Return components of a line fit operation.

    Parameters
    ----------
    data0 : xarray.DataArray
        The dependent coordinates.
    data1 : xarray.DataArray
        The other coordinates.
    dim : str, optional
        The dimension for the regression.
    adjust : bool, optional
        Whether to adjust the slope for autocorrelation effects.
    pctile : float, optional
        The percentile range for the lower and upper uncertainty bounds.

    Returns
    -------
    slope, slope_lower, slope_upper : xarray.DataArray
        The slope estimates with `pctile` lower and upper bounds.
    rsquare : xarray.DataArray
        The variance explained by the fit.
    fit, fit_lower, fit_upper : xarray.DataArray
        The fit to the original points with `pctile` lower and upper bounds.
    """
    # NOTE: Here np.polyfit requires monotonically increasing coordinates. Not
    # sure why... could consider switching to manual slope and stderr calculation.
    dim = dim or data0.dims[0]
    data0, data1 = xr.align(data0, data1)
    axis = data0.dims.index(dim)
    idx = np.argsort(data0.values, axis=axis)
    data0 = data0.isel({dim: idx})
    data1 = data1.isel({dim: idx})
    pctile = 0.5 * (100 - np.atleast_1d(pctile or 95))
    pctile = np.array([pctile, 100 - pctile])  # e.g. [90, 50] --> [[5, 25], [95, 75]]
    slope, sigma, rsquare, fit, fit_lower, fit_upper = linefit(
        data0, data1, dim=dim, adjust=adjust, pctile=pctile,
    )
    dslope_lower, dslope_upper = _get_bounds(sigma, pctile, dof=data0.size - 2)
    dslope_lower = xr.DataArray(dslope_lower, dims=('pctile', *sigma.dims))
    dslope_upper = xr.DataArray(dslope_upper, dims=('pctile', *sigma.dims))
    slope_lower, slope_upper = slope + dslope_lower, slope + dslope_upper
    return slope, slope_lower, slope_upper, rsquare, fit, fit_lower, fit_upper


def _parse_project(data, project=None):
    """
    Return plot labels and facet filter for the project indicator.

    Parameters
    ----------
    data : xarray.Dataset or xarray.DataArray
        The data. Used to search for models from the same institute in other projects.
    project : str, default: 'cmip'
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
    project = project or 'cmip'
    project = project.lower()
    name_to_inst = MODELS_INSTITUTES.copy()
    name_to_inst.update(  # support facets with institutes names instead of models
        {
            (proj, abbrv): inst
            for inst, abbrv in INSTITUTES_LABELS.items()
            for proj in ('CMIP5', 'CMIP6')
        }
    )
    if not project.startswith('cmip'):
        raise ValueError(f'Invalid project indicator {project}. Must contain cmip.')
    _, num = project.split('cmip')
    imax = max(1, len(num))
    if imax > 4:
        raise ValueError(f'Invalid project indicator {project}. Up to 4 numbers allowed.')  # noqa: E501
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
    func.name = project  # WARNING: critical for process_data() detection of 'all_projs'
    return func


def _parse_institute(data, institute=None):
    """
    Return plot labels and facet filter for the institute indicator.

    Parameters
    ----------
    data : xarray.Dataset or xarray.DataArray
        The data. Used to construct the multi-index institute data array.
    institute : str, default: None
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
    inst_to_label = INSTITUTES_LABELS.copy()  # see also _parse_constraints
    model_to_inst = MODELS_INSTITUTES.copy()
    if not institute:
        filt = lambda key: True  # noqa: U100
    elif institute == 'avg':
        insts = [
            model_to_inst.get((key[0], key[1]), 'U')
            for key in data.facets.values
        ]
        facets = [
            (key[0], inst_to_label.get(inst, inst), *key[2:])
            for inst, key in zip(insts, data.facets.values)
        ]
        filt = xr.DataArray(
            pd.MultiIndex.from_tuples(facets, names=data.indexes['facets'].names),
            attrs=data.facets.attrs,
            dims='facets',  # WARNING: critical for groupby() to name this 'facets'
        )
        filt.name = 'facets'
    elif institute == 'flagship':
        inst_to_model = {  # NOTE: flagship models are ordered last in list
            (proj, inst): model for (proj, model), inst in model_to_inst.items()
        }
        filt = lambda key: (
            key[1] == inst_to_model.get((key[0], model_to_inst.get((key[0], key[1]))))
        )
        filt.name = institute  # unnecessary but why not
    elif any(value == institute for pair in inst_to_label.items() for value in pair):
        label_to_inst = {
            abbrv: inst for inst, abbrv in inst_to_label.items()
        }
        filt = lambda key: (
            label_to_inst.get(institute, institute) == model_to_inst.get((key[0], key[1]))  # noqa: E501
        )
        filt.name = institute  # unnecessary but why not
    else:
        raise ValueError(f'Invalid institute name {institute!r}.')
    return filt


def _apply_double(data0, data1, dim=None, method=None, invert=None):
    """
    Apply reduction method for two data arrays.

    Parameters
    ----------
    data0, data1 : xarray.DataArray
        The input data.
    dim : str, default: 'facets'
        The reduction dimension.
    method : str, optional
        The method (see `apply_method`).
    invert : bool, optional
        Whether to invert the direction of composites, projections, and regressions so
        that the first variable is the predictor instead of the second variable.

    Returns
    -------
    data : xarray.DataArray
        The resulting data.
    defaults : dict
        The default attributes and command keyword args.
    """
    data0, data1 = (data1, data0) if invert else (data0, data1)
    method = method or 'cov'
    short = long = None
    name = f'{data0.name}|{data1.name}'  # NOTE: needed for _combine_commands labels
    ndim = max(data0.ndim, data1.ndim)
    dim = dim or 'facets'
    if dim == 'area':
        short_prefix, long_prefix = 'spatial', f'{data1.long_name} spatial'
    elif data0.long_name == data0.long_name:
        short_prefix, long_prefix = data1.short_name, data1.long_name
    else:
        short_prefix, long_prefix = data1.short_name, f'{data1.long_name}/{data0.long_name}'  # noqa: E501
    if method == 'cov':
        data, _ = _components_covariance(data0, data1, dim=dim, both=False)
        data = data.climo.dequantify()
        data.attrs['units'] = f'{data1.units} {data0.units}'
        short = f'{short_prefix} covariance'
        long = f'{long_prefix} covariance'
    elif method == 'slope':  # regression coefficient
        cov, std = _components_covariance(data0, data1, dim=dim, both=False)
        data = cov / std ** 2
        data = data.climo.dequantify()
        data.attrs['units'] = f'{data1.units} / ({data0.units})'
        short = f'{short_prefix} regression coefficient'
        long = f'{long_prefix} regression coefficient'
    elif method == 'rsq':  # correlation coefficient
        cov, std0, std1 = _components_covariance(data0, data1, dim=dim, both=True)
        data = (cov / (std0 * std1)) ** 2
        data = data.climo.to_units('percent').climo.dequantify()
        data.attrs['units'] = '%'
        short = f'{short_prefix} variance explained'
        long = f'{long_prefix} variance explained'
    elif method == 'proj':  # projection onto x
        cov, std = _components_covariance(data0, data1, dim=dim, both=False)
        data = cov / std
        data = data.climo.dequantify()
        data.attrs['units'] = data1.units
        short = f'{short_prefix} projection'
        long = f'{long_prefix} projection'
    elif method == 'corr':  # correlation coefficient
        cov, std0, std1 = _components_covariance(data0, data1, dim=dim, both=True)
        data = cov / (std0 * std1)
        data = data.climo.to_units('dimensionless').climo.dequantify()
        data.attrs['units'] = ''
        short = f'{short_prefix} correlation'
        long = f'{long_prefix} correlation coefficient'
    elif method == 'diff':  # composite difference along first arrays
        assert dim != 'area'
        data_lo, data_hi = _components_composite(data0, data1, dim=dim)
        data = data_hi - data_lo
        data = data.climo.dequantify()
        data.attrs['units'] = data1.units
        short = f'{data1.short_name} composite difference'
        long = f'{data0.long_name}-composite {data1.long_name} difference'
    elif method == 'dist':  # scatter or bars
        assert ndim == 1
        data0, data1 = data0[~data0.isnull()], data1[~data1.isnull()]
        data0, data1 = xr.align(data0, data1)  # intersection-style broadcast
        data = (data0, data1)
        name = None
    else:
        raise ValueError(f'Invalid double-variable method {method}')
    defaults = {'name': name, 'short_name': short, 'long_name': long}
    return data, defaults


def _apply_single(data, dim=None, method=None, pctile=None, std=None):
    """
    Apply reduction method for single data array.

    Parameters
    ----------
    data : xarray.DataArray
        The input data.
    dim : str, default: 'facets'
        The dimension.
    method : str, optional
        The reduction method (see `apply_method`).

    Other Parameters
    ----------------
    pctile : float or sequence, optional
        The percentile range or thresholds for related methods. Set to ``True`` to
        use default values of ``50`` for ``pctile`` and ``95`` for ``avg|med`` shading.
        Pass two values, e.g. ``[50, 95]``, to use both `shade` and `fade` keywords.
        If one is provided an alpha level between `shade` and `fade` is used.
    std : float or sequence, optional
        The standard deviation multiple for related methods. Set to ``True`` to use
        default values of ``1`` for ``std`` and ``3`` for ``avg|med`` shading.
        Pass two values, e.g. ``[1, 3]``, to use both `shade` and `fade` keywords.
        If one is provided an alpha level between `shade` and `fade` is used.

    Returns
    -------
    data : xarray.DataArray
        The resulting data.
    defaults : dict
        The default attributes and command keyword args.
    """
    # NOTE: Here `pctile` is shared between inter-model percentile differences and
    # composites of a second array based on values in the first array.
    # NOTE: Currently proplot will automatically apply xarray tuple multi-index
    # coordinates to bar plot then error out so apply numpy array coords for now.
    method = method or 'avg'
    ndim = data.ndim
    kw = {'dim': dim or 'facets', 'skipna': True}
    name = data.name
    short = long = None
    defaults = {}
    if method == 'avg' or method == 'med':
        key = 'mean' if method == 'avg' else 'median'
        descrip = 'mean' if method == 'avg' else 'median'
        defaults.update({'fadealpha': 0.15, 'shadealpha': 0.3})
        if std is None and pctile is None:
            assert ndim < 4
            cmd = getattr(data, key)
            data = cmd(**kw, keep_attrs=True)
            short = f'{descrip} {data.short_name}'  # only if no range included
            long = f'{descrip} {data.long_name}'
        elif std is not None:
            assert ndim == 2
            # std = [1, 3] if std is True else std
            std = 3 if std is True else std
            std = np.atleast_1d(std)
            shade, fade = std if std.size == 2 else (std.item(), None)
            defaults.update({key: True, 'shadestds': shade, 'fadestds': fade})
        else:
            assert ndim == 2
            # pctile = [50, 95] if pctile is True else pctile
            pctile = 80 if pctile is True else pctile
            pctile = np.atleast_1d(pctile)
            shade, fade = pctile if pctile.size == 2 else (pctile.item(), None)
            defaults.update({key: True, 'shadepctiles': shade, 'fadepctiles': fade})
    elif method == 'pctile':
        assert ndim < 4
        pctile = 50 if pctile is None or pctile is True else pctile
        nums = 0.01 * pctile / 2, 1 - 0.01 * pctile / 2
        with xr.set_options(keep_attrs=True):  # note name is already kept
            data = data.quantile(nums[1], **kw) - data.quantile(nums[0], **kw)
        short = f'{data.short_name} spread'
        long = f'{data.long_name} percentile range'
    elif method == 'std':
        assert ndim < 4
        std = 1 if std is None or std is True else std
        with xr.set_options(keep_attrs=True):  # note name is already kept
            data = std * data.std(**kw)
        short = f'{data.short_name} spread'
        long = f'{data.long_name} standard deviation'
    elif method == 'var':
        assert ndim < 4
        with xr.set_options(keep_attrs=True):  # note name is already kept
            data = data.var(**kw)
        data.attrs['units'] = f'({data.units})^2'
        short = f'{data.short_name} variance'
        long = f'{data.long_name} variance'
    elif method == 'dist':  # bars or boxes
        assert ndim == 1
        data = data[~data.isnull()]
        name = None
    else:
        raise ValueError(f'Invalid single-variable method {method}.')
    defaults.update({'name': name, 'short_name': short, 'long_name': long})
    return data, defaults


def apply_method(*datas, method=None, verbose=False, **kwargs):
    """
    Reduce along the facets coordinate using an arbitrary method.

    Parameters
    ----------
    *datas : xarray.DataArray
        The data array(s).
    method : str, optional
        The reduction method. Here ``dist`` retains the facets dimension for e.g. bar
        and scatter plots (and their multiples), ``avg|med|std|pctile`` reduce the
        facets dimension for a single input argument, and ``corr|diff|proj|slope``
        reduce the facets dimension for two input arguments. See below for details.
    verbose : bool, optional
        Whether to print extra information.
    **kwargs
        Passed to `apply_single` or `apply_double`.

    Returns
    -------
    args : tuple
        The output plotting arrays.
    method : str
        The resulting method used.
    kwargs : dict
        The plotting and `_infer_command` keyword arguments.
    """
    # Apply methods
    # NOTE: Manually recalculate slope error here because faster than polyfit by
    # single columns and often use multi-dimensional data. Leverage fact that
    # offset = ymean - slope * xmean and slope stdeerr = sqrt(resid ** 2 / xanom ** 2
    # / n - 2) where the residual resid = y - ((ymean - slope * xmean) + slope * x)
    # = (y - ymean) - slope * (x - xmean) i.e. very very simple.
    # TODO: Consider adding below to add shading to regression line plots
    # dof = data0.sizes[dim] - 2
    # anom0, anom1 = data0 - data0.mean(dim), data1 - data1.mean(dim)
    # resid = anom1 - data * anom0
    # sigma = np.sqrt((resid ** 2).sum(dim)) / np.sqrt((anom0 ** 2).sum(dim)) / dof  # noqa: E501
    # pctile = np.atleast_1d([50, 95] if pctile is True else pctile)
    # shade, fade = pctile if pctile.size == 2 else (pctile.item(), None)
    # for key, value in zip(('shade', 'fade'), (shade, fade)):
    #     value = 0.5 * (100 - value)
    #     values = np.array([value, 100 - value])
    #     del_lower, del_upper = _get_bounds(sigma, values, dof=dof)
    #     errors = np.array([del_lower, del_upper])  # 2xN array
    #     defaults.update({f'{key}data': errors})
    ndim = max(data.ndim for data in datas)
    datas = tuple(data.copy() for data in datas)  # e.g. for distribution updates
    if ndim == 1:
        method = method or 'dist'  # only possibility
    elif len(datas) == 1:
        method = method or 'avg'
    else:
        method = method or 'rsq'
    keys_double = ('invert',)
    keys_single = ('pctile', 'std')
    kw_double = {key: kwargs.pop(key) for key in keys_double if key in kwargs}
    kw_single = {key: kwargs.pop(key) for key in keys_single if key in kwargs}
    if kwargs:
        raise ValueError(f'Unexpected keyword arguments {kwargs}.')
    if len(datas) == 1:
        data, defaults = _apply_single(*datas, method=method, **kw_single)
    elif len(datas) == 2:
        data, defaults = _apply_double(*datas, method=method, **kw_double)
    else:
        raise ValueError(f'Unexpected argument count {len(datas)}.')

    # Standardize and possibly print information
    # NOTE: Considered re-applying coordinates here but better instead to relegate
    # to process_data so that operators can be retained more easily.
    keys = ('facets', 'time', 'plev', 'lat', 'lon')  # coordinate sorting order
    args = tuple(data) if isinstance(data, tuple) else (data,)
    args = [arg.transpose(..., *(key for key in keys if key in arg.sizes)) for arg in args]  # noqa: E501
    if name := defaults.pop('name', None):
        args[-1].name = name
    if long := defaults.pop('long_name', None):
        args[-1].attrs['long_name'] = long
    if short := defaults.pop('short_name', None):
        args[-1].attrs['short_name'] = short
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


def apply_reduce(data, attrs=None, **kwargs):
    """
    Carry out arbitrary reduction of the given dataset variables.

    Parameters
    ----------
    data : xarray.DataArray
        The data array.
    attrs : dict, optional
        The optional attribute overrides.
    **kwargs
        The reduction selections. Requires a `name`.

    Returns
    -------
    data : xarray.DataArray
        The data array.
    """
    # Apply 'facets' reductions
    # TODO: Might consider adding 'institute' multi-index so e.g. grouby('institute')
    # is possible... then again would not reduce complexity because would now need to
    # load files in special non-alphabetical order to enable selecting 'flaghip' models
    # with e.g. something like .sel(institute=-1)... so don't bother for now.
    institute = kwargs.pop('institute', None)
    project = kwargs.pop('project', None)
    season = kwargs.pop('season', None)
    month = kwargs.pop('month', None)
    if institute is not None:  # WARNING: critical this comes first
        if callable(institute):  # facets filter function
            facets = list(filter(institute, data.facets.values))
            data = data.sel(facets=facets)
        elif isinstance(institute, xr.DataArray):  # groupby multi-index data array
            data = data.groupby(institute).mean(skipna=False, keep_attrs=True)
            dict_ = institute.facets.attrs  # WARNING: avoid overwriting input attrs
            facets = data.indexes['facets']  # xarray bug causes dropped level names
            facets.names = institute.indexes['facets'].names
            facets = xr.DataArray(facets, name='facets', dims='facets', attrs=dict_)
            data = data.assign_coords(facets=facets)  # assign with restored level names
        else:
            raise ValueError(f'Unsupported institute {institute!r}.')
    if project is not None:  # see _parse_project
        if callable(project):
            facets = list(filter(project, data.facets.values))
            data = data.sel(facets=facets)
            data = data.reset_index('project', drop=True)  # models are unique enough
        else:
            raise ValueError(f'Unsupported project {project!r}.')

    # Apply time reductions
    # TODO: Replace 'average_periods' with this and normalize by annual temperature
    # NOTE: The new climopy cell duration calculation will auto-detect monthly and
    # yearly data, but not yet done, so use explicit days-per-month weights for now.
    if season is not None or month is not None:
        if season is not None:
            seasons = data.time.dt.season.str.lower()
            data = data.isel(time=(seasons == season.lower()))
        elif isinstance(month, str):
            months = data.time.dt.strftime('%b').str.lower()
            data = data.isel(time=(months == month.lower()))
        else:
            months = data.time.dt.month
            data = data.isel(time=(months == month))
    if 'time' in data.coords:  # manual weighted average
        days = data.time.dt.days_in_month
        wgts = days.groupby('time.year') / days.groupby('time.year').sum()
        wgts = wgts.astype(np.float32)  # preserve float32 variables
        ones = xr.where(data.isnull(), 0, 1)
        ones = ones.astype(np.float32)  # preserve float32 variables
        with xr.set_options(keep_attrs=True):
            numerator = (data * wgts).groupby('time.year').sum(dim='time')
            denominator = (ones * wgts).groupby('time.year').sum(dim='time')
            data = numerator / denominator

    # Apply default values and possible overrides
    # NOTE: Delay application of defaults until here so that we only include default
    # selections in automatically-generated labels if user explicitly passed them.
    # NOTE: Here None is automatically replaced with default values (e.g. period=None)
    # and default experiment depends on whether this is feedback variable or not.
    name = data.name
    attrs = attrs or {}
    attrs = attrs.copy()  # WARNING: this is critical
    attrs.update({key: val for key, val in data.attrs.items() if key not in attrs})
    defaults = GENERAL_DEFAULTS.copy()
    if 'version' in data.coords:
        defaults.update({'experiment': 'abrupt4xco2', **VERSION_DEFAULTS})
    if 'startstop' in kwargs:  # possibly passed from process_data()
        kwargs['start'], kwargs['stop'] = kwargs.pop('startstop')  # possibly both None
    for key, value in defaults.items():  # apply default reductions
        if key in kwargs and kwargs[key] is None:
            kwargs[key] = value  # used 'None' as default placeholder
        else:
            kwargs.setdefault(key, value)
    if 'version' in data.coords:
        experiment, period, region = kwargs['experiment'], kwargs['period'], kwargs['region']  # noqa: E501
        if period[0] == 'a' and period != 'ann':  # abrupt-only period
            kwargs['period'] = 'ann' if experiment == 'picontrol' else period[1:]
        if region[0] == 'a':  # abrupt-only region
            kwargs['region'] = 'globe' if experiment == 'picontrol' else region[1:]
        if name in ('tpat', 'tabs'):  # others undefined so overwrite
            kwargs['region'] = 'globe'
        if experiment == 'picontrol':  # others undefined so overwrite
            kwargs['start'], kwargs['stop'] = 0, 150

    # Iterate over reductions
    # NOTE: This silently skips dummy selections (e.g. area=None) that may be required
    # to prevent _parse_specs from merging e.g. average and non-average selections.
    # WARNING: Sometimes multi-index reductions can eliminate previously valid coords,
    # so critical to iterate one-by-one and validate selections each time.
    order = list(ORDER_LOGICAL)
    sorter = lambda key: order.index(key) if key in order else len(order)
    spatial = kwargs.get('spatial', None)
    for key in sorted(kwargs, key=sorter):
        # Parse input instructions
        value = kwargs[key]
        opts = list((*data.sizes, 'area', 'volume', 'spatial'))
        opts.extend(level for idx in data.indexes.values() for level in idx.names)
        if value is None or key not in opts:
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

        # Apply reduction
        if spatial and key in ('area', 'start', 'stop', 'experiment'):
            continue
        if key == 'spatial':  # NOTE: implicit 'area' truncation should already be done
            kw0, kw1 = {}, {}
            if not data.sizes.keys() & {'facets', 'experiment'}:
                continue  # e.g. another variable
            if 'version' in data.coords:
                kw0 = {'start': 0, 'stop': 150}
                kw1 = {'start': kwargs.get('start', 0), 'stop': kwargs.get('stop', 150)}
            data0 = data.sel(experiment='picontrol', **kw0)
            data1 = data.sel(experiment='abrupt4xco2', **kw1)
            data, defaults = _apply_double(data0, data1, dim='area', method=value)
            name = defaults.pop('name')  # see below
            attrs.update({**data.attrs, **defaults})
        else:  # apply simple reduction
            try:
                data = data.climo.reduce(**{key: value}).squeeze()
            except Exception:
                raise RuntimeError(f'Failed to reduce data with {key}={value!r}.')

        # Update coordinates
        for multi, levels in zip(('facets', 'version'), (FACETS_LEVELS, VERSION_LEVELS)):  # noqa: E501
            if multi in data.coords:  # multi-index still present
                continue
            levels = data.sizes.keys() & set(levels)  # remaining level
            if not levels:  # no remaining levels (e.g. scalar)
                continue
            coord = data.coords[level := levels.pop()]
            index = pd.MultiIndex.from_arrays((coord.values,), names=(level,))
            data = data.rename({level: multi})
            data = data.assign_coords({multi: index})

    data.name = name
    data.attrs.update(attrs)
    return data


def process_data(dataset, *kws_process, attrs=None, suffix=True, feedback=None):
    """
    Combine the data based on input reduce dictionaries.

    Parameters
    ----------
    dataset : xarray.Dataset
        The source dataset.
    *kws_process : dict
        The reduction keyword dictionaries.
    attrs : dict, optional
        The attribute dictionaries.
    suffix : bool, optional
        Whether to add optional `anomaly` suffix.
    feedback : str, optional
        The feedback name to use in temperature pattern regression.

    Returns
    -------
    args : tuple
        The output plotting arrays.
    method : str
        The method used to reduce the data.
    kwargs : dict
        The plotting and `_infer_command` keyword arguments.
    """
    # Group added/subtracted reduce instructions into separate dictionaries
    # NOTE: Initial kw_red values are formatted as (('[+-]', value), ...) to
    # permit arbitrary combinations of names and indexers (see _parse_specs).
    # WARNING: Here 'product' can be used for e.g. cmip6-cmip5 abrupt4xco2-picontrol
    # but there are some terms that we always want to group together e.g. 'experiment'
    # and 'startstop'. So include some overrides below.
    if len(kws_process) not in (1, 2):
        raise ValueError(f'Expected two process dictionaries. Got {len(kws_process)}.')
    alias_to_name = {
        alias: name for alias, (name, _) in FEEDBACK_TRANSLATIONS.items()
    }
    if 'control' in dataset.experiment:
        alias_to_name.update({'picontrol': 'control', 'abrupt4xco2': 'response'})
    else:
        alias_to_name.update({'control': 'picontrol', 'response': 'abrupt4xco2'})
    kws_reduce, kws_input = [], []
    for kw_process in kws_process:  # iterate method reduce arguments
        # Initial stuff
        # NOTE: See _group_parts comments for details
        kw_reduce = {}
        kw_process = kw_process.copy()
        kw_method = {
            key: kw_process.pop(key) for key in KEYS_METHOD
            if key in kw_process
        }
        kw_input = {
            key: getattr(value, 'name', value) for key, value in kw_process.items()
            if key != 'name' and value is not None
        }
        kw_process = _group_parts(kw_process, keep_operators=True)

        # Get reduce instructions
        for key, value in kw_process.items():
            sels = ['+']
            for part in _ungroup_parts(value):
                if part is None:
                    sel = None
                elif np.iterable(part) and part[0] in ('+', '-', '*', '/'):
                    sel = part[0]
                elif isinstance(part, tuple):  # already mapped integers
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
            kw_reduce[key] = tuple(zip(signs, values))

        # Split instructions along operators
        # TODO: Possibly add more grouping options (similar to _build_specs)
        startstop = 'startstop' in kw_reduce and 'experiment' in kw_reduce
        groups = [('startstop', 'experiment')] if startstop else []
        for group in groups:
            values = _to_lists(*(kw_reduce[key] for key in group))
            kw_reduce.update(dict(zip(group, values)))
        groups.extend((key,) for key in kw_reduce if not any(key in group for group in groups))  # noqa: E501
        kw_product = {group: tuple(zip(*(kw_reduce[key] for key in group))) for group in groups}  # noqa: E501
        ikws_reduce = []
        for values in itertools.product(*kw_product.values()):  # non-grouped coords
            items = {key: val for group, vals in zip(groups, values) for key, val in zip(group, vals)}  # noqa: E501
            signs, values = zip(*items.values())
            sign = -1 if signs.count('-') % 2 else +1
            kw = dict(zip(items.keys(), values))
            kw.update(kw_method)
            ikws_reduce.append((sign, kw))
        kws_reduce.append(ikws_reduce)
        kws_input.append(kw_input)

    # Reduce along facets dimension and carry out operation
    # WARNING: Currently impossible to perform e.g. regressions when have different
    # number of operations in numerator and denominator. Enforce with strict below.
    # TODO: Support operations before reductions instead of after. Should have
    # effect on e.g. correlation, regression results.
    # NOTE: Here 'feedback' is for figures regressing both temperature patterns and
    # feedback patterns on themselves -- temperature patterns are only one where
    # regressor is different from regressee. When just want to show *one* feedback
    # can use e.g. component=('tpat', 'cld'), name=('cld', None), pairs='name'.
    print('.', end='')
    feedback = alias_to_name.get(feedback, feedback or 'rfnt_lam')
    kwargs = {}
    datas_persum = []  # each item part of a summation
    methods_persum = set()
    kws_reduce = _to_lists(*kws_reduce, equal=False)
    if any(len(kws) != len(kws_reduce[0]) for kws in kws_reduce):
        raise ValueError('Operator count mismatch in numerator and denominator.')
    for ikws_reduce in zip(*kws_reduce):
        isigns, ikws_reduce = zip(*ikws_reduce)
        datas, kw_method = [], {}
        for kw_reduce in ikws_reduce:  # iterate method reduce arguments
            kw_reduce = kw_reduce.copy()
            name = kw_reduce.pop('name')  # NOTE: always present
            area = kw_reduce.get('area')
            temps = ('tpat', 'tabs', 'rfnt_ecs')
            experiment = kw_reduce.get('experiment')
            if name in temps[:2] and experiment == 'picontrol':
                name = 'tpat'  # still use rfnt_ecs for e.g. abrupt minus pre-industrial
            if name in temps and area == 'avg' and experiment == 'picontrol' and len(ikws_reduce) == 2:  # noqa: E501
                name = feedback  # regressions of abrupt vs. pre-industrial
            # if name == 'ts' and experiment != 'abrupt4xco2-picontrol' and len(ikws_reduce) == 2:  # noqa: E501
            #     experiment = 'abrupt4xco2-picontrol'
            for key in tuple(kw_reduce):
                if key in KEYS_METHOD:
                    kw_method.setdefault(key, kw_reduce.pop(key))
            if name not in dataset:
                raise ValueError(f'Invalid name {name}. Options are {tuple(dataset.data_vars)}.')  # noqa: E501
            data = dataset[name]
            data = apply_reduce(data, **kw_reduce)
            datas.append(data)
        datas, method, default = apply_method(*datas, **kw_method)
        for key, value in default.items():
            kwargs.setdefault(key, value)
        if len(datas) == 1:  # e.g. regression applied
            isigns = (min(isigns),)
        datas_persum.append((isigns, datas))  # plotting command arguments
        methods_persum.add(method)
        if len(methods_persum) > 1:
            raise RuntimeError(f'Mixed reduction methods {methods_persum}.')

    # Combine arrays specified with reduction '+' and '-' keywords
    # NOTE: The additions below are scaled as *averages* so e.g. project='cmip5+cmip6'
    # gives the average across cmip5 and cmip6 inter-model averages.
    # WARNING: Can end up with all-zero arrays if e.g. doing single scatter or multiple
    # bar regression operations and only the dependent or independent variable relies
    # on an operations between slices (e.g. area 'nino-nina' combined with experiment
    # 'abrupt4xco2-picontrol' creates 4 datas that sum to zero). Use the kludge below.
    print('.', end=' ')
    args = []
    method = methods_persum.pop()
    signs_persum, datas_persum = zip(*datas_persum)
    for signs, datas in zip(zip(*signs_persum), zip(*datas_persum)):  # plot arguments
        idatas = []
        for i, data in enumerate(datas):
            index = data.indexes.get('facets', None)
            project = data.coords.get('project', None)
            if (
                index is not None and project is not None
                and project.size > 1 and np.all(project == project[0])
            ):
                arrays = list(zip(*index.values))  # support institute operations
                arrays[index.names.index('project')] = ['CMIP'] * index.size
                index = pd.MultiIndex.from_arrays(arrays, names=index.names)
                data = data.assign_coords(facets=index)
            idatas.append(data)
        isigns, idatas = signs, xr.align(*idatas)
        sum_scale = sum(sign == 1 for sign in isigns)
        if idatas[0].sizes.get('facets', None) == 0:
            raise RuntimeError(
                'Empty facets dimension. This is most likely due to an '
                'operation across projects without an institute average.'
            )
        with xr.set_options(keep_attrs=True):  # keep e.g. units and short_prefix
            data = sum(s * sdata for s, sdata in zip(isigns, idatas)) / sum_scale
        if len(kws_process) == 1 and (name := kws_process[0].get('name')):
            data.name = name  # e.g. 'ts' minus 'tabs'
        if method == 'dist' and len(idatas) > 1 and np.allclose(data, 0):
            data = idatas[0]
        if suffix and any(sign == -1 for sign in isigns):
            data.attrs['short_suffix'] = data.attrs['long_suffix'] = 'anomaly'
        args.append(data)

    # Align and restore coordinates
    # NOTE: Xarray automatically drops non-matching scalar coordinates (similar to
    # vector coordinate matching utilities) so try to restore them below.
    # NOTE: Commented code at bottom verifies that global average regressions of
    # regional pre-industrial feedbacks onto global pre-industrial feedbacks equal 1
    # even though there are broad positive regions with much larger magnitudes.
    args = xr.align(*args)  # re-align after summation
    if len(args) == len(kws_input):  # one or two (e.g. scatter)
        for arg, kw_input in zip(args, kws_input):
            arg.attrs.update(attrs)
            arg.coords.update(kw_input)
    else:  # create 2-tuple coordinates
        kw_input = {}
        keys = sorted(set(key for kw in kws_input for key in kw))
        for key in keys:
            values = tuple(kw.get(key, None) for kw in kws_input)
            value = values[0]
            if values[0] != values[1]:
                value = np.array(None, dtype=object)
                value[...] = tuple(values)
            kw_input[key] = value
        for arg in args:  # should be singleton
            arg.attrs.update(attrs)
            arg.coords.update(kw_input)
    # if args[0].sizes.keys() & {'lon', 'lat'}:
    #     ic(kws_process, args[0].climo.average('area').item())
    return args, method, kwargs
