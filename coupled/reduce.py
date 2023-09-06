#!/usr/bin/env python3
"""
Coordinate reduction utilities used by plotting functions.
"""
import cftime
import climopy as climo  # noqa: F401
import numpy as np
import pandas as pd
import xarray as xr
from climopy.var import _get_bounds, linefit
from climopy import ureg, vreg  # noqa: F401
from scipy import stats
from icecream import ic  # noqa: F401

from .internals import ORDER_LOGICAL
from .results import FACETS_LEVELS, VERSION_LEVELS
from cmip_data.internals import MODELS_INSTITUTES, INSTITUTES_LABELS

__all__ = ['apply_method', 'apply_reduce']


# Reduction defaults
# NOTE: Default 'style' depends on styles present. If single style is present for
# source 'eraint' then select it. Otherwise prefer 'annual' over 'monthly'.
DEFAULTS_GENERAL = {
    'experiment': 'picontrol',
    'ensemble': 'flagship',
    'period': 'ann',
    'time': 'avg',
}
DEFAULTS_VERSION = {
    'experiment': 'abrupt4xco2',
    'source': 'eraint',
    'start': 0,
    'stop': 150,
    'region': 'globe',
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
    pctile : float, default: 33
        The percentile threshold.

    Returns
    -------
    data_lo, data_hi : xarray.DataArray
        The composite components.
    """
    # NOTE: This can be used to e.g. composite the temperature pattern
    # response on models with high vs. low global average feedbacks.
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


def _components_covariance(data0, data1, resid=False, dim='facets'):
    """
    Return covariance and standard deviations of `data0` and optionally `data1`.

    Parameters
    ----------
    data0 : xarray.DataArray
        The first data. Standard deviation is always returned.
    data1 : xarray.DataArray
        The second data. Standard deviation is optionally returned.
    resid : bool, optional
        Whether to return the standard deviation of `data1` or of the residual.

    Returns
    -------
    covar, std, other : xarray.DataArray
        The covariance of `data0` and `data1`, standard deviation of `data0`, and
        either standard deviation of `data1` or standard deviation of residual.
    """
    # NOTE: Currently masked arrays are used in climopy 'covar' and might also have
    # overhead from metadata stripping stuff and permuting. So go manual here.
    # NOTE: For simplicity this gives the *biased* weighted variance estimator (analogue
    # for unweighted data is SS / N) instead of the complex *unbiased* weighted variance
    # estimator (analogue for unweighted data SS / N - 1). See full formula here:
    # https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Reliability_weights
    # NOTE: Calculation of residual leverages facts that offset = ymean - slope * xmean
    # and slope stderr = sqrt(resid ** 2 / xanom ** 2 / n - 2) where resid =
    # y - ((ymean - slope * xmean) + slope * x) = (y - ymean) - slope * (x - xmean)
    # https://en.wikipedia.org/wiki/Simple_linear_regression#Normality_assumption
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
    std = np.sqrt((weight * anom0 ** 2).sum(dims) / (weight * mask0).sum(dims))
    if not resid:
        other = np.sqrt((weight * anom1 ** 2).sum(dims) / (weight * mask1).sum(dims))
    elif not any(key in dims for key in ('lon', 'lat', 'plev')):
        resid = anom1 - (covar / std ** 2) * anom0  # NOTE: n - 2 is factored in later
        other = np.sqrt((weight * resid ** 2).sum(dims) / (weight * mask0 * mask1).sum(dims))  # noqa: E501
    else:
        raise NotImplementedError('Unsure how to calculate weighted standard error.')
    return covar, std, other


def _components_corr(data0, data1, dim=None, pctile=None):
    """
    Return components of a correlation evalutation.

    Parameters
    ----------
    data0, data1 : xarray.DataArray
        The coordinates to be compared.
    dim : str, optional
        The dimension for the correlation.
    pctile : float, default: 95
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
    pctile = np.atleast_1d(95 if pctile is None else pctile)
    pctile = 0.5 * (100 - pctile)  # e.g. [90, 50] --> [[5, 25], [95, 75]]
    pctile = np.array([pctile, 100 - pctile])
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
    pctile : bool or float, default: 95
        The percentile range for the lower and upper uncertainty bounds.
        Use ``False`` to instead just show standard error ranges.

    Returns
    -------
    slope, slope_lower, slope_upper : xarray.DataArray
        The slope estimates with `pctile` lower and upper bounds.
    rsquare : xarray.DataArray
        The variance explained by the fit.
    fit, fit_lower, fit_upper : xarray.DataArray
        The fit to the original points with `pctile` lower and upper bounds.
    """
    # NOTE: Here np.polyfit requires monotonically increasing coordinates. Not sure
    # why... could consider switching to manual slope and stderr calculation.
    # NOTE: Unlike climopy linefit(), which returns scalar slope standard error and
    # best fit uncertainty range with *optional* percentile dimension, this returns
    # slope estimate uncertainty range with *mandatory* percentile dimension and
    # a single best fit uncertainy range. In this project require former for thin and
    # thick whiskers on _combine_command() bar plots while a single best fit range
    # is sufficient for most scatter and regression plots. Should merge with linefit().
    dim = dim or data0.dims[0]
    data0, data1 = xr.align(data0, data1)
    axis = data0.dims.index(dim)
    isel = {dim: np.argsort(data0.values, axis=axis)}
    data0, data1 = data0.isel(isel), data1.isel(isel)
    slope, sigma, rsquare, fit, fit_lower, fit_upper = linefit(
        data0, data1, dim=dim, adjust=adjust, pctile=pctile,
    )
    coords = {'x': data0, 'y': data1}
    fit.coords.update(coords)
    fit_lower.coords.update(coords)
    fit_upper.coords.update(coords)
    if pctile is False:  # use standard errors
        slope_lower, slope_upper = slope - sigma, slope + sigma
    else:
        pctile = 95 if pctile is None or pctile is True else pctile
        pctile = 0.5 * (100 - np.atleast_1d(pctile))
        pctile = np.array([pctile, 100 - pctile])  # e.g. [90, 50] --> [[5, 25], [95, 75]]  # noqa: E501
        sigma_lower, sigma_upper = _get_bounds(sigma, pctile, dof=data0.size - 2)
        slope_lower = slope + xr.DataArray(sigma_lower, dims=('pctile', *sigma.dims))
        slope_upper = slope + xr.DataArray(sigma_upper, dims=('pctile', *sigma.dims))
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


def _method_single(data, dim=None, method=None, pctile=None, std=None):
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
        use default values of ``80`` for ``avg|med`` and ``50`` for ``pctile``.
        Pass two values, e.g. ``[80, 100]``, to use both `shade` and `fade` keywords.
    std : float or sequence, optional
        The standard deviation multiple for related methods. Set to ``True`` to use
        default values of ``1`` for ``std`` and ``3`` for ``avg|med`` shading.
        Pass two values, e.g. ``[1, 3]``, to use both `shade` and `fade` keywords.

    Returns
    -------
    data : xarray.DataArray
        The resulting data.
    defaults : dict
        The default attributes and command keyword args.
    """
    # NOTE: Here `pctile` is shared between inter-model percentile differences and
    # composites of a second array based on values in the first array.
    method = method or 'avg'
    ndim = data.ndim
    kw = {'dim': dim or 'facets', 'skipna': True}
    name = data.name
    short = long = None
    defaults = {}
    if method == 'avg' or method == 'med':
        key = 'mean' if method == 'avg' else 'median'
        descrip = 'mean' if method == 'avg' else 'median'
        std = std.tolist() if isinstance(std, np.ndarray) else std
        pctile = pctile.tolist() if isinstance(pctile, np.ndarray) else pctile
        if not std and not pctile:
            assert ndim < 4
            cmd = getattr(data, key)
            data = cmd(**kw, keep_attrs=True)
            short = f'{descrip} {data.short_name}'  # only if no range included
            long = f'{descrip} {data.long_name}'
        elif std:  # standard deviation range
            assert ndim == 2
            # std = [1, 3] if std is True else std
            std = 3 if std is True else std
            std = np.atleast_1d(std)
            shade, fade = std if std.size == 2 else (std.item(), None)
            defaults.update({key: True, 'shadestds': shade, 'fadestds': fade})
            defaults.update({'fadealpha': 0.15, 'shadealpha': 0.3})
        elif pctile:  # percentile range
            assert ndim == 2
            # pctile = [50, 95] if pctile is True else pctile
            pctile = 80 if pctile is True else pctile
            pctile = np.atleast_1d(pctile)
            shade, fade = pctile if pctile.size == 2 else (pctile.item(), None)
            defaults.update({key: True, 'shadepctiles': shade, 'fadepctiles': fade})
            defaults.update({'fadealpha': 0.15, 'shadealpha': 0.3})
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
    elif method == 'skew':  # see: https://stackoverflow.com/a/71149901/4970632
        assert ndim < 4
        with xr.set_options(keep_attrs=True):
            data = data.reduce(func=stats.skew, **kw)
        data.attrs['units'] = ''
        short = f'{data.short_name} skewness'
        long = f'{data.long_name} skewness'
    elif method == 'kurt':  # see: https://stackoverflow.com/a/71149901/4970632
        assert ndim < 4
        with xr.set_options(keep_attrs=True):
            data = data.reduce(func=stats.kurtosis, **kw)
        data.attrs['units'] = ''
        short = f'{data.short_name} kurtosis'
        long = f'{data.long_name} kurtosis'
    elif method == 'dist':  # bars or boxes
        assert ndim == 1
        data = data[~data.isnull()]
        name = None
    else:
        raise ValueError(f'Invalid single-variable method {method}.')
    defaults.update({'name': name, 'short_name': short, 'long_name': long})
    return data, defaults


def _method_double(data0, data1, dim=None, method=None, pctile=None, invert=None):
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

    Other Parameters
    ----------------
    pctile : float or sequence, optional
        The percentile range or thresholds for related methods. Set to ``True`` to
        use default values of ``95`` for ``cov|proj|slope|`` and ``33`` for ``diff``.
        Pass two values, e.g. ``[50, 95]``, to use both `shade` and `fade` keywords.
    invert : bool, optional
        Whether to invert the direction of composites, projections, and regressions so
        that the first variable is the predictor instead of the second variable. This
        has no effect on ``cov|corr|rsq``.

    Returns
    -------
    data : xarray.DataArray
        The resulting data.
    defaults : dict
        The default attributes and command keyword args.
    """
    # NOTE: Normalization and anomalies with respect to global average
    # are supported in `apply_method`.
    data0, data1 = (data1, data0) if invert else (data0, data1)
    method = method or 'cov'
    short = long = None
    name = f'{data0.name}|{data1.name}'  # NOTE: needed for _combine_commands labels
    ndim = max(data0.ndim, data1.ndim)
    dim = dim or 'facets'
    defaults = {}
    if dim == 'area':
        short_prefix, long_prefix = 'spatial', f'{data1.long_name} spatial'
    elif data0.long_name == data0.long_name:
        short_prefix, long_prefix = data1.short_name, data1.long_name
    else:
        short_prefix, long_prefix = data1.short_name, f'{data1.long_name}/{data0.long_name}'  # noqa: E501
    if method in ('corr', 'rsq'):
        cov, std0, std1 = _components_covariance(data0, data1, dim=dim, resid=False)
        data = cov / (std0 * std1)
        if method == 'corr':  # correlation coefficient
            data = data.climo.to_units('dimensionless').climo.dequantify()
            data.attrs['units'] = ''
            short = f'{short_prefix} correlation'
            long = f'{long_prefix} correlation coefficient'
        else:  # variance explained
            data = (data ** 2).climo.to_units('percent').climo.dequantify()
            data.attrs['units'] = '%'
            short = f'{short_prefix} variance explained'
            long = f'{long_prefix} variance explained'
    elif method in ('cov', 'proj', 'slope'):
        cov, std, resid = _components_covariance(data0, data1, dim=dim, resid=True)
        factor = 0 if method == 'cov' else 1 if method == 'proj' else 2
        data = (cov / std ** factor).climo.dequantify()
        if method == 'cov':
            data.attrs['units'] = f'{data1.units} {data0.units}'
            short = f'{short_prefix} covariance'
            long = f'{long_prefix} covariance'
        elif method == 'proj':
            data.attrs['units'] = data1.units
            short = f'{short_prefix} projection'
            long = f'{long_prefix} projection'
        else:
            data.attrs['units'] = f'{data1.units} / ({data0.units})'
            short = f'{short_prefix} regression coefficient'
            long = f'{long_prefix} regression coefficient'
        if ndim == 2 and pctile is not None:  # NOTE: implies one dimension was reduced
            dof = data0.sizes[dim] - 2  # must be scalar dimension
            sigma = np.sqrt(resid / std / dof)  # this is sqrt(n)^-1 / sqrt(n)^-1 / dof
            sigma = (sigma * std ** (2 - factor)).climo.dequantify()  # scaled stderr
            pctile = 0.5 * (100 - np.atleast_1d(95 if pctile is True else pctile))
            pctile = np.array([pctile, 100 - pctile])  # e.g. [90, 50] --> [[5, 25], [95, 75]]  # noqa: E501
            dlower, dupper = _get_bounds(sigma, pctile, dof=data0.size - 2)
            bounds = [(data + dl, data + du) for dl, du in zip(dlower, dupper)]
            bounds = [xr.DataArray(b, dims=data.dims) for bnds in bounds for b in bnds]
            data = xr.concat((data, *bounds), dim='pctile')  # pull out after operation
            defaults.update({'fadealpha': 0.15, 'shadealpha': 0.25})
    elif method == 'diff':  # composite difference along first arrays
        assert dim != 'area'
        data_lo, data_hi = _components_composite(data0, data1, pctile=pctile, dim=dim)
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
    defaults.update({'name': name, 'short_name': short, 'long_name': long})
    return data, defaults


def apply_method(*datas, method=None, verbose=False, invert=None, pctile=None, std=None):  # noqa: E501
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
        reduce the facets dimension for two input arguments. Can also end string
        with ``_anom`` or ``_norm`` to take global anomaly or use normalization.
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
    # Apply single or double reduction methods
    # NOTE: This supports on-the-fly anomalies and normalization. Should eventually
    # move this stuff to climopy (already implemented in accessor getter).
    ndim = max(data.ndim for data in datas)
    datas = tuple(data.copy() for data in datas)  # e.g. for distribution updates
    default = 'dist' if ndim == 1 else 'avg' if len(datas) == 1 else 'rsq'
    method = method or default
    method, *options = method.split('_')
    anomaly = 'anom' in options
    normalize = 'norm' in options
    if len(datas) == 1:
        data, defaults = _method_single(*datas, method=method, std=std, pctile=pctile)
    elif len(datas) == 2:
        data, defaults = _method_double(*datas, method=method, invert=invert, pctile=pctile)  # noqa: E501
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
    # TODO: Might consider adding 'institute' multiindex so e.g. .groupby('institute')
    # is possible... then again would not reduce complexity because would need to
    # load files in special non-alphabetical order to enable selecting 'flaghip'
    # models with e.g. something like .sel(institute=-1)... so don't bother for now.
    # WARNING: Critical to do this at end since .groupby() does not seem to handle
    # cell measures that vary across models i.e. cell_height() used for averages.
    institute = kwargs.pop('institute', None)  # apply after end
    project = kwargs.pop('project', None)  # apply after end
    if institute is not None:  # WARNING: critical this comes first
        institute = _parse_institute(data, institute)
        if callable(institute):  # facets filter function
            facets = list(filter(institute, data.facets.values))
            data = data.sel(facets=facets)
        else:  # TODO: group inside separate function?
            height = data.coords.get('cell_height', None)
            data = data.groupby(institute).mean(skipna=False, keep_attrs=True)
            if height is not None and 'facets' in height.dims:  # also varies by model!
                height = height.groupby(institute).mean(skipna=False, keep_attrs=True)
                data.coords['cell_height'] = height
            dict_ = institute.facets.attrs  # WARNING: avoid overwriting input attrs
            facets = data.indexes['facets']  # WARNING: xarray bug drops level names
            facets.names = institute.indexes['facets'].names
            facets = xr.DataArray(facets, name='facets', dims='facets', attrs=dict_)
            data = data.assign_coords(facets=facets)  # assign with restored levels
    if project is not None:  # see _parse_project
        project = _parse_project(data, project)
        facets = list(filter(project, data.facets.values))
        data = data.sel(facets=facets)
        data = data.reset_index('project', drop=True)  # models are unique

    # Apply time reductions and grouped averages
    # TODO: Replace 'average_periods' with this and normalize by annual temperature.
    # Also should add generalized no-op instruction for leaving coordinates alone.
    # NOTE: The new climopy cell duration calculation will auto-detect monthly and
    # yearly data, but not yet done, so use explicit days-per-month weights for now.
    season = kwargs.pop('season', None)
    month = kwargs.pop('month', None)
    time = kwargs.pop('time', None)  # then get average later on
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
    spatial_ignore = ('area', 'spatial', 'start', 'stop', 'experiment')
    order = list(ORDER_LOGICAL)
    sorter = lambda item: order.index(item[0]) if item[0] in order else len(order)  # noqa: U101, E501
    result = []
    if is_dataset := isinstance(data, xr.Dataset):
        datas = data.data_vars.values()
    else:
        datas = (data,)
    for data in datas:
        # Apply default values and possible overrides
        # NOTE: None is automatically replaced with default values (e.g. period=None)
        # and default experiment depends on whether this is feedback variable or not.
        kw = kwargs.copy()
        name = data.name
        attrs = data.attrs.copy()
        attrs.update(attrs or {})
        defaults = DEFAULTS_GENERAL.copy()
        sources = ('zelinka', 'forster', 'geoffroy')  # ignored feedback sources
        if 'version' in data.coords:
            defaults.update(DEFAULTS_VERSION)
        if data.sizes.get('version', 0) > 1:  # skip _derive_data
            values = zip(data.source.values, data.style.values)
            styles = {style for source, style in values if source not in sources}
            if len(styles) > 1:  # prefer 'annual' as result of pop() below
                styles = [style for style in ('monthly', 'annual') if style in styles]
            if len(styles) > 0:
                defaults['style'] = styles.pop()
        for key, value in defaults.items():  # apply default reductions
            if 'startstop' in kw:  # inside loop for aesthetics only
                kw['start'], kw['stop'] = kw.pop('startstop')
            if key in kw and kw[key] is None and key != 'time':
                kw[key] = value  # typically use 'None' as default placeholder
            else:  # for 'time' use 'None' to bypass operation
                kw.setdefault(key, value)
        if 'version' in data.coords:
            experiment, region, period = kw['experiment'], kw['region'], kw['period']
            if period[0] == 'a' and period != 'ann':  # abrupt-only period
                kw['period'] = 'ann' if experiment == 'picontrol' else period[1:]
            if region[0] == 'a':  # abrupt-only region
                kw['region'] = 'globe' if experiment == 'picontrol' else region[1:]
            if name in ('tpat', 'tstd', 'tdev', 'tabs'):  # others undefined so force
                kw['region'] = 'globe'
            if experiment == 'picontrol':  # others undefined so overwrite
                kw['start'], kw['stop'] = 0, 150

        # Iterate over reductions
        # NOTE: This silently skips dummy selections (e.g. area=None) that may be needed
        # to prevent _parse_specs from merging e.g. average and non-average selections.
        # WARNING: Sometimes multi-index reductions can eliminate previously valid
        # coords, so critical to iterate one-by-one and validate selections each time.
        for key, value in sorted(kw.items(), key=sorter):
            opts = [*data.sizes, 'area', 'volume', 'spatial']
            opts.extend(level for idx in data.indexes.values() for level in idx.names)
            dims = data.sizes.keys() & {'lon', 'lat', 'plev'}
            if key not in opts or value is None:
                continue
            if key in spatial_ignore and kw.get('spatial', None):
                continue
            if key == 'area' and not dims - {'plev'} or key == 'volume' and not dims:
                continue
            if key == 'area':
                region = AREA_REGIONS.get(value, None)
                if region is not None:
                    data, value = data.climo.truncate(region), 'avg'
                elif value != 'avg':
                    raise ValueError(f'Unknown averaging region {value!r}.')
            if key == 'time' and 'time' in data.sizes:  # manual weighted average
                if value == 'avg':
                    days = data.time.dt.days_in_month.astype(data.dtype)
                    with xr.set_options(keep_attrs=True):  # average over entire record
                        data = (data * days).sum('time', skipna=False) / days.sum()
                elif value is not None:
                    raise ValueError(f'Unknown time reduction method {time!r}.')
                continue
            if key in data.coords and not isinstance(value, (str, tuple)):
                unit = data.coords[key].climo.units
                if isinstance(value, ureg.Quantity):
                    value = value.to(unit)
                else:
                    value = ureg.Quantity(value, unit)
            data = data.climo.reduce({key: value}, method='interp')
            data = data.squeeze()
            if key not in data.coords:
                data.coords[key] = value  # converts to data array
                data.coords[key] = data.coords[key].climo.dequantify()
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

        # Return after optionally ensuring settings
        # TODO: Check if name or attributes ever do go missing?
        data.name = name
        data.attrs.update(attrs)
        result.append(data)

    # Re-combine and return
    if is_dataset:
        data = xr.Dataset({data.name: data for data in result})
    else:
        data, = result
    return data
