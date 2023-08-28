#!/usr/bin/env python3
"""
Processing utilities used by plotting functions.
"""
import itertools
import math
import re
import warnings

import cftime
import climopy as climo  # noqa: F401
import numpy as np
import pandas as pd
import xarray as xr
from climopy.var import _get_bounds, linefit
from climopy import ureg, vreg  # noqa: F401
from scipy import stats
from icecream import ic  # noqa: F401

from .internals import ALIAS_FEEDBACKS, FEEDBACK_ALIASES, KEYS_METHOD, KEYS_VARIABLE, ORDER_LOGICAL  # noqa: E501
from .internals import _group_parts, _ungroup_parts, _to_lists
from .results import FACETS_LEVELS, REGEX_FLUX, VERSION_LEVELS
from cmip_data.internals import MODELS_INSTITUTES, INSTITUTES_LABELS

__all__ = ['apply_method', 'apply_reduce', 'get_data', 'process_data']

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

# Observational constraints
# TODO: Auto-generate this dictionary or save and commit then parse a text file.
# NOTE: These show best estimate, standard errors, and degrees of freedom for feedback
# regression slopes using either GISTEMP4 or HadCRUT5. See observed.ipynb notebook.
FEEDBACK_CONSTRAINTS = {
    # 'cld': (0.69, (1.04 - 0.31) / 2,  # He et al. with 95% uncertainty
    # 'cld': (0.68, 0.18, 226),  # custom trended estimate
    'cld': (0.35, 0.26, 226),  # custom detrended estimate
    'net': (-0.84, 0.29, 274),
    'sw': (0.86, 0.24, 274),
    'lw': (-1.70, 0.19, 274),
    'cre': (0.27, 0.23, 274),
    'swcre': (0.19, 0.24, 274),
    'lwcre': (0.08, 0.13, 274),
    # 'net': (-0.84, 0.31, 249),  # correlation-adjusted error
    # 'sw': (0.86, 0.25, 242),
    # 'lw': (-1.70, 0.19, 274),
    # 'cre': (0.27, 0.23, 274),
    # 'swcre': (0.19, 0.24, 274),
    # 'lwcre': (0.08, 0.15, 232),
    'hadnet': (-0.84, 0.31, 274),
    'hadsw': (0.88, 0.25, 274),
    'hadlw': (-1.72, 0.21, 274),
    'hadcre': (0.30, 0.24, 274),
    'hadswcre': (0.15, 0.26, 274),
    'hadlwcre': (0.15, 0.14, 274),
    # 'hadnet': (-0.84, 0.31, 274),  # correlation-adjusted error
    # 'hadsw': (0.88, 0.25, 274),
    # 'hadlw': (-1.72, 0.21, 274),
    # 'hadcre': (0.30, 0.24, 274),
    # 'hadswcre': (0.15, 0.26, 274),
    # 'hadlwcre': (0.15, 0.14, 274),
}

# Variable dependencies
# TODO: Instead improve climopy algorithm to only return derivations or
# transformations if dependent variables exist in the dataset.
VARIABLE_DEPENDENCIES = {
    'pt': ('ta',),
    'slope': ('ta',),
    'slope_bulk': ('ta',),
    'dtdy': ('ta',),  # definition related to slope
    'dtdz': ('ta',),  # definition related to slope
    'dptdy': ('ta',),  # definition used for slope
    'dptdz': ('ta',),  # definition used for slope
    'dtdy_bulk': ('ta',),  # definition related to slope_bulk
    'dtdz_bulk': ('ta',),  # definition related to slope_bulk
    'dptdy_bulk': ('ta',),  # definition used for slope_bulk
    'dptdz_bulk': ('ta',),  # definition used for slope_bulk
    'ts_grad': ('ts',),
    'ta_grad': ('ta',),
    'ts_diff': ('ts', 'lset', 'dset'),
    'ta_diff': ('ta', 'lset', 'dset'),
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

# Define additional variables
# TODO: Use this approach for flux addition terms and other stuff
with warnings.catch_warnings():
    # Add definitions
    warnings.simplefilter('ignore')
    vreg.define('ta_grad', 'equator-pole air temperature difference', 'K')
    vreg.define('ts_grad', 'equator-pole surface temperature difference', 'K')
    vreg.define('ta_diff', 'bulk energy transport diffusivity', 'PW / K')
    vreg.define('ts_diff', 'bulk energy transport surface diffusivity', 'PW / K')

    @climo.register_derivation(re.compile(r'\A(ta|ts)_grad\Z'))
    def equator_pole_delta(self, name):
        temp, _ = name.split('_')
        temp = self[temp]  # also quantifies
        if temp.name == 'ta':
            temp = temp.sel(lev=850)  # traditional heat transport pressure
        equator = temp.sel(lat=slice(0, 10)).climo.average('area')
        pole = temp.sel(lat=slice(60, 90)).climo.average('area')
        return equator - pole

    @climo.register_derivation(re.compile(r'\A(ta|ts)_diff\Z'))
    def equator_pole_diffusivity(self, name):
        temp, _ = name.split('_')
        delta = f'{temp}_grad'
        delta = self.get(delta)
        transport = self['lset'] + self['dset']  # moist not directly available
        transport = transport.sel(lat=slice(0, 90)).climo.average('area')
        return transport / delta


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
    # TODO: Update linefit() to return either slope and fit lower and upper bounds
    # or just the slope sigma that can be used to calculate other stuff.
    # TODO: Copy Dessler and Forster methodology for getting effective degrees
    # of freedom for time series with autocorrelated components.
    # NOTE: Here np.polyfit requires monotonically increasing coordinates. Not sure
    # why... could consider switching to manual slope and stderr calculation.
    dim = dim or data0.dims[0]
    data0, data1 = xr.align(data0, data1)
    axis = data0.dims.index(dim)
    idx = np.argsort(data0.values, axis=axis)
    data0 = data0.isel({dim: idx})
    data1 = data1.isel({dim: idx})
    slope, sigma, rsquare, fit, fit_lower, fit_upper = linefit(
        data0, data1, dim=dim, adjust=adjust, pctile=pctile,
    )
    fit.coords.update({'x': data0, 'y': data1})
    fit_lower.coords.update({'x': data0, 'y': data1})
    fit_upper.coords.update({'x': data0, 'y': data1})
    if pctile is False:  # use standard errors
        dslope_lower, dslope_upper = -1 * sigma, sigma
    else:
        pctile = 95 if pctile is None or pctile is True else pctile
        pctile = 0.5 * (100 - np.atleast_1d(pctile))
        pctile = np.array([pctile, 100 - pctile])  # e.g. [90, 50] --> [[5, 25], [95, 75]]  # noqa: E501
        dslope_lower, dslope_upper = _get_bounds(sigma, pctile, dof=data0.size - 2)
        dslope_lower = xr.DataArray(dslope_lower, dims=('pctile', *sigma.dims))
        dslope_upper = xr.DataArray(dslope_upper, dims=('pctile', *sigma.dims))
    slope_lower, slope_upper = slope + dslope_lower, slope + dslope_upper
    return slope, slope_lower, slope_upper, rsquare, fit, fit_lower, fit_upper


def _constrain_response(
    data0, data1, constraint=None, pctile=None, N=None, graphical=False
):
    """
    Return percentile bounds for observational constraint.

    Parameters
    ----------
    data0 : xarray.DataArray
        The predictor data. Must be 1D.
    data1 : xarray.DataArray
        The predictand data. Must be 1D.
    constraint : 2-tuple, default: (0.32, 1.06)
        The 95% bounds on the observational constraint.
    pctile : float, default: 95
        The emergent constraint percentile bounds to be returned.
    N : int, default: 100000
        The number of bootstrapped samples to carry out.
    graphical : bool, optional
        Whether to use graphical intersections instead of bootstrapping.

    Returns
    -------
    observations : 3-tuple
        The lower, mean, and upper bounds for the observational constraint.
    result1 : 3-tuple
        The emergent constraint not accounting for regression uncertainty.
    result2 : 3-tuple
        The emergent constraint accounting for regression uncertainty.
    """
    # NOTE: Below we reverse engineer the t-distribution associated with observational
    # estimate, then use those distribution properties for our bootstrapping.
    # NOTE: Below tried adding 'offset uncertainty' but resulted in equivalent spread
    # compared to standard slop-only regression uncertainty.
    # NOTE: Use N - 2 degrees of freedom for both observed feedback and inter-model
    # coefficients since they are both linear regressions. For now ignore uncertainty
    # of individual feedback regressions that comprise inter-model regression because
    # their sample sizes are larger (150 years) and uncertainties are unc-rrelated so
    # should roughly cancel out across e.g. 30 members of ensemble.
    N = N or 10000  # samples to draw
    pctile = 95 if pctile is None else pctile
    pctile = 0.5 * (100 - pctile)  # e.g. [90, 50] --> [[5, 25], [95, 75]]
    pctile = np.array([pctile, 100 - pctile])
    # steps = 120  # approximate degrees of freedom in Dessler et al.
    # steps = 6 * (2019 - 2001 + 1)  # number of years in He et al. estimate
    if isinstance(constraint, str):
        constraint = FEEDBACK_ALIASES.get(constraint, constraint)
    elif not np.iterable(constraint) or len(constraint) != 3:
        raise ValueError(f'Invalid constraint {constraint}. Must be length 2.')
    if data0.ndim != 1 or data1.ndim != 1:
        raise ValueError(f'Invalid data dims {data0.ndim} and {data1.ndim}. Must be 1D.')  # noqa: E501
    xmean, xscale, xdof = FEEDBACK_CONSTRAINTS[constraint]
    xmin, xmax = stats.t.ppf(0.01 * pctile, scale=xscale, df=xdof)
    observations = (xmin, xmean, xmax)
    data0 = np.array(data0).squeeze()
    data1 = np.array(data1).squeeze()
    idxs = np.argsort(data0, axis=0)
    data0, data1 = data0[idxs], data1[idxs]  # required for graphical error
    mean0, mean1 = data0.mean(), data1.mean()
    bmean, berror, rsquare, fit, fit_lower, fit_upper = linefit(
        data0, data1, adjust=False, pctile=pctile
    )
    if not graphical:  # bootstrapped residual addition
        # xscore = stats.t.isf(0.025, xdof)  # t score associated with 95% bounds
        # xerror = (xmax - xmean) / xscore  # used when percentiles were hardcoded
        rerror = np.sqrt((1 - rsquare) * np.var(data1, ddof=1))  # model residual sigma
        rdof = data0.size - 2  # inter-model regression dof
        xs = stats.t.rvs(xdof, loc=xmean, scale=xscale, size=N)  # observations
        err = stats.t.rvs(rdof, loc=0, scale=rerror, size=N)  # regression
        ys = err + mean1 + bmean * (xs - mean0)
        # amean = mean1 - mean0 * bmean  # see wiki page
        # aerror = berror * np.sqrt(np.sum(data0 ** 2) / data0.size)  # see wiki page
        # bs_ = stats.t.rvs(data0.size - 2, loc=bmean, scale=berror, size=N)
        # as_ = stats.t.rvs(data0.size - 2, loc=amean, scale=aerror, size=N)
        # ys = err + as_ + bs_ * xs  # include both uncertainties
        # ys = mean1 + bs_ * (xs - mean0)  # include slope uncertainty only
        ymean = np.mean(ys)
        ymin, ymax = np.percentile(ys, pctile)
        constrained = (ymin, ymean, ymax)
        ys = mean1 + bmean * (xs - mean0)  # no regression uncertainty
        ymean = np.mean(ys)
        ymin, ymax = np.percentile(ys, pctile)
        alternative = (ymin, ymean, ymax)
    else:  # intersection of shaded regions
        fit_lower = fit_lower.squeeze()
        fit_upper = fit_upper.squeeze()
        xs = np.sort(data0, axis=0)  # linefit returns result for sorted data
        ymin = np.interp(xmin, xs, fit_lower)
        ymax = np.interp(xmax, xs, fit_upper)
        ymean = mean1 + bmean.item() * (xmean - mean0)
        constrained = (ymin, ymean, ymax)
        ymin, ymax = mean1 + bmean * (np.array(observations) - mean0)
        alternative = (ymin, ymean, ymax)  # no regression uncertainty
    return observations, alternative, constrained


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
    # Apply methods
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
    # Apply 'facets' reductions
    # WARNING: Critical to do this at end since .groupby() does not seem to handle
    # cell measures that vary across models i.e. cell_height() used for averages.
    # TODO: Might consider adding 'institute' multiindex so e.g. .groupby('institute')
    # is possible... then again would not reduce complexity because would need to
    # load files in special non-alphabetical order to enable selecting 'flaghip'
    # models with e.g. something like .sel(institute=-1)... so don't bother for now.
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

    # Apply time reductions
    # TODO: Replace 'average_periods' with this and normalize by annual temperature.
    # Also should add generalized no-op instruction for leaving coordinates alone.
    # NOTE: The new climopy cell duration calculation will auto-detect monthly and
    # yearly data, but not yet done, so use explicit days-per-month weights for now.
    season = kwargs.pop('season', None)
    month = kwargs.pop('month', None)
    time = kwargs.pop('time', 'avg')  # same as _apply_single() and climo.reduce()
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
        elif not isinstance(time, str) and time is not None:
            time = time  # should already be datetime
            data = data.sel(time=time, method='nearest')
    if 'time' in data.dims and time == 'avg':  # manual weighted average
        days = data.time.dt.days_in_month
        wgts = days.groupby('time.year') / days.groupby('time.year').sum()
        wgts = wgts.astype(np.float32)  # preserve float32 variables
        ones = xr.where(data.isnull(), 0, 1)
        ones = ones.astype(np.float32)  # preserve float32 variables
        with xr.set_options(keep_attrs=True):
            numerator = (data * wgts).groupby('time.year').sum(dim='time')
            denominator = (ones * wgts).groupby('time.year').sum(dim='time')
            data = numerator / denominator

    # Iterate over data arrays
    # TODO: Also support operations directly on arrays
    result = []
    is_dataset = isinstance(data, xr.Dataset)
    if is_dataset:
        datas = data.data_vars.values()
    else:
        datas = (data,)
    for data in datas:
        # Apply default values and possible overrides
        # NOTE: Delay application of defaults until here so that we only include default
        # selections in automatically-generated labels if user explicitly passed them.
        # NOTE: None is automatically replaced with default values (e.g. period=None)
        # and default experiment depends on whether this is feedback variable or not.
        kw = kwargs.copy()
        name = data.name
        attrs = data.attrs.copy()
        attrs.update(attrs or {})
        defaults = GENERAL_DEFAULTS.copy()
        if 'version' in data.coords:
            defaults.update({'experiment': 'abrupt4xco2', **VERSION_DEFAULTS})
        if 'startstop' in kw:  # possibly passed from process_data()
            kw['start'], kw['stop'] = kw.pop('startstop')  # possibly None
        for key, value in defaults.items():  # apply default reductions
            if key in kw and kw[key] is None:
                kw[key] = value  # used 'None' as default placeholder
            else:
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
        order = list(ORDER_LOGICAL)
        sorter = lambda item: order.index(item[0]) if item[0] in order else len(order)  # noqa: U101, E501
        ignore = ('area', 'spatial', 'start', 'stop', 'experiment')
        ignore = ignore if kw.get('spatial', None) else ()
        for key, value in sorted(kw.items(), key=sorter):
            # Parse input instructions
            opts = list(data.sizes)
            opts.extend(('area', 'volume', 'spatial'))
            opts.extend(level for idx in data.indexes.values() for level in idx.names)  # noqa: E501
            if key not in opts or key in ignore or value is None:
                continue
            if key == 'volume' and not data.sizes.keys() & {'lon', 'lat', 'plev'}:
                continue
            if key == 'area' and not data.sizes.keys() & {'lon', 'lat'}:
                continue
            if key == 'area':
                region = AREA_REGIONS.get(value, None)
                if region is not None:
                    data, value = data.climo.truncate(region), 'avg'
                elif value != 'avg':
                    raise ValueError(f'Unknown averaging region {value!r}.')
            if key in data.coords and not isinstance(value, (str, tuple)):
                unit = data.coords[key].climo.units
                if isinstance(value, ureg.Quantity):
                    value = value.to(unit)
                else:
                    value = ureg.Quantity(value, unit)
            # Carry out reduction and restore coordinates
            # dimensions = ('lon', 'lat', 'plev', 'area', 'volume')
            # if key in dimensions:
            #     data = data.climo.add_cell_measures()
            data = data.climo.reduce({key: value}, method='interp').squeeze()
            data = data.squeeze()
            if key not in data.coords:
                data.coords[key] = value
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


def get_data(dataset, name, attr=None, **kwargs):
    """
    Get derived data component from the dataset.

    Parameters
    ----------
    dataset : xarray.Dataset
        The input dataset.
    name : str
        The variable name.
    attr : str, optional
        The attribute.

    Returns
    -------
    result : xarray.DataArray or str
        If `attr` is ``None`` this is the resulting data array. If `attr` is
        `str` this is the corresponding attribute of the derived variable.
    """
    # Combine input keys and adjust attributes
    # NOTE: When retrieving attributes only return first array in summation (whose
    # attributes would be saved by keep_attrs=True) instead of performing operation.
    # This replicates function of cfvariables for quickly retrieving metadata.
    def _operate_data(keys, search=None, replace=None, signs=None, product=False, **attrs):  # noqa: E501
        signs = signs or (1,) * len(keys)
        if len(signs) != len(keys):
            raise RuntimeError('Mismatch between signs and keys.')
        if attr:  # skip computation and just get the first item
            data = _find_data(keys[0])
        elif product:  # get summation of items
            with xr.set_options(keep_attrs=True):
                data = math.prod(_find_data(key) ** sign for sign, key in zip(signs, keys))  # noqa: E501
        else:
            with xr.set_options(keep_attrs=True):
                data = sum(sign * _find_data(key) for sign, key in zip(signs, keys))
        data = data.copy(deep=False)  # update attributes on derived variable
        for key in ('short_name', 'long_name', 'standard_name'):
            if not search:
                continue
            if key not in data.attrs:
                continue
            value = data.attrs[key]
            if isinstance(search, str):
                value = value.replace(search, replace)
            else:
                value = search.sub(replace, value)
            data.attrs[key] = value
        data.attrs.update(attrs)  # arbitrary overrides
        return data

    # Get variable or derivation
    # TODO: Should combine climopy and this approach by 1) translating instructions,
    # 2) selecting relevant variables, 3) performing scalar selections first (e.g.
    # points, maxima), 4) combining variables (e.g. products, sums), then 5) more
    # complex coordinate operations (e.g. averages, integrals). And optionally order
    # operations around non-linear variable or coordinate operations.
    # NOTE: Here apply_reduce will automatically perform default selections for certain
    # dimensions (only carry out when requesting data). Also gets net flux (longwave
    # plus shortwave), atmospheric flux (top-of-atmosphere plus surface), cloud effect
    # (all-sky minus clear-sky), radiative response (full minus effective forcing),
    # net imbalance (downwelling minus upwelling), and various transport terms.
    def _find_data(key, **attrs):
        key = ALIAS_FEEDBACKS.get(key, key)
        subs = lambda reg, *ss: tuple(reg.sub(s, key).strip('_') for s in ss)
        flux = REGEX_FLUX.search(key)
        if key in dataset.climo:  # get reduced data before adding stuff
            if key in dataset:
                data = dataset[key]
            elif deps := VARIABLE_DEPENDENCIES.get(key, None):
                var = vreg[key]
                names = ('short_name', 'long_name', 'standard_units')
                attrs.update({attr: getattr(var, attr) for attr in names})
                if attr:  # TODO: throughout utilities permit cfvariable attributes
                    data = xr.DataArray([], name=key)
                else:
                    data = dataset.drop_vars(dataset.data_vars.keys() - set(deps))
            else:
                raise ValueError(f'Unknown dependencies for derived variable {key}.')
            if not attr:  # carry out reductions
                data = apply_reduce(data, **kwargs)
                if 'plev' in data.coords:  # TODO: make derivations compatible with plev
                    data = data.rename(plev='lev')
                if hemi and 'lat' in data.sizes:
                    data = data.climo.sel_hemisphere(hemi)
                if isinstance(data, xr.Dataset):  # finally get the derived variable
                    data = data.climo.get(key, **kw_get)
                if 'lev' in data.coords:
                    data = data.rename(lev='plev')
        elif 'ecs2x' in key:
            key = key.replace('ecs2x', 'ecs')
            with xr.set_options(keep_attrs=True):
                data = 0.5 * _find_data(key)
        elif key == 'rluscs' or key == 'rsdtcs':
            key = key[:-2]  # all-sky is same as clear-sky
            data = _find_data(key)
        elif key == 'tabs':
            parts = ('tpat', 'rfnt_ecs')
            data = _operate_data(parts, long_name='effective warming', units='K', product=True)  # noqa: E501
        elif 'total' in key:
            parts = subs(re.compile('total'), 'lse', 'dse', 'ocean')
            data = _operate_data(parts, 'latent', 'total')
        elif 'mse' in key:
            parts = subs(re.compile('mse'), 'lse', 'dse')
            data = _operate_data(parts, 'latent', 'moist')
        elif 'dse' in key:  # possibly missing dry term e.g. dry stationary
            parts = subs(re.compile('dse'), 'hse', 'gse')
            data = _operate_data(parts, 'sensible', 'dry')
        elif flux:
            regex = REGEX_FLUX  # always use this regex
            part, _, wav, net, bnd, sky = flux.groups()
            trad, adj = subs(regex, r'pl\2\3\4\5\6', r'pl*\2\3\4\5\6')
            *_, param = key.split('_')
            if param in ('ts', 'ta'):  # equilibrium temperature metric!
                ratio = key.replace(f'_{param}', '_ratio')
                parts = (param, ratio)  # climate temperature minus implied temperature
                data = _operate_data(parts, 'temperature', 'equilibrium temperature', (1, -1))  # noqa: E501
            elif param in ('ratio',):  # equilibrium temperature metric!
                flux = key.replace(f'_{param}', '').replace(part, '').strip('_')
                feedback = key.replace(f'_{param}', '_lam').strip('_')
                parts = (flux, feedback)
                data = _operate_data(parts, 'flux', 'temperature', (1, -1), units='K', product=True)  # noqa: E501
            elif part == 'atm' and (trad in dataset or adj in dataset):
                if trad in dataset:
                    parts = (trad, *subs(regex, r'lr\2\3\4\5\6', r'hus\2\3\4\5\6'))
                else:
                    parts = (adj, *subs(regex, r'lr*\2\3\4\5\6', r'hur\2\3\4\5\6'))
                search = re.compile(r'(adjusted\s+)?Planck')
                data = _operate_data(parts, search, 'temperature + humidity')
            elif part == 'ncl':
                parts = subs(regex, r'cl\2\3\4\5\6', r'\2\3\4\5\6')
                data = _operate_data(parts, 'cloud', 'non-cloud', (-1, 1))
            elif sky == 'ce':
                parts = subs(regex, r'\1\2\3\4\5cs', r'\1\2\3\4\5')
                data = _operate_data(parts, 'clear-sky', 'cloud', (-1, 1))
            elif bnd == 'a':
                parts = subs(regex, r'\1\2\3\4t\6', r'\1\2\3\4s\6')
                data = _operate_data(parts, 'TOA', 'atmospheric')
            elif net == 'e':  # effective forcing e.g. 'rlet'
                parts = subs(regex, r'\1\2\3n\5\6_erf')
                data = _operate_data(parts, 'effective ', '')  # NOTE: change?
            elif net == 'r':  # radiative response e.g. 'rlrt'
                parts = subs(regex, r'\1\2\3n\5\6', r'\1\2\3n\5\6_erf')
                data = _operate_data(parts, 'flux', 'response', (1, -1))
            elif wav == 'f':  # WARNING: critical to put this last!
                replace = 'net ' if part == '' and sky == '' else ''
                parts = subs(regex, r'\1\2l\4\5\6', r'\1\2s\4\5\6')
                data = _operate_data(parts, 'longwave ', replace)
            elif net == 'n':  # WARNING: only used for ceres currently
                if wav == 'l' and bnd == 't':
                    signs = (-1,)
                    patterns = (r'\1\2\3u\5\6',)
                else:
                    signs = (1, -1)
                    patterns = (r'\1\2\3d\5\6', r'\1\2\3u\5\6')
                search = re.compile('(upwelling|downwelling|incoming|outgoing)')
                parts = subs(regex, *patterns)
                data = _operate_data(parts, search, 'net', signs)
            else:
                opts = ', '.join(s for s in dataset.data_vars if regex.search(s))
                raise ValueError(f'Missing flux variable {key}. Options are: {opts}.')  # noqa: E501
        else:
            opts = ', '.join(s for s in dataset.data_vars)
            raise ValueError(f'Missing variable {key}. Options are: {opts}.')
        data.name = key
        data.attrs.update(attrs)  # arbitrary overrides
        return data

    # Activate recursion and apply special transformations
    # NOTE: This is similar to climopy recursive derivation invocation. Except delays
    # summations until selections are performed in apply_reduce()... probably better
    # than re-calculating full result every time (intensive) or caching full result
    # after calculation (excessive storage). Should add to climopy.
    kw_get = {'quantify': False, 'standardize': True}
    kw_get.update({key: kwargs.pop(key) for key in KEYS_VARIABLE if key in kwargs})
    hemi = kw_get.pop('hemisphere', kw_get.pop('hemi', None))  # handled separately
    data = _find_data(name)
    if spatial := kwargs.get('spatial', None):  # apply spatial correlation
        kw0, kw1 = {}, {}
        if 'version' in data.coords:
            kw0 = {'start': 0, 'stop': 150}
            kw1 = {'start': kwargs.get('start', 0), 'stop': kwargs.get('stop', 150)}
        data0 = data.sel(experiment='picontrol', **kw0)
        data1 = data.sel(experiment='abrupt4xco2', **kw1)
        data, attrs = _method_double(data0, data1, dim='area', method=spatial)
        name = attrs.pop('name')  # see below
        data.attrs.update({**data.attrs, **attrs})
    if attr == 'units':
        return data.climo.units
    elif attr:
        return data.attrs.get(attr, '')
    elif data.size:
        return data
    else:
        raise ValueError(f'Empty result {data.sizes}.')


def process_data(dataset, *kws_process, attrs=None, suffix=True):
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

    Returns
    -------
    args : tuple
        The output plotting arrays.
    method : str
        The method used to reduce the data.
    kwargs : dict
        The plotting and `_infer_command` keyword arguments.
    """
    # Split instructions along operators
    # NOTE: Initial kw_red values are formatted as (('[+-]', value), ...) to
    # permit arbitrary combinations of names and indexers (see _parse_specs).
    # TODO: Possibly keep original 'signs' instead of _group_parts signs? See
    # below where arguments are combined and have to pick a sign.
    kws_group, kws_count, kws_input, kws_method = [], [], [], []
    if len(kws_process) not in (1, 2):
        raise ValueError(f'Expected two process dictionaries. Got {len(kws_process)}.')
    for i, kw_process in enumerate(kws_process):  # iterate method reduce arguments
        kw_process, kw_coords, kw_count = kw_process.copy(), {}, {}
        kw_method = {
            key: kw_process.pop(key) for key in KEYS_METHOD
            if key in kw_process
        }
        kw_input = {  # TODO: should never have data array here?
            key: value for key, value in kw_process.items()
            if key != 'name' and value is not None
        }
        kw_process = _group_parts(kw_process, keep_operators=True)
        for key, value in kw_process.items():
            sels = ['+']
            for part in _ungroup_parts(value):
                if part is None:
                    sel = None
                elif np.iterable(part) and part[0] in ('+', '-', '*', '/'):
                    sel = part[0]
                elif isinstance(part, (str, tuple)):  # already mapped integers
                    sel = part
                else:
                    unit = get_data(dataset, key, 'units')
                    if not isinstance(part, ureg.Quantity):
                        part = ureg.Quantity(part, unit)
                    sel = part.to(unit)
                sels.append(sel)
            signs, values = sels[0::2], sels[1::2]
            kw_coords[key] = tuple(zip(signs, values))
            kw_count[key] = len(values)
        kws_group.append(kw_coords)
        kws_count.append(kw_count)
        kws_input.append(kw_input)
        kws_method.append(kw_method)

    # Group split reduce instructions into separate dictionaries
    # TODO: Add more grouping options (similar to _build_specs). For now just
    # support multiple start-stop and experiment grouping.
    # WARNING: Here 'product' can be used for e.g. cmip6-cmip5 abrupt4xco2-picontrol
    # but there are some terms that we always want to group together e.g. 'experiment'
    # and 'startstop'. So include some overrides below.
    kws_reduce = []
    for key in sorted(set(key for kw in kws_count for key in kw)):
        count = [kw.get(key, 0) for kw in kws_count]
        if count[0] == count[-1]:  # otherwise match correlation pair operators
            continue
        if count[1] % count[0] == 0:
            i, j = 0, 1
        elif count[0] % count[1] == 0:
            i, j = 1, 0
        else:
            raise ValueError(f'Incompatible counts {count[0]} and {count[1]}.')
        values = kws_group[i].get(key, (('+', None),))
        values = values * (count[j] // count[i])  # i.e. *average* with itself
        kws_group[i][key] = values
    for kw_coords, kw_method in zip(kws_group, kws_method):
        startstop = 'startstop' in kw_coords and 'experiment' in kw_coords
        groups = [('startstop', 'experiment')] if startstop else []
        for group in groups:
            values = _to_lists(*(kw_coords[key] for key in group))
            kw_coords.update(dict(zip(group, values)))
        groups.extend(
            (key,) for key in kw_coords if not any(key in group for group in groups)
        )
        kw_product = {
            group: tuple(zip(*(kw_coords[key] for key in group))) for group in groups
        }
        ikws_reduce = []
        for values in itertools.product(*kw_product.values()):  # non-grouped coords
            items = {
                key: val for group, vals in zip(groups, values)
                for key, val in zip(group, vals)
            }
            signs, values = zip(*items.values())
            sign = -1 if signs.count('-') % 2 else +1
            kw_reduce = dict(zip(items.keys(), values))
            kw_reduce.update(kw_method)
            ikws_reduce.append((sign, kw_reduce))
        kws_reduce.append(ikws_reduce)

    # Reduce along facets dimension and carry out operation
    # TODO: Support operations before reductions instead of after. Should have
    # effect on e.g. regional correlation, regression results.
    # WARNING: Here _group_parts modified e.g. picontrol base from late-early
    # to late+early (i.e. average) so try to use sign from non-control experiment.
    print('.', end='')
    warming = ('tpat', 'tdev', 'tstd', 'tabs', 'rfnt_ecs')
    kwargs = {}
    datas_persum = []  # each item part of a summation
    methods_persum = set()
    kws_reduce = _to_lists(*kws_reduce, equal=False)
    if any(len(kws) != len(kws_reduce[0]) for kws in kws_reduce):
        raise ValueError('Operator count mismatch in numerator and denominator.')
    for ikws_reduce in zip(*kws_reduce):  # iterate operators
        isigns, ikws_reduce = zip(*ikws_reduce)
        datas, exps, kw_method = [], [], {}
        for kw_reduce in ikws_reduce:  # iterate method reduce arguments
            kw_reduce = kw_reduce.copy()
            name = kw_reduce.pop('name')  # NOTE: always present
            area = kw_reduce.get('area')
            experiment = kw_reduce.get('experiment')
            if name == 'tabs' and experiment == 'picontrol':
                name = 'tstd'  # use rfnt_ecs for e.g. abrupt minus pre-industrial
            if name in warming and area == 'avg' and len(ikws_reduce) == 2:
                if name in warming[:2] or experiment == 'picontrol':
                    name = 'tstd'  # default to global average temp standard deviation
            for key in tuple(kw_reduce):
                if key in KEYS_METHOD:
                    kw_method.setdefault(key, kw_reduce.pop(key))
            data = get_data(dataset, name, **kw_reduce)
            datas.append(data)
            exps.append(experiment)
        datas, method, default = apply_method(*datas, **kw_method)
        for key, value in default.items():
            kwargs.setdefault(key, value)
        if len(datas) == 1:  # e.g. regression applied
            idxs = tuple(i for i, exp in enumerate(exps) if i != 'picontrol')
            isigns = (isigns[idxs[-1] if idxs else -1],)  # WARNING: see _group_parts
        datas_persum.append((isigns, datas))  # plotting command arguments
        methods_persum.add(method)
        if len(methods_persum) > 1:
            raise RuntimeError(f'Mixed reduction methods {methods_persum}.')

    # Combine arrays specified with reduction '+' and '-' keywords
    # NOTE: The additions below are scaled as *averages* so e.g. project='cmip5+cmip6'
    # gives the average across cmip5 and cmip6 inter-model averages.
    # NOTE: Here the percentile ranges are only supported for 1-dimensional line plots
    # for which we only use one plot argument. So kwargs below are not be overwritten.
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
        if size := data.sizes.get('pctile', None):  # now apply percentile ranges
            for i, key in zip(range(1, size, 2), ('shade', 'fade')):
                kwargs[f'{key}data'] = data.isel(pctile=slice(i, i + 2)).values
            data = data.isel(pctile=0)  # original data
        if len(kws_process) == 1 and (name := kws_process[0].get('name')):
            data.name = name  # e.g. 'ts' minus 'tabs'
        if method == 'dist' and len(idatas) > 1 and np.allclose(data, 0):
            data = idatas[0]
        if suffix and any(sign == -1 for sign in isigns):  # TODO: make optional?
            data.attrs['short_suffix'] = data.attrs['long_suffix'] = 'anomaly'
        args.append(data)

    # Align and restore coordinates
    # WARNING: In some xarray versions seems _method_double with conflicting scalar
    # coordinates will keep the first one instead of discarding. So overwrite below.
    # NOTE: Xarray automatically drops non-matching scalar coordinates (similar to
    # vector coordinate matching utilities) so try to restore them below.
    # NOTE: Commented code at bottom verifies that global average regressions of
    # regional pre-industrial feedbacks onto global pre-industrial feedbacks equal 1
    # even though there are broad positive regions with much larger magnitudes.
    args = xr.align(*args)  # re-align after summation
    if len(args) == len(kws_input):  # one or two (e.g. scatter)
        for arg, kw_input in zip(args, kws_input):
            arg.attrs.update(attrs)
            for key, value in kw_input.items():
                if key not in arg.sizes:
                    arg.coords[key] = value
    else:  # create 2-tuple coordinates
        kw_input = {}
        keys = sorted(set(key for kw in kws_input for key in kw))
        for key in keys:
            values = tuple(kw.get(key, None) for kw in kws_input)
            value = values[0]
            if len(values) == 2 and values[0] != values[1]:
                value = np.array(None, dtype=object)
                value[...] = tuple(values)
            kw_input[key] = value
        for arg in args:  # should be singleton
            arg.attrs.update(attrs)
            for key, value in kw_input.items():  # e.g. regress lat on feedback map
                if key not in arg.sizes:
                    arg.coords[key] = value
    # if args[0].sizes.keys() & {'lon', 'lat'}:
    #     ic(kws_process, args[0].climo.average('area').item())
    return args, method, kwargs
