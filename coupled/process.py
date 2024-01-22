#!/usr/bin/env python3
"""
Processing utilities used by plotting functions.
"""
import itertools
import math
import re
import warnings

import climopy as climo  # noqa: F401
import numpy as np
import xarray as xr
from climopy import var, ureg, vreg  # noqa: F401
from scipy import stats
from icecream import ic  # noqa: F401

from .internals import KEYS_REDUCE
from .internals import _expand_lists, _group_parts, _ungroup_parts
from .results import ALIAS_FEEDBACKS, FEEDBACK_ALIASES, REGEX_FLUX
from .reduce import _reduce_double, reduce_facets, reduce_general
from cmip_data.feedbacks import FEEDBACK_DEPENDENCIES


__all__ = ['get_data', 'process_data']


# Observational constraint mean and spreads
# TODO: Auto-generate this dictionary or save and commit then parse a text file.
# NOTE: These show best estimate, standard errors, and degrees of freedom for feedback
# regression slopes using either GISTEMP4 or HadCRUT5 and using surface temp autocorr
# adjustment (consistent with Dessler et al.). See observed.ipynb notebook.
FEEDBACK_CONSTRAINTS = {
    # 'cld': (0.69, (1.04 - 0.31) / 2,  # He et al. with 95% uncertainty
    # 'cld': (0.68, 0.18, 97),  # He et al. trended estimate (total record is 226)
    'cld': (0.35, 0.25, 108),  # He et al. detrended estimate (total record is 226)
    'net': (-0.85, 0.29, 166),  # custom estimates (total record is 274)
    'sw': (0.86, 0.24, 175),
    'lw': (-1.71, 0.19, 129),
    'cre': (0.27, 0.23, 131),
    'swcre': (0.19, 0.24, 140),
    'lwcre': (0.08, 0.13, 184),
    'hadnet': (-0.84, 0.31, 166),
    'hadsw': (0.89, 0.26, 175),
    'hadlw': (-1.73, 0.21, 129),
    'hadcre': (0.30, 0.24, 131),
    'hadswcre': (0.15, 0.26, 140),
    'hadlwcre': (0.15, 0.14, 184),
}

# Model-estimated bootstrapped feedback uncertainty
# NOTE: See templates.py _calc_bootstrap for calculation details
# TODO: Record degrees of freedom and autocorrelation across successive bootstrap
# feedback estimates (e.g. year 0-20 estimate will be related to year 5-25). For now
# just assume implied distribution is Gaussian in _constrain_data() below.
INTERNAL_VARIABILITY = {
    'cld': {20: np.nan, 40: np.nan, 60: np.nan},
    'net': {20: 0.38, 40: 0.22, 50: 0.19, 60: 0.13},
    'cs': {20: 0.24, 40: 0.15, 50: 0.13, 60: 0.10},
    'cre': {20: 0.31, 40: 0.20, 50: 0.18, 60: 0.12},
    'lw': {20: 0.24, 40: 0.15, 50: 0.15, 60: 0.10},
    'lwcs': {20: 0.13, 40: 0.07, 50: 0.07, 60: 0.05},
    'lwcre': {20: 0.15, 40: 0.10, 50: 0.09, 60: 0.07},
    'sw': {20: 0.32, 40: 0.19, 50: 0.17, 60: 0.13},
    'swcs': {20: 0.17, 40: 0.11, 50: 0.10, 60: 0.08},
    'swcre': {20: 0.35, 40: 0.21, 50: 0.18, 60: 0.14},
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

# Variable derivations
# NOTE: See timescales definitions.py (eventually will be copied to climopy)
def equator_pole_delta(self, name):  # noqa: E302
    temp, _ = name.split('_')
    temp = self[temp]  # also quantifies
    if temp.name == 'ta':
        temp = temp.sel(lev=850)  # traditional heat transport pressure
    equator = temp.sel(lat=slice(0, 10)).climo.average('area')
    pole = temp.sel(lat=slice(60, 90)).climo.average('area')
    return equator - pole
def equator_pole_diffusivity(self, name):  # noqa: E302
    temp, _ = name.split('_')
    delta = f'{temp}_grad'
    delta = self.get(delta)
    transport = self['lset'] + self['dset']  # moist not directly available
    transport = transport.sel(lat=slice(0, 90)).climo.average('area')
    return transport / delta

# Register variables
# TODO: Restore this after updating climopy and move to definitions file
# TODO: Use this approach for flux addition terms and other stuff
# climo.register_derivation(re.compile(r'\A(ta|ts)_grad\Z'))(equator_pole_delta)
# climo.register_derivation(re.compile(r'\A(ta|ts)_diff\Z'))(equator_pole_diffusivity)
# with warnings.catch_warnings():  # noqa: E305
# warnings.simplefilter('ignore')
# vreg.define('ta_grad', 'equator-pole air temperature difference', 'K')
# vreg.define('ts_grad', 'equator-pole surface temperature difference', 'K')
# vreg.define('ta_diff', 'bulk energy transport diffusivity', 'PW / K')
# vreg.define('ts_diff', 'bulk energy transport surface diffusivity', 'PW / K')


def _constrain_data(
    data0, data1, constraint=None, pctile=None, observed=None, variability=None, graphical=False, N=None,  # noqa: E501
):
    """
    Return percentile bounds for observational constraint.

    Parameters
    ----------
    data0 : xarray.DataArray
        The predictor data. Must be 1D.
    data1 : xarray.DataArray
        The predictand data. Must be 1D.
    constraint : str or 2-tuple,
        The 95% bounds on the observational constraint.
    pctile : float, default: 95
        The emergent constraint percentile bounds to be returned.
    observed : int, default: 20
        Number of years to assume for observational uncertainty estimate.
    variability : bool or int, optional
        Number of years for variability variability estimate.
    graphical : bool, optional
        Whether to use graphical intersections instead of residual bootstrapping.
    N : int, default: 100000
        The number of bootstrapped samples to carry out.

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
    # estimate, then use those distribution properties for our bootstrapping. Tried
    # using 'offset uncertainty' but was similar to slope-only regression uncertainty.
    # NOTE: Use N - 2 degrees of freedom for both observed feedback and inter-model
    # coefficients since they are both linear regressions. For now ignore uncertainty
    # of individual feedback regressions that comprise inter-model regression because
    # their sample sizes are larger (150 years) and uncertainties are uncorrelated so
    # should roughly cancel out across e.g. 30 members of ensemble.
    name = constraint if isinstance(constraint, str) else None
    name = FEEDBACK_ALIASES.get(name, name)
    pctile = 95 if pctile is None or pctile is True else pctile
    pctile = np.array([50 - 0.5 * pctile, 50 + 0.5 * pctile])
    observed = observed or 20
    variability = observed if variability is None or variability is True else variability  # noqa: E501
    variability = INTERNAL_VARIABILITY[name][variability] if variability else None
    constraint = 'cld' if constraint is None or constraint is True else constraint
    kwargs = dict(pctile=pctile, adjust=False)  # no correlation across ensembles
    N = N or 10000  # samples to draw
    if name is not None:
        constraint = FEEDBACK_CONSTRAINTS[name]
    if not np.iterable(constraint) or len(constraint) != 3:
        raise ValueError(f'Invalid constraint {constraint}. Must be (mean, sigma, dof).')  # noqa: E501
    if data0.ndim != 1 or data1.ndim != 1:
        raise ValueError(f'Invalid data dims {data0.ndim} and {data1.ndim}. Must be 1D.')  # noqa: E501
    xmean, xscale, xdof = constraint  # note dof will have small effect
    xdof *= observed / 20  # increased degrees of freedom
    xscale /= np.sqrt(observed / 20)  # reduced regression error
    nbounds = 1 if variability is None else 2  # number of bounds returned
    xmin, xmax = stats.t(df=xdof, loc=xmean, scale=xscale).ppf(0.01 * pctile)
    observations = np.array([xmin, xmean, xmax])
    if variability is not None:
        xs = stats.t(df=xdof, loc=xmean, scale=xscale).rvs(N)
        es = stats.norm(loc=0, scale=variability).rvs(N)  # implied variability error
        observations = np.insert(np.percentile(xs + es, pctile), 1, observations)
    data0 = np.array(data0).squeeze()
    data1 = np.array(data1).squeeze()
    idxs = np.argsort(data0, axis=0)
    data0, data1 = data0[idxs], data1[idxs]  # required for graphical error
    mean0, mean1 = data0.mean(), data1.mean()
    bmean, berror, _, fit, fit_lower, fit_upper = var.linefit(data0, data1, **kwargs)
    if graphical:  # intersection of shaded regions
        fit_lower = fit_lower.squeeze()
        fit_upper = fit_upper.squeeze()
        xs = np.sort(data0, axis=0)  # linefit returns result for sorted data
        ymean = mean1 + bmean.item() * (xmean - mean0)
        ymins = np.interp(observations[:nbounds], xs, fit_lower)
        ymaxs = np.interp(observations[-nbounds:], xs, fit_upper)
        constrained = (*ymins, ymean, *ymaxs)
        alternative = mean1 + bmean * (observations - mean0)
    else:  # bootstrapped residual addition
        # amean = mean1 - mean0 * bmean  # see wiki page
        # aerror = berror * np.sqrt(np.sum(data0 ** 2) / data0.size)  # see wiki page
        # bs_ = stats.t(loc=bmean, scale=berror, df=data0.size - 2).rvs(N)
        # as_ = stats.t(loc=amean, scale=aerror, df=data0.size - 2).rvs(N)
        # ys = err + as_ + bs_ * xs  # include both uncertainties
        # ys = mean1 + bs_ * (xs - mean0)  # include slope uncertainty only
        rscale = np.std(data1 - fit, ddof=1)  # model residual sigma
        rdof = data0.size - 2  # inter-model regression dof
        xs = stats.t(df=xdof, loc=xmean, scale=xscale).rvs(N)  # observations
        rs = stats.t(df=rdof, loc=0, scale=rscale).rvs(N)  # regression
        ys = rs + mean1 + bmean * (xs - mean0)
        constrained = np.insert(np.percentile(ys, pctile), 1, np.mean(ys))
        ys = mean1 + bmean * (xs - mean0)  # no regression uncertainty
        alternative = np.insert(np.percentile(ys, pctile), 1, np.mean(ys))
        if variability is not None:
            xs = stats.t(df=xdof, loc=xmean, scale=xscale).rvs(N)
            es = stats.norm(loc=0, scale=variability).rvs(N)  # implied variability
            ys = rs + mean1 + bmean * (xs + es - mean0)
            constrained = np.insert(np.percentile(ys, pctile), 1, constrained)
            ys = mean1 + bmean * (xs + es - mean0)
            alternative = np.insert(np.percentile(ys, pctile), 1, alternative)
    return observations, alternative, constrained


def _find_data(
    dataset, name, attr=None, attrs=None, hemi=None,
    scaled=False, quantify=False, standardize=True, **kwargs
):
    """
    Find or derive the variable or variable attribute.

    Parameters
    ----------
    dataset : xarray.Dataset
        The input dataset.
    name : str
        The requested variable.
    attr : str, optional
        The requested variable attribute.
    attrs : dict, optional
        The variable attribute overrides.
    hemi : str, optional
        The hemisphere to select.
    scaled : bool, optional
        Whether ecs/erf data has been scaled.
    quantify : bool, optional
        Whether to quantify the data.
    standardize : str, optional
        Whether to standardize the data.
    **kwargs
        Additional reduction instructions.

    Returns
    -------
    data : xarray.DataArray
        The result. If `attr` is not ``None`` this is an empty array.
    """
    # WARNING: Critical to only apply .reduce() operations to variables required for
    # derivation or else takes too long. After climopy refactoring this will be taken
    # into account automatically and derivations will require explicitly specifying
    # the dependencies upon registration (no more retrieveing inside the function).
    # NOTE: Here reduce_general will automatically perform selections for certain
    # dimensions (only carry out when requesting data). Also gets net flux (longwave
    # plus shortwave), atmospheric flux (top-of-atmosphere plus surface), cloud effect
    # (all-sky minus clear-sky), radiative response (full minus effective forcing),
    # net imbalance (downwelling minus upwelling), and various transport terms.
    name = ALIAS_FEEDBACKS.get(name, name)
    attrs = attrs or {}
    scaled = scaled or 'ecs' not in name and 'erf' not in name
    if scaled and name in dataset:
        data = dataset[name]
    elif not scaled or name not in dataset.climo:  # coupled-model derivation
        data = _derive_data(dataset, name, attr=attr, scaled=scaled, **kwargs)
    elif name in VARIABLE_DEPENDENCIES:  # any registered climopy derivation
        var = vreg[name]  # specific supported derivation
        keys = ('short_name', 'long_name', 'standard_units')
        deps = set(VARIABLE_DEPENDENCIES[name])  # climopy derivation dependencies
        attrs.update({attr: getattr(var, attr) for attr in keys})
        if attr:  # TODO: throughout utilities permit cfvariable attributes
            data = xr.DataArray([], name=name)
        else:  # derive below with get()
            data = dataset.drop_vars(dataset.data_vars.keys() - set(deps))
    else:  # note in future will
        raise RuntimeError(f'Climopy derivation {name} has unknown dependencies.')
    if not attr:  # carry out reductions
        data = reduce_general(data, **kwargs)
        if 'plev' in data.coords:  # TODO: make derivations compatible with plev
            data = data.rename(plev='lev')
        if hemi and 'lat' in data.sizes:
            data = data.climo.sel_hemisphere(hemi)
        if isinstance(data, xr.Dataset):  # finally get the derived variable
            data = data.climo.get(name, quantify=quantify, standardize=standardize)
        if 'lev' in data.coords:
            data = data.rename(lev='plev')
    data.name = name
    data.attrs.update(attrs)  # arbitrary overrides
    return data


def _derive_data(dataset, name, scaled=False, **kwargs):  # noqa: E501
    """
    Get or derive the variable or its attribute.

    Parameters
    ----------
    dataset : xarray.Dataset
        The source dataset.
    name : str
        The requested variable.
    scaled : bool, optional
        Whether ecs/erf data has been scaled.
    **kwargs
        Additional reduction instructions.

    Returns
    -------
    data : xarray.DataArray
        The result with appropriate attributes.
    """
    # TODO: This is ad hoc implementation of cfvariable-like scheme for quickly
    # retrieving names. In future should implement these as derivations with
    # standardized cfvariable properties when derivation scheme is more sophisticated.
    # NOTE: Previously had quadruple=True default option to load all feedbacks as
    # quadrupled versions, but meant same variable name could have different meaning
    # depending on input arguments. Now implement scaling as a derivation below and
    # add explicit indicator if '2x' or '4x' is specified. Preserve default of always
    # *displaying* the unscaled experiment results by translating 'erf' or 'ecs' below.
    signs = search = replace = None
    product = relative = False
    regex = REGEX_FLUX  # always use this regex
    attrs = {}  # attribute overrides
    subs = lambda reg, *ss: tuple(reg.sub(s, name).strip('_') for s in ss)
    head = wav = net = bnd = sky = trad = adjust = param = 'NA'
    if flux := regex.search(name):
        head, _, wav, net, bnd, sky = flux.groups()
        trad, adjust = subs(regex, r'pl\2\3\4\5\6', r'pl*\2\3\4\5\6')  # see below
        *_, param = name.split('_')
    if 'rluscs' in name or 'rsdtcs' in name:
        parts = name.replace('rluscs', 'rlus')  # all-sky is same as clear-sky
        parts = parts.replace('rsdtcs', 'rsdt')  # all-sky is same as clear-sky
    elif 'total' in name:
        parts = subs(re.compile('total'), 'lse', 'dse', 'ocean')
        search, replace = 'latent', 'total'
    elif 'mse' in name:
        parts = subs(re.compile('mse'), 'lse', 'dse')
        search, replace = 'latent', 'moist'
    elif 'dse' in name:  # possibly missing dry term e.g. dry stationary
        parts = subs(re.compile('dse'), 'hse', 'gse')
        search, replace = 'sensible', 'dry'
    elif name == 'tdev':  # TODO: define other 'relative' variables?
        parts, relative = 'tstd', True
        attrs.update(units='K', long_name='relative warming')
    elif name == 'tabs':
        parts, product = ('tpat', 'ecs'), True
        attrs.update(units='K', long_name='effective warming')
        kwargs.update(scaled=False)
    elif param == 'ts' or param == 'ta':  # eqtemp = temp - (temp - eqtemp) from below
        delta = name.replace('_' + param, '_dt')  # climate minus implied deviation
        parts, signs, product = (param, delta), (1, -1), False
        search, replace = 'temperature', 'equilibrium temperature'
    elif param == 'dt':  # temp - eqtemp anomaly inferred from feedback
        denom = name.replace('_dt', '_lam')  # denominator
        numer = name.replace('_dt', '').replace(head + '_', '')
        parts, signs, product = (numer, denom), (1, -1), True
        search, replace = 'flux', 'temperature'
        attrs.update(units='K', long_name='temperature difference')
    elif not scaled and re.search(temp := r'(ecs|erf)(?:([0-9.-]+)x)?', name):
        default = 4  # display 4xCO2 by default when no scale is specified
        scale = re.search(temp, name).group(2)
        signs = np.log2(float(scale or default))
        parts = subs(re.compile(temp), r'\1')
        search = re.compile(r'\A')  # prepend to existing label
        replace = rf'{scale}$\\times$CO$_2$ ' if scale else ''
    elif head == 'atm':
        if trad in dataset:
            parts = (trad, *subs(regex, r'lr\2\3\4\5\6', r'hus\2\3\4\5\6'))
        elif adjust in dataset:
            parts = (adjust, *subs(regex, r'lr*\2\3\4\5\6', r'hur\2\3\4\5\6'))
        else:
            raise ValueError("Missing non-cloud components for 'atm' feedback.")
        search = re.compile(r'(adjusted\s+)?Planck')
        replace = 'temperature + humidity'
    elif head == 'ncl':
        parts = subs(regex, r'cl\2\3\4\5\6', r'\2\3\4\5\6')
        search, replace = 'cloud', 'non-cloud'
        signs = (-1, 1)
    elif sky == 'ce':
        parts = subs(regex, r'\1\2\3\4\5cs', r'\1\2\3\4\5')
        search, replace = 'clear-sky', 'cloud'
        signs = (-1, 1)
    elif bnd == 'a':  # net atmosheric (surface and TOA are positive into atmosphere)
        parts = subs(regex, r'\1\2\3\4t\6', r'\1\2\3\4s\6')
        search, replace = 'TOA', 'atmospheric'
        signs = (1, 1)
    elif net == 'e':  # effective forcing e.g. 'rlet'
        parts = subs(regex, r'\1\2\3n\5\6_erf')
        search, replace = 'effective ', ''  # NOTE: change?
        signs = (1, 1)
    elif net == 'r':  # radiative response e.g. 'rlrt'
        parts = subs(regex, r'\1\2\3n\5\6', r'\1\2\3n\5\6_erf')
        signs = (1, -1)
        search, replace = 'flux', 'response'
    elif wav == 'f':  # WARNING: critical to add wavelengths last
        deps = {'longwave': True, 'shortwave': True}  # e.g. atm_rfnt depends on both
        deps = FEEDBACK_DEPENDENCIES.get(head, deps)
        wavs = [wav[0] for wav, names in deps.items() if names or head == '']
        patterns = [rf'\1\2{wav}\4\5\6' for wav in wavs]
        parts = subs(regex, *patterns)
        search = re.compile('(longwave|shortwave) ')
        replace = 'net ' if head == sky == '' else ''
    elif net == 'n':  # WARNING: only used for CERES data currently
        if wav == 'l' and bnd == 't':
            signs, patterns = (-1,), (r'\1\2\3u\5\6',)
        else:
            signs, patterns = (1, -1), (r'\1\2\3d\5\6', r'\1\2\3u\5\6')
        parts = subs(regex, *patterns)
        search = re.compile('(upwelling|downwelling|incoming|outgoing)')
        replace = 'net'
    else:
        opts = ', '.join(s for s in dataset.data_vars if regex.search(s))
        raise ValueError(f'Missing flux variable {name}. Options are: {opts}.')
    kwargs.setdefault('scaled', True)  # see 'tabs'
    kwargs.update(product=product, relative=relative)
    data = _operate_data(dataset, parts, signs, search, replace, **kwargs)
    data.attrs.update(attrs)  # manual attribute overrides
    data.name = name
    return data


def _operate_data(
    dataset, names, signs=None, search=None, replace=None, *,
    attr=None, product=False, relative=False, **kwargs,
):
    """
    Operate over input variables with sums or products and adjust their attributes.

    Parameters
    ----------
    dataset : xarray.Dataset
        The input dataset.
    names : str or sequence
        The variables to operate along.
    signs : int or sequence, optional
        The signs or exponents of the variables added or multiplied.
    search, replace : str or re.Pattern, optional
        The attribute search pattern and replacement.
    attr : str, optional
        The requested variable attribute.
    product : bool, optional
        Whether to add or multiply the variables.
    relative : bool, optional
        Whether to subtract or divide the global average.
    **kwargs
        Passed to `reduce_general`.

    Returns
    -------
    data : xarray.DataArray
        The result with appropriate attributes.
    """
    # NOTE: When retrieving attributes only return first array in summation (whose
    # attributes would be saved by keep_attrs=True) instead of performing operation.
    # data = data.climo.add_cell_measures(('width', 'depth'))
    names = (names,) if isinstance(names, str) else names
    signs = (1,) * len(names) if signs is None else np.atleast_1d(signs).tolist()
    if len(signs) != len(names):
        raise RuntimeError('Length mismatch between signs and names.')
    names = names if attr is None else names[:1]  # may just need first array attrs
    datas = [_find_data(dataset, name, attr=attr, **kwargs) for name in names]
    if not relative or attr is not None:
        bases = (1 if product else 0,) * len(names)
    else:
        bases = [data.climo.average('area') for data in datas]
    with xr.set_options(keep_attrs=True):
        parts = zip(datas, signs, bases)
        if attr is not None or len(datas) == 1 and signs[0] == 1 and not relative:
            data = datas[0]
        elif product:  # get products or ratios
            data = math.prod(data ** sign / base for data, sign, base in parts)
        else:  # get sums or differences
            data = sum(sign * data - base for data, sign, base in parts)
    data = data.copy(deep=False)  # update attributes on derived variable
    for name in ('short_name', 'long_name', 'standard_name'):
        if not search:
            continue
        if name not in data.attrs:
            continue
        value = data.attrs[name]
        if isinstance(search, str):
            value = value.replace(search, replace)
        else:
            value = search.sub(replace, value)
        data.attrs[name] = value
    return data


def get_data(dataset, name, attr=None, **kwargs):
    """
    Get derived data component from the dataset.

    Parameters
    ----------
    dataset : xarray.Dataset
        The input dataset.
    name : str
        The requested variable name.
    attr : str, optional
        The requested variable attribute.
    **kwargs
        Passed to `reduce_general`.

    Returns
    -------
    result : xarray.DataArray or str
        The result. If `attr` is not ``None`` this is the corresponding attribute.
    """
    # TODO: This is similar to climopy recursive derivation invocation. Except delays
    # summations until selections are performed in reduce_general()... better than
    # re-calculating full result every time (intensive) or caching full result after
    # calculation (excessive storage). Should add to climopy by 1) translating
    # instructions, 2) selecting relevant variables, 3) performing scalar selections
    # first (e.g. points, maxima), 4) combining variables (e.g. products, sums), then
    # 5) more complex coordinate operations (e.g. averages, integrals). And optionally
    # order operations around non-linear variable or coordinate operations.
    data = _find_data(dataset, name, attr=attr, **kwargs)
    spatial = kwargs.get('spatial', None)
    start = kwargs.get('start', 0)
    stop = kwargs.get('stop', 150)
    if spatial:  # only apply custom spatial correlation
        kw0, kw1 = {}, {}
        if 'version' in data.coords:
            kw0, kw1 = {'start': 0, 'stop': 150}, {'start': start, 'stop': stop}
        data0 = data.sel(experiment='picontrol', **kw0)
        data1 = data.sel(experiment='abrupt4xco2', **kw1)
        data, attrs = _reduce_double(data0, data1, dim='area', method=spatial)
        data.attrs.update({**data.attrs, **attrs})
        del data.attrs['name']  # only used in reduce_facets()
    if attr == 'units':
        result = data.climo.units
    elif attr:
        result = data.attrs.get(attr, '')
    elif data.size:
        result = data
    else:
        raise ValueError(f'Empty result {data.sizes}.')
    return result


def process_data(dataset, *kws_process, attrs=None, suffix=True):
    """
    Combine the data based on input reduce dictionaries.

    Parameters
    ----------
    dataset : xarray.Dataset
        The input dataset.
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
    defaults : dict
        The default plotting arguments.
    """
    # Split instructions along operators
    # NOTE: Initial kw_red values are formatted as (('[+-]', value), ...) to
    # permit arbitrary combinations of names and indexers (see _parse_specs).
    # TODO: Possibly keep original 'signs' instead of _group_parts signs? See
    # below where arguments are combined and have to pick a sign.
    kws_group, kws_count, kws_input, kws_reduce = [], [], [], []
    if len(kws_process) not in (1, 2):
        raise ValueError(f'Expected two process dictionaries. Got {len(kws_process)}.')
    for i, kw_process in enumerate(kws_process):  # iterate method reduce arguments
        kw_process, kw_coords, kw_count = kw_process.copy(), {}, {}
        kw_reduce = {key: kw_process.pop(key) for key in KEYS_REDUCE if key in kw_process}  # noqa: E501
        kw_input = {key: val for key, val in kw_process.items() if key != 'name' and val is not None}  # noqa: E501
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
        kws_reduce.append(kw_reduce)

    # Group split reduce instructions into separate dictionaries
    # TODO: Add more grouping options (similar to _build_specs). For now just
    # support multiple start-stop and experiment grouping.
    # WARNING: Here 'product' can be used for e.g. cmip6-cmip5 abrupt4xco2-picontrol
    # but there are some terms that we always want to group together e.g. 'experiment'
    # and 'startstop'. So include some overrides below.
    kws_data = []
    for key in sorted(set(key for kw in kws_count for key in kw)):
        count = [kw.get(key, 0) for kw in kws_count]
        if count[0] == count[-1]:  # otherwise match correlation pair operators
            continue
        if count[1] % count[0] == 0:
            idx1, idx2 = 0, 1
        elif count[0] % count[1] == 0:
            idx1, idx2 = 1, 0
        else:
            raise ValueError(f'Incompatible counts {count[0]} and {count[1]}.')
        values = kws_group[idx1].get(key, (('+', None),))
        values = values * (count[idx2] // count[idx1])  # i.e. *average* with itself
        kws_group[idx1][key] = values
    for kw_coords, kw_reduce in zip(kws_group, kws_reduce):
        startstop = 'startstop' in kw_coords and 'experiment' in kw_coords
        groups = [('startstop', 'experiment')] if startstop else []
        for group in groups:
            values = _expand_lists(*(kw_coords[key] for key in group))
            kw_coords.update(dict(zip(group, values)))
        groups.extend(
            (key,) for key in kw_coords if not any(key in group for group in groups)
        )
        kw_product = {
            group: tuple(zip(*(kw_coords[key] for key in group))) for group in groups
        }
        ikws_data = []
        for values in itertools.product(*kw_product.values()):  # non-grouped coords
            items = {
                key: val for group, vals in zip(groups, values)
                for key, val in zip(group, vals)
            }
            signs, values = zip(*items.values())
            sign = -1 if signs.count('-') % 2 else +1
            kw_data = dict(zip(items.keys(), values))
            kw_data.update(kw_reduce)
            ikws_data.append((sign, kw_data))
        kws_data.append(ikws_data)

    # Reduce along facets dimension and carry out operation
    # TODO: Support operations before reductions instead of after. Should have
    # effect on e.g. regional correlation, regression results.
    # WARNING: Here _group_parts modified e.g. picontrol base from late-early
    # to late+early (i.e. average) so try to use sign from non-control experiment.
    print('.', end='')
    warming = ('tpat', 'tdev', 'tstd', 'tabs', 'rfnt_ecs')
    defaults = {}  # default plotting arguments
    datas_persum = []  # each item part of a summation
    methods_persum = set()
    kws_data = _expand_lists(*kws_data)
    for ikws_data in zip(*kws_data):  # iterate operators
        signs, ikws_data = zip(*ikws_data)
        datas, exps, kw_reduce = [], [], {}
        for kw_data in ikws_data:  # iterate reduce arguments
            kw_data = kw_data.copy()
            name = kw_data.pop('name')  # NOTE: always present
            area = kw_data.get('area')
            experiment = kw_data.get('experiment')
            if name == 'tabs' and experiment == 'picontrol':
                name = 'tstd'  # use rfnt_ecs for e.g. abrupt minus pre-industrial
            if name in warming and area == 'avg' and len(ikws_data) == 2:
                if name in warming[:2] or experiment == 'picontrol':
                    name = 'tstd'  # default to global average temp standard deviation
            for key in tuple(kw_data):
                if key in KEYS_REDUCE:
                    kw_reduce.setdefault(key, kw_data.pop(key))
            data = get_data(dataset, name, **kw_data)
            datas.append(data)
            exps.append(experiment)
        datas, method, default = reduce_facets(*datas, **kw_reduce)
        for key, value in default.items():
            defaults.setdefault(key, value)
        if len(datas) == 1:  # e.g. regression applied
            idxs = tuple(i for i, exp in enumerate(exps) if i != 'picontrol')
            signs = (signs[idxs[-1] if idxs else -1],)  # WARNING: see _group_parts
        datas_persum.append((signs, datas))  # plotting command arguments
        methods_persum.add(method)
        if len(methods_persum) > 1:
            raise RuntimeError(f'Mixed reduction methods {methods_persum}.')

    # Combine arrays specified with reduction '+' and '-' keywords
    # WARNING: Can end up with all-zero arrays if e.g. doing single scatter or multiple
    # bar regression operations and only the dependent or independent variable relies
    # on an operations between slices (e.g. area 'nino-nina' combined with experiment
    # 'abrupt4xco2-picontrol' creates 4 datas that sum to zero). Use the kludge below.
    # NOTE: The additions below are scaled as *averages* so e.g. project='cmip5+cmip6'
    # gives the average across cmip5 and cmip6 inter-model averages. Also previously
    # we supported percentile ranges and standard deviations of differences between
    # individual institutes from different projects by passing 2D data to line() with
    # e.g. 'shadepctiles' but now for consistency with line() plots of regression
    # coefficients we use sum of variances (see _reduce_single and _reduce_double
    # for details). Institute differences are now only supported for scalar plots.
    print('.', end=' ')
    args = []
    method = methods_persum.pop()
    signs_persum, datas_persum = zip(*datas_persum)
    for signs, datas in zip(zip(*signs_persum), zip(*datas_persum)):  # plot arguments
        datas = xr.align(*datas)
        parts = tuple(data.isel(sigma=0) if 'sigma' in data.dims else data for data in datas)  # noqa: E501
        sum_scale = sum(sign == 1 for sign in signs)
        if parts[0].sizes.get('facets', None) == 0:
            raise RuntimeError('Empty facets dimension.')
        with xr.set_options(keep_attrs=True):  # keep e.g. units and short_prefix
            arg = sum(sign * part for sign, part in zip(signs, parts)) / sum_scale
        if method == 'dist' and len(parts) > 1 and np.allclose(arg, 0):
            arg = parts[0]  # kludge for e.g. control late minus control early
        if len(parts) == 1 and (name := kws_process[0].get('name')):
            arg.name = name  # kluge for e.g. 'ts' minus 'tstd'
        if suffix and any(sign == -1 for sign in signs):
            suffix = 'anomaly' if suffix is True else suffix
            arg.attrs['short_suffix'] = arg.attrs['long_suffix'] = suffix
        if 'sigma' in datas[0].dims:  # see _apply_single and _apply_double
            dof = defaults.pop('dof', None)
            sigma = sum(data.isel(sigma=1) for data in datas) ** 0.5
            for which in ('shade', 'fade'):
                pctile = defaults.pop(f'{which}pctiles', None)
                std = defaults.pop(f'{which}stds', None)
                if pctile is not None:  # should be scalar
                    bounds = var._dist_bounds(sigma, pctile, dof=dof)
                elif std is not None:
                    bounds = (-std * sigma, std * sigma)
                else:
                    continue
                bounds = (arg + bounds[0], arg + bounds[1])
                bounds = xr.concat(bounds, dim='bounds')
                defaults[f'{which}data'] = bounds.transpose('bounds', ...).values
        args.append(arg)

    # Align and restore coordinates
    # WARNING: In some xarray versions seems _reduce_double with conflicting scalar
    # coordinates will keep the first one instead of discarding. So overwrite below.
    # NOTE: Xarray automatically drops non-matching scalar coordinates (similar
    # to vector coordinate matching utilities) so try to restore them below.
    # NOTE: Global average regressions of local pre-industrial feedbacks onto global
    # pre-industrial feedbacks equal one despite regions with much larger magnitudes.
    # if args[0].sizes.keys() & {'lon', 'lat'}:  # ensure average is one
    #     ic(kws_process, args[0].climo.average('area').item())
    args = xr.align(*args)  # re-align after summation
    if len(args) == len(kws_input):  # one or two (e.g. scatter)
        for arg, kw_input in zip(args, kws_input):
            names = [name for index in arg.indexes.values() for name in index.names]
            arg.attrs.update(attrs)
            for key, value in kw_input.items():
                if key not in names and key not in arg.sizes:
                    arg.coords[key] = value
                delta = (
                    key == 'experiment' and isinstance(value, str)
                    and 'picontrol' in value and 'abrupt4xco2' in value
                )
                if delta and 'style' in arg.coords and 'style' not in kw_input:
                    del arg.coords['style']  # i.e. default was requested
    else:  # create 2-tuple coordinates
        keys = sorted(set(key for kw in kws_input for key in kw))
        kw_input = {}
        for key in keys:
            values = tuple(kw.get(key, None) for kw in kws_input)
            value = values[0]
            dtype = np.asarray(getattr(value, 'magnitude', value)).dtype
            isnan = np.issubdtype(dtype, float) and np.isnan(value)
            if not isnan and len(values) == 2 and values[0] != values[1]:
                value = np.array(None, dtype=object)
                value[...] = tuple(values)
            kw_input[key] = value
        for arg in args:  # should be singleton
            arg.attrs.update(attrs)
            for key, value in kw_input.items():  # e.g. regress lat on feedback map
                if key not in arg.sizes:
                    arg.coords[key] = value
    return args, method, defaults
