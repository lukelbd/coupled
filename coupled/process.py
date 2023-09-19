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
import pandas as pd
import xarray as xr
from climopy.var import linefit
from climopy import ureg, vreg  # noqa: F401
from scipy import stats
from icecream import ic  # noqa: F401

from .internals import KEYS_METHOD
from .internals import _group_parts, _ungroup_parts, _to_lists
from .results import ALIAS_FEEDBACKS, FEEDBACK_ALIASES, REGEX_FLUX
from .reduce import _method_double, apply_method, apply_reduce
from cmip_data.feedbacks import FEEDBACK_DEPENDENCIES


__all__ = ['get_data', 'process_data']


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
    pctile = 90 if pctile is None else pctile
    # pctile = 95 if pctile is None else pctile
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
    xmin, xmax = stats.t.ppf(0.01 * pctile, loc=xmean, scale=xscale, df=xdof)
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
        ymin, ymean, ymax = mean1 + bmean * (np.array(observations) - mean0)
        alternative = (ymin, ymean, ymax)  # no regression uncertainty
    return observations, alternative, constrained


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
        Passed to `apply_reduce`.

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
    # NOTE: Here apply_reduce will automatically perform default selections for certain
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
        data = apply_reduce(data, **kwargs)
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
        Passed to `apply_reduce`.

    Returns
    -------
    result : xarray.DataArray or str
        The result. If `attr` is not ``None`` this is the corresponding attribute.
    """
    # TODO: This is similar to climopy recursive derivation invocation. Except delays
    # summations until selections are performed in apply_reduce()... better than
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
    if spatial:  # only now apply spatial correlation
        if 'version' in data.coords:
            kw0, kw1 = {'start': 0, 'stop': 150}, {'start': start, 'stop': stop}
        else:
            kw0, kw1 = {}, {}
        data0 = data.sel(experiment='picontrol', **kw0)
        data1 = data.sel(experiment='abrupt4xco2', **kw1)
        data, attrs = _method_double(
            data0, data1, dim='area', method=spatial
        )
        name = attrs.pop('name')  # only used in apply_method()
        data.attrs.update({**data.attrs, **attrs})
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
    else:  # create 2-tuple coordinates
        keys = sorted(set(key for kw in kws_input for key in kw))
        kw_input = {}
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
    return args, method, kwargs
