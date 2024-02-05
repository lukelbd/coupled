#!/usr/bin/env python3
"""
Utilities for processing coupled model data variables.
"""
import itertools
import math
import re
from collections import namedtuple

import climopy as climo  # noqa: F401
import numpy as np
import xarray as xr
from climopy import var, ureg, vreg  # noqa: F401
from scipy import stats
from icecream import ic  # noqa: F401

from cmip_data.feedbacks import FEEDBACK_DEPENDENCIES
from .reduce import _reduce_datas, reduce_facets, reduce_general
from .specs import _expand_lists, _expand_parts, _group_parts, _pop_kwargs

__all__ = ['get_parts', 'get_result', 'process_constraint', 'process_data']


# Observational constraint mean and spreads
# TODO: Auto-generate this dictionary or save and commit then parse a text file. For
# now copied from jupyterlab observed.ipynb cell.
# NOTE: These show best estimate, standard errors, and degrees of freedom for feedback
# regression slopes using either GISTEMP4 or HadCRUT5 and using surface temp autocorr
# adjustment (consistent with Dessler et al.). See observed.ipynb notebook.
FEEDBACK_CONSTRAINTS = {
    # 'cld': (0.69, (1.04 - 0.31) / 2,  # He et al. with 95% uncertainty
    # 'cld': (0.68, 0.26, 110),  # He et al. trended estimate (total record is 226)
    'cld': (0.35, 0.35, 111),  # He et al. detrended estimate (total record is 226)
    'net': (-0.85, 0.37, 177),  # custom estimates (total record is 274)
    'sw': (0.86, 0.29, 185),
    'lw': (-1.71, 0.23, 198),
    'cre': (0.27, 0.33, 129),
    'swcre': (0.19, 0.34, 140),
    'lwcre': (0.08, 0.16, 182),
    'hadnet': (-0.84, 0.39, 177),
    'hadsw': (0.89, 0.31, 183),
    'hadlw': (-1.73, 0.25, 191),
    'hadcre': (0.30, 0.35, 129),
    'hadswcre': (0.15, 0.36, 140),
    'hadlwcre': (0.15, 0.17, 182),
}

# Model-estimated bootstrapped uncertainty (see _calc_bootstrap)
# TODO: Record degrees of freedom and autocorrelation across successive bootstrap
# feedback estimates (e.g. year 0-20 estimate will be related to year 5-25). For now
# just assume implied distribution is Gaussian in process_constraint() below.
# NOTE: Include bootstraps longer than observational record so that we can
# test what uncertainty *would be* if input observed feedback was calculated
# from longer record. Will also adjust dof on regression estimate uncertainty.
INTERNAL_UNCERTAINTY = {
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


def get_parts(dataset, name, scaled=False):  # noqa: E501
    """
    Return the components needed to build the requested variable.

    Parameters
    ----------
    dataset : xarray.Dataset
        The source dataset.
    name : str
        The requested variable.
    scaled : bool, optional
        Whether ecs/erf data has been scaled.

    Returns
    -------
    parts : namedtuple
        Tuple named `name` containing the components required by `get_result`.
    """
    # NOTE: This is ad hoc implementation of cfvariable-like scheme for quickly
    # retrieving names. In future should implement these as derivations with standard
    # cfvariable properties and recursively combine / check for availability using
    # nested lists of derivation functions rather than this restricted approach.
    # NOTE: Previously had quadruple=True default option to load all feedbacks as
    # quadrupled versions, but meant same variable name could have different meaning
    # depending on input arguments. Now implement scaling as a derivation below and
    # add explicit indicator if '2x' or '4x' is specified. Preserve default of always
    # *displaying* the unscaled experiment results by translating 'erf' or 'ecs' below.
    def _get_names(regex, *patterns):
        if isinstance(regex, str):
            regex = re.compile(regex)
        names = [regex.sub(pattern, name) for pattern in patterns]
        names = [name.strip('_') for name in names]
        return names
    from .feedbacks import ALIAS_VARIABLES as aliases
    from .feedbacks import REGEX_FLUX as regex
    name = aliases.get(name, name)
    names, signs = (name,), None
    search = replace = rescale = product = relative = None
    scaled = scaled or 'ecs' not in name and 'erf' not in name
    attrs = {}  # attribute overrides
    keys = ('short_name', 'long_name', 'standard_units')
    head = wav = net = bnd = sky = param = 'NA'
    if flux := regex.search(name):
        head, _, wav, net, bnd, sky = flux.groups()
        *_, param = name.split('_')
    if scaled and name in dataset:
        pass
    elif 'rluscs' in name or 'rsdtcs' in name:
        names = (name[:-2],)  # all-sky same as clear-sky
    elif name in dataset.climo and name not in dataset:
        attrs.update({attr: getattr(vreg[name], attr) for attr in keys})
    elif 'total' in name:
        names = _get_names('total', 'lse', 'dse', 'ocean')
        search, replace = 'latent', 'total'
    elif 'mse' in name:
        names = _get_names('mse', 'lse', 'dse')
        search, replace = 'latent', 'moist'
    elif 'dse' in name:  # possibly missing dry term e.g. dry stationary
        names = _get_names('dse', 'hse', 'gse')
        search, replace = 'sensible', 'dry'
    elif name == 'tdev':  # TODO: define other 'relative' variables?
        names, relative = ('tstd',), (True,)
        attrs.update(units='K', long_name='relative warming')
    elif name == 'tabs':  # warming pattern for scaled sensitivity
        rescale = True  # re-apply scaling
        names, product = ('tpat', 'ecs'), (True,)
        attrs.update(units='K', long_name='effective warming')
    elif param == 'ts' or param == 'ta':  # eqtemp = temp - (temp - eqtemp) (see below)
        delta = name.replace('_' + param, '_dt')  # climate minus implied deviation
        names, signs = (param, delta), (1, -1)
        search, replace = 'temperature', 'equilibrium temperature'
    elif param == 'dt':  # temp - eqtemp anomaly inferred from feedback (see above)
        denom = name.replace('_dt', '_lam')  # feedback in denominator
        numer = name.replace('_dt', '').replace(f'{head}_', '')
        names, signs, product = (numer, denom), (1, -1), (True,)
        search, replace = 'flux', 'temperature'
        attrs.update(units='K', long_name='temperature difference')
    elif not scaled and re.search(temp := r'(ecs|erf)(?:([0-9.-]+)x)?', name):
        names = _get_names(temp, r'\1')
        scale = re.search(temp, name).group(2)
        signs = (np.log2(float(scale or 4)),)  # default to 4xCO2 scale
        search = r'\A'  # prepend to existing label
        replace = rf'{scale}$\\times$CO$_2$ ' if scale else ''
    elif head == 'atm':  # atmospheric kernels
        names1 = _get_names(regex, r'pl\2\3\4\5\6', r'lr\2\3\4\5\6', r'hus\2\3\4\5\6')
        names2 = _get_names(regex, r'pl*\2\3\4\5\6', r'lr*\2\3\4\5\6', r'hur\2\3\4\5\6')
        names = names1 if names1[0] in dataset else names2
        search = r'(adjusted\s+)?Planck'
        replace = 'temperature + humidity'
    elif head == 'ncl':  # non-cloud kernels
        names = _get_names(regex, r'cl\2\3\4\5\6', r'\2\3\4\5\6')
        search, replace = 'cloud', 'non-cloud'
        signs = (-1, 1)
    elif sky == 'ce':  # raw cloud effect
        names = _get_names(regex, r'\1\2\3\4\5cs', r'\1\2\3\4\5')
        search, replace = 'clear-sky', 'cloud'
        signs = (-1, 1)
    elif bnd == 'a':  # net atmosheric (surface and TOA are positive into atmosphere)
        names = _get_names(regex, r'\1\2\3\4t\6', r'\1\2\3\4s\6')
        search, replace = 'TOA', 'atmospheric'
        signs = (1, 1)
    elif net == 'e':  # effective forcing e.g. 'rlet'
        names = _get_names(regex, r'\1\2\3n\5\6_erf')
        search, replace = 'effective ', ''  # NOTE: change?
        signs = (1, 1)
    elif net == 'r':  # radiative response e.g. 'rlrt'
        names = _get_names(regex, r'\1\2\3n\5\6', r'\1\2\3n\5\6_erf')
        signs = (1, -1)
        search, replace = 'flux', 'response'
    elif wav == 'f':  # WARNING: critical to add wavelengths last
        opts = FEEDBACK_DEPENDENCIES.get(head, {'longwave': True, 'shortwave': True})
        wavs = [key[0] for key, val in opts.items() if val or head == '']
        names = _get_names(regex, *(rf'\1\2{wav}\4\5\6' for wav in wavs))
        search = '(longwave|shortwave) '
        replace = 'net ' if head == sky == '' else ''
    elif net == 'n':  # WARNING: only used for CERES data currently
        if wav == 'l' and bnd == 't':
            signs, names = (-1,), (r'\1\2\3u\5\6',)
        else:  # upwelling downwelling components
            signs, names = (1, -1), (r'\1\2\3d\5\6', r'\1\2\3u\5\6')
        names = _get_names(regex, *names)
        search = '(upwelling|downwelling|incoming|outgoing)'
        replace = 'net'
    else:
        opts = ', '.join(s for s in dataset.data_vars if regex.search(s))
        raise KeyError(f'Required variable {name} not found. Options are: {opts}.')
    parts = []
    for iname in names:
        if iname in dataset or iname not in dataset.climo:
            if scaled and iname in dataset:
                part = dataset[iname]
            else:  # instruction tuple
                part = get_parts(dataset, iname, scaled=not rescale)
        else:
            if deps := VARIABLE_DEPENDENCIES.get(iname, ()):
                part = dataset.drop_vars(dataset.data_vars.keys() - set(deps))
            else:
                raise KeyError(f'Climopy derivation {iname} has unknown dependencies.')
        parts.append(part)
    keys = ('parts', 'names', 'signs', 'product', 'relative', 'attrs', 'search', 'replace')  # noqa: E501
    signs = signs or (1,) * len(names)
    search = search and re.compile(search)  # note re.compile(re.compile(...)) allowed
    values = (parts, names, signs, product, relative, attrs, search, replace)
    parts = namedtuple('parts', keys)(*values)
    parts.__class__.__name__ = name  # permit non-idenfitier
    return parts


def get_result(
    *args, attr=None, spatial=None, hemi=None, hemisphere=None, quantify=False, standardize=True, **kwargs,  # noqa: E501
):
    """
    Return the requested variable or attribute.

    Parameters
    ----------
    *args : namedtuple
        The variable arguments. If not a `namedtuple` then passed to `get_parts`.
    attr : str, optional
        The attribute to retrieve. Can be passed as the final positional argument.
    spatial : str, optional
        The spatial reduction option.
    hemi, hemisphere : str, optional
        Passed to `.sel_hemisphere`.
    quantify, standardize : bool, optional
        Passed to `accessor.get`.
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
    def _get_array(parts, attr=None, hemi=None, **kwargs):
        keys_climo = ('quantify', 'standardize')
        kw_climo = {key: kwargs.pop(key) for key in keys_climo if key in kwargs}
        datas, name = [], parts.__class__.__name__
        for part in parts.parts:
            data = part
            if isinstance(part, tuple):  # recurse
                data = _get_array(part, attr=attr, **kwargs)
            elif attr is not None:  # skip reduce
                if isinstance(part, xr.DataArray):  # data variable
                    data = part
                else:  # climopy derivation
                    data = xr.DataArray([], name=name)
            else:  # apply reduce
                data = reduce_general(data, **kwargs)
                if hemi and 'lat' in data.dims:
                    data = data.climo.sel_hemisphere(hemi)
                if 'plev' in data.coords:
                    data = data.rename(plev='lev')
                if isinstance(data, xr.Dataset):  # climopy derivation
                    data = data.climo.get(name, **kw_climo)
                if 'lev' in data.coords:
                    data = data.rename(lev='plev')
            datas.append(data)
        if attr is None:
            datas, names, signs = datas, parts.names, parts.signs
        else:
            datas, names, signs = datas[:1], parts.names[:1], parts.signs[:1]
        if not parts.relative or attr is not None:
            bases = (1 if parts.product else 0,) * len(names)
        else:
            bases = [data.climo.average('area') for data in datas]
        iter_ = zip(datas, signs, bases, strict=True)
        with xr.set_options(keep_attrs=True):
            if attr is not None or len(datas) == 1 and signs[0] == 1 and not parts.relative:  # noqa: E501
                data = datas[0]
            elif parts.product:  # get products or ratios
                data = math.prod(data ** sign / base for data, sign, base in iter_)
            else:  # get sums or differences
                data = sum(sign * data - base for data, sign, base in iter_)
        data = data.copy(deep=False)  # update attributes on derived variable
        data.name = name
        data.attrs.update(parts.attrs)
        for key in ('long_name', 'short_name', 'standard_name'):
            value = data.attrs.get(key, None)
            if parts.search and isinstance(value, str):
                data.attrs[key] = parts.search.sub(parts.replace, value)
        return data
    if not args or len(args) > 3:
        raise TypeError(f'Expected 1 to 3 positional arguments but got {len(args)}.')  # noqa: E501
    if len(args) == 1 and not isinstance(args[0], tuple):
        raise TypeError('Input argument must be namedtuple returned by get_parts().')
    kwargs['hemi'] = hemi or hemisphere  # permit either
    kwargs['quantify'] = quantify  # detectable with _pop_kwargs
    kwargs['standardize'] = standardize  # detectable with _pop_kwargs
    nargs = isinstance(args[0], xr.Dataset) + 1
    args, attrs = args[:nargs], args[nargs:]
    parts = args[0] if nargs == 1 else get_parts(*args)
    attr = attr or attrs and attrs[0] or None
    data = _get_array(parts, attr=attr, **kwargs)
    if spatial is not None:  # only apply custom spatial correlation
        val0, val1 = kwargs.get('start', 0), kwargs.get('stop', 150)
        kw0 = {'start': 0, 'stop': 150} if 'version' in data.coords else {}
        kw1 = {'start': val0, 'stop': val1} if 'version' in data.coords else {}
        data0 = data.sel(experiment='picontrol', **kw0)
        data1 = data.sel(experiment='abrupt4xco2', **kw1)
        data, attrs = _reduce_datas(data0, data1, dim='area', method=spatial)
        data.attrs.update({**data.attrs, **attrs})
        del data.attrs['name']  # only used in reduce_facets()
    if attr == 'units':
        result = data.climo.units
    elif attr:
        result = data.attrs.get(attr, '')
    elif data.size:
        result = data
    else:
        raise ValueError(f'Empty result {data.name} with sizes {data.sizes}.')
    return result


def process_constraint(
    data0, data1, constraint=None, observed=None, internal=False, graphical=False, pctile=None, N=None,  # noqa: E501
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
    observed : int, optional
        The number of years to use for observational uncertainty.
    internal : bool or int, optional
        Whether to include model-estimated observed uncertainty due to variability. If
        ``True`` then ``20`` is used. If passed the bounds are with and without model
        inferred spread due to internal variability. Also adjusts the observed spread.
    graphical : bool, optional
        Whether to use graphical intersections instead of residual bootstrapping. If
        ``False`` then constrained uncertainty estimated by adding t-distribution with
        standard error calculated from residuals of regression relationship.
    pctile : float, default: 95
        The emergent constraint percentile bounds to be returned.
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
    from .feedbacks import VARIABLE_ALIASES
    constraint = 'cld' if constraint is None or constraint is True else constraint
    observed = observed or 20
    internal = observed if internal is None or internal is True else internal
    pctile = 95 if pctile is None or pctile is True else pctile
    pctile = np.array([50 - 0.5 * pctile, 50 + 0.5 * pctile])
    kwargs = dict(pctile=pctile, correct=False)  # no correlation across ensembles
    name = constraint if isinstance(constraint, str) else None
    name = VARIABLE_ALIASES.get(name, name)
    N = N or 10000  # samples to draw
    if name is not None:
        constraint = FEEDBACK_CONSTRAINTS[name]
    if not np.iterable(constraint) or len(constraint) != 3:
        raise ValueError(f'Invalid constraint {constraint}. Must be (mean, sigma, dof).')  # noqa: E501
    if data0.ndim != 1 or data1.ndim != 1:
        raise ValueError(f'Invalid data dims {data0.ndim} and {data1.ndim}. Must be 1D.')  # noqa: E501
    xmean, xscale, xdof = constraint  # note dof will have small effect
    nbounds = 1 if internal is None else 2  # number of bounds returned
    xdof *= observed / 20  # increased degrees of freedom
    xscale /= np.sqrt(observed / 20)  # reduced regression error
    xmin, xmax = stats.t(df=xdof, loc=xmean, scale=xscale).ppf(0.01 * pctile)
    observations = np.array([xmin, xmean, xmax])
    if internal is not False:
        internal = INTERNAL_UNCERTAINTY[name][internal]
        xs = stats.t(df=xdof, loc=xmean, scale=xscale).rvs(N)
        es = stats.norm(loc=0, scale=internal).rvs(N)  # implied variability error
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
        if internal is not False:
            xs = stats.t(df=xdof, loc=xmean, scale=xscale).rvs(N)
            es = stats.norm(loc=0, scale=internal).rvs(N)  # implied variability error
            ys = rs + mean1 + bmean * (xs + es - mean0)
            constrained = np.insert(np.percentile(ys, pctile), 1, constrained)
            ys = mean1 + bmean * (xs + es - mean0)
            alternative = np.insert(np.percentile(ys, pctile), 1, alternative)
    return observations, alternative, constrained


def process_data(dataset, *specs, attrs=None, suffix=True):
    """
    Combine the data based on input reduce dictionaries.

    Parameters
    ----------
    dataset : xarray.Dataset
        The input dataset.
    *specs : dict
        The variable specifiers.
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
    kws_group, kws_count, kws_input, kws_facets = [], [], [], []
    if len(specs) not in (1, 2):
        raise ValueError(f'Expected two process dictionaries. Got {len(specs)}.')
    for i, spec in enumerate(specs):  # iterate method reduce arguments
        kw_data, kw_count, kw_group = spec.copy(), {}, {}
        kw_facets = _pop_kwargs(kw_data, reduce_facets)
        kw_input = {key: val for key, val in kw_data.items() if key != 'name' and val is not None}  # noqa: E501
        kw_data = _group_parts(kw_data, keep_operators=True)
        for key, value in kw_data.items():
            sels = ['+']
            for part in _expand_parts(value):
                if part is None:
                    sel = None
                elif np.iterable(part) and part[0] in ('+', '-', '*', '/'):
                    sel = part[0]
                elif isinstance(part, (str, tuple)):  # already mapped integers
                    sel = part
                else:  # retrieve unit
                    unit = get_result(dataset, key, 'units')
                    if not isinstance(part, ureg.Quantity):
                        part = ureg.Quantity(part, unit)
                    sel = part.to(unit)
                sels.append(sel)
            signs, values = sels[0::2], sels[1::2]
            kw_count[key] = len(values)
            kw_group[key] = tuple(zip(signs, values))
        kws_count.append(kw_count)
        kws_group.append(kw_group)
        kws_input.append(kw_input)
        kws_facets.append(kw_facets)

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
    for kw_group, kw_facets in zip(kws_group, kws_facets):
        startstop = 'startstop' in kw_group and 'experiment' in kw_group
        groups = [('startstop', 'experiment')] if startstop else []
        for group in groups:
            values = _expand_lists(*(kw_group[key] for key in group))
            kw_group.update(dict(zip(group, values)))
        groups.extend(
            (key,) for key in kw_group if not any(key in group for group in groups)
        )
        kw_product = {
            group: tuple(zip(*(kw_group[key] for key in group))) for group in groups
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
            kw_data.update(kw_facets)
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
        datas, exps, kw_facets = [], [], {}
        for kw_data in ikws_data:  # iterate reduce arguments
            kw_data = kw_data.copy()
            name = kw_data.pop('name')  # NOTE: always present
            area = kw_data.get('area')
            experiment = kw_data.get('experiment')
            for key, value in _pop_kwargs(kw_data, reduce_facets).items():
                kw_facets.setdefault(key, value)
            if name == 'tabs' and experiment == 'picontrol':
                name = 'tstd'  # use rfnt_ecs for e.g. abrupt minus pre-industrial
            if name in warming and area == 'avg' and len(ikws_data) == 2:
                if name in warming[:2] or experiment == 'picontrol':
                    name = 'tstd'  # default to global average temp standard deviation
            data = get_result(dataset, name, **kw_data)
            datas.append(data)
            exps.append(experiment)
        datas, method, default = reduce_facets(*datas, **kw_facets)
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
    # coefficients we use sum of variances (see _reduce_data and _reduce_datas for
    # details). Institute differences are now only supported for scalar plots.
    print('.', end=' ')
    args = []
    method = methods_persum.pop()
    signs_persum, datas_persum = zip(*datas_persum)
    for signs, datas in zip(zip(*signs_persum), zip(*datas_persum)):  # plot arguments
        datas = xr.align(*datas)
        parts = tuple(data.isel(sigma=0) if 'sigma' in data.dims else data for data in datas)  # noqa: E501
        sum_scale = sum(sign == 1 for sign in signs)
        if parts[0].sizes.get('facets', None) == 0:
            raise RuntimeError('Empty model facets dimension.')
        with xr.set_options(keep_attrs=True):  # keep e.g. units and short_prefix
            arg = sum(sign * part for sign, part in zip(signs, parts)) / sum_scale
        if method == 'dist' and len(parts) > 1 and np.allclose(arg, 0):
            arg = parts[0]  # kludge for e.g. control late minus control early
        if len(parts) == 1 and (name := specs[0].get('name')):
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
    # WARNING: In some xarray versions seems _reduce_datas with conflicting scalar
    # coordinates will keep the first one instead of discarding. So overwrite below.
    # NOTE: Xarray automatically drops non-matching scalar coordinates (similar
    # to vector coordinate matching utilities) so try to restore them below.
    # NOTE: Global average regressions of local pre-industrial feedbacks onto global
    # pre-industrial feedbacks equal one despite regions with much larger magnitudes.
    # if args[0].sizes.keys() & {'lon', 'lat'}:  # ensure average is one
    #     ic(specs, args[0].climo.average('area').item())
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
