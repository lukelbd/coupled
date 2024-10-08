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
from .reduce import _reduce_data, _reduce_datas, reduce_facets, reduce_general
from .specs import _expand_lists, _expand_parts, _group_parts, _pop_kwargs

__all__ = ['get_parts', 'get_result', 'process_constraint', 'process_data']


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
# TODO: See timescales definitions.py (eventually will be copied to climopy). Should
# also use this approach for flux addition terms and other stuff.
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
# climo.register_derivation(re.compile(r'\A(ta|ts)_grad\Z'))(equator_pole_delta)
# climo.register_derivation(re.compile(r'\A(ta|ts)_diff\Z'))(equator_pole_diffusivity)
# vreg.define('ta_grad', 'equator-pole air temperature difference', 'K')
# vreg.define('ts_grad', 'equator-pole surface temperature difference', 'K')
# vreg.define('ta_diff', 'bulk energy transport diffusivity', 'PW / K')
# vreg.define('ts_diff', 'bulk energy transport surface diffusivity', 'PW / K')


def _get_parts(
    name=None, parts=None, signs=None, product=None, relative=None,
    search=None, replace=None, absolute=None, scaled=True, **attrs,
):
    """
    Get the options associated with the input name.

    Parameters
    ----------
    name : str, optional
        The name whose components should be inferred.
    parts : str or sequence of str, optional
        The components required for the input `name`.
    signs : float or sequnce of float, optional
        The scalings to apply when combining the components.
    product : bool, optional
        Whether to combine components as a product or sum.
    relative : bool, optional
        Whether to subtract a global average from the result.
    search, replace : str, optional
        The regular expressions to use to replace components.
    absolute : bool, optional
        Whether to build atmospheric feedbacks from absolute components.
    scaled : bool, optional
        Whether to consider sensitivity and forcing components already scaled.
    **attrs
        The attributes to optionally add to the result.

    Returns
    -------
    parts : sequence of str
        The variable components.
    options : dict
        The combination options.
    """
    # TODO: Fix possible bug where (e.g.) file with shortwave longwave cloud effect
    # but no full cloud effect fails to read components, instead builds from net/cs.
    # NOTE: This is ad hoc implementation of cfvariable-like scheme for quickly
    # retrieving names. In future should implement these as derivations with standard
    # cfvariable properties and recursively combine / check for availability using
    # nested lists of derivation functions rather than this restricted approach.
    from .feedbacks import REGEX_FLUX as regex
    name = name and name.replace('rluscs', 'rlus')  # equivalent by construction
    name = name and name.replace('rsdtcs', 'rsdt')  # equivalent by construction
    trans = lambda arg, *subs: [re.sub(arg, sub, name).strip('_') for sub in subs]
    if name and (flux := regex.search(name)):
        head, _, wav, net, bnd, sky, *_, param = (*flux.groups(), *name.split('_'))
    else:
        head = wav = net = bnd = sky = param = 'NA'
    if not name:  # user-input only
        parts = parts or ()
    elif 'total' in name:
        parts = trans('total', 'lse', 'dse', 'ocean')
        search, replace = 'latent', 'total'
    elif 'mse' in name:
        parts = trans('mse', 'lse', 'dse')
        search, replace = 'latent', 'moist'
    elif 'dse' in name:  # possibly missing dry term e.g. dry stationary
        parts = trans('dse', 'hse', 'gse')
        search, replace = 'sensible', 'dry'
    elif name == 'tdev':  # TODO: define other 'relative' variables?
        parts, relative = ('tstd',), (True,)  # TODO: revisit labels
        attrs.update(units='K / sigma', long_name='temperature departure', short_name='departure')  # noqa: E501
    elif name == 'tabs':  # warming pattern for scaled sensitivity
        parts, product = ('tpat', 'ecs'), (True,)  # TODO: revisit
        attrs.update(units='K', long_name='temperature response', short_name='response')  # noqa: E501
    elif param == 'ts' or param == 'ta':  # eqtemp = temp - (temp - eqtemp) (see below)
        delta = name.replace('_' + param, '_dt')  # climate minus implied deviation
        parts, signs = (param, delta), (1, -1)
        search, replace = 'temperature', 'equilibrium temperature'
    elif param == 'dt':  # temp - eqtemp anomaly inferred from feedback (see above)
        value = name.replace('_dt', '').replace(f'{head}_', '')  # flux anomaly
        slope = name.replace('_dt', '_lam')  # feedback in denominator
        parts, signs, product = (value, slope), (1, -1), (True,)
        search, replace = 'flux', 'temperature'
        attrs.update(units='K', long_name='temperature deviation')
    elif not scaled and re.search(temp := r'(ecs|erf)(?:([0-9.-]+)x)?', name):
        parts = trans(temp, r'\1')
        scale = re.search(temp, name).group(2)
        signs = (np.log2(float(scale or 4)),)  # default to 4xCO2 scale
        search = r'\A'  # prepend to existing label
        replace = rf'{scale}$\\times$CO$_2$ ' if scale else ''
    elif head == 'atm':  # atmospheric kernels
        names1 = trans(regex, r'pl\2\3\4\5\6', r'lr\2\3\4\5\6', r'hus\2\3\4\5\6')
        names2 = trans(regex, r'pl*\2\3\4\5\6', r'lr*\2\3\4\5\6', r'hur\2\3\4\5\6')
        parts = names1 if absolute else names2
        search = r'(adjusted\s+)?Planck'
        replace = 'temperature + humidity'
    elif head == 'ncl':  # non-cloud kernels
        parts = trans(regex, r'cl\2\3\4\5\6', r'\2\3\4\5\6')
        search, replace = 'cloud', 'non-cloud'
        signs = (-1, 1)
    elif head == 'msk':  # cloud masking = cloud minus cloud effect
        parts = trans(regex, r'cl\2\3\4\5\6', r'\2\3\4\5ce')
        search, replace = 'cloud', 'cloud masking'
        signs = (1, -1)
    elif sky == 'ce':  # raw cloud effect
        parts = trans(regex, r'\1\2\3\4\5cs', r'\1\2\3\4\5')
        search, replace = 'clear-sky', 'cloud effect'
        signs = (-1, 1)
    elif bnd == 'a':  # net atmosheric (surface and TOA are positive into atmosphere)
        parts = trans(regex, r'\1\2\3\4t\6', r'\1\2\3\4s\6')
        search, replace = 'TOA', 'atmospheric'
        signs = (1, 1)
    elif net == 'e':  # effective forcing e.g. 'rlet'
        parts = trans(regex, r'\1\2\3n\5\6_erf')
        search, replace = 'effective ', ''  # NOTE: change?
        signs = (1, 1)
    elif net == 'r':  # radiative response e.g. 'rlrt'
        parts = trans(regex, r'\1\2\3n\5\6', r'\1\2\3n\5\6_erf')
        signs = (1, -1)
        search, replace = 'flux', 'response'
    elif wav == 'f':  # WARNING: critical to add wavelengths last
        opts = FEEDBACK_DEPENDENCIES.get(head, {'longwave': True, 'shortwave': True})
        wavs = [key[0] for key, val in opts.items() if val or head == '']
        parts = trans(regex, *(rf'\1\2{wav}\4\5\6' for wav in wavs))
        search = '(longwave|shortwave) '
        replace = 'net ' if head == sky == '' else ''
    elif net == 'n':  # WARNING: only used for CERES data currently
        idxs = slice(1, None) if wav == 'l' and bnd == 't' else slice(None, None)
        parts, signs = (r'\1\2\3d\5\6', r'\1\2\3u\5\6'), (1, -1)
        parts, signs = trans(regex, *parts[idxs]), signs[idxs]
        search = '(upwelling|downwelling|incoming|outgoing)'
        replace = 'net'
    else:  # defer error message
        parts = parts or (name,)
    parts = (parts,) if isinstance(parts, str) else tuple(parts)
    signs = np.atleast_1d(1 if signs is None else signs)
    signs = tuple(signs) * (1 if len(signs) == len(parts) else len(parts))
    search = search and re.compile(search)  # re.compile(re.compile(...)) allowed
    options = dict(signs=signs, product=product, relative=relative)
    options = dict(**options, attrs=attrs, search=search, replace=replace)
    return parts, options


def _get_result(parts, operate=True, hemisphere=None, **kwargs):
    """
    Get the data array associated with the input parts.

    Parameters
    ----------
    parts : namedtuple
        The (possibly nested) components and instructions returned by `_get_parts`.
    operate : str, optional
        Whether to carry out required operations or just return the first component.
    hemisphere : bool or str, optional
        The hemisphere to optionally select.
    **kwargs
        The reduce instructions passed to `reduce_general`.

    Returns
    -------
    xarray.DataArray
        The array used
    """
    # TODO: This is similar to climopy recursive derivation invocation. Except delays
    # summations until selections are performed in reduce_general()... better than
    # re-calculating full result every time (intensive) or caching full result after
    # calculation (excessive storage). Should add to climopy (see below).
    keys_climo = ('quantify', 'standardize')
    kw_climo = {key: kwargs.pop(key) for key in keys_climo if key in kwargs}
    datas, name = [], parts.__class__.__name__
    for part in parts.parts:
        data = part
        if isinstance(part, tuple):  # recurse
            data = _get_result(part, operate=operate, **kwargs)
        elif not operate:  # skip reduce
            data = part if isinstance(part, xr.DataArray) else xr.DataArray([], name=name)  # noqa: E501
        else:  # apply reduce
            data = reduce_general(data, **kwargs)
            if hemisphere and 'lat' in data.dims:
                data = data.climo.sel_hemisphere(hemisphere)
            if 'plev' in data.coords:
                data = data.rename(plev='lev')
            if isinstance(data, xr.Dataset):  # climopy derivation
                data = data.climo.get(name, **kw_climo)
            if 'lev' in data.coords:
                data = data.rename(lev='plev')
        datas.append(data)
    if operate:
        datas, signs = datas, parts.signs
    else:
        datas, signs = datas[:1], parts.signs[:1]
    if operate and parts.relative:
        bases = [data.climo.average('area') for data in datas]
    else:
        bases = (1 if parts.product else 0,) * len(signs)
    iter_ = zip(datas, signs, bases, strict=True)
    with xr.set_options(keep_attrs=True):
        if not operate or len(datas) == 1 and signs[0] == 1 and not parts.relative:
            data = datas[0]
        elif parts.product:  # get products or ratios
            data = math.prod(data ** sign / base for data, sign, base in iter_)
        else:  # get sums or differences
            data = sum(sign * data - base for data, sign, base in iter_)
    data = data.copy(deep=False)  # update attributes on derived variable
    data.name = name
    data.attrs.update(parts.attrs)
    data.attrs.setdefault('units', '')  # missing units
    data.attrs.setdefault('short_name', name)  # e.g. 'model'
    data.attrs.setdefault('long_name', name)  # e.g. 'model'
    for key in ('long_name', 'short_name', 'standard_name'):
        value = data.attrs.get(key, None)
        if parts.search and isinstance(value, str):
            data.attrs[key] = parts.search.sub(parts.replace, value)
    return data


def get_parts(dataset, name, scaled=None, **kwargs):
    """
    Return the components needed to build the requested variable.

    Parameters
    ----------
    dataset : xarray.Dataset
        The source dataset.
    name : str
        The requested variable.
    scaled : bool, optional
        Whether CO$_2$ scaling has been applied.
    **kwargs
        Additional settings passed to `_get_parts`.

    Returns
    -------
    parts : namedtuple
        Tuple named `name` containing the components required by `get_result`.
    """
    # NOTE: Previously had quadruple=True default option to load all feedbacks as
    # quadrupled versions, but meant same variable name could have different meaning
    # depending on input arguments. Now implement scaling as a derivation below and
    # add explicit indicator if '2x' or '4x' is specified. Preserve default of always
    # *displaying* the unscaled experiment results by translating 'erf' or 'ecs' below.
    from .feedbacks import ALIAS_VARIABLES as aliases
    from .feedbacks import REGEX_FLUX as regex
    name = aliases.get(name, name)
    names = [key for idx in dataset.indexes.values() for key in idx.names]
    if scaled is None:  # scaling not necessary
        scaled = 'ecs' not in name and 'erf' not in name
    absolute = all(name[:3] != 'pl*' for name in dataset)
    kwargs.update(scaled=scaled, absolute=absolute)
    if scaled and name in dataset or name in names or name in dataset.climo:
        parts, options = _get_parts(parts=[name], **kwargs)  # manual parts
    else:  # automatic parts
        parts, options = _get_parts(name, **kwargs)
    if name not in dataset and name in dataset.climo:
        attrs = ('short_name', 'long_name', 'standard_units')
        attrs = {attr: getattr(vreg[name], attr) for attr in attrs}
        options['attrs'].update(attrs)
    fluxes = ', '.join(s for s in dataset.data_vars if regex.search(s))
    results = []
    for part in parts:
        deps = VARIABLE_DEPENDENCIES.get(name, ())
        if part in dataset or part in names:  # note 'tabs' not saved
            part = dataset[part]
        elif part in dataset.climo:  # climopy derivation
            part = dataset.drop_vars(dataset.data_vars.keys() - set(deps))
        elif not scaled or part != name:  # valid derivation
            part = get_parts(dataset, part, scaled=True)
        else:  # print available
            raise KeyError(f'Required variable {name} not found. Options are: {fluxes}.')  # noqa: E501
        results.append(part)
    parts = namedtuple('parts', ('parts', *options))
    parts.__name__ = name  # permit non-idenfitier
    parts = parts(results, *options.values())
    return parts


def get_result(
    *args, spatial=None, quantify=False, standardize=True, **kwargs,
):
    """
    Return the requested variable or attribute.

    Parameters
    ----------
    *args : namedtuple or xarray.Dataset, str
        The `get_parts` `namedtuple` (or dataset plus variable name) and attribute.
    spatial : str, optional
        The spatial reduction option.
    quantify, standardize : bool, optional
        Passed to `accessor.get`.
    **kwargs
        Passed to `reduce_general`.

    Returns
    -------
    result : xarray.DataArray or str
        The result. If `attr` is not ``None`` this is the corresponding attribute.
    """
    # NOTE: This returns both data arrays and attributes. Should add to climopy by
    # 1) translating instructions, 2) selecting relevant variables, 3) performing scalar
    # selections first (e.g. points, maxima), 4) combining variables (e.g. products,
    # sums), then 5) more complex coordinate operations (e.g. averages, integrals). And
    # optionally order operations around non-linear variable or coordinate operations.
    from observed.arrays import mask_region
    scaled = kwargs.pop('scaled', None)
    kwargs.update(quantify=quantify, standardize=standardize)
    if args and isinstance(args[0], xr.Dataset):
        args = (get_parts(*args[:2], scaled=scaled), *args[2:])
    if len(args) == 1:
        part, attr = *args, None
    elif len(args) == 2:
        part, attr = args
    else:
        raise TypeError(f'Expected one to three input arguments but got {len(args)}.')
    if not isinstance(part, tuple):
        raise TypeError(f'Invalid input arguments {args!r}. Should be data or tuple.')
    if attr or not spatial:  # calculate result
        kwarg = {**kwargs, 'operate': attr is None}
        data = _get_result(part, **kwarg)
    else:  # only apply custom spatial correlation
        kwarg = _pop_kwargs(kwargs, mask_region)
        kwarg.setdefault('mask', kwargs.pop('area', None))
        kwarg.update(method=spatial, dim=('lon', 'lat'))
        data0 = _get_result(part, **{**kwargs, 'experiment': 'picontrol'})
        data1 = _get_result(part, **{**kwargs, 'experiment': 'abrupt4xco2'})
        data, attrs = _reduce_datas(data0, data1, **kwarg)
        data.attrs.update({key: val for key, val in attrs.items() if key != 'name'})
    if attr and attr == 'units':
        result = data.climo.units
    elif attr:
        result = data.attrs.get(attr, '')
    elif data.size:
        result = data
    else:
        raise ValueError(f'Empty result {data.name} with sizes {data.sizes}.')
    return result


def process_constraint(
    data0, data1, constraint=None, constraint_kw=None, observed=None, extended=None,
    dataset=None, noresid=False, nosigma=False, pctile=None, N=None,
    perturb=False, internal=False, combined=False, graphical=False, **kwargs,
):
    """
    Return percentile bounds for observational constraint.

    Parameters
    ----------
    data0 : xarray.DataArray
        The predictor data. Must be 1D.
    data1 : xarray.DataArray
        The predictand data. Must be 1D.
    dataset : xarray.Dataset, optional
        The source dataset used for inferring masking (if necessary).
    constraint : str or 2-tuple,
        The 95% bounds on the observational constraint.
    constraint_kw : dict, optional
        The keyword arguments passed to the loading function.
    observed : int, optional
        The number of years to use for observational uncertainty.
    extended : int, optional
        The number of additional years to apply shading.
    noresid : bool, optional
        Whether to ignore the regression residuals.
    nosigma : bool, optional
        Whether to ignore the regression standard error.
    pctile : float, default: 95
        The emergent constraint percentile bounds to be returned.
    N : int, default: 1000000
        The number of bootstrapped samples to carry out.

    Other Parameters
    ----------------
    perturb : bool, optional
        Whether to perturb the model data with uncertainty of each regression. Since
        errors are uncorrelated effect is very small (even slight strengthening).
    internal : bool or int, optional
        Whether to include model-estimated observed uncertainty due to variability. If
        ``True`` then ``20`` is used. If passed alternative bounds exclude variability.
    combined : bool, optional
        Whether to include alternative bootstrapping procedure based on single
        sampling of uncertain slope for the estimate and its offset.
    graphical : bool, optional
        Whether to use graphical intersections instead of residual bootstrapping. If
        ``False`` then bootstrapped slope times observations plus residual is used.
    **kwargs
        Passed to `_get_regression`.

    Returns
    -------
    observations : 3-tuple
        The lower, mean, and upper bounds for the observational constraint.
    result1 : 3-tuple
        The emergent constraint not accounting for regression uncertainty.
    result2 : 3-tuple
        The emergent constraint accounting for regression uncertainty.
    """
    # Initial stuff
    # NOTE: Here preserve regression properties between internal feedbacks and
    # constraint. Use 'external' and 'cld' for He et al. datasets
    from .reduce import _get_regression
    N = N or 10000000  # {{{
    np.random.seed(51423)  # see: https://stackoverflow.com/a/63980788/4970632
    historical = 24 + 4 / 12  # March 2000 to June 2024
    constraint = 'cre' if constraint is None or constraint is True else constraint
    observed = observed or historical
    internal = observed if internal is None or internal is True else internal
    pctile = 95 if pctile is None or pctile is True else pctile
    pctile = np.array([50 - 0.5 * pctile, 50 + 0.5 * pctile])
    keys = ('statistic', 'initial', 'detrend', 'remove', 'correct', 'error')
    name = constraint if isinstance(constraint, str) else 'cre'
    cloud, masks = (data0.name or '')[:2] == 'cl', ()
    kw_scalar = {key: data0.coords[key].item() for key in keys if key in data0.coords}
    kw_scalar.update(constraint_kw or {})
    if dataset is None and cloud:
        suffix, kw_scalar['source'] = 'cld', 'external'  # see observed/feedbacks.py
    elif not cloud or dataset is not None:  # possibly auto-infer masking
        suffix, masks = 'cre', ('mask', 'swmask', 'lwmask') if cloud else ()
    else:
        raise ValueError(f'Source dataset required for constraint {data0.name!r}.')

    # Read observational constraints
    # NOTE: Here optionally offset constraint estimate using average of internal
    # cloud feedback masking adjustments.
    if isinstance(constraint, str):  # {{{
        from .feedbacks import REGEX_FLUX as regex
        from .datasets import open_scalar
        stat = kw_scalar.get('statistic', None) or 'slope'
        scalar = open_scalar(ceres=True, suffix=suffix)  # TODO: update this
        project = data0.coords.get('project') or xr.DataArray('CMIP6')
        kw_mask = dict(area='avg', experiment='picontrol', project=project.item())
        kw_mean = dict(method='avg', weight=kwargs.get('weight'), skipna=True)
        for mask in masks:  # masking components
            print(f'Computing cloud masking: {mask!r}')
            (data,), *_ = reduce_facets(get_result(dataset, mask, **kw_mask), **kw_mean)
            icld = re.sub(regex, r'cl\2\3\4\5\6', data.name).strip('_')
            icre = re.sub(regex, r'\2\3\4\5ce', data.name).strip('_')
            scalar[icld] = scalar[icre].copy(deep=True)
            scalar[icld].loc[{'statistic': 'slope'}] += data.item()
        if stat == 'slope':  # constrain slope
            kw_scalar['statistic'] = ['slope', 'sigma', 'dof']
            scalar = get_result(scalar, constraint, **kw_scalar)
            constraint = scalar.values.tolist()
        else:
            kw_scalar['statistic'] = stat
            scalar = get_result(scalar, constraint, **kw_scalar)
            return [scalar.item()], [np.nan], [np.nan]
    if not np.iterable(constraint) or len(constraint) != 3:
        raise ValueError(f'Invalid constraint {constraint}. Must be (slope, sigma, dof).')  # noqa: E501
    if data0.ndim != 1 or data1.ndim != 1:
        raise ValueError(f'Invalid data dims {data0.ndim} and {data1.ndim}. Must be 1D.')  # noqa: E501
    xmean, xscale, xdof = constraint  # note dof will have small effect
    xdof *= observed / historical  # increased degrees of freedom
    xscale /= np.sqrt(observed / historical)  # reduced regression error
    extended = 50 if extended is True else extended or observed
    escale = extended / observed  # extended version
    xs1 = stats.t(df=xdof, loc=xmean, scale=xscale)
    xs2 = stats.t(df=escale * xdof, loc=xmean, scale=xscale / np.sqrt(escale))
    xmin1, xmax1 = xs1.ppf(0.01 * pctile)
    xmin2, xmax2 = xs2.ppf(0.01 * pctile)
    print(f' observed: mean {xmean:.2f} sigma {xscale:.2f} dof {xdof:.0f}')

    # Get bootstrapped uncertainty distributions
    # NOTE: Use N - 2 degrees of freedom for both observed feedback and inter-model
    # coefficients since they are both linear regressions. For now ignore uncertainty
    # of individual feedback regressions that comprise inter-model regression because
    # their sample sizes are larger (150 years) and uncertainties are uncorrelated so
    # should roughly cancel out across e.g. 30 members of ensemble.
    idxs = data0.argsort().values  # 'fit' has same coordinates  # {{{
    data0 = data0.isel({data0.dims[0]: idxs})
    data1 = data1.isel({data0.dims[0]: idxs})
    if not internal and extended == observed:
        ixs, observations = 0, np.array([xmin1, xmean, xmax1])
    elif not internal:
        ixs, observations = 0, np.array([xmin1, xmin2, xmean, xmax2, xmax1])
    else:
        internal = 20 if internal is True else internal
        kw_internal = dict(**kw_scalar, experiment='picontrol')
        kw_internal.update(period=f'{internal}yr', error='internal', statistic='sigma')
        dataset = open_scalar(ceres=True, suffix='cre')
        iscales = get_result(dataset, name, **kw_internal)
        iscale = iscales.mean().item()
        xs = stats.t(df=xdof, loc=xmean, scale=xscale).rvs(N)
        ixs = stats.norm(loc=0, scale=iscale).rvs(N)
        observations = np.insert(np.percentile(xs + ixs, pctile), 1, observations)
    if not perturb:  # skip perturbing model data
        bmean = None
        mean0 = data0.mean().values
        mean1 = data1.mean().values
    else:  # include regression error
        mscale, mdof, M = 0.11, 1000, data0.size  # TODO: use errors from file
        offset0 = stats.t(df=mdof, loc=0, scale=mscale).rvs((N, M))
        offset1 = stats.t(df=mdof, loc=0, scale=mscale).rvs((N, M))
        with xr.set_options(keep_attrs=True):
            datas0 = data0 + xr.DataArray(offset0, dims=('sample', 'facets'))
            datas1 = data1 + xr.DataArray(offset1, dims=('sample', 'facets'))
        bmean, *_ = _get_regression(datas0, datas1, pctile=pctile, nofit=True, **kwargs)
        mean0 = datas0.mean(dim='facets').values
        mean1 = datas1.mean(dim='facets').values
    result = _get_regression(data0, data1, pctile=False, **kwargs)
    slope, slope1, slope2, fit, fit1, fit2, rse, _, rdof = result
    bmean = (slope if bmean is None else bmean).values  # possibly include spread
    bscale = 0.5 * abs(slope2 - slope1)
    rdof, fit, fit1, fit2 = (da.values for da in (rdof, fit, fit1, fit2))  # noqa: E501

    # Propagate observational uncertainty through regression uncertainty
    # NOTE: This was adapted from Simpson et al. methdology. Reverse engineer the
    # t-distribution associated with observational estimate, then use distribution
    # properties for our bootstrapping. Tried only a/b uncertainty but too small.
    # NOTE: Offset error 'oe' = be * sqrt(sum(x ** 2) / n) = be * sqrt(mean(x ** 2))
    # and since mean(x ** 2) = mean((xm + xe) ** 2) = mean(xm) ** 2 + mean(xe) ** 2
    # this is -> be * xm + be * sqrt(mean(xe ** 2)). First part yields higher
    # offset uncerainty when farther from zero which makes sense, but second part
    # when combined with slope error formula 'be' yields simply the standard error
    # of residuals, i.e. the Simpson et al. approach! So never combine 'offset
    # uncertainty' with 'residual uncertainty' as they are identical, but should add
    # the slope uncertainty, otherwise constraint could be 'effective' when it has
    # slope near zero i.e. small residuals with large x-coordinate error.
    if graphical:  # intersection of shaded regions
        nbounds = 2 if internal else 1  # {{{
        fit1 = fit1.squeeze()
        fit2 = fit2.squeeze()
        xs = np.sort(data0, axis=0)  # linefit returns result for sorted data
        ymean = mean1 + bmean * (xmean - mean0)
        ymins = np.interp(observations[:nbounds], xs, fit1)
        ymaxs = np.interp(observations[-nbounds:], xs, fit2)
        constrained = (*ymins, ymean, *ymaxs)
        alternative = mean1 + bmean * (observations - mean0)  # }}}
    else:  # bootstrapping
        xs1, xs2 = xs1.rvs(N), xs2.rvs(N)  # {{{
        if not nosigma:
            bs = stats.t(loc=bmean, scale=bscale, df=rdof).rvs(N)
        else:  # ignore slope error
            bs = bmean
        if noresid:  # ignore residual
            rs1 = rs2 = 0
        elif not combined:  # regression standard error
            rs1 = rs2 = stats.t(loc=0, scale=rse, df=rdof).rvs(N)
        else:  # combined slope intercept
            nsq = np.sqrt(data0.size)  # infer weighted x-standard deviation
            std = rse.item() / bscale.item() / nsq  # compare to np.std
            bs, ibs = bmean, bs - bmean  # slope anomaly
            rs1, rs2 = ibs * (xs1 + std * nsq), ibs * (xs2 + std * nsq)
        ys1 = rs1 + mean1 + bs * (xs1 + ixs - mean0)  # include both
        ys2 = rs2 + mean1 + bs * (xs2 + ixs - mean0)
        if internal:  # alternative has no internal error
            as1 = rs1 + mean1 + bs * (xs1 - mean0)
            as2 = rs2 + mean1 + bs * (xs2 - mean0)
        else:  # alternative has no regression error
            as1 = rs1 + mean1 + bmean * (xs1 - mean0)
            as2 = rs2 + mean1 + bmean * (xs2 - mean0)
        ymean = np.mean(np.append(ys1, ys2))
        amean = np.mean(np.append(as1, as2))
        (ymin1, ymax1) = np.percentile(ys1, pctile)  # TODO: fix for multi pctiles
        (ymin2, ymax2) = np.percentile(ys2, pctile)
        (amin1, amax1) = np.percentile(as1, pctile)
        (amin2, amax2) = np.percentile(as2, pctile)
        if extended == observed:
            ymins, ymean, ymaxs = (ymin1,), ymean, (ymax1,)
            amins, amean, amaxs = (amin1,), ymean, (amax1,)
        else:
            ymins, ymean, ymaxs = (ymin1, ymin2), ymean, (ymax2, ymax1)
            amins, amean, amaxs = (amin1, amin2), amean, (amax2, amax1)
        constrained = np.array([*ymins, ymean, *ymaxs])
        alternative = np.array([*amins, amean, *amaxs])  # }}}
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
    # TODO: Return single arrays or 2-tuples of arrays with method and argument
    # attributes instead of multiple return values.
    # TODO: Possibly keep original 'signs' instead of _group_parts signs (see
    # below where arguments are combined and have to pick a sign).
    # NOTE: Initial kw_red values are formatted as (('[+-]', value), ...) to
    # permit arbitrary combinations of names and indexers (see _parse_specs).
    reduces = (get_result, reduce_facets, _reduce_data, _reduce_datas)
    kws_group, kws_count, kws_input, kws_facets = [], [], [], []  # {{{
    if len(specs) not in (1, 2):
        raise ValueError(f'Expected two process dictionaries. Got {len(specs)}.')
    for i, spec in enumerate(specs):  # iterate method reduce arguments
        kw_data, kw_count, kw_group = spec.copy(), {}, {}
        kw_facets = _pop_kwargs(kw_data, *reduces)
        kw_input = {key: val for key, val in kw_data.items() if key != 'name' and val is not None}  # noqa: E501
        kw_data = _group_parts(kw_data, keep_operators=True)
        for key, value in kw_data.items():
            sels = ['+']
            for part in _expand_parts(key, value):
                if part is None:
                    sel = None
                elif np.iterable(part) and part[:1] in ('+', '-', '*', '/'):
                    sel = part[:1]
                elif isinstance(part, (str, bool, tuple)):
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
    kws_data = []  # {{{
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
    print('.', end='')  # {{{
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
            kw_reduce = _pop_kwargs(kw_data, reduce_facets, _reduce_data, _reduce_datas)
            for key, value in kw_reduce.items():
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
        default.pop('symbol', None)
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
    print('.', end=' ')  # {{{
    args, method = [], methods_persum.pop()
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
        if not arg.name and len(datas) == 1 and (name := specs[0].get('name')):
            arg.name = name  # TODO: revisit
        if suffix and any(sign == -1 for sign in signs):
            suffix = 'anomaly' if suffix is True else suffix
            arg.attrs['long_suffix'] = arg.attrs['short_suffix'] = ''
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
    args = xr.align(*args)  # re-align after summation  # {{{
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
