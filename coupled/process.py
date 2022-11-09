#!/usr/bin/env python3
"""
Internal helper functions for figure templates.
"""
import itertools

import climopy as climo  # noqa: F401
import numpy as np
import pandas as pd
import xarray as xr
from climopy import ureg, vreg  # noqa: F401
from icecream import ic  # noqa: F401

from .output import FEEDBACK_TRANSLATIONS
from .labels import ORDER_LOGICAL, REGEX_FLOAT, REGEX_SPLIT
from cmip_data.internals import MODELS_INSTITUTES, INSTITUTES_LABELS

__all__ = ['apply_reduce', 'apply_method', 'get_data']

# Reduce presets
# See (WPG and ENSO): https://doi.org/10.1175/JCLI-D-12-00344.1
# See (WPG and ENSO): https://doi.org/10.1038/s41598-021-99738-3
# See (tropical ratio): https://doi.org/10.1175/JCLI-D-18-0843.1
# See (feedback regions): https://doi.org/10.1175/JCLI-D-17-0087.1
# https://climatedataguide.ucar.edu/climate-data/nino-sst-indices-nino-12-3-34-4-oni-and-tni
AREA_REGIONS = {
    'trop': {'lat_lim': (-30, 30), 'lon_lim': (0, 360)},
    'pool': {'lat_lim': (-30, 30), 'lon_lim': (50, 200)},
    'wlam': {'lat_lim': (-15, 15), 'lon_lim': (150, 170)},
    'elam': {'lat_lim': (-30, 0), 'lon_lim': (260, 280)},
    'nina': {'lat_lim': (0, 10), 'lon_lim': (130, 150)},
    'nino': {'lat_lim': (-5, 5), 'lon_lim': (190, 240)},
    'nino3': {'lat_lim': (-5, 5), 'lon_lim': (210, 270)},
    'nino4': {'lat_lim': (-5, 5), 'lon_lim': (160, 210)},
}


def _parse_project(data, project=None):
    """
    Return plot labels and facet filter for the project indicator.

    Parameters
    ----------
    data : xarray.Dataset or xarray.DataArray
        The data. Must contain a ``'facets'`` coordinate.
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
    if imax not in (1, 2, 4):
        raise ValueError(f'Invalid project indicator {project}. 1/2/4 numbers allowed.')
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
    func.name = project  # WARNING: critical for get_data() detection of 'all_projs'
    return func


def _parse_institute(data, institute=None):
    """
    Return plot labels and facet filter for the institute indicator.

    Parameters
    ----------
    data : xarray.Dataset or xarray.DataArray
        The data. Must contain a ``'facets'`` coordinate.
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
    inst_to_abbrv = INSTITUTES_LABELS.copy()  # see also _parse_constraints
    model_to_inst = MODELS_INSTITUTES.copy()
    if not institute:
        filt = lambda key: True  # noqa: U100
    elif institute == 'avg':
        insts = [
            model_to_inst.get((key[0], key[1]), 'U')
            for key in data.facets.values
        ]
        facets = [
            (key[0], inst_to_abbrv.get(inst, inst), key[2], key[3])
            for inst, key in zip(insts, data.facets.values)
        ]
        filt = xr.DataArray(
            pd.MultiIndex.from_tuples(facets, names=data.indexes['facets'].names),
            attrs=data.facets.attrs,
            dims='facets',  # WARNING: critical fro groupby() to name this 'facets'
        )
        filt.name = 'facets'
    elif institute == 'flagship':  # flagship from either cmip5 or cmip6 series
        inst_to_model = {
            (proj, inst): model for (proj, model), inst in model_to_inst.items()
        }
        filt = lambda key: (
            key[1]
            == inst_to_model.get((key[0], model_to_inst.get((key[0], key[1]))))
        )
        filt.name = institute  # unnecessary but why not
    else:
        abbrv_to_inst = {
            abbrv: inst for inst, abbrv in inst_to_abbrv.items()
        }
        filt = lambda key: (
            abbrv_to_inst.get(institute, institute)
            == model_to_inst.get((key[0], key[1]))
        )
        filt.name = institute  # unnecessary but why not
    return filt


def _parts_composite(data0, data1, pctile=None):  # noqa: E301
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
    thresh = 33 if pctile is None else pctile
    data0, data1 = xr.broadcast(data0, data1)
    comp_lo = np.nanpercentile(data0, thresh)
    comp_hi = np.nanpercentile(data0, 100 - thresh)
    mask_lo, = np.where(data0 <= comp_lo)
    mask_hi, = np.where(data0 >= comp_hi)
    data_hi = data1.isel(facets=mask_hi)
    data_lo = data1.isel(facets=mask_lo)
    with np.errstate(all='ignore'):
        data_hi = data_hi.mean('facets', keep_attrs=True)
        data_lo = data_lo.mean('facets', keep_attrs=True)
    data_hi = data_hi.climo.quantify()
    data_lo = data_lo.climo.quantify()
    return data_lo, data_hi


def _parts_covariance(data0, data1, both=True):
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
    data0, data1 = xr.broadcast(data0, data1)
    data0 = data0.climo.quantify()
    data1 = data1.climo.quantify()
    with np.errstate(all='ignore'):
        mean0 = data0.mean(dim='facets', skipna=True)
        mean1 = data1.mean(dim='facets', skipna=True)
        anom0 = data0 - mean0
        anom1 = data1 - mean1
        covar = (anom0 * anom1).sum(dim='facets', skipna=True)
        std0 = (anom0 ** 2).sum(dim='facets', skipna=True)
        std0 = np.sqrt(std0)
        if both:
            std1 = (anom1 ** 2).sum(dim='facets', skipna=True)
            std1 = np.sqrt(std1)
    return (covar, std0, std1) if both else (covar, std0)


def apply_reduce(data, attrs=None, **kwargs):
    """
    Carry out arbitrary reduction of the given dataset variables.

    Parameters
    ----------
    data : xarray.DataArray
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
    # Apply special reductions
    # TODO: Might consider adding 'institute' multi-index so e.g. grouby('institute')
    # is possible... then again would not reduce complexity because would now need to
    # load files in special non-alphabetical order to enable selecting 'flaghip' models
    # with e.g. something like .sel(institute=-1)... so don't bother for now.
    project = kwargs.pop('project', None)
    if project is not None:  # see _parse_project
        if callable(project):
            facets = list(filter(project, data.facets.values))
            data = data.sel(facets=facets)
            data.coords['project'] = project.name
        else:
            raise RuntimeError(f'Unsupported project {project!r}.')
    institute = kwargs.pop('institute', None)
    if institute is not None:  # ignore dummy None
        if callable(institute):  # facets filter function
            facets = list(filter(institute, data.facets.values))
            data = data.sel(facets=facets)
            data.coords['institute'] = institute.name
        elif isinstance(institute, xr.DataArray):  # groupby multi-index data array
            if project:  # filter works with both models and institutes
                bools = list(map(project, institute.facets.values))
                institute = institute[{'facets': bools}]
            data = data.groupby(institute).mean(skipna=False, keep_attrs=True)
            facets = data.indexes['facets']  # xarray bug causes dropped level names
            facets.names = institute.indexes['facets'].names
            facets = xr.DataArray(
                facets,
                name='facets',
                dims='facets',
                attrs=institute.facets.attrs  # WARNING: avoid overwriting input kwarg
            )
            data = data.assign_coords(facets=facets)
            data.coords['institute'] = 'avg'
        else:
            raise RuntimeError(f'Unsupported institute {institute!r}.')

    # Apply defaults and iterate over options
    # NOTE: This silently skips dummy selections (e.g. area=None) that may be required
    # to prevent _parse_specs from merging e.g. average and non-average selections.
    # WARNING: Sometimes multi-index reductions can eliminate previously valid coords,
    # so critical to iterate one-by-one and validate selections each time.
    name = data.name
    attrs = attrs or {}
    attrs = attrs.copy()  # WARNING: critical
    defaults = {'period': 'ann', 'ensemble': 'flagship', 'experiment': 'picontrol'}
    versions = {'source': 'eraint', 'statistic': 'slope', 'region': 'globe'}
    if 'version' in data.coords:
        defaults.update({'experiment': 'abrupt4xco2', **versions})
    for key, value in defaults.items():
        kwargs.setdefault(key, value)
    for key, value in data.attrs.items():
        attrs.setdefault(key, value)
    order = list(ORDER_LOGICAL)
    sorter = lambda key: order.index(key) if key in order else len(order)
    for key in sorted(kwargs, key=sorter):
        value = kwargs[key]
        options = list(data.sizes)
        options.extend(name for idx in data.indexes.values() for name in idx.names)
        options.extend(('area', 'volume'))
        if value is None:
            continue
        if key not in options:
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
        try:
            data = data.climo.reduce(**{key: value})
        except Exception:
            raise RuntimeError(f'Failed to reduce data with {key}={value!r}.')
        if key not in data.coords and isinstance(value, str):
            data.coords[key] = value  # used for box and bar plot colors
        data = data.squeeze()
    data.name = name
    data.attrs.update(attrs)
    return data


def apply_method(
    *datas, method=None, std=None, pctile=None, invert=False, verbose=False
):
    """
    Reduce along the facets coordinate using an arbitrary method.

    Parameters
    ----------
    *datas : xarray.DataArray
        The data array(s).
    method : str, optional
        The method. ``dist``, ``dstd``, and ``dpctile`` retain the facets dimension
        before plotting. ``avg``, ``std``, and ``pctile`` reduce the facets dimension
        for a single input argument. ``corr``, ``diff``, ``proj``, and ``slope``
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
    args : tuple
        The output plotting arrays.
    method : str
        The resulting method used.
    kwargs : dict
        The plotting and `_infer_command` keyword arguments.
    """
    # Combine one array along facets dimension
    # NOTE: Here `pctile` is shared between inter-model percentile differences and
    # composites of a second array based on values in the first array.
    # NOTE: Currently proplot will automatically apply xarray tuple multi-index
    # coordinates to bar plot then error out so apply numpy array coords for now.
    defaults = {}
    datas = tuple(data.copy() for data in datas)
    ndims = tuple(data.ndim for data in datas)
    args = name = long = short = None
    if invert:
        datas = datas[::-1]
    if max(ndims) == 1:
        method = method or 'dist'  # only possibility
    elif len(datas) == 1:
        method = method or 'avg'
    else:
        method = method or 'rsq'
    if len(datas) == 1:
        data, = datas
        name = data.name
        kw = {'dim': 'facets', 'skipna': True}
        if method == 'avg' or method == 'med':
            key = 'mean' if method == 'avg' else 'median'
            std = 3.0 if std is True else std
            pctile = 90.0 if pctile is True else pctile
            if std:
                assert max(ndims) == 2
                defaults.update({key: True, 'shadestds': std})
            elif pctile:
                assert max(ndims) == 2
                defaults.update({key: True, 'shadepctiles': pctile})
            else:
                assert max(ndims) < 4
                data = getattr(data, key)(**kw)
        elif method == 'pctile':
            assert max(ndims) < 4
            pctile = 50.0 if pctile is None else pctile
            nums = 0.01 * pctile / 2, 1 - 0.01 * pctile / 2
            with xr.set_options(keep_attrs=True):  # note name is already kept
                data = data.quantile(nums[1], **kw) - data.quantile(nums[0], **kw)
            short = f'{data.short_name} percentile range'
            long = f'{data.long_name} percentile range'
        elif method == 'std':
            assert max(ndims) < 4
            std = 1.0 if std is None else std
            with xr.set_options(keep_attrs=True):  # note name is already kept
                data = std * data.std(**kw)
            short = f'{data.short_name} standard deviation'
            long = f'{data.long_name} standard deviation'
        elif method == 'dist':  # horizontal lines
            assert max(ndims) == 1
            data = data[~data.isnull()]
            args = (data,)
        else:
            raise ValueError(f'Invalid single-variable method {method}.')

    # Combine two arrays along facets dimension
    # NOTE: The idea for 'diff' reductions is to build the feedback-based composite
    # difference defined ``data[feedback > 100 - pctile] - data[feedback < pctile]``.
    elif len(datas) == 2:
        data0, data1 = datas
        name = f'{data0.name}-{data1.name}'
        if method == 'rsq':  # correlation coefficient
            cov, std0, std1 = _parts_covariance(*datas, both=True)
            data = (cov / (std0 * std1)) ** 2
            data = data.climo.to_units('percent').climo.dequantify()
            short = f'{data1.short_name} variance explained'
            long = f'{data0.long_name}-{data1.long_name} variance explained'
        elif method == 'corr':  # correlation coefficient
            cov, std0, std1 = _parts_covariance(*datas, both=True)
            data = cov / (std0 * std1)
            data = data.climo.to_units('dimensionless').climo.dequantify()
            short = f'{data1.short_name} correlation'
            long = f'{data0.long_name}-{data1.long_name} correlation coefficient'  # noqa: E501
        elif method == 'proj':  # projection onto x
            cov, std = _parts_covariance(*datas, both=False)
            data = cov / std
            data = data.climo.dequantify()
            short = f'{data1.short_name} projection'
            long = f'{data1.long_name} vs. {data0.long_name}'
        elif method == 'slope':  # regression coefficient
            cov, std = _parts_covariance(*datas, both=False)
            data = cov / std ** 2
            data = data.climo.dequantify()
            short = f'{data1.short_name} regression coefficient'
            long = f'{data1.long_name} vs. {data0.long_name} regression coefficient'
        elif method == 'diff':  # composite difference along first arrays
            data_lo, data_hi = _parts_composite(*datas)
            data = data_hi - data_lo
            data = data.climo.dequantify()
            short = f'{data1.short_name} composite difference'
            long = f'{data0.long_name}-composite {data1.long_name} difference'  # noqa: E501
        elif method == 'dist':  # scatter points
            assert max(ndims) == 1
            data0, data1 = data0[~data0.isnull()], data1[~data1.isnull()]
            data0, data1 = xr.align(data0, data1)  # intersection-style broadcast
            args = (data0, data1)
            name = None
        else:
            raise ValueError(f'Invalid double-variable method {method}')
    else:
        raise ValueError(f'Unexpected argument count {len(datas)}.')

    # Standardize the result and print information
    # NOTE: This modifies
    args = args or (data,)
    args = list(args)
    if name:
        args[-1].name = name
    if long:
        args[-1].attrs['long_name'] = long
    if short:
        args[-1].attrs['short_name'] = short
    keys = ('facets', 'time', 'plev', 'lat', 'lon')  # sorting order
    args = [arg.transpose(..., *(key for key in keys if key in arg.sizes)) for arg in args]  # noqa: E501
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


def get_data(dataset, *kws_dat, attrs=None):
    """
    Combine the data based on input reduce dictionaries.

    Parameters
    ----------
    dataset : xarray.Dataset
        The source dataset.
    *kws_dat : dict
        The reduction keyword dictionaries.
    attrs : dict, optional
        The attribute dictionaries.

    Returns
    -------
    args : tuple
        The output plotting arrays.
    method : str
        The resulting method used.
    kwargs : dict
        The plotting and `_infer_command` keyword arguments.
    """
    # Group added/subtracted reduce instructions into separate dictionaries
    # NOTE: Initial kw_red values are formatted as (('[+-]', value), ...) to
    # permit arbitrary combinations of names and indexers (see _parse_specs).
    alias_to_name = {
        alias: name for alias, (name, _) in FEEDBACK_TRANSLATIONS.items()
    }
    if 'control' in dataset.experiment:
        alias_to_name.update({'picontrol': 'control', 'abrupt4xco2': 'response'})
    else:
        alias_to_name.update({'control': 'picontrol', 'response': 'abrupt4xco2'})
    kws_method = []  # each item represents a method argument
    keys_method = ('method', 'std', 'pctile', 'invert')  # special method keywords
    for kw_dat in kws_dat:
        scale = 1
        kw_reduce = {}
        kw_method = {key: kw_dat.pop(key) for key in keys_method if key in kw_dat}
        for key, value in kw_dat.items():
            sels = ['+']
            parts = REGEX_SPLIT.split(value) if isinstance(value, str) else (value,)
            for i, part in enumerate(parts):
                if isinstance(part, str) and REGEX_FLOAT.match(part):  # e.g. 850-250hPa
                    part = float(part)
                if part in (None, '+', '-', '*', '/'):  # dummy coordinate or operator
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
            scale *= sum(sign == '+' for sign in signs)
            kw_reduce[key] = tuple(zip(signs, values))
        kws_reduce = []
        for values in itertools.product(*kw_reduce.values()):
            signs, values = zip(*values)
            sign = -1 if signs.count('-') % 2 else +1
            kw = dict(zip(kw_reduce, values))
            kw.update(kw_method)
            kws_reduce.append((sign, kw))
        kws_method.append((scale, kws_reduce))

    # Reduce along facets dimension and carry out operation
    # TODO: Add other possible reduction methods, e.g. covariance
    # or regressions instead of normalized correlation.
    # NOTE: Here 'skip' prevents subtracting or averaging identical selections
    # from identical selections (i.e. all-zero array generation).
    scales, kws_method = zip(*kws_method)
    if len(set(scales)) > 1:
        raise RuntimeError(f'Mixed reduction scalings {scales}.')
    all_membs = all_projs = True
    kws_method = list(kws_method)
    skip = None
    if len(kws_method) == 2 and len(kws_method[0]) == 1 and len(kws_method[1]) != 1:
        skip = 0
        kws_method[0] = kws_method[0] * len(kws_method[1])
    if len(kws_method) == 2 and len(kws_method[1]) == 1 and len(kws_method[0]) != 1:
        skip = 1
        kws_method[1] = kws_method[1] * len(kws_method[0])
    kws_persum = zip(*kws_method)
    kwargs = {}
    datas_persum = []  # each item part of a summation
    methods_persum = set()
    for kws_reduce in kws_persum:
        kw_method = {}
        keys = ('std', 'pctile', 'invert', 'method')
        datas = []
        signs, kws_reduce = zip(*kws_reduce)
        if skip is not None:  # ignore scalar indices
            signs = (signs[1 - skip],)
        if len(set(signs)) > 1:
            raise RuntimeError(f'Mixed reduction signs {signs}.')
        for kw in kws_reduce:  # two for e.g. 'corr', one for e.g. 'avg'
            kw_reduce = kw.copy()
            for key in tuple(kw_reduce):
                if key in keys:
                    kw_method.setdefault(key, kw_reduce.pop(key))
            institute = kw_reduce.get('institute', None)
            project = getattr(kw_reduce.get('project'), 'name', '')
            all_membs = all_membs and institute is None
            all_projs = all_projs and len(project) not in (5, 6)
            data = dataset[kw_reduce.pop('name')]
            data = apply_reduce(data, attrs=attrs, **kw_reduce)
            datas.append(data)
        datas, method, default = apply_method(*datas, **kw_method)
        for key, value in default.items():
            kwargs.setdefault(key, value)
        datas_persum.append((signs[0], datas))  # plotting command arguments
        methods_persum.add(method)
        if len(methods_persum) > 1:
            raise RuntimeError(f'Mixed reduction methods {methods_persum}.')

    # Combine arrays specified with reduction '+' and '-' keywords
    # NOTE: The default inferred prefixes associated with reductions are passed
    # here with 'attrs' so only need to possibly modify the suffix.
    # NOTE: The additions below are scaled as *averages* so e.g. project='cmip5+cmip6'
    # gives the average across cmip5 and cmip6 inter-model averages.
    # NOTE: The operations below use a dummy 'cmip' project before acting to support
    # specific case of e.g. project='cmip6-cmip5' with institute='avg'.
    def _replace_project(data, project=None):  # temporarily replace project
        project = project or 'CMIP'
        projects = data.project.values if hasattr(data, 'project') else np.array([])
        original = projects[0] if len(projects) > 0 else None
        if len(projects) > 1 and all(proj == original for proj in projects):
            facets = [(project, *key[1:]) for key in data.facets.values]
            facets = xr.DataArray(
                pd.MultiIndex.from_tuples(facets, names=data.indexes['facets'].names),
                attrs=data.facets.attrs,
                dims='facets',
                name='facets',
            )
            data = data.assign_coords(facets=facets)
        return data, original
    args = []
    signs, datas_persum = zip(*datas_persum)
    kwargs.update({'projects': all_projs, 'members': all_membs})
    for s, datas in enumerate(zip(*datas_persum)):
        if s == skip and method == 'dist':
            data = datas[0]
        else:
            datas, projs = zip(*map(_replace_project, datas))
            datas = xr.align(*datas)
            if 'facets' in datas[0].coords and not datas[0].facets.size:
                raise RuntimeError(
                    'Empty facets dimension. This is most likely due to an '
                    'operation across projects without an institute average.'
                )
            with xr.set_options(keep_attrs=True):
                data = sum(sign * sdata for sign, sdata in zip(signs, datas))
                data = data / scales[0]
            proj = 'CMIP6' if len(set(projs)) > 1 else projs[0]
            data, _ = _replace_project(data, proj)
            if any(sign == -1 for sign in signs):
                data.attrs['long_suffix'] = 'anomaly'
                data.attrs['short_suffix'] = 'anomaly'
        args.append(data)
    args = xr.align(*args)  # re-align after summation
    return args, method, kwargs
