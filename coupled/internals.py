#!/usr/bin/env python3
"""
Internal helper functions for figure templates.
"""
import climopy as climo  # noqa: F401
import numpy as np
import xarray as xr
from climopy import ureg, vreg  # noqa: F401
from icecream import ic  # noqa: F401

__all__ = [
    'apply_composite',
    'apply_variance',
    'apply_reduce',
    'apply_method',
    'specs_breakdown',
]

UNITS_LABELS = {
    'K': 'temperature',
    'hPa': 'pressure',
    'dam': 'surface height',
    'mm': 'liquid depth',
    'mm day^-1': 'accumulation',
    'm s^-1': 'wind speed',
    'Pa': 'wind stress',  # default tau units
    'g kg^-1': 'concentration',
    'W m^-2': 'flux',
    'PW': 'transport'
}


def apply_composite(data0, data1, pctile=None):  # noqa: E301
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


def apply_variance(data0, data1, both=True, skipna=True):
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
    skipna : bool, optional
        Whether to skip nan-values in the summation.
    """
    # NOTE: Currently masked arrays are used in climopy 'covar' and might also have
    # overhead from metadata stripping stuff and permuting. So go manual here.
    data0, data1 = xr.broadcast(data0, data1)
    data0 = data0.climo.quantify()
    data1 = data1.climo.quantify()
    with np.errstate(all='ignore'):
        mean0 = data0.mean(dim='facets', skipna=skipna)
        mean1 = data1.mean(dim='facets', skipna=skipna)
        anom0 = data0 - mean0
        anom1 = data1 - mean1
        covar = (anom0 * anom1).sum(dim='facets', skipna=skipna)
        std0 = (anom0 ** 2).sum(dim='facets', skipna=skipna)
        std0 = np.sqrt(std0)
        if both:
            std1 = (anom1 ** 2).sum(dim='facets', skipna=skipna)
            std1 = np.sqrt(std1)
    return (covar, std0, std1) if both else (covar, std0)


def apply_reduce(dataset, attrs=None, **kwargs):
    """
    Carry out arbitrary reduction of the given dataset variables.

    Parameters
    ----------
    dataset : xarray.Dataset
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
    # NOTE: Sometimes multi-index reductions can eliminate previously valid
    # coordinates, so iterate one-by-one and validate selections each time.
    # NOTE: This silently skips dummy selections (e.g. area=None) that may be
    # required to prevent _parse_bulk from merging variable specs that differ
    # only in that one contains a selection and the other doesn't (generally
    # when constraining local feedbacks vs. global feedbacks).
    defaults = {
        'period': 'ann',
        'experiment': 'picontrol',
        'ensemble': 'flagship',
        'source': 'eraint',
        'statistic': 'slope',
        'region': 'globe',
    }
    for key, value in defaults.items():
        kwargs.setdefault(key, value)
    name = kwargs.pop('name', None)
    if name is None:
        raise ValueError('Variable name missing from reduction specification.')
    data = dataset[name]
    if facets := kwargs.pop('facets', None):  # see _parse_projects
        data = data.sel(facets=list(filter(facets, data.facets.values)))
    attrs = attrs or {}
    for key, value in data.attrs.items():
        attrs.setdefault(key, value)
    for key, value in kwargs.items():
        if value is None:
            continue
        data = data.squeeze()
        options = [*data.sizes, 'area', 'volume']
        options.extend(name for idx in data.indexes.values() for name in idx.names)
        if key in options:
            data = data.climo.reduce(**{key: value})
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
        The user-declared reduction method. The methods ``dist``, ``dstd``, and
        ``dpctile`` retain the facets dimension before plotting. The methods
        ``avg``, ``std``, and ``pctile`` reduce the facets dimension for a single
        input argument. The methods ``corr``, ``diff``, ``proj``, and ``slope``.
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
    datas : tuple
        The data array(s). Two are returned for `hist2d` plots.
    """
    # Infer short names from the unit strings (standardized by open_climate) for use
    # with future colorbar and legend labels (long names are used for grid labels).
    # TODO: Update climopy and implement functionality with new cf variables.
    datas = tuple(data.copy() for data in datas)
    ndims = tuple(data.ndim for data in datas)
    for data in datas:
        for array in (data, *data.coords.values()):
            if 'units' not in array.attrs or 'long_name' not in array.attrs:
                continue
            if 'short_name' in array.attrs:
                continue
            long = array.long_name
            units = array.units
            if 'feedback' in long:
                short = 'feedback'
            elif 'forcing' in long:
                short = 'forcing'
            else:
                short = UNITS_LABELS.get(units, long)
            array.attrs['short_name'] = short

    # Combine one array along facets dimension
    # NOTE: Here `pctile` is shared between inter-model percentile differences and
    # composites of a second array based on values in the first array.
    # NOTE: Currently proplot will automatically apply xarray tuple multi-index
    # coordinates to bar plot then error out so apply numpy array coords for now.
    if invert:
        datas = datas[::-1]
    if max(ndims) == 1:
        method = 'dist'  # only possibility
    elif len(datas) == 1:
        method = method or 'avg'
    else:
        method = method or 'rsq'
    if len(datas) == 1:
        if method == 'dist':  # horizontal lines
            data, = datas
            data = data[~data.isnull()]
            data = (np.arange(data.size), data.sortby(data, ascending=False))
            assert max(ndims) == 1
        elif method == 'dstd':
            data, = datas
            kw = {'means': True, 'shadestds': 3.0 if std is None else 3.0}
            data = (data, kw)
            assert max(ndims) == 2
        elif method == 'dpctile':
            data, = datas
            kw = {'medians': True, 'shadepctiles': 90.0 if pctile is None else pctile}
            data = (data, kw)
            assert max(ndims) == 2
        elif method == 'avg':
            data = datas[0].mean('facets', skipna=True, keep_attrs=True)
            data.name = datas[0].name
            data.attrs['units'] = datas[0].units
        elif method == 'std':
            std = 1.0 if std is None else 1.0
            data = std * datas[0].std('facets', skipna=True)
            data.name = datas[0].name
            data.attrs['short_name'] = f'{datas[0].short_name} standard deviation'
            data.attrs['long_name'] = f'{datas[0].long_name} standard deviation'
            data.attrs['units'] = datas[0].units
        elif method == 'pctile':
            pctile = 25.0 if pctile is None else pctile
            lo_vals = np.nanpercentile(datas[0], pctile)
            hi_vals = np.nanpercentile(datas[0], 100 - pctile)
            lo_vals = hi_vals.mean('facets') - lo_vals.mean('facets')
            data = hi_vals - lo_vals
            data.attrs['short_name'] = f'{datas[0].short_name} percentile range'
            data.attrs['long_name'] = f'{datas[0].long_name} percentile range'
            data.attrs['units'] = datas[0].units
        else:
            raise ValueError(f'Invalid single-variable method {method}.')

    # Combine two arrays along facets dimension
    # NOTE: The idea for 'diff' reductions is to build the feedback-based composite
    # difference defined ``data[feedback > 100 - pctile] - data[feedback < pctile]``.
    elif len(datas) == 2:
        if method == 'dist':  # scatter points
            datas = xr.broadcast(*datas)
            mask = ~datas[0].isnull() & ~datas[1].isnull()
            data = (datas[0][mask], datas[1][mask])
            assert max(ndims) == 1
        elif method == 'rsq':  # correlation coefficient
            cov, std0, std1 = apply_variance(*datas, both=True)
            data = (cov / (std0 * std1)) ** 2
            data = data.climo.to_units('percent').climo.dequantify()
            short_name = 'variance explained'
            long_name = f'{datas[0].long_name}-{datas[1].long_name} variance explained'
            data.name = f'{datas[0].name}-{datas[1].name}'
            data.attrs['short_name'] = short_name
            data.attrs['long_name'] = long_name
        elif method == 'corr':  # correlation coefficient
            cov, std0, std1 = apply_variance(*datas, both=True)
            data = cov / (std0 * std1)
            data = data.climo.to_units('dimensionless').climo.dequantify()
            short_name = 'correlation coefficient'
            long_name = f'{datas[0].long_name}-{datas[1].long_name} correlation coefficient'  # noqa: E501
            data.name = f'{datas[0].name}-{datas[1].name}'
            data.attrs['short_name'] = short_name
            data.attrs['long_name'] = long_name
        elif method == 'proj':  # projection onto x
            cov, std = apply_variance(*datas, both=False)
            data = cov / std
            data = data.climo.dequantify()
            short_name = f'{datas[1].short_name} projection'
            long_name = f'{datas[1].long_name} vs. {datas[0].long_name}'
            data.name = f'{datas[0].name}-{datas[1].name}'
            data.attrs['short_name'] = short_name
            data.attrs['long_name'] = long_name
        elif method == 'slope':  # regression coefficient
            cov, std = apply_variance(*datas, both=False)
            data = cov / std ** 2
            data = data.climo.dequantify()
            short_name = f'{datas[1].short_name} regression'
            long_name = f'{datas[1].long_name} vs. {datas[0].long_name}'
            data.name = f'{datas[0].name}-{datas[1].name}'
            data.attrs['short_name'] = short_name
            data.attrs['long_name'] = long_name
        elif method == 'diff':  # composite difference along first arrays
            data_lo, data_hi = apply_composite(*datas)
            data = data_hi - data_lo
            data = data.climo.dequantify()
            short_name = f'{datas[1].short_name} difference'
            long_name = f'{datas[0].long_name}-composite {datas[1].long_name} difference'  # noqa: E501
            data.name = f'{datas[0].name}-{datas[1].name}'
            data.attrs['short_name'] = short_name
            data.attrs['long_name'] = long_name
        else:
            raise ValueError(f'Invalid double-variable method {method}')
    else:
        raise ValueError(f'Unexpected argument count {len(datas)}.')

    # Print information and standardize result. Shows both the
    # available models and the intersection once they are combined.
    # print('input!', data, 'result!', *datas, sep='\n')
    order = ('facets', 'time', 'plev', 'lat', 'lon')
    result = tuple(
        part.transpose(..., *(key for key in order if key in part.sizes))
        if isinstance(part, xr.DataArray) else part
        for part in (data if isinstance(data, tuple) else (data,))
    )
    if verbose:
        masks = [
            (~data.isnull()).any(data.sizes.keys() - {'facets'})
            for data in datas if isinstance(data, xr.DataArray)
        ]
        valid = invalid = ''
        if len(masks) == 2:
            mask = masks[0] & masks[1]
            valid, invalid = f' ({np.sum(mask).item()})', f' ({np.sum(~mask).item()})'
        for mask, data in zip(masks, datas[len(datas) - len(masks):]):
            min_, max_, mean = data.min().item(), data.mean().item(), data.max().item()
            print(format(f'{data.name} {method}:', ' <20s'), end=' ')
            print(f'min {min_: <+7.2f} max {max_: <+7.2f} mean {mean: <+7.2f}', end=' ')
            print(f'valid {np.sum(mask).item()}{valid}', end=' ')
            print(f'invalid {np.sum(~mask).item()}{invalid}', end='\n')
    return result, method


def specs_breakdown(
    breakdown='net',
    feedbacks=True,
    forcing=False,
    adjust=False,
    sensitivity=False,
    maxcols=4,
):
    """
    Return a list of feedback parameters and suggested `gridskip` value.

    Parameters
    ----------
    breakdown : str
        The breakdown format.
    feedbacks, forcing, adjust, sensitivity : bool, optional
        Whether to include various components.
    maxcols : int, default: 4
        The maximum number of columns (influences order of the specs).
    """
    # Interpret keyword arguments from suffixes
    # NOTE: Idea is to automatically remove feedbacks, filter out all-none
    # rows and columns at the end, and then infer the 'gridskip' from them.
    if '_all' in breakdown:
        breakdown, *_ = breakdown.split('_all')
        sensitivity = forcing = True
    if '_lam' in breakdown:
        breakdown, *_ = breakdown.split('_lam')
        feedbacks = True
    if '_erf' in breakdown:
        breakdown, *_ = breakdown.split('_erf')
        forcing, adjust = True, False  # effective forcing *without* rapid adjustments
    if '_adj' in breakdown:
        breakdown, *_ = breakdown.split('_erf')
        adjust, forcing = True, False  # effective forcing *and* rapid adjustments
    if '_ecs' in breakdown:
        breakdown, *_ = breakdown.split('_ecs')
        sensitivity = True
    if not feedbacks and not forcing and not sensitivity:
        raise RuntimeError

    # Three variable breakdowns
    # NOTE: Options include 'wav', 'atm', 'alb', 'res', 'all', 'atm_wav', 'alb_wav'
    # with the 'wav' suffixes including longwave and shortwave cloud components
    # instead of a total cloud feedback. Strings denote successively adding atmospheric,
    # albedo, residual, and remaining temperature/humidity feedbacks with options.
    def _get_specs(cols):
        specs = np.array([[None] * cols] * 25)
        iflat = specs.flat
        return specs, iflat
    if breakdown in ('atm', 'wav'):  # shortwave longwave
        if breakdown == 'atm':
            lams = ['net', 'cld', 'atm']
            erfs = ['erf', 'cl_rfnt_erf', 'atm_rfnt_erf']
        else:
            lams = ['net', 'sw', 'lw']
            erfs = ['erf', 'rsnt_erf', 'rlnt_erf']
        if maxcols == 2:
            specs, iflat = _get_specs(maxcols)
            if sensitivity:
                iflat[0] = 'ecs'
            if feedbacks and adjust:
                specs[1:4, 0] = lams
                specs[1:4, 1] = erfs
            elif feedbacks and forcing:
                specs[1, :] = lams[:1] + erfs[:1]
                specs[2, :] = lams[1:]
            elif feedbacks:
                iflat[1] = lams[0]
                specs[1, :] = lams[1:]
            elif adjust:
                iflat[1] = erfs[0]
                specs[1, :] = erfs[1:]
        else:
            offset = 0
            maxcols = 1 if maxcols == 1 else 3
            specs, iflat = _get_specs(maxcols)
            if sensitivity:
                iflat[offset] = 'ecs'
            if forcing:  # adjust 'erf' handled below
                iflat[offset + 1] = 'erf'
            if feedbacks:
                idx = 2 * maxcols
                iflat[idx:idx + 3] = lams
            if adjust:
                idx = 3 * maxcols
                iflat[idx:idx + 3] = erfs

    # Four variable breakdowns
    elif breakdown in ('alb', 'atm_wav'):
        if breakdown == 'alb':
            lams = ['net', 'cld', 'alb', 'atm']
            erfs = ['erf', 'cl_rfnt_erf', 'alb_rfnt_erf', 'atm_rfnt_erf']
        else:
            lams = ['net', 'swcld', 'lwcld', 'atm']
            erfs = ['erf', 'cl_rsnt_erf', 'cl_rlnt_erf', 'atm_rfnt_erf']
        if maxcols == 2:
            specs, iflat = _get_specs(maxcols)
            if sensitivity:
                iflat[0] = 'ecs'
            if feedbacks and adjust:
                specs[1:5, 0] = lams
                specs[1:5, 1] = erfs
            elif feedbacks:
                iflat[1] = 'erf' if forcing else None
                specs[1, :] = lams[::3]
                specs[2, :] = lams[1:3]
            elif adjust:
                specs[1, :] = erfs[::3]
                specs[2, :] = erfs[1:3]
        else:
            offset = 0 if maxcols == 1 else 1
            maxcols = 1 if maxcols == 1 else 3
            specs, iflat = _get_specs(maxcols)
            if sensitivity:
                iflat[-offset % 3] = 'ecs'
            if forcing or adjust:
                iflat[2 - offset] = 'erf'
            if feedbacks:
                idx = 3
                iflat[1 - offset] = lams[0]
                iflat[idx:idx + 3] = lams[1:]
            if adjust:
                idx = 3 + 3
                iflat[idx:idx + 3] = erfs[1:]

    # Five variable breakdowns
    elif breakdown in ('res', 'alb_wav'):
        if breakdown == 'res':
            lams = ['net', 'cld', 'alb', 'resid', 'atm']
            erfs = ['erf', 'cl_rfnt_erf', 'alb_rfnt_erf', 'atm_rfnt_erf', 'resid_rfnt_erf']  # noqa: E501
        else:
            lams = ['net', 'swcld', 'lwcld', 'alb', 'atm']
            erfs = ['erf', 'cl_rsnt_erf', 'cl_rlnt_erf', 'alb_rfnt_erf', 'atm_rfnt_erf']
        if maxcols == 2:
            specs, iflat = _get_specs(maxcols)
            if sensitivity:
                iflat[0] = 'ecs'
            if feedbacks and adjust:
                specs[1:5, 0] = lams
                specs[1:5, 1] = erfs  # noqa: E501
            elif feedbacks and forcing:
                specs[1, :] = lams[:1] + erfs[:1]
                specs[2, :] = lams[1:3]
                specs[3, :] = lams[3:5]
            elif feedbacks:
                specs[0, 1] = lams[0]
                specs[1, :] = lams[1:3]
                specs[2, :] = lams[3:5]
            elif adjust:
                specs[0, 1] = erfs[0]
                specs[1, :] = erfs[1:3]
                specs[2, :] = erfs[3:5]
        else:
            offset = 0 if maxcols == 1 else 1
            maxcols = 1 if maxcols == 1 else 4  # disallow 3 columns
            specs, iflat = _get_specs(maxcols)
            if sensitivity:
                iflat[-offset % 3] = 'ecs'  # either before or after net and erf
            if forcing or adjust:
                iflat[2 - offset] = erfs[0]
            if feedbacks:
                idx = 4  # could leave empty single-column row
                iflat[1 - offset] = lams[0]
                iflat[idx:idx + 4] = lams[1:]
            if adjust:
                idx = 4 + 4
                iflat[idx:idx + 4] = erfs[1:]

    # Full breakdown
    elif breakdown == 'all':
        lams = ['net', 'atm', 'cld', 'swcld', 'lwcld', 'alb', 'resid']
        erfs = ['erf', 'cl_rfnt_erf', 'cl_rsnt_erf', 'cl_rlnt_erf', 'atm_rfnt_erf', 'alb_rfnt_erf', 'resid_rfnt_erf']  # noqa: E501
        hums = ['wv', 'rh', 'lr', 'lr*', 'pl', 'pl*']
        if maxcols == 2:
            specs, iflat = _get_specs(maxcols)
            if sensitivity:
                iflat[0] = 'ecs'
            if feedbacks and adjust:
                specs[1:8, 0] = lams
                specs[1:8, 1] = erfs  # noqa: E501
                iflat[16:22] = hums
            elif feedbacks:
                iflat[0] = lams[0]
                iflat[1] = 'erf' if forcing else None
                specs[2, :] = lams[1:3]
                specs[3, :] = lams[3:5]
            elif adjust:
                iflat[0] = erfs[0]
                specs[1, :] = erfs[1:3]
                specs[2, :] = erfs[3:5]
        else:
            offset = 0 if maxcols == 1 else 1
            maxcols = 1 if maxcols == 1 else 4  # disallow 3 columns
            specs, iflat = _get_specs(maxcols)
            if sensitivity:
                iflat[-offset % 3] = 'ecs'  # either before or after net and erf
            if forcing or adjust:
                iflat[2 - offset] - erfs[0]
            if feedbacks:
                idx = 4  # could leave empty single-column row
                iflat[1 - offset] = lams[0]
                iflat[idx:idx + 4] = lams[1:5]
                if maxcols == 1:
                    idx = 4 + 4
                    iflat[idx:idx + 8] = lams[5:7] + hums[:]
                else:
                    idx = 4 + 4
                    iflat[idx:idx + 4] = lams[5:6] + hums[0::2]
                    idx = 4 + 2 * 4
                    iflat[idx:idx + 4] = lams[6:7] + hums[1::2]
            if adjust:
                idx = 4 + 3 * 4
                iflat[idx:idx + 6] = erfs[1:7]
    else:
        specs = [breakdown]
        gridskip = None

    # Remove all-none segments and determine gridskip
    idx, = np.where(np.any(specs != None, axis=0))  # noqa: E711
    specs = np.take(specs, idx, axis=1)
    idx, = np.where(np.any(specs != None, axis=1))  # noqa: E711
    specs = np.take(specs, idx, axis=0)
    idxs = np.where(specs == None)  # noqa: E711
    gridskip = np.ravel_multi_index(idxs, specs.shape)
    specs = specs.ravel().tolist()
    specs = [spec for spec in specs if spec is not None]
    return specs, maxcols, gridskip
