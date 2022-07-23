#!/usr/bin/env python3
"""
Utilities for plotting coupled model output.
"""
import itertools
import re
import pandas as pd
from pathlib import Path

import numpy as np
import xarray as xr

import climopy as climo
import proplot as pplt
from climopy import ureg, vreg  # noqa: F401
from cmip_data import output

# Global constants
FEEDBACK_ALIASES = {
    key: name
    for key, (name, _) in output.FEEDBACK_DEFINITIONS.items()
}
REDUCE_DEFAULTS = {  # default feedback selections
    'period': 'ann',
    'ensemble': 'flagship',
    'project': 'all',
    'author': 'davis',
    'series': 'response',
    'statistic': 'regression',
    'numerator': 'globe',
    'denominator': 'globe',
}
REDUCE_DESCRIPTIONS = {  # default is title-case of input
    '+': 'plus',
    '-': 'minus',
    '*': 'times',
    '/': 'over',
    'ann': 'Annual',
    'djf': 'DJF',
    'mam': 'MAM',
    'jja': 'JJA',
    'son': 'SON',
    'avg': 'Average',
    'int': 'Integral',
    'absmin': 'Minimum',
    'absmax': 'Maximum',
    'globe': 'Global',
    'point': 'Local',
    'latitude': 'Zonal',
    'hemisphere': 'Hemisphere',
}


def _parse_indexers(dataset, **indexers):
    """
    Standardize the indexers and translate into a title suitable for
    figures. Include special behavior for project indexers.

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset. If ``project`` is passed then this should
        contain a ``'source'`` `~pandas.MultiIndex` coordinate.
    **indexers
        The indexers. Numeric are assumed to be the units of the array and
        string indexers can have arithmetic operators ``+*/-`` to perform
        operations between multiple selections (e.g. seasonal differences
        relative to annual average or one project relative to another).

    Note
    ----
    Here `project` is treated specially. Values can optionally start with ``'cmip'``
    and must end with integers indicating the source values. No integer indicates all
    CMIP5 and CMIP6 models, ``5`` (``6``) indicates just CMIP5 (CMIP6) models, ``56``
    (``65``) indicate CMIP5 (CMIP6) models filtered to those from the same institutions
    as CMIP6 (CMIP5), and ``55`` (``66``) indicates models from institutions found only
    in CMIP5 (CMIP6). A common combination might be ``5``, ``65``, ``66``. As with
    other indexers, can use arithmetic operators for combinations (e.g. ``'6-5'``).
    """
    # NOTE: Integers are added to titles later to indicate the number of values
    # without NaNs. May be fewer when feedback components are unavailable.
    # TODO: Add 'add_sel', 'sub_sel', etc. methods to climopy and similar
    # 'add_<dim>', 'sub_<dim>' methods to reduce() on macbook. For now use
    # arithmetic indicators in strings for this sort of selection.
    operators = set()
    titles, values = {}, {}
    for dim, coord in indexers.items():
        tits, vals = [], []
        iter_ = re.split('([+*/-])', coord) if isinstance(coord, str) else (coord,)
        if len(iter_) not in (1, 3):  # maximum one operator
            raise ValueError(f'Unexpected coordinate format {coord!r}.')
        for i, value in enumerate(iter_):
            if dim == 'project' and value not in '+*/-':
                regex = re.compile(r'\A([a-zA-Z_]+).*\Z')
                _, num = value.lower().split('cmip')
                if not num:
                    title = 'CMIP models'
                    func = lambda key: True  # noqa: U100
                elif num in ('5', '6'):
                    title = f'CMIP{num} models'
                    func = lambda key: num == key[0][-1]
                elif num in ('65', '66', '56', '66'):
                    idx = len(set(num)) - 1  # zero if only one unique integer
                    num1, num2 = num[0], '65'[idx]
                    title = ('Other', 'Matched')[idx] + f' CMIP{num1} models'
                    func = lambda key: idx == any(
                        re.match(inst, other[1], re.IGNORECASE)
                        or inst[:3].lower() == other[1][:3].lower() == 'inm'
                        for other in dataset.source.values  # iterate over keys
                        if key[0][-1] == num2 and (inst := regex.sub(r'\1', key[1]))
                    )
                else:
                    raise ValueError(f'Invalid project {num!r}.')
                key = 'source'
                value = list(filter(func, dataset.source.values))
            elif isinstance(value, str):
                title = REDUCE_DESCRIPTIONS.get(value, value.title())
                if value in '+*/-':
                    operators.add(value)
                    if len(operators) > 1:
                        raise ValueError(f'Conflicting selection {operators=}.')
            else:
                coords = dataset[dim]
                unit = coords.climo.units
                if not isinstance(value, ureg.Quantity):
                    value = ureg.Quantity(value, unit)
                value = value.to(unit)
                title = f'${value:~L}$'
            tits.append(title)
            if value not in operators:
                vals.append(value)
        titles[key] = tits
        values[key] = vals
    if 'numerator' in titles and 'denominator' in titles:
        numers, denoms = titles.pop('numerator'), titles.pop('denominator')
        if len(numers) == 1:
            numers = numers * len(denoms)
        if len(denoms) == 1:
            denoms = denoms * len(numers)
        if len(numers) != len(denoms):
            raise ValueError(f'Unexpected {numers=} and {denoms=}.')
        regions = tuple(
            numer if i == 1 else f'{numer} vs. {denom}'
            for i, (numer, denom) in enumerate(zip(numers, denoms))
        )
        titles = {'region': regions, **titles}
    operator = operators.pop() if operators else None
    titles = {key: ' '.join(tits) for tits in titles}
    values = {key: vals[0] if len(vals) == 1 else tuple(vals) for key, vals in values.items()}  # noqa: E501
    return titles, operator, values


def _parse_specs(dataset, inspecs, **kwargs):
    """
    Parse variable and project specifications and auto-determine row and column
    labels based on the unique names and/or keywords in the spec lists.

    Parameters
    ----------
    dataset : xarray.Dataset
        The source dataset.
    inspecs : list of name, tuple, dict, or list thereof
        The variable specification(s) per subplot slot.
    **kwargs
        Additional options shared across all specs.
    """
    # TODO: Migrate and merge the spec parser and the row/col scheme
    # into the 'idealized' package. And perhaps rename that package.
    # NOTE: Here we delay reducing along keywords since we want to
    # share across different rows and column.
    outspecs = []
    kw_global = {}  # apply to entire spec list
    if not isinstance(inspecs, list):
        inspecs = [inspecs]
    if isinstance(inspecs[-1], dict) and not all(isinstance(_, dict) for _ in inspecs[:-1]):  # noqa: E501
        *inspecs, kw_global = inspecs
    for ispecs in inspecs:
        ospecs = []
        kw_local = {}  # apply to this sub-list
        if not isinstance(ispecs, list):
            ispecs = [ispecs]
        if isinstance(ispecs[-1], dict) and not all(isinstance(_, dict) for _ in ispecs[:-1]):  # noqa: E501
            *ispecs, kw_local = ispecs
        for spec in ispecs:
            if spec is None:
                name, kw = None, {}
            elif isinstance(spec, str):
                name, kw = spec, {}
            elif isinstance(spec, dict):
                name, kw = None, spec
            else:  # 2-tuple required
                name, kw = spec
            kw = {**kwargs, **kw_global, **kw_local, **kw}  # add shared keywords
            idxs = [idx for idx in dataset.indexes.values() if isinstance(idx, pd.MultiIndex)]  # noqa: E501
            names, kw_red, kw_fmt, kw_plt, kw_bar, kw_leg = [], {}, {}, {}, {}, {}
            fmt_prefixes = ('x', 'y', 'lon', 'lat', 'abc', 'title')
            bar_prefixes = ('extend', 'tick', 'locator', 'formatter', 'minor')
            leg_prefixes = ('ncol', 'order', 'frame', 'handle', 'border', 'column')
            if name:  # TODO: climopy get() add methods instead?
                names = [FEEDBACK_ALIASES.get(n, n) for n in name.split('+')]
            for key, value in kw.items():
                if key in dataset.sizes or any(key in idx.names for idx in idxs):
                    kw_red[key] = value  # includes 'project'
                elif any(key.startswith(prefix) for prefix in fmt_prefixes):
                    kw_fmt[key] = value
                elif any(key.startswith(prefix) for prefix in bar_prefixes):
                    kw_bar[key] = value
                elif any(key.startswith(prefix) for prefix in leg_prefixes):
                    kw_leg[key] = value
                else:  # arbitrary plotting keywords
                    kw_plt[key] = value
            titles, operator, kw_red = _parse_indexers(dataset, **kw_red)
            spec = (names, operator, titles, kw_red, kw_fmt, kw_plt, kw_bar, kw_leg)
            ospecs.append(spec)
        outspecs.append(ospecs)
    labels = []
    names, operators, titles, *kws = zip(*(ospecs[0] for ospecs in outspecs))
    dims = tuple(dim for tits in titles for dim in tits)
    for dim in dims:
        tits = tuple(tits.get(dim, None) for tits in titles)
        if len(set(tits)) == len(tits):  # all unique entries
            labels.append(tits)
    labels = ['\n'.join(labs) for labs in zip(*labels)]
    return dict(zip(labels, zip(names, operators, *kws)))


def plot_bulk(
    dataset,
    rowspecs,
    colspecs,
    method1='avg',
    method2='corr',
    hcolorbar='right',
    vcolorbar='bottom',
    dcolorbar='bottom',
    hlegend='bottom',
    vlegend='bottom',
    dlegend='bottom',
    title=None,
    refwidth=None,
    maxcols=5,
    absmax=0.5,
    pctile=33.3,
    proj='eqearth',
    save=True,
    **kwargs
):
    """
    Plot any combination of variables across rows and columns.

    Parameters
    ----------
    dataset : xarray.Dataset
        A dataset generated by `open_bulk`.
    rowspecs, colspecs : list of 2-tuple
        Tuples containing the ``(variable, kwargs)`` passed to ``ClimoAccessor.get``
        used to generate plotting data in rows and columns. See below for details.
    maxcols : int, optional
        The maximum number of columns. Used only if one of the row or variable
        specs is scalar (in which case there are no row or column labels).
    method1 : {'avg', 'std'}, optional
        The method for reducing the source dimension on individual non-scalar
        variables. The default is an average.
    method2 : {'corr', 'diff'}, optional
        The method for reducing the source dimension on paired non-scalar
        variables. The default is an average.
    hcolorbar, hlegend : {'right', 'left', 'bottom', 'top'}
        The location for colorbars or legends annotating horizontal rows.
        Automatically placed along the relevant axes.
    vcolorbar, vlegend : {'bottom', 'top', 'right', 'left'}
        The location for colorbars or legends annotating vertical columns,
        Automatically placed along the relevant axes.
    dcolorbar : {'bottom', 'right', 'top', 'left'}
        The default location for colorbars or legends annotating a mix of
        axes. Placed with the figure colorbar or legend method.
    pctile : float, optional
        The percentiles to use for ``'diff'`` composite differences.
    proj : str, optional
        The cartopy projection for longitude-latitude type plots.
    save : bool, optional
        Whether to save the results.
    **kwargs
        Also parsed with `rowspecs` and `colspecs`.

    Notes
    -----
    If `variable` is defined for both a row and column entry then an inter-model
    correlation between the two variables is taken. If it is not defined for one entry
    then an inter-model average is taken of the defined variable (must be defined
    for at least one). Use nested lists to show more than one item in the same axes.
    The keyword arguments for each row and column slot will be combined and filtered
    to valid indexers for the data. They can also contain a `method` keyword.

    The data resulting from each ``ClimoAccessor.get`` operation must be less
    than 3D. 2D data will be plotted with `pcolor`, then darker contours, then
    lighter contours; 1D data will be plotted with `line`, then on an alternate
    axes (so far just two axes are allowed); and 0D data will omit the average
    or correlation step and plot each model with `hist2d` (if both variables
    are defined) or `hist` (if only one model is defined).
    """
    # Initital stuff and figure out geometry
    # NOTE: The default behavior is always to draw a map... and users
    # are required to select either lon='avg' or plev=sel for 3D data.
    rowspecs = _parse_specs(dataset, rowspecs, **kwargs)
    colspecs = _parse_specs(dataset, colspecs, **kwargs)
    nrows, ncols = len(rowspecs), len(colspecs)
    if len(rowspecs) == 1 or len(colspecs) == 1:
        naxes = max(len(rowspecs), len(colspecs))
        ncols = min(naxes, maxcols)
        nrows = 1 + (ncols - 1) // naxes

    # Iterate over axes and plots
    # NOTE: Critical to disable 'grouping' so that e.g. colorbars or legends that
    # extend into other panel slots are not considered in the tight layout algorithm.
    gs = pplt.GridSpec(nrows, ncols)
    fig = pplt.figure(refwidth=refwidth, hgroup=False, wgroup=False)
    commands = {}
    for (i, rspecs), (j, cspecs) in itertools.product(
        enumerate(rowspecs.values()), enumerate(colspecs.values())
    ):
        # Get the variables
        ax = None
        ndims = set()
        contours = ('gray8', 'gray3')  # contour colors
        for k, ((rnames, rsource, *rkws), (cnames, csource, *ckws)) in enumerate(
            itertools.zip_longest(rspecs, cspecs)
        ):
            # Get the data arrays
            # NOTE: Assumption for now is that 'feedback' variables (i.e. the thing
            # we are trying to predict) are on rows and 'climate' variables on
            # columns. Use this fact when naming correlations and getting composites.
            datas = []
            source = sorted(set(rsource) & set(csource))
            kw_red = {**REDUCE_DEFAULTS, **rkws[0], **ckws[0]}
            kw_fmt = {**rkws[1], **ckws[1]}
            kw_plt = {**rkws[2], **ckws[2]}
            kw_bar = {**rkws[3], **ckws[3]}
            kw_leg = {**rkws[4], **ckws[4]}
            m1 = m2 = kw_plt.pop('method', None)  # input method
            for names in (rnames, cnames):
                if not names:
                    continue
                with xr.set_options(keep_attrs=True):  # e.g. pl+lr+hus
                    data = sum(dataset.climo.get(n, quantify=False) for n in names)
                idxs = [
                    idx for idx in data.indexes.values()
                    if isinstance(idx, pd.MultiIndex)
                ]
                reduce = {
                    key: value for key, value in kw_red.items()
                    if key in data.sizes or any(key in idx.names for idx in idxs)
                }
                if reduce:
                    data = data.climo.reduce(**reduce)
                data.name = '+'.join(names)
                mask = (~data.isnull()).any(data.sizes.keys() - {'source'})
                print(f'Variable: {data.name!r}.')
                print('Available models:', *data.model[mask].values.flat)
                print('Missing models:', *data.model[~mask].values.flat)
                datas.append(data.sel(source=source))
            if not datas:
                continue

            # Reduce along the source dimension
            # TODO: Add other possible reduction methods, e.g. covariance
            # or average addition or subtraction of variables.
            if len(datas) == 1:
                method = m1 or method1
                if datas[0].ndim == 1:  # 1D histogram with scatter points
                    data = (datas,)
                elif method == 'avg':
                    data = datas[0].mean('source', skipna=True, keep_attrs=True)
                    data.name = datas[0].name
                    data.attrs['long_name'] = f'{data.long_name} multi-model mean'
                elif method == 'std':
                    data = datas[0].std(dim='model', skipna=True, keep_attrs=True)
                    data.name = datas[0].name
                    data.attrs['long_name'] = f'{data.long_name} inter-model stdev'
                else:
                    raise ValueError(f'Invalid single-variable method {method}.')
            else:
                method = m2 or method2
                if datas[0].ndim == 1:  # 2D histogram with scatter points
                    data = (datas[0], datas[1])
                elif method == 'corr':  # correlation coefficient
                    da0, da1 = xr.broadcast(*datas)
                    _, data = climo.corr(*datas, dim='source')  # updates units
                    data = data.isel(lag=0)
                    long_name = f'{datas[0].long_name}-{datas[1].long_name} correlation'
                    data.name = 'corr'
                    data.attrs['long_name'] = long_name
                elif method == 'diff':  # composite difference (feedbacks on rows)
                    lo_comp = np.nanpercentile(datas[0], pctile)
                    hi_comp = np.nanpercentile(datas[0], 100 - pctile)
                    lo_mask, = np.where(datas[0] <= lo_comp)
                    hi_mask, = np.where(datas[0] >= hi_comp)
                    hi_data = datas[1].isel(source=hi_mask).mean('source', keep_attrs=True)  # noqa: E501
                    lo_data = datas[1].isel(source=lo_mask).mean('source', keep_attrs=True)  # noqa: E501
                    with xr.set_options(keep_attrs=True):  # keep units
                        data = hi_data - lo_data
                    long_name = f'{datas[0].long_name}-composite {datas[1].long_name} difference'  # noqa: E501
                    data.name = datas[1].name
                    data.attrs['long_name'] = long_name
                else:
                    raise ValueError(f'Invalid double-variable method {method}')

            # Queue the command
            # TODO: Should support only adding axes if the x or y units of the
            # data arrays do not match previous units... consider e.g. using
            # built-in matplotlib-pint unit conversions.
            dims = data.sizes.keys() - {'source'}
            projection = proj if dims == {'lon', 'lat'} else 'cartesian'
            if ax is None:
                ax = iax = fig.add_subplot(gs[i, j], projection=projection)
            elif hasattr(ax, 'alty') != (projection == 'cartesian'):
                raise ValueError(f'Conflicting projection types for dimensions {dims}.')
            ndims.add(ndim := len(data.sizes.keys() - {'source'}))
            if len(ndims) > 1:
                raise ValueError(f'Conflicting plot types for sizes {ndims}.')
            if ndim == 0:
                cmd = 'hist2d' if len(datas) == 2 else 'hist'
            elif ndim == 1:
                cmd, iax = 'line', ax if k == 0 else ax.alty()
            elif ndim == 2:
                cmd = ('pcolormesh', 'contour')[min(k, 1)]
            if cmd == 'contours':
                kw_plt.setdefault('color', contours[k - 1])
            key = (data.name, method, cmd)
            args = data if isinstance(data, tuple) else (data,)
            cmds = commands.setdefault(key, None)
            cmds.append((iax, args, kw_plt, kw_bar, kw_leg))
            min_, max_, mean = data.min().item(), data.mean().item(), data.max().item()
            print(f'{method.title()} range:', end=' ')
            print(f'min {min_:.02f} max {max_:.02f} mean {mean:.02f}')
            iax.format(**kw_fmt)

    # Carry out the plotting commands
    # NOTE: Axes are always added top-to-bottom and left-to-right so leverage
    # this fact below when selecting axes for legends and colorbars.
    for (name, method, cmd), parts in commands.items():
        # Call plotting commands
        axs = tuple(part[0] for part in parts)
        kws_plt, kws_bar, kws_leg = zip(*parts[2:])
        kw_plt = {key: val for part in parts for key, val in part[3].items()}
        kw_bar = {key: val for part in parts for key, val in part[3].items()}
        kw_leg = {key: val for part in parts for key, val in part[4].items()}
        if cmd in ('pcolormesh', 'contour', 'hist2d'):
            guide = 'colorbar'
            hori, vert, default, kw_guide = hcolorbar, vcolorbar, dcolorbar, kw_bar
            args = [arg for _, args, *_ in parts for arg in args]
            x, y = (args[0].coords[dim] for dim in args[0].dims)  # restrict limits
            norm, *_ = axs[0]._parse_level_vals(x, y, *args, **kw_plt)
#         else:
#  kwargs = commands[0][2].copy()  # identical for all experiments
#  if all(arg.shape == args[0].shape for arg in args):
#      x, y = (args[0].coords[dim] for dim in args[0].dims)
#  else:
#      x = y = None  # not restricted to in-bounds
#  if hasattr(ax, '_parse_level_lim'):
#      vmin, vmax, _ = ax._parse_level_lim(x, y, *args)
#  else:
#      vmin, vmax, _ = ax._parse_vlim(x, y, *args)
#         else:
#             guide = 'legend'
#             hori, vert, default, kw_guide = hlegend, vlegend, dlegend, kw_leg
#         for ax in axs:

        # Generate colorbars and legends
        rows = set(ax._range_subplotspec('x') for ax in axs)
        rows = rows.pop() if len(rows) == 1 else None
        cols = set(ax._range_subplotspec('y') for ax in axs)
        cols = cols.pop() if len(cols) == 1 else None
        if not (rows is None) ^ (cols is None):
            obj = fig
            loc = default
        elif rows is None:  # single column
            if loc[0] in 'tb':
                obj = axs[0] if loc[0] == 't' else axs[-1]
            else:
                obj = axs[len(axs) // 2]  # TODO: support even-numbered axes
            loc = vert
        else:
            if loc[0] in 'lr':
                obj = axs[0] if loc[0] == 'l' else axs[-1]
            else:
                obj = axs[len(axs) // 2]  # TODO: support even-numbered axes
            loc = hori
        getattr(obj, guide)(h, queue=True, **kw_guide)

    # Format the axes and optionally save
    kwargs.setdefault('xscale', 'sine')
    kwargs.update(
        suptitle=title,
        rowlabels=tuple(rowspecs),
        collabels=tuple(colspecs),
    )
    fig.format(**kwargs)
    return fig, fig.subplotgrid
    # if save:
    #     autoname = lambda *s: '_'.join('-'.join(sorted(set(_))).strip('-') for _ in s) + '.pdf'  # noqa: E501
    #     base = Path(__file__).parent.parent / 'figures'
    #     proj = (part for arg in args for part in re.findall(r'\w*', arg['project']) if 'project' in arg)  # noqa: E501
    #     src = (arg['src'] for arg in args if 'src' in arg)
    #     term = (arg['term'] for arg in args if 'term' in arg)
    #     var = (arg['var'] for arg in args if 'var' in arg)
    #     parts = [(mode,), proj, src, term, var]
    #     parts.extend(((dims,),) if dims else ())
    #     filename = autoname(*parts)
    #     print(f'Saving figure {filename!r}...')
    #     fig.save(base / filename)


def plot_drift(
    *specs,
    pointwise=False,
    relative=None,
    refwidth=None,
    title=None,
    proj='eqearth',
    path='~/data/cmip-constants',
    save=True,
):
    """
    Plot summaries of the drift corrections, both in absolute terms and relative
    to the offset (similar to a normalization with respect to climate).

    Parameters
    ----------
    *variables : str or tuple
        The variables to plot. Can be strings or 2-tuples where the
        second item in the tuple is a dictionary of indexers.
    pointwise : bool, optional
        If ``True`` the pointwise inter-model min, max, and mean are shown. If ``False``
        the spatial min, max, and mean for each model are shown with scatter plots.
    relative : {None, True, False}
        Whether to standardize the trends by offsets. Default is to show both
        in separate rows or columns.
    refwidth : float, optional
        The reference axes width. Default depends on `pointwise`.
    title : str, optional
        The figure title. Default is no title.
    proj : str, optional
        The cartopy projection if `pointwise` is ``True``.
    path : str, optional
        The path to search.
    save : bool, optional
        Whether to save the result.
    """
    # Initial stuff
    print(f'Creating {title} figure...')
    refwidth = 2.5 if pointwise else 2.0
    bools = (False, True) if relative is None else (relative,)
    specs = tuple((spec, {}) if isinstance(spec, str) else spec for spec in specs)
    specs = tuple((*spec, b) for spec, b in itertools.product(specs, bools))
    suffix = '-'.join(sorted(set(spec[0] for spec in specs)))
    suffix += '' if relative is None else ('-absolute', '-relative')[relative]
    fig = pplt.figure(refwidth=refwidth, sharex=0, sharey=1, span=False)
    gs = pplt.GridSpec(nrows=len(specs), ncols=3)
    path = Path(path).expanduser()
    collabels = ['Average', 'Minimum', 'Maximum']
    rowlabels = []
    for i, (variable, kwargs, relative) in enumerate(specs):
        # Load the data
        print(f'Loading {variable} ({relative})...')
        offsets, slopes = {}, {}
        models = tuple(
            file.name.split('_')[1]
            for name in ('slope', 'offset')
            for file in path.glob(f'{variable}_*_{name}.nc')
        )
        for model in sorted(set(models)):
            for dest, name in ((offsets, 'offset'), (slopes, 'slope')):
                file = path / f'{variable}_{model}_{name}_standard.nc'
                if not file.is_file():
                    print(f'Missing file(s) for model {model!r}.')
                    continue
                data = output.open_file(file, validate=False)
                if variable not in data:
                    print(f'Missing {variable} for model {model!r}.')
                    continue
                data = output._adjust_moisture_terms(data)
                data = output._adjust_standard_units(data)
                data = data[variable]
                isel = {dim: 0 for dim, size in data.sizes.items() if size == 1}
                drop = tuple(key for key, coord in data.coords.items() if coord.size == 1)  # noqa: E501
                data = data.isel(**isel).drop_vars(drop)
                data = data.climo.standardize_coords()
                if not all(key in data.sizes for key in kwargs):
                    raise ValueError(f'Invalid selection {kwargs} for {data.dims=}.')
                if kwargs:
                    data = data.climo.sel(**kwargs, method='nearest', drop=True)
                if data.dims != ('lat', 'lon'):
                    raise ValueError(f'Invalid dimensions {data.dims=}.')
                dest[model] = data

        # Combine the data
        print(f'Plotting {variable} ({relative})...')
        models = xr.DataArray(list(offsets), dims='model', name='model')
        offset = xr.concat(offsets.values(), dim=models, combine_attrs='override')
        slope = xr.concat(slopes.values(), dim=models, combine_attrs='override')
        offset = offset.climo.add_cell_measures()
        slope = slope.climo.add_cell_measures()
        descrip = 'relative' if relative else 'absolute'
        if True:
            with xr.set_options(keep_attrs=True):
                slope = 150 * slope
        if relative:
            slope.values[np.isclose(offset.values, 0, atol=1e-3)] = np.nan
            with xr.set_options(keep_attrs=True):
                slope = 100 * slope / offset
            slope.attrs['units'] = '%'

        # Plot the data
        label = offset.climo.long_name
        label = ' '.join(s if s == 'TOA' else s.lower() for s in label.split())
        label = label if label[:3] == 'TOA' else label[0].upper() + label[1:]
        label = re.sub('(upwelling|downwelling|outgoing|incident) ', '', label)
        rowlabels.append(label.replace('radiation', 'flux'))
        for j, operator in enumerate(('mean', 'min', 'max')):
            if pointwise:
                ax = fig.add_subplot(
                    gs[i, j],
                    proj=proj,
                    coast=True,
                    lonlines=30,
                    latlines=30,
                )
                label = f'{descrip} trend ({slope.climo.units_label} / 150 years)'
                slp = getattr(slope, operator)('model', keep_attrs=True)
                ax.pcolormesh(
                    slp,
                    extend='both',
                    diverging=True,
                    colorbar='b',
                    colorbar_kw={'label': label},
                    levels=20,
                    robust=99,
                )
            else:
                ax = fig.add_subplot(
                    gs[i, j],
                    xlabel=f'offset ({offset.climo.units_label})',
                    ylabel=f'{descrip} trend ({slope.climo.units_label} / 150 years)',
                )
                if operator == 'mean':
                    off = offset.climo.average('area')
                    slp = slope.climo.average('area')
                else:
                    off = getattr(offset, operator)(('lon', 'lat'), keep_attrs=True)
                    slp = getattr(slope, operator)(('lon', 'lat'), keep_attrs=True)
                c = pplt.get_colors('tableau', N=len(models))
                ax.set_prop_cycle('tableau')
                ax.axhline(0, color='k', lw=1)
                ax.scatter(off, slp, c=c, m='x', ms=8 ** 2)
                kw = dict(textcoords='offset points', ha='left', va='top')
                for model in models.values:
                    o = off.sel(model=model, drop=True)
                    s = slp.sel(model=model, drop=True)
                    ax.annotate(model, (o, s), (5, -5), **kw)
                xpad = 12 * pplt.rc.fontsize / (refwidth * 72)
                ypad = pplt.rc.fontsize / (refwidth * 72)
                xlim, ylim = ax.get_xlim(), ax.get_ylim()
                ax.set_ylim((ylim[0] - ypad * (ylim[1] - ylim[0]), ylim[1]))
                ax.set_xlim((xlim[0], xlim[1] + xpad * (xlim[1] - xlim[0])))

    # Save the result
    title = title and title + ' ' or ''
    fig.format(
        suptitle=f'Control climate {title}drift',
        collabels=collabels,
        rowlabels=rowlabels,
    )
    if save:
        type_ = 'pointwise' if pointwise else 'scatter'
        path = Path(__file__).parent.parent / 'background'
        path = path / f'drift_{type_}_{suffix}.pdf'
        print(f'Saving {path}...')
        fig.save(path)
    return fig, fig.subplotgrid


def plot_demo(path='~/data/cmip-constants', save=True):
    """
    Plot demo of drift correction results for surface temperature data.

    Parameters
    ----------
    path : path-like, optional
        The path. Should contain the files ``nodrift_(ann|mon)[0-4].nc`` where
        0 is the uncorrected data, 1 is the offset, 2 is the trend, 3 is the
        corrected data, and 4 is an alternative correction (see notebook).
    save : bool, optional
        Whether to save the result.
    """
    # Generate figure
    path = Path('path').expanduser()
    fig, axs = pplt.subplots(
        ncols=2, span=0, share=4, axwidth=2.3, figtitle='Drift correction demo'
    )
    for ax, name, title in zip(axs, ('ann', 'mon'), ('Annual', 'Monthly')):
        # Standardize data
        files = [path / f'nodrift_{name}{i}.nc' for i in range(0, 5)]
        temps = [xr.open_dataset(file, decode_times=True)['ts'] for file in files]
        temps = [
            temp if temp.time.size == 1 else temp.resample(time='AS').mean('time')
            for temp in temps
        ]
        temps = [
            temp if temp.time.size == 1 else temp.climo.runmean(time=30, center=True)
            for temp in temps
        ]
        temps = [temp.data.squeeze() for temp in temps]  # TODO: why is this necessary
        times = np.arange(temps[0].size)

        # Plot the corrections
        ax.plot(
            times,
            temps[0],
            color='C0',
            label='Original data',
        )
        ax.plot(
            times,
            temps[1] + temps[3],
            color='C1',
            label='Corrected data',
        )
        ax.plot(
            times,
            temps[1] + temps[4],
            color='C5',
            label='Alternative correction',
            zorder=0.5,
        )  # noqa: E501

        # Plot the slopes
        ax.axhline(
            np.nanmean(temps[0]),
            color='C0',
            ls='--',
            label='Original mean',
        )
        ax.axhline(
            np.nanmean(temps[1] + temps[3]),
            color='C1',
            ls='--',
            label='Corrected mean',
        )
        ax.axhline(
            np.nanmean(temps[1] + temps[4]),
            color='C5',
            ls='--',
            zorder=0.5,
            label='Alternative mean',
        )

        # Plot the terms and format
        scale = 12 if name == 'mon' else 1
        ax.plot(
            times,
            temps[1] + temps[2] * np.arange(0, temps[0].size) * scale,
            color='k',
            ls='-',
            alpha=0.5,
            label='Trend',
        )
        ax.axhline(
            temps[1],
            color='k',
            ls='--',
            label='Offset',
            alpha=0.5,
        )
        ax.format(
            title=f'{title} trends',
            ylabel='temperature (K)',
            xlabel='year',
        )
        if ax is axs[0]:
            fig.legend(loc='bottom', ncols=3, order='F')

    # Save figure
    if save:
        path = Path(__file__).parent.parent / 'background'
        path = path / 'drift_demo_annual-monthly.pdf'
        print(f'Saving {path}...')
        fig.save(path)
    return fig, fig.subplotgrid
