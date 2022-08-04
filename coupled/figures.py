#!/usr/bin/env python3
"""
Utilities for plotting coupled model output.
"""
import collections
import itertools
import re
from pathlib import Path

import numpy as np
import xarray as xr

import climopy as climo
import proplot as pplt
from climopy import ureg, vreg  # noqa: F401
from .output import FEEDBACK_DEFINITIONS, MODELS_INSTITUTIONS
from .output import _update_climate_moisture, _update_climate_units, open_file


# Variable constants
VARIABLE_ALIASES = {
    key: name for key, (name, _) in FEEDBACK_DEFINITIONS.items()
}

# Reduce constants
REDUCE_ABBREV = {
    'point': 'local',
    'globe': 'globe',
    'latitude': 'zone',
    'hemisphere': 'hemi',
}
REDUCE_LABELS = {  # default is title-case of input
    'ann': 'annual',
    'djf': 'DJF',
    'mam': 'MAM',
    'jja': 'JJA',
    'son': 'SON',
    'avg': 'average',
    'int': 'integral',
    'absmin': 'minimum',
    'absmax': 'maximum',
    'point': 'local',
    'globe': 'global',
    'latitude': 'zonal',
    'hemisphere': 'hemisphere',
}

# Keyword argument constants
KWARGS_DEFAULT = {  # default feedback selections
    'format': {
        'coast': True,
        'xscale': 'sine',
    },
    'reduce': {
        'period': 'ann',
        'experiment': 'control',
        'ensemble': 'flagship',
        'source': 'eraint',
        'statistic': 'regression',
        'region': 'globe',
    },
}
KWARGS_TUPLE = collections.namedtuple(
    'kwargs',
    (
        'reduce',
        'format',
        'colorbar',
        'legend',
        'plot'
    )
)
KWARGS_SORT = (
    'lon',  # regions
    'lat',
    'area',
    'plev',
    'period',
    'feedback',  # feedback multi-index
    'source',
    'statistic',
    'region',
    'facets',  # facet multi-index
    'project',
    'model',
    'ensemble',
    'experiment',
)


def _parse_project(dataset, project):
    """
    Return plot labels and facet tuples for the project indicator.

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset. Must contain a ``'facets'`` coordinates.
    project : str
        The project selection. Values can optionally start with ``'cmip'`` and must end
        with integers indicating the facet values. No integer indicates all CMIP5 and
        CMIP6 models, ``5`` (``6``) indicates just CMIP5 (CMIP6) models, ``56`` (``65``)
        indicates CMIP5 (CMIP6) models filtered to those from the same institutions as
        CMIP6 (CMIP5), and ``55`` (``66``) indicates models from institutions found only
        in CMIP5 (CMIP6). A common combination might be ``5``, ``65``, ``66``. As with
        other indexers, can use arithmetic operators for combinations (e.g. ``'6-5'``).

    Returns
    -------
    abbrev : str
        The file name abbreviation.
    label : str
        The column or row label string.
    filter : callable
        Function for filtering ``facets`` coordinates.
    """
    # NOTE: Critical that 'facets' selection is a list because accessor reduce method
    # will pass tuples to '.get()' for interpolation onto variable-derived locations.
    s1, s2 = object(), object()  # sentinels
    _, num = project.lower().split('cmip')
    if not num:
        filter = lambda key: True  # noqa: U100
        label = 'CMIP'
        abbrev = 'cmip'
    elif num in ('5', '6'):
        filter = lambda key: key[0][-1] == num
        label = f'CMIP{num}'
        abbrev = f'cmip{num}'
    elif num in ('65', '66', '56', '55'):
        idx = len(set(num)) - 1  # zero if only one unique integer
        opp = '5' if num[0] == '6' else '6'  # opposite project number
        filter = lambda key: key[0][-1] == num[0] and idx == any(
            MODELS_INSTITUTIONS.get(key[1], s1) == MODELS_INSTITUTIONS.get(other[1], s2)
            for other in dataset.facets.values  # iterate over keys
            if other[0][-1] == opp
        )
        label = ('Other', 'Matched')[idx] + f' CMIP{num[0]}'
        abbrev = f'cmip{num}'
    else:
        raise ValueError(f'Invalid project {num!r}.')
    return abbrev, label, filter


def _parse_reduce(dataset, **indexers):
    """
    Standardize the indexers and translate into labels suitable for figures.

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset.
    **indexers
        The indexers. Numeric are assumed to be the units of the array and
        string indexers can have arithmetic operators ``+-`` to perform
        operations between multiple selections (e.g. seasonal differences
        relative to annual average or one project relative to another). Note
        the special key `project` is handled by `_parse_project`.

    Returns
    -------
    abbrevs : dict
        Abbreviations for the coordinate reductions. This is destined
        for use in file names.
    labels : dict
        Labels for the coordinate reductions. This is destined for
        use in row and column figure labels.
    operator : str or None
        The operator used between 2-tuples of selections. This is commonly
        used for project differences, e.g. ``cmip6-cmip5``.
    reduce : dict
        The reduction selections. Contains either scalar sanitized selections
        or two-tuples of selections for operations.
    """
    # NOTE: Integers are added to titles later to indicate the number of values
    # without NaNs. May be fewer when feedback components are unavailable.
    # NOTE: Critical that we iterate through indexers in order since they the order
    # is standardized in _parse_spec. Otherwise get different labels and file names.
    abbrevs, labels, reduce = {}, {}, {}
    operators = set()
    for dim, coord in indexers.items():
        abvs, labs, sels = [], [], []
        parts = re.split('([+-])', coord) if isinstance(coord, str) else (coord,)
        if len(parts) not in (1, 3):  # maximum one operator
            raise ValueError(f'Unexpected coordinate format {coord!r}.')
        for i, sel in enumerate(parts):
            if not isinstance(sel, str):
                coords = dataset[dim]
                unit = coords.climo.units
                if not isinstance(sel, ureg.Quantity):
                    sel = ureg.Quantity(sel, unit)
                sel = sel.to(unit)
                label = f'${sel:~L}$'
                abbrev = f'{sel:~.0f}'.replace('/', 'p').replace(' ', '')
            elif dim == 'project':
                dim = 'facets'
                abbrev, label, sel = _parse_project(dataset, sel)
            elif sel in '+-':
                abbrev, label = sel, 'plus' if sel == '+' else 'minus'
                operators.add('add' if sel == '+' else 'sub')
                if len(operators) > 1:
                    raise ValueError(f'Conflicting selection {operators=}.')
            else:
                label = REDUCE_LABELS.get(sel, sel)
                abbrev = REDUCE_ABBREV.get(sel, sel.lower())
            abvs.append(abbrev)
            labs.append(label)
            if not isinstance(sel, str) or sel not in '+-':
                sels.append(sel)
        abbrevs[dim] = '-'.join(abvs)
        labels[dim] = ' '.join(labs)
        reduce[dim] = sels[0] if len(sels) == 1 else tuple(sels)
    operator = operators.pop() if operators else None  # only single operator permitted
    return abbrevs, labels, operator, reduce


def _parse_spec(dataset, spec, **kwargs):
    """
    Parse the variable specification.

    Parameters
    ----------
    dataset : `xarray.Dataset`
        The dataset.
    spec : sequence, str, or dict
        The specification.
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    name : str
        The short name. Summation of multiple variables (e.g. feedback terms)
        can be indicated with e.g. ``pl+lr+wv``. Shorthand aliases are translated.
        This is destined for use in file names.
    long : str
        The long name. Names are combined for summation of multiple variables
        using e.g. ``Planck + lapse rate + water vapor``. This is destined for
        use in figure row or column labels.
    kwargs : namedtuple of dict
        A named tuple containing keyword arguments for different plotting-related
        commands. The keys are as follows:

          * ``reduce``: Passed to `.climo.reduce` for index reductions.
          * ``format``: Passed to `.format` for cartesian or geographic formatting.
          * ``colorbar``: Passed to `.colorbar` for scalar mappable outputs.
          * ``legend``: Passed to `ax.legend` for other artist outputs.
          * ``plot``: Passed to the plotting command.

        The first four categories are detected automatically using the dataset
        and a list of possible argument prefixes while the the last category
        contains all remaining keyword arguments.
    """
    options = (*dataset.sizes, 'area', 'volume')
    options += tuple(name for idx in dataset.indexes.values() for name in idx.names)
    if spec is None:
        name, kw = None, {}
    elif isinstance(spec, str):
        name, kw = spec, {}
    elif isinstance(spec, dict):
        name, kw = None, spec
    else:  # 2-tuple required
        name, kw = spec
    kw = {**kwargs, **kw}  # add shared keywords
    long, kw_red, kw_fmt, kw_plt, kw_bar, kw_leg = None, {}, {}, {}, {}, {}
    fmt_detect = ('x', 'y', 'lon', 'lat', 'abc', 'title', 'coast')
    bar_detect = ('extend', 'tick', 'locator', 'formatter', 'minor')
    leg_detect = ('ncol', 'order', 'frame', 'handle', 'border', 'column')
    if name:  # TODO: climopy get() add methods instead?
        name = '+'.join(VARIABLE_ALIASES.get(n, n) for n in name.split('+'))
        long = ' + '.join(dataset[n].long_name for n in name.split('+'))
    sort = lambda key: (
        KWARGS_SORT.index(key) if key in KWARGS_SORT else len(KWARGS_SORT)
    )
    for key in sorted(kw, key=sort):
        value = kw[key]  # sort for name and label standardization
        if key in options:
            kw_red[key] = value  # e.g. for averaging
        elif any(key.startswith(prefix) for prefix in fmt_detect):
            kw_fmt[key] = value
        elif any(key.startswith(prefix) for prefix in bar_detect):
            kw_bar[key] = value
        elif any(key.startswith(prefix) for prefix in leg_detect):
            kw_leg[key] = value
        else:  # arbitrary plotting keywords
            kw_plt[key] = value
    kwargs = KWARGS_TUPLE(kw_red, kw_fmt, kw_bar, kw_leg, kw_plt)
    return name, long, kwargs


def _parse_specs(
    dataset, rowspecs, colspecs, method1='avg', method2='corr', **kwargs
):
    """
    Parse variable and project specifications and auto-determine row and column
    labels based on the unique names and/or keywords in the spec lists.

    Parameters
    ----------
    dataset : xarray.Dataset
        The source dataset.
    rowspecs : list of name, tuple, dict, or list thereof
        The variable specification(s) per subplot slot.
    colspecs : list of name, tuple, dict, or list thereof
        The variable specification(s) per subplot slot.
    method1 : str, optional
        The default `facets` reduction method for single-variable specifications.
    method2 : str, optional
        The default `facets` reduction method for double-variable specifications.
    **kwargs
        Additional options shared across all specs.

    Returns
    -------
    gridspecs : list of list
        Row and column labels suitable for the figure grid.
    filespecs : list of str
        String segments suitable for the file name.
    plotspecs : list of dict
        List of per-subplot lists of plot specifications. Indicates the facet reduce
        `method`, the reduce-selection `operator`, the row varaible `rname`, the
        column variable `cname`, and the joint keyword args named tuple `kwargs`.
    """
    # Parse variable specs per gridspec row or column and per subplot, and generate
    # abbreviated figure labels and file names based on the first entries.
    # NOTE: This permits sharing keywords across each group with trailing dicts
    # in either the primary gridspec list or any of the subplot sub-lists.
    filespecs, gridspecs, allspecs = [], [], []
    for inspecs in (rowspecs, colspecs):
        # Get general specifications
        outspecs = []  # specs containing general information
        if not isinstance(inspecs, list):
            inspecs = [inspecs]
        kw_global = {}  # apply to entire spec list
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
                kw = {**kwargs, **kw_global, **kw_local}
                name, long, kw = _parse_spec(dataset, spec, **kw)
                abbrevs, labels, operator, reduce = _parse_reduce(dataset, **kw.reduce)
                kw.reduce.clear()
                kw.reduce.update(reduce)
                ospecs.append((name, long, abbrevs, labels, operator, kw))
            outspecs.append(ospecs)
        allspecs.append(outspecs)

        # Get file labels
        names, longs, abbrevs, labels, *_ = zip(*(ospecs[0] for ospecs in outspecs))
        seen = set()
        reverse = {
            name: key for key, name in VARIABLE_ALIASES.items()
            if name not in seen and not seen.add(name)
        }
        names = tuple(
            '+'.join(reverse.get(n, n) for n in name.split('+'))
            for name in filter(None, names)
        )
        seen = set()
        fspecs = (*names, *(abv for abvs in abbrevs for abv in abvs.values()))
        fspecs = (spec for spec in fspecs if spec not in seen and not seen.add(spec))
        fspecs = (spec.replace('_', '') for spec in filter(None, fspecs))
        filespecs.append(list(fspecs))

        # Get row and column gridspec labels
        gspecs = []  # row/column grid label specs
        if len(set(tuple(labs.items()) for labs in labels)) > 1:
            seen = set()
            dims = [
                dim for labs in labels for dim in labs
                if dim not in seen and not seen.add(dim)
            ]
            labels = [tuple(labs.get(dim, None) for labs in labels) for dim in dims]
            labels = [labs for labs in labels if any(lab != labs[0] for lab in labs)]
            labels = [' '.join(filter(None, labs)) for labs in zip(*labels)]
            gspecs.append(labels)
        if len(set(longs)) > 1:  # unique variables
            replace_always = (' parameter', ' effective')
            replace_repeated = (' feedback', ' forcing')
            for string in replace_always:
                longs = [long.replace(string, '') for long in longs]
            for string in replace_repeated:  # NOTE: can add to this
                if all(string in long for long in longs):
                    longs = [long.replace(string, '') for long in longs]
            gspecs.append(longs)
        gspecs = [' '.join(filter(None, specs)) for specs in zip(*gspecs)]
        gspecs = [spec[0].upper() + spec[1:] if spec.split()[0].islower() else spec for spec in gspecs]  # noqa: E501
        gridspecs.append(gspecs or [''] * len(outspecs))

    # Combine row and column specifications for plotting and file naming
    # NOTE: Multiple plotted values per subplot can be indicated in either the
    # row or column list, and the specs from the other list will be repeated.
    methods, plotspecs = [], []
    nrows, ncols = map(len, allspecs)
    for (i, rspecs), (j, cspecs) in itertools.product(*map(enumerate, allspecs)):
        if len(rspecs) == 1:
            rspecs = list(rspecs) * len(cspecs)
        if len(cspecs) == 1:
            cspecs = list(cspecs) * len(rspecs)
        if len(rspecs) != len(cspecs):
            raise ValueError(
                f'Incompatible per-subplot spec count: {len(rspecs)} from '
                f'the row specs, {len(cspecs)} from the column specs.'
            )
        pspecs = []
        for k, (rspec, cspec) in enumerate(zip(rspecs, cspecs)):
            rname, _, _, _, roperator, rkwargs = rspec
            cname, _, _, _, coperator, ckwargs = cspec
            operator = {roperator, coperator}.pop()
            kwargs = KWARGS_TUPLE({}, {}, {}, {}, {})
            for key in ('reduce', 'format', 'colorbar', 'legend', 'plot'):
                kw = getattr(kwargs, key)
                kw.update(KWARGS_DEFAULT.get(key, {}))
                kw.update(getattr(rkwargs, key))
                kw.update(getattr(ckwargs, key))
            m1 = m2 = kwargs.plot.pop('method', None)  # input method
            if rname is None and cname is None:
                raise ValueError(f'Empty variable specification in slot {(i, j)}.')
            elif rname is not None and cname is not None:
                method = m2 or method2
            else:
                method = m1 or method1
            if k == 0 and method not in methods:  # record first item in subplot
                methods.append(method)
            pspecs.append((method, operator, rname, cname, kwargs))
        plotspecs.append(pspecs)

    # Return specifications
    # NOTE: Grid specs have two entires: the row labels and the column labels. File
    # specs have four entires: the methods employed, the row-column global specs, the
    # row-specific specs, and the col-specific specs. Plot specs have as many entries
    # as there are subplots in the figure (nested for multiple subplots).
    fboth = [spec for spec in filespecs[0] if spec in filespecs[1]]
    frows = [spec for spec in filespecs[0] if spec not in fboth]
    fcols = [spec for spec in filespecs[1] if spec not in fboth]
    filespecs = ['-'.join(methods), '-'.join(fboth), '-'.join(frows), '-'.join(fcols)]
    return gridspecs, filespecs, plotspecs


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
        The method for reducing the facets dimension on individual non-scalar
        variables. The default is an average.
    method2 : {'corr', 'diff'}, optional
        The method for reducing the facets dimension on paired non-scalar
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
        The percentiles for the ``'diff'`` composite differences, constructed
        roughly as ``data[feedback > 100 - pctile] - data[feedback < pctile]``.
    proj : str, optional
        The cartopy projection for longitude-latitude type plots. Default
        is the area-weighted projection ``'eqearth'``.
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
    kwargs.setdefault('coast', True)
    kwargs.setdefault('xscale', 'sine')
    (rowlabels, collabels), filespecs, plotspecs = _parse_specs(
        dataset, rowspecs, colspecs, method1=method1, method2=method2, **kwargs
    )
    nrows, ncols = len(rowlabels), len(collabels)
    if len(rowlabels) == 1 or len(collabels) == 1:
        naxes = max(len(rowlabels), len(collabels))
        ncols = min(naxes, maxcols)
        nrows = 1 + (ncols - 1) // naxes
    def _safe_reduce(data, **kwargs):  # noqa: E301
        if facets := kwargs.pop('facets', None):  # see _parse_projects
            data = data.sel(facets=list(filter(facets, data.facets.values)))
        kwargs = {
            key: value for key, value in kwargs.items()
            if key in data.sizes
            or any(key in idx.names for idx in data.indexes.values())  # noqa: E501
        }
        if kwargs:
            data = data.sel(**kwargs)
        return data

    # Iterate over axes and plots
    # NOTE: Critical to disable 'grouping' so that e.g. colorbars or legends that
    # extend into other panel slots are not considered in the tight layout algorithm.
    fig = pplt.figure(refwidth=refwidth, hgroup=False, wgroup=False)
    gridspec = pplt.GridSpec(nrows, ncols)
    contours = ('gray8', 'gray3')  # contour colors
    commands = {}
    for i, pspecs in enumerate(plotspecs):
        ax = None  # restart the axes
        ndims = set()
        subplotspec = gridspec[i]
        for j, pspec in enumerate(pspecs):
            # Get the data arrays
            # NOTE: See `_parse_specs` for more information.
            datas, infos = [], {}
            method, operator, rname, cname, kwargs = pspec
            for name in (rname, cname):
                if not name:
                    continue
                with xr.set_options(keep_attrs=True):  # e.g. pl+lr+hus
                    data = sum(dataset[n] for n in name.split('+'))
                if not operator:
                    data = _safe_reduce(data, **kwargs.reduce)
                else:
                    kw1, kw2 = {}, {}
                    for dim, sel in kwargs.reduce.items():
                        if isinstance(sel, tuple) and len(sel) == 2:
                            kw1[dim], kw2[dim] = sel
                        else:
                            kw1[dim] = kw2[dim] = sel
                    data2 = _safe_reduce(data, **kw2)
                    data1 = _safe_reduce(data, **kw1)
                    data = getattr(data2, f'__{operator}__')(data1)  # e.g. subtract
                data.name = name
                datas.append(data)
                mask = (~data.isnull()).any(data.sizes.keys() - {'facets'})
                min_, max_, mean = data.min().item(), data.mean().item(), data.max().item()  # noqa: E501
                infos[data.name] = (mask, min_, max_, mean)
            if not datas:
                continue

            # Reduce along the source dimension
            # TODO: Add other possible reduction methods, e.g. covariance
            # or average addition or subtraction of variables.
            for name, (mask, min_, max_, mean) in infos.items():
                print(f'Variable {data.name!r} range: ', end=' ')
                print(f'min {min_:.02f} max {max_:.02f} mean {mean:.02f}')
                print(f'Variable {data.name!r} facets:', end=' ')
                print(f'{np.sum(mask).item()} valid {np.sum(~mask).item()} invalid')
            if len(infos) == 2:
                name = '-'.join(infos.keys())
                mask = np.logical_and(*(mask for mask, *_ in infos.values()))
                print(f'Joined {data.name!r} facets:', end=' ')
                print(f'{np.sum(mask).item()} valid {np.sum(~mask).item()} invalid')
            if len(datas) == 1:
                if datas[0].ndim == 1:  # 1D histogram with scatter points
                    data = (datas,)
                elif method == 'avg':
                    data = datas[0].mean('facets', skipna=True, keep_attrs=True)
                    data.name = datas[0].name
                    data.attrs['long_name'] = f'{data.long_name} multi-model mean'
                elif method == 'std':
                    data = datas[0].std(dim='model', skipna=True, keep_attrs=True)
                    data.name = datas[0].name
                    data.attrs['long_name'] = f'{data.long_name} inter-model stdev'
                else:
                    raise ValueError(f'Invalid single-variable method {method}.')
            else:
                if datas[0].ndim == 1:  # 2D histogram with scatter points
                    data = (datas[0], datas[1])
                elif method == 'corr':  # correlation coefficient
                    da0, da1 = xr.broadcast(*datas)
                    _, data = climo.corr(*datas, dim='facets')  # updates units
                    data = data.isel(lag=0)
                    long_name = f'{datas[0].long_name}-{datas[1].long_name} correlation'
                    data.name = f'{datas[0].name}-{datas[1].name}'
                    data.attrs['long_name'] = long_name
                elif method == 'diff':  # composite difference (feedbacks on rows)
                    lo_comp = np.nanpercentile(datas[0], pctile)
                    hi_comp = np.nanpercentile(datas[0], 100 - pctile)
                    lo_mask, = np.where(datas[0] <= lo_comp)
                    hi_mask, = np.where(datas[0] >= hi_comp)
                    hi_data = datas[1].isel(facets=hi_mask).mean('facets', keep_attrs=True)  # noqa: E501
                    lo_data = datas[1].isel(facets=lo_mask).mean('facets', keep_attrs=True)  # noqa: E501
                    with xr.set_options(keep_attrs=True):  # keep units
                        data = hi_data - lo_data
                    long_name = f'{datas[0].long_name}-composite {datas[1].long_name} difference'  # noqa: E501
                    data.name = f'{datas[0].name}-{datas[1].name}'
                    data.attrs['long_name'] = long_name
                else:
                    raise ValueError(f'Invalid double-variable method {method}')

            # Queue the command and store auto-generated filename info
            # TODO: Should support only adding axes if x or y units do not match
            # previous units... consider e.g. using matplotlib-pint unit conversions.
            # NOTE: For e.g. correlation between Planck, lapse rate, water vapor
            # feedbacks and surface temperature name will be 'corr_pl+lr+wv-ts_cmip5'
            dims = data.sizes.keys() - {'facets'}
            ndims.add(ndim := len(data.sizes.keys() - {'facets'}))
            projection = proj if dims == {'lon', 'lat'} else 'cartesian'
            if ax is None:
                ax = jax = fig.add_subplot(subplotspec, projection=projection)
            if hasattr(ax, 'alty') != (projection == 'cartesian'):
                raise ValueError(f'Conflicting projection types for dimensions {dims}.')
            if len(ndims) > 1:
                raise ValueError(f'Conflicting plot types for sizes {ndims}.')
            if ndim == 0:
                command = 'hist2d' if len(datas) == 2 else 'hist'
            elif ndim == 1:
                command, jax = 'line', ax if j == 0 else ax.alty()
            elif ndim == 2:
                command = ('pcolormesh', 'contour')[min(j, 1)]
            if command == 'contours':
                kwargs.plot.setdefault('color', contours[j - 1])
            args = data if isinstance(data, tuple) else (data,)
            name = '-'.join(da.name for da in args)
            values = commands.setdefault((name, method, command), [])
            values.append((jax, args, kwargs))
            if projection == 'cartesian':
                ignore = ('lon', 'lat', 'coast')
            else:
                ignore = ('x', 'y')
            jax.format(**{
                key: value for key, value in kwargs.format.items()
                if not any(key.startswith(string) for string in ignore)
            })

    # Carry out the plotting commands
    # NOTE: Axes are always added top-to-bottom and left-to-right so leverage
    # this fact below when selecting axes for legends and colorbars.
    print(nrows, ncols, fig.subplotgrid)
    for (name, method, command), values in commands.items():
        # Call plotting commands
        print(f'Creating plot(s): {name} {method} {command}')
        axs, args, kw_bar, kw_leg, kw_plt = [], [], {}, {}, {}
        for ax, arg, kwargs in values:
            axs.append(ax)
            args.append(arg)
            kw_bar.update(kwargs.colorbar)
            kw_leg.update(kwargs.legend)
            kw_plt.update(kwargs.plot)
        if command in ('pcolor', 'pcolormesh', 'contour', 'contourf', 'hist2d'):
            guide = 'colorbar'
            hori, vert, default, kw_obj = hcolorbar, vcolorbar, dcolorbar, kw_bar
            kw_obj['label'] = args[0][-1].climo.units_label
        else:
            guide = 'legend'
            hori, vert, default, kw_obj = hlegend, vlegend, dlegend, kw_leg
            kw_plt['label'] = args[0][-1].climo.units_label
        if guide == 'colorbar':
            kw_plt['levels'], *_, = axs[0]._parse_level_vals(
                *(args[0][-1].coords[dim] for dim in args[0][-1].dims),
                *(a for arg in args for a in arg),
                norm_kw={},
                **kw_plt
            )
        for key in ('robust', 'vmin', 'vmax', 'values', 'N'):
            kw_plt.pop(key, None)
        for ax, arg in zip(axs, args):
            obj = getattr(ax, command)(*arg, **kw_plt)  # plot and return guide object

        # Generate colorbars and legends
        rows = set(ax._range_subplotspec('y') for ax in axs)
        rows = rows.pop() if len(rows) == 1 else None
        cols = set(ax._range_subplotspec('x') for ax in axs)
        cols = cols.pop() if len(cols) == 1 else None
        if not (rows is None) ^ (cols is None):  # not single column or single row
            src = fig
            loc = default
        elif cols is not None:  # single column
            if vert[0] in 'tb':
                src = axs[0] if vert[0] == 't' else axs[-1]
            else:
                src = axs[len(axs) // 2]  # TODO: support even-numbered axes
            loc = vert
        else:  # single row
            if hori[0] in 'lr':
                src = axs[0] if hori[0] == 'l' else axs[-1]
            else:
                src = axs[len(axs) // 2]  # TODO: support even-numbered axes
            loc = hori
        getattr(src, guide)(obj, loc=loc, queue=True, **kw_obj)

    # Format the axes and optionally save
    fig.format(
        suptitle=title,
        rowlabels=rowlabels or None,
        collabels=collabels or None,
    )
    if save:
        path = Path(__file__).parent.parent / 'figures'
        path = path / ('_'.join(filespecs) + '.pdf')
        print(f'Saving {path.parent}/{path.name}...')
        fig.save(path)
    return fig, fig.subplotgrid


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
                data = open_file(file, validate=False)
                if variable not in data:
                    print(f'Missing {variable} for model {model!r}.')
                    continue
                data = _update_climate_moisture(data)
                data = _update_climate_units(data)
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
