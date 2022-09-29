#!/usr/bin/env python3
"""
Templates for figures detailing coupled model output.
"""
import collections
import itertools
import re
import warnings
from pathlib import Path

import numpy as np
import xarray as xr
from icecream import ic  # noqa: F401

import proplot as pplt
from climopy import decode_units, format_units, ureg, var, vreg  # noqa: F401
from .internals import apply_method, apply_reduce, apply_variance
from .internals import _parse_institute, _parse_labels, _parse_project, _parse_reduce


# Keyword argument constants
# NOTE: For sake of grammar we place 'ensemble' before 'experiment' here
KWARGS_ALL = collections.namedtuple(
    'kwargs', (
        'figure',
        'gridspec',
        'axes',
        'colorbar',
        'legend',
        'plot',
        'attrs'
    )
)
KWARGS_DEFAULT = {
    'geo': {
        'coast': True,
        'lonlines': 30,
        'latlines': 30,
        'refwidth': 2.3,
    },
    'lat': {
        'xlabel': 'latitude',
        'xformatter': 'deg',
        'xlocator': 30,
        'xscale': 'linear',
        'xlim': (-89, 89),
        'refwidth': 1.5,
    },
    'plev': {
        'ylocator': 200,
        'yreverse': True,
        'ylabel': 'pressure (hPa)',
        'refwidth': 1.5,
    },
}


def _parse_dicts(dataset, spec, **kwargs):
    """
    Parse the variable name specification.

    Parameters
    ----------
    dataset : `xarray.Dataset`
        The dataset.
    spec : sequence, str, or dict
        The variable specification. Can be a ``(name, kwargs)``, a naked ``name``,
        or a naked ``kwargs``. The name can be omitted or specified inside `kwargs`
        with the `name` dictionary, and it will be specified when intersected with
        other specifications (i.e. the row-column framework; see `_parse_specs`).
    **kwargs
        Additional keyword arguments, used as defaults for the unset keys
        in the variable specifications.

    Returns
    -------
    kw_red : dict
        The indexers used to reduce the data variable with `.reduce`. This is
        parsed specially compared to other keywords, and its keys are restricted
        to ``'name'`` and any coordinate or multi-index names.
    kw_all : namedtuple of dict
        A named tuple containing keyword arguments for different plotting-related
        commands. The tuple fields are as follows:

          * ``figure``: Passed to `Figure` when the figure is instantiated.
          * ``gridspec``: Passed to `GridSpec` when the gridfspec is instantiated.
          * ``axes``: Passed to `.format` for cartesian or geographic formatting.
          * ``colorbar``: Passed to `.colorbar` for scalar mappable outputs.
          * ``legend``: Passed to `.legend` for other artist outputs.
          * ``plot``: Passed to the plotting command (the default field).
          * ``attrs``: Added to `.attrs` for use in resulting plot labels.

        These keywords are applied at different points throughout the `plot_bulk`
        function. The figure and gridspec ones are only passed on instantiation.
    """
    # NOTE: For subsequent processing we put the variables being combined (usually
    # just one) inside the 'name' key in kw_red (here `short` is shortened relative
    # to actual dataset names and intended for file names only). This helps when
    # merging variable specifications between row and column specifications and
    # between tuple-style specifications (see _parse_specs).
    options = list(dataset.sizes)
    options.extend(name for idx in dataset.indexes.values() for name in idx.names)
    options.extend(('area', 'volume', 'institute'))  # see _parse_institute
    options.extend(('method', 'invert', 'pctile', 'std'))  # see apply_method
    if spec is None:
        name, kw = None, {}
    elif isinstance(spec, str):
        name, kw = spec, {}
    elif isinstance(spec, dict):
        name, kw = None, spec
    else:  # length-2 iterable
        name, kw = spec
    kw = kw.copy()  # critical
    alt = kw.pop('name', None)
    name = name or alt  # see below
    kw = {**kwargs, **kw}
    kw_red, kw_att = {}, {}
    kw_fig, kw_grd, kw_axs = {}, {}, {}
    kw_plt, kw_cba, kw_leg = {}, {}, {}
    keys = ('space', 'ratio', 'group', 'equal', 'left', 'right', 'bottom', 'top')
    att_detect = ('short', 'long', 'standard')
    fig_detect = ('fig', 'ref', 'space', 'share', 'span', 'align')
    grd_detect = tuple(s + key for key in keys for s in ('w', 'h', ''))
    axs_detect = ('x', 'y', 'lon', 'lat', 'abc', 'title', 'coast')
    bar_detect = ('extend', 'tick', 'locator', 'formatter', 'minor', 'label')
    leg_detect = ('ncol', 'order', 'frame', 'handle', 'border', 'column')
    for key, value in kw.items():  # NOTE: sorting performed in _parse_labels
        if key in options:
            kw_red[key] = value  # e.g. for averaging
        elif any(key.startswith(prefix) for prefix in att_detect):
            kw_att[key] = value
        elif any(key.startswith(prefix) for prefix in fig_detect):
            kw_fig[key] = value
        elif any(key.startswith(prefix) for prefix in grd_detect):
            kw_grd[key] = value
        elif any(key.startswith(prefix) for prefix in axs_detect):
            kw_axs[key] = value
        elif any(key.startswith(prefix) for prefix in bar_detect):
            kw_cba[key] = value
        elif any(key.startswith(prefix) for prefix in leg_detect):
            kw_leg[key] = value
        else:  # arbitrary plotting keywords
            kw_plt[key] = value
    kw_all = KWARGS_ALL(kw_fig, kw_grd, kw_axs, kw_cba, kw_leg, kw_plt, kw_att)
    if isinstance(name, str):
        kw_red['name'] = name  # always place last for gridspec labels
    return kw_red, kw_all


def _parse_specs(dataset, rowspecs, colspecs, **kwargs):
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
    **kwargs
        Additional options shared across all specs.

    Returns
    -------
    filespecs : list of list of str
        Strings suitable for the figure file name. See `_parse_labels`.
    gridspecs : list of list of str
        Strings suitable for the figure grid labels. See `_parse_labels`.
    plotspecs : list of list of tuple
        Specifications for plotting in subplots. Returned as list totaling the number
        of axes with length-3 sublists containing the reduction method, a length-1
        or length-2 list of coordinate reduction keyword arguments, and a named tuple
        storing the shared plotting-related keyword arguments.
    """
    # Parse variable specs per gridspec row or column and per subplot, and generate
    # abbrviated figure labels and file names based on the first entries.
    # NOTE: This permits sharing keywords across each group with trailing dicts
    # in either the primary gridspec list or any of the subplot sub-lists.
    # NOTE: The two data arrays required for two-argument methods can be indicated with
    # either 2-tuples in spec lists or conflicting row and column names or reductions.
    filespecs, gridspecs, plotspecs = [], [], []
    for inspecs in (rowspecs, colspecs):
        outspecs = []  # specs containing general information
        if not isinstance(inspecs, list):
            inspecs = [inspecs]
        for ispecs in inspecs:  # specs per figure
            ospecs = []
            if not isinstance(ispecs, list):
                ispecs = [ispecs]
            for ispec in ispecs:  # specs per subplot
                ospec = []
                if isinstance(ispec, (str, dict)):
                    ispec = (ispec,)
                elif len(ispec) != 2:
                    raise ValueError(f'Invalid variable specs {ispec}.')
                elif type(ispec[0]) != type(ispec[1]):  # noqa: E721  # i.e. (str, dict)
                    ispec = (ispec,)
                else:
                    ispec = tuple(ispec)
                for spec in ispec:  # specs per correlation pair
                    kw_red, kw_all = _parse_dicts(dataset, spec, **kwargs)
                    abbrvs, labels, kw_red = _parse_reduce(dataset, **kw_red)
                    ospec.append((abbrvs, labels, kw_red, kw_all))
                ospecs.append(ospec)
            outspecs.append(ospecs)
        fspecs, gspecs = _parse_labels(outspecs, refwidth=kwargs.get('refwidth'))
        plotspecs.append(outspecs)
        filespecs.append(fspecs)
        gridspecs.append(gspecs)

    # Combine row and column specifications for plotting and file naming
    # NOTE: More than one plotted values per subplot can be indicated in either the
    # row or column list, and the specs from the other list are repeated below.
    # WARNING: Critical to make copies of dictionaries or create new ones
    # here since itertools product repeats the same spec multiple times.
    def _to_hash(value):
        if not isinstance(value, tuple):
            return value
        elif not all(isinstance(val, tuple) for val in value):
            return value
        else:
            hashes = []
            for val in value:
                hash = []
                for v in val:
                    if callable(v):  # filter name
                        h = v.__name__
                    elif isinstance(v, xr.DataArray):  # groupby array
                        h = v.name
                    else:  # e.g. 'avg' string
                        h = v
                    hash.append(h)
                hashes.append(tuple(hash))
            return tuple(hashes)
    iter_ = itertools.product(*map(enumerate, plotspecs))
    nrows, ncols = map(len, plotspecs)
    plotspecs = []
    for (i, rspecs), (j, cspecs) in iter_:
        if len(rspecs) == 1:
            rspecs = list(rspecs) * len(cspecs)
        if len(cspecs) == 1:
            cspecs = list(cspecs) * len(rspecs)
        if len(rspecs) != len(cspecs):
            raise ValueError(
                'Incompatible per-subplot spec count.'
                + f'\nRow specs ({len(rspecs)}): \n' + '\n'.join(map(repr, rspecs))
                + f'\nColumn specs ({len(cspecs)}): \n' + '\n'.join(map(repr, cspecs))
            )
        pspecs = []
        for k, (rspec, cspec) in enumerate(zip(rspecs, cspecs)):  # subplot entries
            _, _, rkws_red, rkws_all = zip(*rspec)
            _, _, ckws_red, ckws_all = zip(*cspec)
            kws = []
            for field in KWARGS_ALL._fields:
                kw = {}  # NOTE: previously applied default values here
                for ikws_all in (rkws_all, ckws_all):
                    for ikw_all in ikws_all:  # correlation pairs
                        for key, value in getattr(ikw_all, field).items():
                            kw.setdefault(key, value)  # prefer row entries
                kws.append(kw)
            kw_all = KWARGS_ALL(*kws)
            rkws_red = tuple(kw.copy() for kw in rkws_red)  # NOTE: copy is critical
            ckws_red = tuple(kw.copy() for kw in ckws_red)
            for ikws_red, jkws_red in ((rkws_red, ckws_red), (ckws_red, rkws_red)):
                for key in ikws_red[0]:
                    if key not in ikws_red[-1]:
                        continue
                    if _to_hash(ikws_red[0][key]) == _to_hash(ikws_red[-1][key]):
                        for kw in jkws_red:  # possible correlation pair
                            kw.setdefault(key, ikws_red[0][key])
            kws_red = {}  # filter unique specifications
            for kw_red in (*rkws_red, *ckws_red):
                keyval = tuple(
                    (key, _to_hash(kw_red[key])) for key in sorted(kw_red)
                    if key not in ('method',)
                )
                if tuple(keyval) in kws_red:
                    continue
                others = tuple(other for other in kws_red if set(keyval) < set(other))
                if others:
                    continue
                others = tuple(other for other in kws_red if set(other) < set(keyval))
                for other in others:  # prefer more selection keywords
                    kws_red.pop(other)
                kws_red[tuple(keyval)] = kw_red
            kws_red = tuple(kws_red.values())
            if len(kws_red) > 2:
                raise RuntimeError(
                    'Expected 1-2 specs for combining with apply_method but '
                    f'got {len(kws_red)} specs: ' + '\n'.join(map(repr, kws_red))
                )
            pspecs.append((kws_red, kw_all))
        plotspecs.append(pspecs)

    # Return the specifications
    both = [spec for spec in filespecs[0] if spec in filespecs[1]]
    rows = [spec for spec in filespecs[0] if spec not in both]
    cols = [spec for spec in filespecs[1] if spec not in both]
    filespecs = [both, rows, cols]
    return filespecs, gridspecs, plotspecs


def plot_bulk(
    dataset,
    rowspecs,
    colspecs,
    maxcols=3,
    argskip=None,
    figtitle=None,
    gridskip=None,
    rowlabels=None,
    collabels=None,
    hcolorbar='right',
    vcolorbar='bottom',
    dcolorbar='bottom',
    hlegend='bottom',
    vlegend='bottom',
    dlegend='bottom',
    horizontal=False,
    standardize=False,
    annotate=False,
    linefit=False,
    oneone=False,
    pcolor=False,
    cycle=None,
    proj='kav7',
    proj_kw=None,
    save=None,
    **kwargs
):
    """
    Plot any combination of variables across rows and columns.

    Parameters
    ----------
    dataset : xarray.Dataset
        A dataset generated by `open_bulk`.
    rowspecs, colspecs : list of 2-tuple
        Tuples containing the ``(name, kwargs)`` passed to ``ClimoAccessor.get``
        used to generate data in rows and columns. See `_parse_specs` for details.
    figtitle, rowlabels, collabels : optional
        The figure settings. The labels are determined automatically from
        the specs but can be overridden in a pinch.
    maxcols : int, optional
        The maximum number of columns. Used only if one of the row or variable
        specs is scalar (in which case there are no row or column labels).
    argskip : int or sequence, optional
        The axes indices to omit from auto color scaling in each group of axes
        that shares the same colorbar. Can be used to let columns saturate.
    gridskip : int of sequence, optional
        The gridspec slots to skip. Can be useful for uneven grids when we
        wish to leave earlier slots empty.
    hcolorbar, hlegend : {'right', 'left', 'bottom', 'top'}
        The location for colorbars or legends annotating horizontal rows.
        Automatically placed along the relevant axes.
    vcolorbar, vlegend : {'bottom', 'top', 'right', 'left'}
        The location for colorbars or legends annotating vertical columns.
        Automatically placed along the relevant axes.
    dcolorbar : {'bottom', 'right', 'top', 'left'}
        The default location for colorbars or legends annotating a mix of
        axes. Placed with the figure colorbar or legend method.
    standardize : bool, optional
        Whether to standardize axis limits to span the same range for all
        plotted content with the same units.
    annotate : bool, optional
        Whether to annotate scatter plots and bar plots with model names
        associated with each point.
    linefit : bool, optional
        Whether to draw best-fit lines for scatter plots of two arbitrary
        variables. Uses `climo.linefit`.
    oneone : bool, optional
        Whether to draw dashed one-one lines for scatter plots of two variables
        with the same units.
    pcolor : bool, optional
        Whether to use `pcolormesh` for the default first two-dimensional plot
        instead of `contourf`. Former is useful for noisy data.
    cycle : cycle-spec, optional
        The color cycle used for line plots. Default is to iterate
        over open-color colors.
    proj : str, optional
        The cartopy projection for longitude-latitude type plots. Default
        is the shape-preserving projection ``'kav7'``.
    proj_kw : dict, optional
        The cartopy projection keyword arguments for longitude-latitude
        type plots. Default is ``{'lon_0': 180}``.
    save : path-like, optional
        The save folder base location. Stored inside a `figures` subfolder.
    **kw_specs
        Passed to `_parse_specs`.
    **kw_method
        Passed to `apply_method`.

    Notes
    -----
    The data resulting from each ``ClimoAccessor.get`` operation must be less
    than 3D. 2D data will be plotted with `pcolor`, then darker contours, then
    lighter contours; 1D data will be plotted with `line`, then on an alternate
    axes (so far just two axes are allowed); and 0D data will omit the average
    or correlation step and plot each model with `scatter` (if both variables
    are defined) or `barh` (if only one model is defined).
    """
    # Initital stuff and figure out geometry
    # TODO: Support e.g. passing 2D arrays to line plotting methods with built-in
    # shadestd, shadepctile, etc. methods instead of using map. See apply_method.
    argskip = np.atleast_1d(() if argskip is None else argskip)
    gridskip = np.atleast_1d(() if gridskip is None else gridskip)
    filespecs, gridspecs, plotspecs = _parse_specs(dataset, rowspecs, colspecs, **kwargs)  # noqa: E501
    nrows, ncols = map(len, gridspecs)
    nrows, ncols = max(nrows, 1), max(ncols, 1)
    titles = (None,) * nrows * ncols
    if nrows == 1 or ncols == 1:
        naxes = gridskip.size + max(nrows, ncols)
        ncols = min(naxes, maxcols)
        nrows = 1 + (naxes - 1) // ncols
        titles = max(gridspecs, key=lambda labels: len(labels))
        gridspecs = (None, None)
    cycle = pplt.Cycle(cycle or ['blue7', 'red7', 'yellow7', 'gray7'])
    colors = pplt.get_colors(cycle)
    kw_annotate = {'fontsize': 'x-small', 'textcoords': 'offset points'}
    kw_contour = {'robust': 96, 'nozero': True, 'linewidth': pplt.rc.metawidth}
    kw_contourf = {'levels': 20, 'extend': 'both'}
    kws_contour = []
    kws_contour.append({'color': 'gray8', 'linestyle': None, **kw_contour})
    kws_contour.append({'color': 'gray3', 'linestyle': ':', **kw_contour})
    kw_bar = {'linewidth': pplt.rc.metawidth, 'edgecolor': 'black', 'width': 1.0}
    kw_line = {'linestyle': '-', 'linewidth': 1.5 * pplt.rc.metawidth}
    kw_scatter = {'color': 'gray7', 'linewidth': 1.5 * pplt.rc.metawidth}
    kw_scatter.update({'marker': 'x', 'markersize': 0.1 * pplt.rc.fontsize ** 2})
    kw_violin = {'means': True, 'boxmarkercolor': 'w'}

    # Iterate over axes and plots
    # NOTE: Critical to disable 'grouping' so that e.g. colorbars or legends that
    # extend into other panel slots are not considered in the tight layout algorithm.
    fig = gs = None  # delay instantiation
    proj = pplt.Proj(proj, **(proj_kw or {'lon_0': 180}))
    methods = []
    commands = {}
    iterator = zip(titles, plotspecs)
    print('Getting data...', end=' ')
    for i in range(nrows * ncols):
        if i in gridskip:
            continue
        print(f'{i + 1}/{nrows * ncols}', end=' ')
        ax = None  # restart the axes
        cmds = []
        aunits = set()
        asizes = set()
        try:
            title, pspecs = next(iterator)
        except StopIteration:
            continue
        for j, (kws_red, kw_all) in enumerate(pspecs):
            # Group added/subtracted reduce instructions into separate dictionaries
            # NOTE: Initial kw_red values are formatted as (('[+-]', value), ...) to
            # permit arbitrary combinations of names and indexers (see _parse_specs).
            kws_method = []  # each item represents a method argument
            for kw in kws_red:
                kw_extra, kw_reduce, scale = {}, {}, 1
                for key, value in kw.items():
                    filt = lambda val: isinstance(val, str) and val == '+'
                    if isinstance(value, tuple) and isinstance(value[0], tuple):
                        kw, count = kw_reduce, sum(filt(_) for vals in value for _ in vals)  # noqa: E501
                    else:
                        kw, count = kw_extra, 1  # e.g. a 'std' or 'pctile' keyword
                    kw[key] = value
                    scale *= count
                kws_persum = []
                for values in itertools.product(*kw_reduce.values()):
                    signs, values = zip(*values)
                    sign = -1 if signs.count('-') % 2 else +1
                    kw = dict(zip(kw_reduce, values))
                    kw.update(kw_extra)
                    kws_persum.append((sign, kw))
                kws_method.append((scale, kws_persum))

            # Reduce along facets dimension and carry out operation
            # TODO: Add other possible reduction methods, e.g. covariance
            # or regressions instead of normalized correlation.
            scales, kws_method = zip(*kws_method)
            if len(set(scales)) > 1:
                raise RuntimeError(f'Mixed reduction scalings {scales}.')
            all_membs, all_projs = True, False
            kws_persum = zip(*kws_method)
            datas_persum = []  # each item represents part of a summation
            methods_persum = set()
            for kws_reduce in kws_persum:
                kw_method = {}
                keys = ('std', 'pctile', 'invert', 'method')
                datas = []
                signs, kws_reduce = zip(*kws_reduce)
                if len(set(signs)) > 1:
                    raise RuntimeError(f'Mixed reduction signs {signs}.')
                for kw_reduce in kws_reduce:  # two for e.g. 'corr', one for e.g. 'avg'
                    for key in tuple(kw_reduce):
                        if key in keys:
                            kw_method.setdefault(key, kw_reduce.pop(key))
                    institute = kw_reduce.get('institute', None)
                    project = getattr(kw_reduce.get('project'), '__name__', None)
                    all_membs = all_membs and institute is None
                    all_projs = all_projs or project in ('cmip', 'cmip5665', 'cmip6556')
                    attrs = kw_all.attrs.copy()
                    data = apply_reduce(dataset, attrs=attrs, **kw_reduce)
                    datas.append(data)
                datas, method = apply_method(*datas, **kw_method)
                if isinstance(datas[-1], dict):
                    *datas, kw = datas
                    for key, val in kw.items():
                        kw_all.plot.setdefault(key, val)
                datas_persum.append((signs[0], datas))  # plotting command arguments
                methods_persum.add(method)
                if len(methods_persum) > 1:
                    raise RuntimeError(f'Mixed reduction methods {methods_persum}.')
            args = []
            signs, datas_persum = zip(*datas_persum)
            for datas in zip(*datas_persum):
                with xr.set_options(keep_attrs=True):
                    arg = sum(sign * data for sign, data in zip(signs, datas))
                    arg = arg / scales[0]
                if isinstance(arg, xr.DataArray):
                    names = (data.name for data in datas if hasattr(data, 'name'))
                    longs = (data.long_name for data in datas if hasattr(data, 'long_name'))  # noqa: E501
                    arg.name = '-'.join(names)
                    arg.attrs['long_name']
                args.append(arg)

            # Instantiate and setup the figure, gridspec, axes
            # NOTE: Here creation is delayed so we can pass arbitrary loose keyword
            # arguments for relevant objects and parse them in _parse_dicts.
            sizes = args[-1].sizes.keys() - {'facets', 'version', 'period'}
            asizes.add(tuple(sorted(sizes)))
            sharex = True if 'lat' in sizes or 'plev' in sizes else 'labels'
            sharey = True if 'lat' in sizes or 'plev' in sizes else 'labels'
            kw_fig = {'sharex': sharex, 'sharey': sharey, 'spanx': False, 'spany': False}  # noqa: E501
            kw_axs = {'title': title}  # possibly none
            dims = ('geo',) if sizes == {'lon', 'lat'} else sizes & {'lat', 'plev'}
            projection = proj if sizes == {'lon', 'lat'} else 'cartesian'
            kw_def = {key: val for dim in dims for key, val in KWARGS_DEFAULT[dim].items()}  # noqa: E501
            kw_fig.update(refwidth=kw_def.pop('refwidth', None))
            kw_axs.update(kw_def)
            for key, value in kw_fig.items():
                kw_all.figure.setdefault(key, value)
            for key, value in kw_axs.items():
                kw_all.axes.setdefault(key, value)
            if fig is None:
                fig = pplt.figure(**kw_all.figure)
            if gs is None:
                gs = pplt.GridSpec(nrows, ncols, **kw_all.gridspec)
            if ax is None:
                ax = jax = fig.add_subplot(gs[i], projection=projection, **kw_all.axes)
            if len(asizes) > 1:
                raise ValueError(f'Conflicting plot types with spatial coordinates {asizes}.')  # noqa: E501
            if hasattr(ax, 'alty') != (projection == 'cartesian'):
                raise ValueError(f'Invalid projection for spatial coordinates {sizes}.')
            cmds.append((jax, args, method, kw_all))  # TODO: use this

            # Apply default plotting command arguments
            # NOTE: This automatically generates alternate axes depending on the
            # units of the datas. Currently works only for 1-dimensional plots.
            nunits = len(aunits)
            units = args[-1].attrs['units']  # avoid numpy coordinates
            aunits.add(units)
            if len(sizes) == 0 and len(pspecs) == 1:
                if all_membs:
                    _, _, filt = _parse_institute(dataset, 'flagship')
                    bools_flags = [filt(facet) for facet in args[-1].facets.values]
                if all_projs:
                    _, _, filt = _parse_project(dataset, 'cmip65')
                    bools_cmip65 = [filt(facet) for facet in args[-1].facets.values]
                    _, _, filt = _parse_project(dataset, 'cmip66')
                    bools_cmip66 = [filt(facet) for facet in args[-1].facets.values]
                if len(args) == 2 and isinstance(args[0], xr.DataArray):
                    command = 'scatter'
                    color = kw_all.plot.get('color', 'gray7')
                    size = (0.5 * pplt.rc['lines.markersize']) ** 2
                    if all_projs:  # faded colors for cmip5 project
                        color = [
                            pplt.set_alpha('gray8', 1 if b66 else 0.6 if b65 else 0.2)
                            for b65, b66 in zip(bools_cmip65, bools_cmip66)
                        ]
                        kw_all.plot.setdefault('color', color)
                    if all_membs:  # larger markers for institution flagships
                        size = [(0.5 * size, 1.5 * size)[b] for b in bools_flags]
                        kw_all.plot.update(sizes=size, absolute_size=True)
                    for key, value in kw_scatter.items():
                        kw_all.plot.setdefault(key, value)
                else:
                    command = 'barh' if horizontal else 'bar'
                    ax.format(**{'ylocator' if horizontal else 'xlocator': 'null'})
                    color = [('blue7', 'red7')[val > 0] for val in args[-1].values]
                    color = ['gray8' if units == 'W m^-2' else c for c in color]
                    color = ['gray8' if units == 'K' else c for c in color]
                    kw_all.plot.setdefault('color', color)
                    if all_projs:  # faded colors for cmip5 project
                        color = [
                            pplt.set_alpha(c, 1 if b66 else 0.6 if b65 else 0.2)
                            for c, b65, b66 in zip(color, bools_cmip65, bools_cmip66)
                        ]
                        edgecolor = [
                            pplt.set_alpha('k', 1 if b66 else 0.6 if b65 else 0.2)
                            for c, b65, b66 in zip(color, bools_cmip65, bools_cmip66)
                        ]
                        linewidth = [
                            pplt.rc.metawidth * (1 if b66 else 1 if b65 else 0.4)
                            for b65, b66 in zip(bools_cmip65, bools_cmip66)
                        ]
                        kw_all.plot.update(color=color, edgecolor=edgecolor, linewidth=linewidth)  # noqa: E501
                    if all_membs:  # hatching for institution flagships
                        hatch = [(None, '//////')[b] for b in bools_flags]
                        kw_all.plot.update(hatch=hatch)
                    for key, value in kw_bar.items():
                        kw_all.plot.setdefault(key, value)
            elif len(sizes) == 0 and len(pspecs) > 1:
                if len(args) == 1 or not isinstance(args[0], xr.DataArray):
                    command = 'violinh' if horizontal else 'violin'
                    color = colors[j % len(colors)]
                    args = (j, args[-1:])
                    for key, value in kw_violin.items():
                        kw_all.plot.setdefault(key, value)
                else:  # comparison of correlations or r-squared values
                    command = 'barh' if horizontal else 'bar'
                    cov, std1, std2 = apply_variance(*args, both=True)
                    corr = cov / (std1 * std2)
                    err = (1 - corr ** 2) / np.sqrt(args[0].size - 2)
                    args = (j, corr)  # TODO: add labels
                    kw_all.plot.setdefault('barstd', err)
                    for key, value in kw_bar.items():
                        kw_all.plot.setdefault(key, value)
            elif len(sizes) == 1:
                if 'plev' in sizes:
                    command = 'linex'
                    color = colors[j % len(colors)]
                    jax = ax
                    if nunits and nunits != len(aunits):
                        jax = ax.altx(color=color)  # noqa: E501
                    else:
                        jax = ax
                    kw_all.plot.setdefault('color', color)
                    for key, value in kw_line.items():
                        kw_all.plot.setdefault(key, value)
                else:
                    command = 'line'
                    color = colors[j % len(colors)]
                    jax = ax
                    if nunits and nunits != len(aunits):
                        jax = ax.alty(color=color)
                    else:
                        jax = ax
                    kw_all.plot.setdefault('color', color)
            elif len(sizes) == 2:
                if 'hatches' in kw_all.plot:
                    command = 'contourf'
                elif j == 0:
                    command = 'pcolormesh' if pcolor else 'contourf'
                    kw_all.plot.setdefault('robust', 98)
                    for key, value in kw_contourf.items():
                        kw_all.plot.setdefault(key, value)
                else:
                    command = 'contour'
                    kw_all.plot.setdefault('labels', True)
                    for key, value in kws_contour[j - 1].items():
                        kw_all.plot.setdefault(key, value)
            else:
                raise ValueError(f'Invalid dimension count {len(sizes)} and sizes {sizes}.')  # noqa: E501

            # Queue the plotting command
            # NOTE: This will automatically allocate separate colorbars for
            # variables with different declared level-restricting arguments.
            args = tuple(args)
            name = '-'.join(arg.name for arg in args if isinstance(arg, xr.DataArray))
            cmap = kw_all.plot.get('cmap', None)
            cmap = tuple(cmap) if isinstance(cmap, list) else cmap
            color = kw_all.plot.get('color', None)
            color = tuple(color) if isinstance(color, list) else color
            key = (name, method, command, cmap, color)
            values = commands.setdefault(key, [])
            values.append((jax, args, kw_all))
            if method not in methods:
                methods.append(method)

    # Carry out the plotting commands
    # NOTE: Axes are always added top-to-bottom and left-to-right so leverage
    # this fact below when selecting axes for legends and colorbars.
    print('\nPlotting data...', end=' ')
    axs_objs = {}
    axs_units = {}  # axes grouped by units
    for k, (key, values) in enumerate(commands.items()):
        # Get guide and plotting arguments
        # NOTE: Here 'argskip' is isued to skip arguments with vastly different
        # ranges when generating levels that annotate multiple different subplots.
        print(f'{k + 1}/{len(commands)}', end=' ')
        name, method, command, cmap, color = key
        axs, args, kw_all = zip(*values)
        kw_cba, kw_leg, kw_plt = {}, {}, {}
        for kw in kw_all:
            kw_cba.update(kw.colorbar)
            kw_leg.update(kw.legend)
            kw_plt.update(kw.plot)
        if command in ('contour', 'contourf', 'pcolormesh'):
            xy = (args[0][-1].coords[dim] for dim in args[0][-1].dims)
            zs = (a for l, arg in enumerate(args) for a in arg if l % ncols not in argskip)  # noqa: E501
            kw_add = {key: kw_plt[key] for key in ('extend',) if key in kw_plt}
            levels, *_, kw_plt = axs[0]._parse_level_vals(*xy, *zs, norm_kw={}, **kw_plt)  # noqa: E501
            kw_plt.update({**kw_add, 'levels': levels})
            kw_plt.pop('robust', None)
        if command in ('contourf', 'pcolormesh') and 'hatches' not in kw_plt:
            guide, kw_guide = 'colorbar', kw_cba
            label = args[0][-1].climo.short_label
            label = re.sub(r' \(', '\n(', label)
            locator = pplt.DiscreteLocator(levels, nbins=7)
            minorlocator = pplt.DiscreteLocator(levels, nbins=7, minor=True)
            kw_guide.setdefault('locator', locator)
            kw_guide.setdefault('minorlocator', minorlocator)  # scaled internally
            kw_guide.setdefault('extendsize', 1.2 + 0.6 * (ax._name != 'cartesian'))
        else:  # TODO: permit short *or* long
            guide, kw_guide = 'legend', kw_leg
            accessor = args[0][-1].climo
            label = accessor.short_label if command == 'contour' else accessor.long_name
            label = None if 'hatches' in kw_plt else label
            keys = ('robust', 'symmetric', 'diverging', 'levels', 'locator', 'extend')
            keys = () if 'contour' in command or 'pcolor' in command else keys
            keys += ('cmap', 'norm', 'norm_kw')
            kw_plt = {key: val for key, val in kw_plt.items() if key not in keys}
            kw_guide.setdefault('ncols', 1)
            kw_guide.setdefault('frame', False)

        # Iterate over axes and arguments
        obj = result = None
        for l, (ax, arg) in enumerate(zip(axs, args)):
            # Add plotted content and formatting
            # NOTE: Bar plots support 'list of' hatch arguments alongside linewidth,
            # edgecolor, and color, even though it is not listed in documentation.
            # TODO: Support hist and hist2d plots in addition to scatter and barh plots
            # (or just hist since, hist2d usually looks ugly with so little data)
            if ax._name == 'cartesian':  # x-axis formatting
                if command in ('bar',):
                    x = None
                elif command in ('barh',):
                    x = arg[-1]
                elif command in ('linex', 'scatter'):
                    x = arg[0]
                else:
                    x = arg[-1].coords[arg[-1].dims[-1]]  # e.g. contour() is y by x
                units = getattr(x, 'units', None)
                axes = axs_units.setdefault(('x', units), [])
                axes.append(ax)
                xlabel = x.climo.short_label if 'units' in getattr(x, 'attrs', {}) else None  # noqa: E501
                if not ax.get_xlabel():
                    rows = ax._get_topmost_axes()._range_subplotspec('y')
                    if ax == ax._get_topmost_axes() or max(rows) == nrows - 1:
                        ax.set_xlabel(xlabel)
            if ax._name == 'cartesian':  # y-axis formatting
                if command in ('barh',):
                    y = None
                elif command in ('bar', 'line', 'scatter'):
                    y = arg[-1]
                else:
                    y = arg[-1].coords[arg[-1].dims[0]]
                units = getattr(y, 'units', None)
                axes = axs_units.setdefault(('y', units), [])
                axes.append(ax)
                ylabel = y.climo.short_label if 'units' in getattr(y, 'attrs', {}) else None  # noqa: E501
                if not ax.get_ylabel():
                    cols = ax._get_topmost_axes()._range_subplotspec('x')
                    if ax == ax._get_topmost_axes() or max(cols) == ncols - 1:
                        ax.set_ylabel(ylabel)
            with warnings.catch_warnings():  # ignore 'masked to nan'
                warnings.simplefilter('ignore', UserWarning)
                result = getattr(ax, command)(*arg, **kw_plt)
                if 'line' in command:  # silent list or tuple
                    obj = result[0][1] if isinstance(result[0], tuple) else result[0]
                elif command == 'contour' and result.collections:
                    obj = result.collections[-1]
                elif command in ('contourf', 'pcolormesh'):
                    obj = result
                else:  # e.g. bar, violin, scatter
                    pass

            # Add other content
            # NOTE: Want to disable autoscaling based on zero line but not currently
            # possible with convenience functions axvline and axhline. Instead use
            # manual plot(). See: https://github.com/matplotlib/matplotlib/issues/14651
            # NOTE: Using set_in_layout False significantly improves appearance since
            # generally don't mind overlapping with tick labels for scatter plots and
            # improves draw time since tight bounding box calculation is expensive.
            if ax == ax._get_topmost_axes() and ('bar' in command or 'line' in command):
                kw = {
                    'color': 'black',
                    'scalex': False,
                    'scaley': False,
                    'linestyle': '-',
                    'linewidth': 1.25 * pplt.rc.metawidth,
                }
                if command in ('bar', 'line', 'vlines'):
                    transform = ax.get_yaxis_transform()
                    ax.plot([0, 1], [0, 0], transform=transform, **kw)
                else:
                    transform = ax.get_xaxis_transform()
                    ax.plot([0, 0], [0, 1], transform=transform, **kw)
            if command == 'scatter':
                if annotate:
                    xlim, ylim = ax.get_xlim(), ax.get_ylim()
                    width, _ = ax._get_size_inches()
                    diff = (pplt.rc.fontsize / 72) * (max(xlim) - min(xlim)) / width
                    xmax = xlim[1] + 5 * diff if ax.get_autoscalex_on() else None
                    ymin = ylim[0] - 1 * diff if ax.get_autoscaley_on() else None
                    ax.format(xmax=xmax, ymin=ymin)  # skip if overridden by user
                    for x, y in zip(*arg):  # iterate over scalar arrays
                        kw = {'ha': 'left', 'va': 'top', **kw_annotate}
                        tup = x.facets.item()  # multi-index is destroyed
                        model = tup[1] if 'CMIP' in tup[0] else tup[0]
                        res = ax.annotate(model, (x.item(), y.item()), (2, -2), **kw)
                        res.set_in_layout(False)
                if oneone:
                    lim = (*ax.get_xlim(), *ax.get_ylim())
                    lim = (min(lim), max(lim))
                    avg = 0.5 * (lim[0] + lim[1])
                    span = lim[1] - lim[0]
                    ones = (avg - 1e3 * span, avg + 1e3 * span)
                    ax.format(xlim=lim, ylim=lim)  # autoscale disabled
                    ax.plot(ones, ones, ls='--', lw=1.5 * pplt.rc.metawidth, color='k')
                if linefit:  # https://en.wikipedia.org/wiki/Simple_linear_regression
                    idx = np.argsort(arg[0].values)
                    x, y = arg[0].values[idx], arg[1].values[idx]
                    slope, stderr, rsquare, fit, lower, upper = var.linefit(x, y, adjust=False)  # noqa: E501
                    rsquare = ureg.Quantity(rsquare.item(), '').to('percent')
                    ax.format(ultitle=rf'$R^2 = {rsquare:~L.1f}$'.replace('%', r'\%'))
                    ax.plot(x, fit, color='r', ls='-', lw=1.5 * pplt.rc.metawidth)
                    ax.area(x, lower, upper, color='r', alpha=0.5 ** 2, lw=0)
            if 'bar' in command or 'lines' in command:
                for container in result:
                    container = container if np.iterable(container) else (container,)
                    for artist in container:
                        artist.sticky_edges.x.clear()
                        artist.sticky_edges.y.clear()
                if annotate:
                    xlim, ylim = ax.get_xlim(), ax.get_ylim()
                    if command == 'bar':
                        height, _ = ax._get_size_inches()
                        diff = (pplt.rc.fontsize / 72) * (max(ylim) - min(ylim)) / height  # noqa: E501
                        along, across = 5 * diff, 0.5 * diff
                        ymin = ylim[0] - along * np.any(arg[-1] < 0)
                        ymax = ylim[1] + along * np.any(arg[-1] > 0)
                        xmin, xmax = xlim[0] - across, xlim[1] + across
                    else:
                        width, _ = ax._get_size_inches()
                        diff = (pplt.rc.fontsize / 72) * (max(xlim) - min(xlim)) / width
                        along, across = 5 * diff, 0.5 * diff
                        xmin = xlim[0] - along * np.any(arg[-1] < 0)
                        xmax = xlim[1] + along * np.any(arg[-1] > 0)
                        ymin, ymax = ylim[0] - across, ylim[1] + across
                    xlim = (xmin, xmax) if ax.get_autoscalex_on() else None
                    ylim = (ymin, ymax) if ax.get_autoscaley_on() else None
                    ax.format(xlim=xlim, ylim=ylim)  # skip if overridden by user
                    for i, a in enumerate(arg[-1]):  # iterate over scalar arrays
                        if command == 'bar':
                            va = 'bottom' if a > 0 else 'top'
                            kw = {'ha': 'center', 'va': va, 'rotation': 90, **kw_annotate}  # noqa: E501
                            xy = (i, a.item())
                            xytext = (0, 2 if a > 0 else -2)
                        else:
                            ha = 'left' if a > 0 else 'right'
                            kw = {'ha': ha, 'va': 'center', **kw_annotate}
                            xy = (a.item(), i)
                            xytext = (2 if a > 0 else -2, 0)
                        tup = a.facets.item()  # multi-index is destroyed
                        model = tup[1] if 'CMIP' in tup[0] else tup[0]
                        res = ax.annotate(model, xy, xytext, **kw)
                        res.set_in_layout(False)

        # Update legend and colorbar queues
        # NOTE: Commands are grouped so that levels can be synchronized between axes
        # and referenced with a single colorbar... but for contour and other legend
        # entries only the unique labels and handle properties matter. So re-group
        # here into objects with unique labels by the rows and columns they span.
        # WARNING: Must record rows and columns here instead of during iteration
        # over legends and colorbars because hidden panels will change index.
        if not obj or not label:
            continue
        if command in ('contourf', 'pcolormesh'):
            key = (name, method, command, cmap, guide, label)
        else:
            key = (command, color, guide, label)
        rows = [n for ax in axs for n in ax._get_topmost_axes()._range_subplotspec('y')]
        cols = [n for ax in axs for n in ax._get_topmost_axes()._range_subplotspec('x')]
        objs = axs_objs.setdefault(key, [])
        objs.append((axs, rows, cols, obj, kw_guide))

    # Add colorbar and legend objects
    # TODO: Should support colorbars spanning multiple columns or rows in the
    # center of the gridspec in addition to figure edges.
    print('\nAdding guides...')
    for key, objs in axs_objs.items():
        *_, guide, label = key
        axs, rows, cols, objs, kws_guide = zip(*objs)
        kw_guide = {key: val for kw in kws_guide for key, val in kw.items()}
        if guide == 'colorbar':
            hori, vert, default = hcolorbar, vcolorbar, dcolorbar
        else:
            hori, vert, default = hlegend, vlegend, dlegend
        axs = [ax for iaxs in axs for ax in iaxs]
        rows = set(n for row in rows for n in row)
        cols = set(n for col in cols for n in col)
        if len(rows) != 1 and len(cols) != 1:
            src = fig
            loc = default
        elif len(rows) == 1:  # single row
            loc = hori
            if loc[0] == 'l':
                src = fig if min(cols) == 0 else axs[0]
            elif loc[0] == 'r':
                src = fig if max(cols) == ncols - 1 else axs[-1]
            elif loc[0] == 't':
                src = fig if min(rows) == 0 else axs[len(axs) // 2]
            elif loc[0] == 'b':
                src = fig if max(rows) == nrows - 1 else axs[len(axs) // 2]
            else:
                raise ValueError(f'Invalid location {loc!r}.')
        else:  # single column
            loc = vert
            if loc[0] == 't':
                src = fig if min(rows) == 0 else axs[0]
            elif loc[0] == 'b':
                src = fig if max(rows) == nrows - 1 else axs[-1]
            elif loc[0] == 'l':
                src = fig if min(cols) == 0 else axs[len(axs) // 2]
            elif loc[0] == 'r':
                src = fig if max(cols) == ncols - 1 else axs[len(axs) // 2]
            else:
                raise ValueError(f'Invalid location {loc!r}.')
        if src is not fig:
            pass
        elif loc[0] in 'lr':
            kw_guide['rows'] = (min(rows) + 1, max(rows) + 1)
        else:
            kw_guide['cols'] = (min(cols) + 1, max(cols) + 1)
        if guide == 'colorbar':
            result = src.colorbar(objs[0], label=label, loc=loc, **kw_guide)
        else:
            result = src.legend(objs[0], label, loc=loc, queue=True, **kw_guide)

    # Format the axes and optionally save
    # NOTE: Here default labels are overwritten with non-none 'rowlabels' or
    # 'collabels', and the file name can be overwritten with 'save'.
    kw = {}
    custom = {'rowlabels': rowlabels, 'collabels': collabels}
    default = dict(zip(('rowlabels', 'collabels'), gridspecs))
    for (key, clabels), (_, dlabels) in zip(custom.items(), default.items()):
        nlabels = nrows if key == 'rowlabels' else ncols
        clabels = clabels or [None] * nlabels
        dlabels = dlabels or [None] * nlabels
        if len(dlabels) != nlabels or len(clabels) != nlabels:
            raise RuntimeError(f'Expected {nlabels} labels but got {len(dlabels)} and {len(clabels)}.')  # noqa: E501
        kw[key] = [clab or dlab for clab, dlab in zip(clabels, dlabels)]
    fig.format(figtitle=figtitle, **kw)
    if gridskip.size:  # kludge to center super title above empty slots
        for i in gridskip:
            ax = fig.add_subplot(gs[i])
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            ax.patch.set_visible(False)
            for spine in ax.spines.values():
                spine.set_visible(False)
    if standardize:
        for (axis, _), axes in axs_units.items():
            lims = [getattr(ax, f'get_{axis}lim')() for ax in axes]
            span = max(abs(lim[1] - lim[0]) for lim in lims)
            avgs = [0.5 * (lim[0] + lim[1]) for lim in lims]
            lims = [(avg - 0.5 * span, avg + 0.5 * span) for avg in avgs]
            for ax, lim in zip(axes, lims):
                getattr(ax, f'set_{axis}lim')(lim)
    if save:
        path = Path(save).expanduser()
        if '.pdf' not in path.name:
            name = '-'.join(methods) + '_'
            name += '_'.join('-'.join(specs) for specs in filespecs if specs)
            path.mkdir(exist_ok=True)
            path = path / 'figures' / f'{name}.pdf'
        print(f'Saving {path.parent}/{path.name}...')
        fig.save(path)
    return fig, fig.subplotgrid
