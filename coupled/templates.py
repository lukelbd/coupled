#!/usr/bin/env python3
"""
Templates for figures detailing coupled model output.
"""
import collections
import itertools
import warnings
from pathlib import Path

import numpy as np
import xarray as xr
from icecream import ic  # noqa: F401

import proplot as pplt
from climopy import decode_units, format_units, ureg, var, vreg  # noqa: F401
from .internals import _apply_method, _parse_institute, _parse_project, get_data
from .internals import _infer_abbrevs, _infer_labels, _infer_newlines


# Figure and axes default keyword arguments
KWARGS_DEFAULT = {
    'fig': {
        'refwidth': 1.5,
        'abcloc': 'left',
        'abc': 'A.',
    },
    'geo': {
        'coast': True,
        'lonlines': 30,
        'latlines': 30,
        'refwidth': 2.3,
        'abcloc': 'upper left',
    },
    'lat': {
        'xlabel': 'latitude',
        'xformatter': 'deg',
        'xlocator': 30,
        'xscale': 'sine',  # or sine scale
        'xlim': (-89, 89),
    },
    'plev': {
        'ylocator': 200,
        'yreverse': True,
        'ylabel': 'pressure (hPa)',
    },
}


def _adjust_bars(ax, data, collection, horizontal=False, annotate=False):
    """
    Adjust and optionally add content to bar plots.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes.
    data : xarray.DataArray
        The original data.
    collection : matplotlib.collection.BarCollection
        The collection.
    horizontal : bool, optional
        Whether the bars were plotted horizontally.
    annotate : bool, optional
        Whether to add annotations to the bars.
    """
    # NOTE: Using set_in_layout False significantly improves appearance since
    # generally don't mind overlapping with tick labels for bar plots and
    # improves draw time since tight bounding box calculation is expensive.
    kw_annotate = {'fontsize': 'x-small', 'textcoords': 'offset points'}
    for container in collection:
        container = container if np.iterable(container) else (container,)
        for artist in container:
            artist.sticky_edges.x.clear()
            artist.sticky_edges.y.clear()
    if annotate:
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        if not horizontal:
            height, _ = ax._get_size_inches()
            diff = (pplt.rc.fontsize / 72) * (max(ylim) - min(ylim)) / height  # noqa: E501
            along, across = 5 * diff, 0.5 * diff
            ymin = ylim[0] - along * np.any(data < 0)
            ymax = ylim[1] + along * np.any(data > 0)
            xmin, xmax = xlim[0] - across, xlim[1] + across
        else:
            width, _ = ax._get_size_inches()
            diff = (pplt.rc.fontsize / 72) * (max(xlim) - min(xlim)) / width
            along, across = 5 * diff, 0.5 * diff
            xmin = xlim[0] - along * np.any(data < 0)
            xmax = xlim[1] + along * np.any(data > 0)
            ymin, ymax = ylim[0] - across, ylim[1] + across
        xlim = (xmin, xmax) if ax.get_autoscalex_on() else None
        ylim = (ymin, ymax) if ax.get_autoscaley_on() else None
        ax.format(xlim=xlim, ylim=ylim)  # skip if overridden by user
        for i, a in enumerate(data):  # iterate over scalar arrays
            if not horizontal:
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


def _adjust_points(ax, data0, data1, oneone=False, linefit=False, annotate=False):
    """
    Adjust and optionally add content to scatter plots.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes.
    data0, data1 : xarray.DataArray
        The original data.
    collection : matplotlib.collection.BarCollection
        The collection.
    oneone : bool, optional
        Whether to add a one-one line.
    linefit : bool, optional
        Whether to add a least-squares fit line.
    annotate : bool, optional
        Whether to add annotations to the bars.
    """
    # NOTE: Using set_in_layout False significantly improves appearance since
    # generally don't mind overlapping with tick labels for scatter plots and
    # improves draw time since tight bounding box calculation is expensive.
    kw_annotate = {'fontsize': 'x-small', 'textcoords': 'offset points'}
    if annotate:
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        width, _ = ax._get_size_inches()
        diff = (pplt.rc.fontsize / 72) * (max(xlim) - min(xlim)) / width
        xmax = xlim[1] + 5 * diff if ax.get_autoscalex_on() else None
        ymin = ylim[0] - 1 * diff if ax.get_autoscaley_on() else None
        ax.format(xmax=xmax, ymin=ymin)  # skip if overridden by user
        for x, y in zip(data0, data1):  # iterate over scalar arrays
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
        x, y = xr.align(data0, data1)
        idx = np.argsort(x.values)
        x, y = x.values[idx], y.values[idx]
        slope, stderr, rsquare, fit, lower, upper = var.linefit(x, y, adjust=False)  # noqa: E501
        sign = '(\N{MINUS SIGN})' if slope < 0 else ''  # point out negative r-squared
        rsquare = ureg.Quantity(rsquare.item(), '').to('percent')
        ax.format(ultitle=rf'$R^2 = {sign}{rsquare:~L.1f}$'.replace('%', r'\%'))
        ax.plot(x, fit, color='r', ls='-', lw=1.5 * pplt.rc.metawidth)
        ax.area(x, lower, upper, color='r', alpha=0.5 ** 2, lw=0)


def _adjust_format(ax, command, *args, refwidth=None):
    """
    Adjust x and y axis labels and possibly add zero lines.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes.
    command : str
        The plotting command.
    *args : array-like
        The plotting arguments.

    Returns
    -------
    xunits, yunits : str
        The associated x and y-axis units.
    """
    # Add zero line
    # NOTE: Want to disable autoscaling based on zero line but not currently
    # possible with convenience functions axvline and axhline. Instead use
    # manual plot(). See: https://github.com/matplotlib/matplotlib/issues/14651
    top = ax._get_topmost_axes()
    args = list(args)
    nrows, ncols, *_ = top.get_subplotspec()._get_geometry()  # custom geometry function
    refwidth = 1.5 * ax.figure._refwidth  # permit overlap
    kw_zero = {
        'color': 'black',
        'scalex': False,
        'scaley': False,
        'zorder': 0.5,
        'linestyle': '-',
        'linewidth': 1.25 * pplt.rc.metawidth,
    }
    if ax == top:
        if command in ('bar', 'box', 'violin', 'line', 'vlines'):
            transform = ax.get_yaxis_transform()
            ax.plot([0, 1], [0, 0], transform=transform, **kw_zero)
        if command in ('barh', 'boxh', 'violinh', 'linex', 'hlines'):
            transform = ax.get_xaxis_transform()
            ax.plot([0, 0], [0, 1], transform=transform, **kw_zero)

    # Handle x and y axis labels
    # TODO: Handle this automatically with climopy and proplot autoformatting
    if command in ('bar', 'box', 'violin'):
        x = None
    elif command in ('barh', 'boxh', 'violinh'):
        x = args[-1]
    elif command in ('linex', 'scatter'):
        x = args[0]
    else:
        x = args[-1].coords[args[-1].dims[-1]]  # e.g. contour() is y by x
    if x is None:
        xunits = None
        ax.format(xlocator='null')
        # if 'bar' in command:
        #     ax.format(xlocator='null')
        # else:
        #     ax.format(xgrid=False, xlocator=1, xtickminor=False, xformatter='null')
    else:
        x = x.copy()
        x.attrs.pop('short_prefix', None)  # ignore legend indicators
        xunits = x.attrs.get('units', None)
        xprefix = x.attrs.get('xlabel_prefix', None)  # keep scatter prefix
        if command == 'scatter' and len(xprefix) < 20:  # see _parse_bulk
            x.attrs['short_prefix'] = xprefix
        xlabel = x.climo.short_label if xunits is not None else None
        if ncols > 1:
            xlabel = _infer_newlines(xlabel, refwidth=refwidth)
        if not ax.get_xlabel():
            rows = top._range_subplotspec('y')
            if ax == top or not ax.figure._sharex or min(rows) == 0:  # topmost axes
                ax.set_xlabel(xlabel)
    if command in ('barh', 'boxh', 'violinh'):
        y = None
    elif command in ('bar', 'box', 'violin', 'line', 'scatter'):
        y = args[-1]
    else:
        y = args[-1].coords[args[-1].dims[0]]
    if y is None:
        yunits = None
        ax.format(ylocator='null')
        # if 'bar' in command:
        #     ax.format(ylocator='null')
        # else:
        #     ax.format(ygrid=False, ylocator=1, ytickminor=False, yformatter='null')
    else:
        y = y.copy()
        y.attrs.pop('short_prefix', None)  # ignore legend indicators
        yunits = y.attrs.get('units', None)
        yprefix = y.attrs.get('ylabel_prefix', None)  # keep scatter prefix
        if command == 'scatter' and len(yprefix) < 20:  # see _parse_bulk
            y.attrs['short_prefix'] = yprefix
        ylabel = y.climo.short_label if yunits is not None else None
        if nrows > 1:
            ylabel = _infer_newlines(ylabel, refwidth=refwidth)
        if not ax.get_ylabel():
            cols = top._range_subplotspec('x')
            if ax == top or not ax.figure._sharey or max(cols) == ncols - 1:
                ax.set_ylabel(ylabel)
    return xunits, yunits


def _infer_command(
    *args,
    ax=None,
    num=None,
    offset=None,
    alternate=False,
    multiple=False,
    horizontal=False,
    projects=False,
    members=False,
    pcolor=False,
    cycle=None,
    label=False,
    **kwargs,
):
    """
    Infer the plotting command from the input arguments and apply settings.

    Parameters
    ----------
    *args : xarray.DataArray
        The plotting arguments.
    ax : matplotlib.axes.Axes, optional
        The axes instance. Possibly turned into a duplicate.
    num : int, optional
        The index in the subplot.
    offset : float, optional
        The optional offset for the coordinate.
    alternate : bool, optional
        Whether to create an alternate axes.
    multiple : bool, optional
        Whether multiple distributions or correlations should be plotted.
    horizontal : bool, optional
        Whether to use horizontal or vertical bars and violins.
    projects : bool, optoinal
        Whether to color-code cmip5 and cmip6 models.
    members : bool, optoinal
        Whether to color-code flagship and non-flagship models.
    pcolor : bool, optional
        Whether to use `pcolormesh` instead of `contourf`.
    cycle : cycle-spec, optional
        The default cycle color-spec.
    label : bool, optional
        Whether to label violin and bar plots.
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The plotting axes.
    command : str
        The plotting command.
    args : tuple
        The possibly modified plotting arguments.
    kwargs : dict
        The possibly modified plotting keyword arguments.
    """
    # Initial stuff
    # NOTE: Bar plots support 'list of' hatch arguments alongside linewidth,
    # edgecolor, and color, even though it is not listed in documentation.
    def _ints_member(data):
        ints = []
        filt_flag = _parse_institute(data, 'flagship')
        for facet in args[-1].facets.values:
            int_ = int(filt_flag(facet))
            ints.append(int_)
        return ints
    def _ints_project(data):  # noqa: E301
        ints = []
        filt_cmip65 = _parse_project(data, 'cmip65')
        filt_cmip66 = _parse_project(data, 'cmip66')
        for facet in args[-1].facets.values:
            int_ = 2 if filt_cmip66(facet) else 1 if filt_cmip65(facet) else 0
            ints.append(int_)
        return ints
    args = list(args)
    cycle = pplt.Cycle(cycle or ['blue7', 'red7', 'yellow7', 'gray7'])
    colors = pplt.get_colors(cycle)
    flier = {'markersize': 2, 'marker': 'x'}
    kw_contour = {'robust': 96, 'nozero': True, 'linewidth': pplt.rc.metawidth}
    kw_contourf = {'robust': 98, 'levels': 20, 'extend': 'both'}
    kws_contour = []
    kws_contour.append({'color': 'gray8', 'linestyle': None, **kw_contour})
    kws_contour.append({'color': 'gray3', 'linestyle': ':', **kw_contour})
    kw_bar = {'linewidth': pplt.rc.metawidth, 'edgecolor': 'black', 'width': 1.0}
    kw_line = {'linestyle': '-', 'linewidth': 1.5 * pplt.rc.metawidth}
    kw_scatter = {'color': 'gray7', 'linewidth': 1.5 * pplt.rc.metawidth}
    kw_scatter.update({'marker': 'x', 'markersize': 0.1 * pplt.rc.fontsize ** 2})
    kw_box = {'means': True, 'whis': (5, 95), 'widths': 0.33, 'flierprops': flier}
    sizes = args[-1].sizes.keys() - {'facets', 'version', 'period'}

    # Infer commands
    # TODO: Support hist and hist2d perlots in addition to scatter and barh plots
    # (or just hist since, hist2d usually looks ugly with so little data)
    num = num or 0
    offset = offset or 0
    defaults = {}
    if len(sizes) == 2:
        if 'hatches' in kwargs:
            command = 'contourf'
        elif num == 0:
            command = 'pcolormesh' if pcolor else 'contourf'
            defaults = kw_contourf.copy()
        else:
            command = 'contour'
            defaults = {'labels': True, **kws_contour[num - 1]}
    elif len(sizes) == 1:
        if 'plev' in sizes:
            command = 'linex'
            color0 = colors[max(num - 1, 0) % len(colors)]
            color1 = colors[num % len(colors)]
            defaults = {'color': color1, **kw_line}
            if alternate:  # TODO: handle parent axis color inside plot_bulk()
                ax.format(xcolor=color0)
                ax = ax.altx(color=color1)
        else:
            command = 'line'
            color0 = colors[max(num - 1, 0) % len(colors)]
            color1 = colors[num % len(colors)]
            defaults = {'color': color1, **kw_line}
            if alternate:  # TODO: handle parent axis color inside plot_bulk()
                ax.format(ycolor=color0)
                ax = ax.alty(color=color1)
    elif len(sizes) == 0 and multiple:
        if len(args) == 2:  # TODO: stack regression coefficient and r-squared values
            command = 'barh' if horizontal else 'bar'
            project = args[0].project.values[0]
            (slope,), *_ = _apply_method(*args, method='slope')
            short_prefix = project
            if hasattr(args[0], 'short_prefix') and hasattr(args[1], 'short_prefix'):
                short_prefix += f' {args[1].short_prefix} vs. {args[0].short_prefix}'
            slope.attrs['short_prefix'] = short_prefix
            args = (np.array([num - offset]), slope)  # TODO: add error bars
            color = colors[num % len(colors)]
            defaults = {'color': color, **kw_bar}
        else:  # compare distributions
            # command = 'violinh' if horizontal else 'violin'
            command = 'boxh' if horizontal else 'box'
            args = (np.array([num - offset]), args[-1].expand_dims('num', axis=1))
            color = colors[num % len(colors)]
            defaults = {'facecolor': color, **kw_box}
        data = args[-1].copy()
        data.attrs['units'] = ''
        string = data.climo.short_label
        string = _infer_newlines(string.replace('\n', ' '), 25 * pplt.rc.fontsize / 72)
        if label:
            if horizontal:
                trans = ax.get_yaxis_transform()
                align = {'ha': 'right', 'va': 'center', 'rotation': 0}
                ax.text(-0.05, num - offset, string, transform=trans, **align)
            else:
                trans = ax.get_xaxis_transform()
                align = {'ha': 'center', 'va': 'top', 'rotation': 90}
                ax.text(num - offset, -0.05, string, transform=trans, **align)
    elif len(sizes) == 0:
        if len(args) == 2:  # plot correlation
            # TODO: repair this... currently sizes will conflict
            command = 'scatter'
            color = kwargs.get('color', 'gray7')
            size = (0.5 * pplt.rc['lines.markersize']) ** 2
            defaults = kw_scatter.copy()
            if members:  # larger markers for institution flagships
                ints = _ints_member(args[-1])
                size = [(0.5 * size, 1.5 * size)[i] for i in ints]
                defaults.update(sizes=size, absolute_size=True)
            if projects:  # faded colors for cmip5 project
                ints = _ints_project(args[-1])
                color = [pplt.set_alpha('k', (0.2, 0.6, 1)[i]) for i in ints]
                defaults.setdefault('color', color)
        else:  # plot distribution
            command = 'barh' if horizontal else 'bar'
            ascending = False if horizontal else True
            data = args[-1]
            name = data.name
            args.insert(0, np.arange(data.size))
            args[-1] = data.sortby(data, ascending=ascending)
            color = [('blue7', 'red7')[val > 0] for val in args[-1].values]
            color = ['gray8' if name == 'ecs_rfnt' else c for c in color]
            color = ['gray8' if name == 'erf_rfnt' else c for c in color]
            defaults = kw_bar.copy()
            defaults['color'] = color
            if members:  # hatching for institution flagships
                ints = _ints_member(args[-1])
                hatch = [(None, '//////')[i] for i in ints]
                defaults.update(hatch=hatch)
            if projects:  # faded colors for cmip5 project
                ints = _ints_project(args[-1])
                color = [pplt.set_alpha(c, (0.2, 0.6, 1)[i]) for c, i in zip(color, ints)]  # noqa: E501
                edgecolor = [pplt.set_alpha('k', (0.2, 0.6, 1)[i]) for i in ints]
                linewidth = [pplt.rc.metawidth * (0.5, 1, 1)[i] for i in ints]
                defaults.update(color=color, edgecolor=edgecolor, linewidth=linewidth)
    else:
        raise ValueError(f'Invalid dimension count {len(sizes)} and sizes {sizes}.')  # noqa: E501
    for key, value in defaults.items():
        kwargs.setdefault(key, value)
    return ax, command, args, kwargs


def _parse_item(dataset, spec, **kwargs):
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
        other specifications (i.e. the row-column framework; see `_parse_bulk`).
    **kwargs
        Additional keyword arguments, used as defaults for the unset keys
        in the variable specifications.

    Returns
    -------
    kw_dat : dict
        The indexers used to reduce the data variable with `.reduce`. This is
        parsed specially compared to other keywords, and its keys are restricted
        to ``'name'`` and any coordinate or multi-index names.
    kw_plt : namedtuple of dict
        A named tuple containing keyword arguments for different plotting-related
        commands. The tuple fields are as follows:

          * ``figure``: Passed to `Figure` when the figure is instantiated.
          * ``gridspec``: Passed to `GridSpec` when the gridfspec is instantiated.
          * ``axes``: Passed to `.format` for cartesian or geographic formatting.
          * ``colorbar``: Passed to `.colorbar` for scalar mappable outputs.
          * ``legend``: Passed to `.legend` for other artist outputs.
          * ``command``: Passed to the plotting command (the default field).
          * ``attrs``: Added to `.attrs` for use in resulting plot labels.
    """
    # NOTE: For subsequent processing we put the variables being combined (usually
    # just one) inside the 'name' key in kw_red (here `short` is shortened relative
    # to actual dataset names and intended for file names only). This helps when
    # merging variable specifications between row and column specifications and
    # between tuple-style specifications (see _parse_bulk).
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
    kw = {**kwargs, **kw}  # copy
    alt = kw.pop('name', None)
    name = name or alt  # see below
    kw_dat, kw_att = {}, {}
    kw_fig, kw_grd, kw_axs = {}, {}, {}
    kw_cmd, kw_cba, kw_leg = {}, {}, {}
    keys = ('space', 'ratio', 'group', 'equal', 'left', 'right', 'bottom', 'top')
    att_detect = ('short', 'long', 'standard', 'xlabel_', 'ylabel_')
    fig_detect = ('fig', 'ref', 'space', 'share', 'span', 'align')
    grd_detect = tuple(s + key for key in keys for s in ('w', 'h', ''))
    axs_detect = ('x', 'y', 'lon', 'lat', 'abc', 'title', 'coast')
    bar_detect = ('extend', 'tick', 'locator', 'formatter', 'minor', 'label')
    leg_detect = ('ncol', 'order', 'frame', 'handle', 'border', 'column')
    for key, value in kw.items():  # NOTE: sorting performed in _parse_labels
        if key in options:
            kw_dat[key] = value  # e.g. for averaging
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
            kw_cmd[key] = value
    if isinstance(name, str):
        kw_dat['name'] = name  # always place last for gridspec labels
    keys = ('method', 'std', 'pctile', 'invert')
    kw_dat.update({key: kwargs.pop(key) for key in keys if key in kwargs})
    fields = ('figure', 'gridspec', 'axes', 'command', 'attrs', 'colorbar', 'legend')
    kwargs = collections.namedtuple('kwargs', fields)
    kw_plt = kwargs(kw_fig, kw_grd, kw_axs, kw_cmd, kw_att, kw_cba, kw_leg)
    return kw_dat, kw_plt


def _parse_bulk(dataset, rowspecs=None, colspecs=None, **kwargs):
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
    dataspecs : list of list of tuple of dict
        The reduction keyword argument specifications.
    plotspecs : list of list of kwargs
        The keyword arguments used for plotting.
    figlabel : str, optional
        The default figure title.
    gridlabels : list of list of str
        The default row and column labels.
    pathlabels : list of list of str
        The default strings for file naming.
    """
    # Parse variable specs per gridspec row or column and per subplot
    # NOTE: This permits sharing keywords across each group with trailing dicts
    # in either the primary gridspec list or any of the subplot sub-lists.
    # NOTE: The two arrays required for two-argument methods can be indicated with
    # either 2-tuples in spec lists or conflicting row and column names or reductions.
    refwidth = None
    dataspecs, plotspecs, gridlabels, pathlabels = [], [], [], []
    for inspecs in (rowspecs, colspecs):
        datspecs, pltspecs = [], []
        if not isinstance(inspecs, list):
            inspecs = [inspecs]
        for ispecs in inspecs:  # specs per figure
            dspecs, pspecs = [], []
            if not isinstance(ispecs, list):
                ispecs = [ispecs]
            for ispec in ispecs:  # specs per subplot
                dspec, pspec = [], []
                if ispec is None:
                    ispec = (None,)  # possibly construct from keyword args
                elif isinstance(ispec, (str, dict)):
                    ispec = (ispec,)
                elif len(ispec) != 2:
                    raise ValueError(f'Invalid variable specs {ispec}.')
                elif type(ispec[0]) != type(ispec[1]):  # noqa: E721  # i.e. (str, dict)
                    ispec = (ispec,)
                else:
                    ispec = tuple(ispec)
                for spec in ispec:  # specs per correlation pair
                    kw_dat, kw_plt = _parse_item(dataset, spec, **kwargs)
                    if value := kw_plt.figure.get('refwidth', None):
                        refwidth = value
                    dspec.append(kw_dat)
                    pspec.append(kw_plt)
                dspecs.append(tuple(dspec))  # tuple to identify as correlation-pair
                pspecs.append(tuple(pspec))
            datspecs.append(dspecs)
            pltspecs.append(pspecs)
        zerospecs = [dspecs[0] if dspecs else {} for dspecs in datspecs]
        pthlabels = _infer_abbrevs(dataset, *zerospecs, identical=False)
        grdlabels = _infer_labels(
            dataset,
            *zerospecs,
            identical=False,
            long_names=True,
            title_case=True,
            refwidth=refwidth,
        )
        gridlabels.append(grdlabels)
        pathlabels.append(pthlabels)
        dataspecs.append(datspecs)
        plotspecs.append(pltspecs)

    # Combine row and column specifications for plotting and file naming
    # NOTE: Several plotted values per subplot can be indicated in either the
    # row or column list, and the specs from the other list are repeated below.
    # WARNING: Critical to make copies of dictionaries or create new ones
    # here since itertools product repeats the same spec multiple times.
    bothspecs = [
        [list(zip(dspecs, pspecs)) for dspecs, pspecs in zip(datspecs, pltspecs)]
        for datspecs, pltspecs in zip(dataspecs, plotspecs)
    ]
    kw_infer = dict(
        keeppairs=False,
        identical=False,
        skip_area=True,
        skip_name=True,
        long_names=False,
        title_case=False
    )
    dataspecs, plotspecs = [], []
    for (i, rspecs), (j, cspecs) in itertools.product(*map(enumerate, bothspecs)):
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
        dspecs, pspecs = [], []
        for k, (rspec, cspec) in enumerate(zip(rspecs, cspecs)):  # subplot entries
            rkws_dat, rkws_plt, ckws_dat, ckws_plt = *rspec, *cspec
            kwargs = type((rkws_plt or ckws_plt)[0])
            kws = []
            for field in kwargs._fields:
                kw = {}  # NOTE: previously applied default values here
                for ikws_plt in (rkws_plt, ckws_plt):
                    for ikw_plt in ikws_plt:  # correlation pairs
                        for key, value in getattr(ikw_plt, field).items():
                            kw.setdefault(key, value)  # prefer row entries
                kws.append(kw)
            kw_plt = kwargs(*kws)
            rkws_dat = tuple(kw.copy() for kw in rkws_dat)  # NOTE: copy is critical
            ckws_dat = tuple(kw.copy() for kw in ckws_dat)
            for ikws_dat, jkws_dat in ((rkws_dat, ckws_dat), (ckws_dat, rkws_dat)):
                for key in ikws_dat[0]:
                    if key not in ikws_dat[-1]:
                        continue
                    if ikws_dat[0][key] == ikws_dat[-1][key]:
                        for kw in jkws_dat:  # possible correlation pair
                            kw.setdefault(key, ikws_dat[0][key])
            kws_dat = {}  # filter unique specifications
            for kw_dat in (*rkws_dat, *ckws_dat):
                keys = ('method', 'std', 'pctile', 'invert')
                keys = sorted(key for key in kw_dat if key not in keys)
                keyval = tuple((key, kw_dat[key]) for key in keys)
                if tuple(keyval) in kws_dat:
                    continue
                others = tuple(other for other in kws_dat if set(keyval) < set(other))
                if others:
                    continue
                others = tuple(other for other in kws_dat if set(other) < set(keyval))
                for other in others:  # prefer more selection keywords
                    kws_dat.pop(other)
                kws_dat[tuple(keyval)] = kw_dat
            kws_dat = tuple(kws_dat.values())  # possible correlation tuple
            if len(kws_dat) > 2:
                raise RuntimeError(
                    'Expected 1-2 specs for combining with get_data but got '
                    f'{len(kws_dat)} specs: ' + '\n'.join(map(repr, kws_dat))
                )
            dspecs.append(kws_dat)
            pspecs.append(kw_plt)
        labels = _infer_labels(dataset, *dspecs, **kw_infer)
        xlabels = _infer_labels(dataset, *(kws[:1] for kws in dspecs), **kw_infer)
        ylabels = _infer_labels(dataset, *(kws[1:] for kws in dspecs), **kw_infer)
        ylabels = ylabels or [None] * len(xlabels)
        for label, xlabel, ylabel, pspec in zip(labels, xlabels, ylabels, pspecs):
            if label:  # use identical label as fallback
                pspec.attrs.setdefault('short_prefix', label)
            if xlabel:
                pspec.attrs.setdefault('xlabel_prefix', xlabel)
            if ylabel:
                pspec.attrs.setdefault('ylabel_prefix', ylabel)
        dataspecs.append(dspecs)
        plotspecs.append(pspecs)
    ncols = len(colspecs) if len(colspecs) > 1 else len(rowspecs) if len(rowspecs) > 1 else 4  # noqa: E501
    refwidth = refwidth or pplt.rc['subplots.refwidth']
    zerospecs = [dspecs[0] if dspecs else {} for dspecs in dataspecs]
    pthlabels = _infer_abbrevs(dataset, *zerospecs, identical=True)
    pathlabels.insert(0, pthlabels)  # insert reductions shared across rows-columns
    subspecs = [dspec for dspecs in dataspecs for dspec in dspecs]
    figlabels = _infer_labels(
        dataset,
        *subspecs,
        identical=True,
        long_names=True,
        title_case='first',  # special value
        refwidth=2 * ncols * refwidth,  # larger value
    )
    figlabel = ' '.join(figlabels)
    return dataspecs, plotspecs, figlabel, gridlabels, pathlabels


def plot_bulk(
    dataset,
    rowspecs=None,
    colspecs=None,
    maxcols=None,
    argskip=None,
    figtitle=None,
    figprefix=None,
    figsuffix=None,
    gridskip=None,
    rowlabels=None,
    collabels=None,
    hcolorbar='right',
    vcolorbar='bottom',
    dcolorbar='bottom',
    hlegend='bottom',
    vlegend='bottom',
    dlegend='bottom',
    standardize=False,
    annotate=False,
    linefit=False,
    oneone=False,
    proj='eqearth',
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
        used to generate data in rows and columns. See `_parse_bulk` for details.
    figtitle, rowlabels, collabels : optional
        The figure settings. The labels are determined automatically from
        the specs but can be overridden in a pinch.
    figprefix, figsuffix : str, optional
        Optional modifications to the default figure title determined
        from shared reduction instructions.
    maxcols : int, optional
        The default number of columns. Used only if one of the row or variable
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
    cycle, pcolor, horizontal : optional
        Passed to `_infer_command`.
    **kw_specs
        Passed to `_parse_bulk`.
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
    dataspecs, plotspecs, figlabel, gridlabels, pathlabels = _parse_bulk(
        dataset, rowspecs, colspecs, **kwargs
    )
    figtitle = figtitle or figlabel
    nrows, ncols = map(len, gridlabels)
    nrows, ncols = max(nrows, 1), max(ncols, 1)
    titles = (None,) * nrows * ncols
    if figsuffix:
        figtitle = ' '.join((figtitle, figsuffix))
    if figprefix:
        end = figtitle if figtitle[:2].isupper() else figtitle[0].lower() + figtitle[1:]  # noqa: E501
        start = figprefix if figprefix[:1].isupper() else figprefix[0].upper() + figprefix[1:]  # noqa: E501
        figtitle = ' '.join((start, end))
    if nrows == 1 or ncols == 1:
        naxes = gridskip.size + max(nrows, ncols)
        ncols = min(naxes, maxcols or 4)
        nrows = 1 + (naxes - 1) // ncols
        titles = max(gridlabels, key=lambda labels: len(labels))
        titles = titles or (None,) * nrows * ncols
        gridlabels = (None, None)

    # Iterate over axes and plots
    # NOTE: Critical to disable 'grouping' so that e.g. colorbars or legends that
    # extend into other panel slots are not considered in the tight layout algorithm.
    fig = gs = None  # delay instantiation
    proj = pplt.Proj(proj, **(proj_kw or {'lon_0': 180}))
    methods = []
    commands = {}
    iterator = zip(titles, dataspecs, plotspecs)
    print('Getting data...', end=' ')
    for i in range(nrows * ncols):
        if i in gridskip:
            continue
        print(f'{i + 1}/{nrows * ncols}', end=' ')
        ax = None  # restart the axes
        aunits = set()
        asizes = set()
        try:
            title, dspecs, pspecs = next(iterator)
        except StopIteration:
            continue
        for j, (kws_dat, kw_plt) in enumerate(zip(dspecs, pspecs)):
            # Generate the data arrays and plotting keyword arguments
            # NOTE: Here `kws_dat` should be singleton tuple or several tuples
            # for taking e.g. correlations or regressions of two quantities.
            attrs = kw_plt.attrs.copy()
            args, method, default = get_data(dataset, *kws_dat, attrs=attrs)
            for key, value in default.items():
                kw_plt.command.setdefault(key, value)
            sizes = args[-1].sizes.keys() - {'facets', 'version', 'period'}
            asizes.add(tuple(sorted(sizes)))
            sharex = True if 'lat' in sizes or 'plev' in sizes else 'labels'
            sharey = True if 'lat' in sizes or 'plev' in sizes else 'labels'
            kw_fig = {'sharex': sharex, 'sharey': sharey, 'spanx': False, 'spany': False}  # noqa: E501
            kw_axs = {'title': title}  # possibly none
            defaults = ['fig']  # default kwargs
            if sizes == {'lon', 'lat'}:
                projection = proj
                defaults.append('geo')
            else:
                projection = 'cartesian'
                defaults.extend(sizes & {'lat', 'plev'})
            kw_def = {key: val for k in defaults for key, val in KWARGS_DEFAULT[k].items()}  # noqa: E501
            kw_fig.update(refwidth=kw_def.pop('refwidth', None))
            kw_axs.update(kw_def)
            for key, value in kw_fig.items():
                kw_plt.figure.setdefault(key, value)
            for key, value in kw_axs.items():
                kw_plt.axes.setdefault(key, value)

            # Instantiate objects and infer the plotting command
            # NOTE: Delay object creation until this block so we can pass arbitrary
            # loose keyword arguments and simply parse them in _parse_item.
            if fig is None:
                fig = pplt.figure(**kw_plt.figure)
            if gs is None:
                gs = pplt.GridSpec(nrows, ncols, **kw_plt.gridspec)
            if ax is None:
                ax = jax = fig.add_subplot(gs[i], projection=projection, **kw_plt.axes)
            if len(asizes) > 1:
                raise ValueError(f'Conflicting plot types with spatial coordinates {asizes}.')  # noqa: E501
            if hasattr(ax, 'alty') != (projection == 'cartesian'):
                raise ValueError(f'Invalid projection for spatial coordinates {sizes}.')
            nunits = len(aunits)
            aunits.add(args[-1].attrs['units'])
            kw = kw_plt.command.copy()
            horizontal = kw.get('horizontal', False)
            kw['alternate'] = nunits and nunits != len(aunits)
            kw['multiple'] = len(pspecs) > 1
            kw['label'] = i % ncols == 0 if horizontal else i >= (nrows - 1) * ncols
            jax, command, args, kw = _infer_command(*args, num=j, ax=ax, **kw)
            kw_plt.command.clear()
            kw_plt.command.update(kw)

            # Queue the plotting command
            # NOTE: This will automatically allocate separate colorbars for
            # variables with different declared level-restricting arguments.
            args = tuple(args)
            name = '_'.join(arg.name for arg in args if isinstance(arg, xr.DataArray))
            cmap = kw_plt.command.get('cmap', None)
            cmap = tuple(cmap) if isinstance(cmap, list) else cmap
            color = kw_plt.command.get('color', None)
            color = tuple(color) if isinstance(color, list) else color
            size = kw_plt.command.get('sizes', None)
            size = tuple(size) if isinstance(size, list) else size
            key = (name, method, command, cmap, color, size)
            values = commands.setdefault(key, [])
            values.append((jax, args, kw_plt))
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
        name, method, command, cmap, color, *_ = key
        axs, args, kws_plt = zip(*values)
        kw_cba = {key: val for kw_plt in kws_plt for key, val in kw_plt.colorbar.items()}  # noqa: E501
        kw_leg = {key: val for kw_plt in kws_plt for key, val in kw_plt.legend.items()}  # noqa: E501
        kws_cmd = [kw_plt.command.copy() for kw_plt in kws_plt]
        hatches = any('hatches' in kw for kw in kws_cmd)
        if command in ('contour', 'contourf', 'pcolormesh'):
            xy = (args[0][-1].coords[dim] for dim in args[0][-1].dims)
            zs = (a for l, arg in enumerate(args) for a in arg if l % ncols not in argskip)  # noqa: E501
            kw = {key: val for kw_plt in kws_plt for key, val in kw_plt.command.items()}
            keep = {key: kw[key] for key in ('extend',) if key in kw}
            levels, *_, kw = axs[0]._parse_level_vals(*xy, *zs, norm_kw={}, **kw)  # noqa: E501
            kw.update({**keep, 'levels': levels})
            kw.pop('robust', None)
            kws_cmd = [kw] * len(args)
        if not hatches and command in ('contourf', 'pcolormesh'):
            guide, kw_guide = 'colorbar', kw_cba
            label = args[0][-1].climo.short_label
            label = _infer_newlines(label, refwidth=ax.figure._refwidth, scale=0.8)
            locator = pplt.DiscreteLocator(levels, nbins=7)
            minorlocator = pplt.DiscreteLocator(levels, nbins=7, minor=True)
            kw_guide.setdefault('locator', locator)
            kw_guide.setdefault('minorlocator', minorlocator)  # scaled internally
            kw_guide.setdefault('extendsize', 1.2 + 0.6 * (ax._name != 'cartesian'))
        else:  # TODO: permit short *or* long
            guide, kw_guide = 'legend', kw_leg
            variable = args[0][-1].climo.cfvariable
            label = variable.short_label if command == 'contour' else variable.short_name  # noqa: E501
            label = None if hatches else label
            keys = ['cmap', 'norm', 'norm_kw']
            if 'contour' not in command and 'pcolor' not in command:
                keys.extend(('robust', 'symmetric', 'diverging', 'levels', 'locator', 'extend'))  # noqa: E501
            kw_guide.setdefault('ncols', 1)
            kw_guide.setdefault('frame', False)
            kws_cmd, kws = [], kws_cmd
            for kw in kws:
                kws_cmd.append({key: val for key, val in kw.items() if key not in keys})

        # Add plotted content and queue guide instructions
        # NOTE: Commands are grouped so that levels can be synchronized between axes
        # and referenced with a single colorbar... but for contour and other legend
        # entries only the unique labels and handle properties matter. So re-group
        # here into objects with unique labels by the rows and columns they span.
        # WARNING: Must record rows and columns here instead of during iteration
        # over legends and colorbars because hidden panels will change index.
        obj = result = None
        for l, (ax, dats, kw_cmd) in enumerate(zip(axs, args, kws_cmd)):
            cmd = getattr(ax, command)
            vals = dats
            if command in ('bar', 'barh', 'box', 'boxh', 'violin', 'violinh'):
                vals = [da.values if isinstance(da, xr.DataArray) else da for da in dats]  # noqa: E501
            with warnings.catch_warnings():  # ignore 'masked to nan'
                warnings.simplefilter('ignore', UserWarning)
                result = cmd(*vals, **kw_cmd)
            if 'line' in command:  # silent list or tuple
                obj = result[0][1] if isinstance(result[0], tuple) else result[0]
            elif command == 'contour' and result.collections:
                obj = result.collections[-1]
            elif command in ('contourf', 'pcolormesh'):
                obj = result
            if ax._name == 'cartesian':
                xunits, yunits = _adjust_format(ax, command, *dats)
                axs_units.setdefault(('x', xunits), []).append(ax)
                axs_units.setdefault(('y', yunits), []).append(ax)
            if 'bar' in command or 'lines' in command:
                kw = dict(horizontal=(command == 'barh'), annotate=annotate)
                _adjust_bars(ax, dats[1], result, **kw)
            if command == 'scatter':
                kw = dict(annotate=annotate, oneone=oneone, linefit=linefit)
                _adjust_points(ax, *dats, **kw)
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
    handles = {}
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
            handles.setdefault(src, []).append((objs[0], label, loc, kw_guide))
    for src, tups in handles.items():  # support 'queue' for figure legends
        objs, labels, locs, kws = zip(*tups)
        kw = {key: val for kw in kws for key, val in kw.items()}
        src.legend(list(objs), list(labels), loc=locs[0], **kw)

    # Format the axes and optionally save
    # NOTE: Here default labels are overwritten with non-none 'rowlabels' or
    # 'collabels', and the file name can be overwritten with 'save'.
    kw = {}
    custom = {'rowlabels': rowlabels, 'collabels': collabels}
    default = {'rowlabels': gridlabels[0], 'collabels': gridlabels[1]}
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
        if save is True:
            path = Path()  # current directory
        else:
            path = Path(save).expanduser()
        if '.pdf' not in path.name:
            path = path / 'figures'
            path.mkdir(exist_ok=True)
            name = '-'.join(methods) + '_'
            name += '_'.join('-'.join(labs) for labs in pathlabels if labs)
            path = path / f'{name}.pdf'
        print(f'Saving {path.parent}/{path.name}...')
        fig.save(path)
    return fig, fig.subplotgrid
