#!/usr/bin/env python3
"""
Templates for figures detailing coupled model output.
"""
import copy
import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from icecream import ic  # noqa: F401

import proplot as pplt
from climopy import decode_units, format_units, ureg, vreg  # noqa: F401
from .process import _components_slope, _parse_institute, _parse_project, get_data
from .internals import _infer_labels, _wrap_label, parse_specs

pplt.rc.load(Path(__file__).parent.parent / 'proplotrc')

__all__ = ['breakdown_feedbacks', 'breakdown_transport', 'create_plot']


# Default format keyword arguments
KWARGS_FIG = {
    'refwidth': 1.5,
    'abcloc': 'left',
    'abc': 'A.',
}
KWARGS_GEO = {
    'proj': 'hammer',
    'proj_kw': {'lon_0': 180},
    'lonlines': 30,
    'latlines': 30,
    'refwidth': 2.3,
    'abcloc': 'upper left',
    'coast': True,
}
KWARGS_LAT = {
    'xlabel': 'latitude',
    'xformatter': 'deg',
    'xlocator': 30,
    'xscale': 'sine',  # or sine scale
    'xlim': (-89, 89),
}
KWARGS_PLEV = {
    'yminorlocator': 100,
    'ylocator': 200,
    'yreverse': True,
    'ylabel': 'pressure (hPa)',
}

# Default plotting keyword arguments
KWARGS_BAR = {
    'color': 'gray8',
    'edgecolor': 'black',
    'width': 1.0,
    'linewidth': pplt.rc.metawidth,
}
KWARGS_BOX = {
    'means': True,
    'whis': (5, 95),
    'widths': 0.7,
    'flierprops': {'markersize': 2, 'marker': 'x'},
}
KWARGS_CONTOUR = {
    'globe': True,
    'robust': 96,
    'nozero': True,
    'linewidth': pplt.rc.metawidth,
}
KWARGS_HASH = {
    'globe': True,
    'colors': 'none',
    'edgecolor': 'black'
}
KWARGS_LINE = {
    'linestyle': '-',
    'linewidth': 2.5 * pplt.rc.metawidth,
}
KWARGS_SCATTER = {
    'color': 'gray8',
    'linewidth': 1.5 * pplt.rc.metawidth,
    'marker': 'x',
    'markersize': 0.1 * pplt.rc.fontsize ** 2
}
KWARGS_SHADING = {
    'globe': True,
    'robust': 98,
    'levels': 20,
    'extend': 'both',
}
KWARGS_ZERO = {
    'color': 'black',
    'scalex': False,
    'scaley': False,
    'zorder': 0.5,
    'linestyle': '-',
    'linewidth': 1.25 * pplt.rc.metawidth,
}


def _get_colors(cycle=None, shade=None):
    """
    Get colors for individual or multiple plotting commands.

    Parameters
    ----------
    cycle : cycle-spec, optional
        The color cycle.
    shade : int, optional
        The shading to use.

    Returns
    -------
    colors : list
        The resulting colors.
    """
    shade = shade or 7
    colors = ['blue', 'red', 'gray', 'yellow', 'cyan', 'pink', 'orange', 'indigo']
    colors = [f'{color}{shade}' for color in cycle]
    colors = pplt.get_colors(cycle or colors)
    return colors


def _get_command(args, kw_plt, shading=True, contour=None):
    """
    Infer the single plotting command from the input arguments.

    Parameters
    ----------
    args : xarray.DataArray
        Input arguments.
    kw_plt : namedtuple
        Input keyword arguments.
    shading : bool, optional
        Whether to use shading or contours for 2D data.
    contour : dict, optional
        Additional contour properties for 2D data.

    Other Parameters
    ----------------
    shade : int, optional
        The shading for the default color cycle (taken from `.other`).
    cycle : cycle-spec, optional
        The manual color cycle (taken from `.other`).
    pcolor : bool, optional
        Whether to use `pcolormesh` for 2D shaded plot (taken from `.other`).
    horizontal : bool, optional
        Whether to use vertical or horizontal orientation (taken from `.other`).

    Results
    -------
    command : str
        The default command. Note `kwargs` is updated in-place.
    args : tuple
        The possibly updated positional arguments.
    kw_plt : namedtuple
        The possibly updated keyword arguments.
    """
    # Helper functions
    # TODO: Support hist and hist2d plots in addition to scatter and barh plots
    # (or just hist since, hist2d usually looks ugly with so little data)
    def _model_flagships(data):
        bools = []
        filt_flag = _parse_institute(data, 'flagship')
        for facet in args[-1].facets.values:
            b = bool(filt_flag(facet))
            bools.append(b)
        return bools
    def _model_projects(data):  # noqa: E306
        projects = []
        filt_cmip65 = _parse_project(data, 'cmip65')
        filt_cmip66 = _parse_project(data, 'cmip66')
        for facet in args[-1].facets.values:
            if filt_cmip66(facet):
                project = 'cmip66'
            elif filt_cmip65(facet):
                project = 'cmip65'
            else:
                project = 'cmip5'  # noqa: E501
            projects.append(project)
        return projects

    # Get command defaults
    colors = _get_colors(cycle=kw_plt.other.get('cycle'), shade=kw_plt.other.get('shade'))  # noqa: E501
    contour = contour or {}  # count-specific properties
    horizontal = kw_plt.other.get('horizontal', False)
    pcolor = kw_plt.other.get('pcolor', False)
    ignore = {'facets', 'version', 'period'}  # additional keywords
    dims = tuple(sorted(args[-1].sizes.keys() - ignore))
    kw_plt = copy.deepcopy(kw_plt)
    if dims:
        # Plots with longitude, latitude, and/or pressure dimension
        # NOTE: Line plot arrays may be 2D with 'facets' dimension and shade kwargs
        if dims == ('plev',):
            command = 'linex'
            defaults = KWARGS_LINE.copy()
        elif len(dims) == 1:
            command = 'line'
            defaults = KWARGS_LINE.copy()
        elif 'hatches' in kw_plt.command:
            command = 'contourf'
            defaults = KWARGS_HASH.copy()
        elif shading:
            command = 'pcolormesh' if pcolor else 'contourf'
            defaults = KWARGS_SHADING.copy()
        else:
            command = 'contour'
            defaults = {'labels': True, **KWARGS_CONTOUR, **contour}

    elif len(dims) == 0:  # always 1D with 'facets' or 'components' dimension
        # Get default bar and box properties
        index = args[-1].indexes[args[-1].dims[0]]
        levels = [
            level for name, level in zip(index.names, index.levels)
            if name not in ('project', 'institute')
        ]
        size = (0.5 * pplt.rc['lines.markersize']) ** 2
        tuples = list(map(tuple, zip(*levels)))
        keys_alpha = {'cmip5': 0.2, 'cmip56': 0.2, 'cmip65': 0.6}
        keys_width = {'cmip5': 0.5 * pplt.rc.metawidth, 'cmip56': 0.5 * pplt.rc.metawidth}  # noqa: E501
        keys_sizes = {False: 0.5 * size, True: 1.5 * size}
        keys_hatch = {False: None, True: '//////'}
        keys_color = {}
        for key in tuples:  # unique tuples minus project
            if key not in keys_color:
                keys_color[key] = colors[len(keys_color) % len(colors)]

        # Convert to lists of properties
        if 'facets' in args[-1].sizes:
            negpos = args[-1].name[:3] not in ('ecs', 'erf')  # overwrites input color
            usecolors = False  # do not use colors array
            flagship = _model_flagships(args[-1])
            project = _model_projects(args[-1])
            args[-1] = args[-1].sortby(args[-1], ascending=not horizontal)
        else:
            negpos = False
            usecolors = True
            flagship = np.zeros(args[-1].size)
            project = np.atleast_1d(args[-1].coords.get('project', None), dtype='O')
        if not any(key in project for key in ('cmip65', 'cmip66')):
            keys_alpha['cmip5'] = keys_alpha['cmip56'] = 0.4  # higher default alpha
        alphas = [keys_alpha.get(key, 1.0) for key in project]
        alphas = None if len(set(alphas)) <= 1 else alphas
        widths = [keys_width.get(key, pplt.rc.metawidth) for key in project]
        widths = None if len(set(widths)) <= 1 else widths
        hatches = [keys_hatch.get(key, None) for key in flagship]
        hatches = None if len(set(hatches)) <= 1 else hatches
        sizes = [keys_sizes.get(key, None) for key in flagship]
        sizes = None if len(set(sizes)) <= 1 else sizes
        colors = [keys_color.get(key, None) for key in tuples]
        colors = None if len(set(colors)) <= 1 else colors

        # Add command-specific settings
        nargs = sum(isinstance(arg, xr.DataArray) for arg in args)
        multiple = args[-1].ndim == 2 or args[-1].ndim == 1 and args[-1].dtype == 'O'
        if nargs == 2:
            command = 'scatter'
            defaults = {
                **KWARGS_SCATTER,
                'alpha': alphas,
                'sizes': sizes,
                'absolute_size': True,
            }
        elif multiple:  # box plot
            # command = 'violinh' if horizontal else 'violin'
            command = 'boxh' if horizontal else 'box'
            kw_plt.command.pop('color', None)
            defaults = {
                **KWARGS_BOX,
                'color': colors,
                'hatch': hatches,
                'alpha': alphas,
            }
        else:  # bar plot
            command = 'barh' if horizontal else 'bar'
            kw_plt.command.pop('color', None)
            defaults = {
                **KWARGS_BAR,
                'negpos': negpos,
                'alpha': alphas,
                'linewidth': widths,
                'hatch': hatches,
            }
            if usecolors:  # generally regression bar plots
                defaults['color'] = colors
    else:
        raise ValueError(f'Invalid dimension count {len(sizes)} and sizes {sizes}.')  # noqa: E501
    for key, value in defaults.items():
        kw_plt.command.setdefault(key, value)
    return command, args, kw_plt


def _merge_commands(dataset, inputs, kwargs):
    """
    Merge several distribution plots into a single box or bar plot instruction.

    Parameters
    ----------
    dataset : xarray.Dataset
        The source dataset.
    inputs : list of tuple
        The input arguments.
    kwargs : list of namedtuple
        The keyword arguments.

    Returns
    -------
    args : tuple
        The merged arguments.
    kwargs : namedtuple
        The merged keyword arguments.
    """
    # Combine keyword arguments and coordinates
    # NOTE: Considered removing this and having _parse_spec support 'lists of lists
    # of specs' per subplot but this is complicated, and would still have to split
    # then re-combine scalar coordinates e.g. 'project', so not much more elegant.
    nargs = set(map(len, inputs))
    if len(nargs) != 1:
        raise RuntimeError(f'Unexpected combination of argument counts {nargs}.')
    if (nargs := nargs.pop()) not in (1, 2):
        raise RuntimeError(f'Unexpected number of arguments {nargs}.')
    coords, outers, inners = [], [], []
    for args in zip(*inputs):
        keys = ['name', *sorted(set(key for arg in args for key in arg.coords))]
        coord, outer, inner = {}, {}, {}
        for key in keys:
            vals = [arg.coords.get(key, '') for arg in args]
            if not all(isinstance(val, str) for val in vals):
                continue
            if all(val == vals[0] for val in vals):
                continue
            b = any(val1 == val2 for val1, val2 in zip(vals[:1], vals[:-1]))
            src = outer if b else inner
            coord[key] = src[key] = vals
        coords.append(coord)
        outers.append([dict(zip(outer, vals)) for vals in zip(*outer.values())])
        inners.append([dict(zip(inner, vals)) for vals in zip(*inner.values())])

    # Get tick locations and labels for outer group
    # TODO: Also support colors along *inner* group instead of *outer*
    kw_plt = copy.deepcopy(kwargs[0])  # deep copy of namedtuple
    for kw in kwargs:
        for field in kw._fields:
            getattr(kw_plt, field).update(getattr(kw, field))
    axis = 'y' if kw_plt.other.get('horizontal', False) else 'x'
    offset = kw_plt.other.get('offset', 0.5)  # additional offset coordinate
    kw_infer = dict(identical=False, long_names=True, title_case=False)
    outers = _infer_labels(dataset, list(zip(*outers)), **kw_infer)
    inners = _infer_labels(dataset, list(zip(*inners)), **kw_infer)
    num, locs, labels = 0, [], []
    for grp, (label, group) in enumerate(itertools.groupby(outers)):
        group = list(group)  # convert itertools._group object
        idxs = [idx for idx in range(num, num + len(group))]
        avg = 0.5 * (idxs[0] + idxs[-1])
        num += len(group)
        locs.extend((grp * offset + idx for idx in idxs))
        labels.append((grp * offset + avg, label))
    ticks, labels = zip(*labels)
    kw_plt.axes.update({f'{axis}ticks': ticks, f'{axis}ticklabels': labels})
    kw_plt.axes.update({f'{axis}ticklen': 0, f'{axis}tickpad': '1em'})

    # Merge arrays and infer slope products
    # TODO: Somehow combine regression component coordinates
    locs = np.array(locs, dtype=float)
    coord = coords[-1]  # take *independent* variable coordinates for regressions
    if nargs == 1:
        values = np.array([tuple(arg.values) for (arg,) in inputs], dtype=object)
    else:
        values, labels, bardata = [], [], []
        for args in inputs:
            slope, slope_lower, slope_upper, rsquare, _, _, _ = _components_slope(*args)
            rsquare = ureg.Quantity(rsquare.item(), '').to('percent')
            label = f'${rsquare:~L.1f}$'.replace('%', r'\%')  # no R^2 symbol here
            values.append(slope)
            labels.append(label)
            bardata.append([slope_lower, slope_upper])
        coord.update(label=label)  # then used in _setup_bars
        kwargs[0].command.update(bardata=np.array(bardata).T)
    index = pd.MultiIndex.from_arrays(coord.values(), names=coord.keys())
    values = xr.DataArray(values, dims='components', coords={'components': index})
    args = (locs, values)
    return args, kw_plt


def _infer_commands(dataset, inputs, kwargs, fig=None, gs=None, geom=None, title=None):
    """
    Infer the plotting command from the input arguments and apply settings.

    Parameters
    ----------
    dataset : xarray.Dataset
        The input dataset.
    inputs : list of tuple of xarray.DataArray
        The plotting arguments.
    kwargs : list of namedtuple of dict
        The keyword arguments for each command.
    fig : pplt.Figure, optional
        The existing figure.
    gs : pplt.GridSpec, optional
        The existing gridspec.
    geom : tuple, optional
        The ``(nrows, ncols, index)`` geometry.
    title : str, optional
        The default title of the axes.

    Other Parameters
    ----------------
    shade : int, optional
        The shading for the default color cycle (taken from `.other`).
    cycle : cycle-spec, optional
        The manual color cycle (taken from `.other`).
    offset : float, optional
        The additional offset for bar/box groups (taken from `.other`).

    Returns
    -------
    fig, gs : object
        The figure and gridspec objects.
    axs : matplotlib.axes.Axes
        The plotting axes.
    command : str
        The plotting commands.
    inputs : tuple
        The possibly modified plotting arguments.
    kwargs : dict
        The possibly modified plotting keyword arguments.
    """
    # Instantiate figure and subplot
    # NOTE: Delay object creation until this block so we can pass arbitrary
    # loose keyword arguments and simply parse them in _parse_item.
    ignore = {'facets', 'version', 'period'}  # additional keywords
    sizes = [tuple(sorted(args[-1].sizes.keys() - ignore)) for args in inputs]
    if len(set(sizes)) == 1:
        sizes = set(sizes[0])
    else:
        raise RuntimeError(f'Conflicting dimensionalities in single subplot: {sizes}')
    geom = geom or (1, 1, 0)  # total subplotspec geometry
    defaults = KWARGS_FIG.copy()
    if sizes == {'lon', 'lat'}:
        defaults.update(KWARGS_GEO)
    else:
        defaults.update(KWARGS_LAT if 'lat' in sizes else {})
        defaults.update(KWARGS_PLEV if 'plev' in sizes else {})
    width = defaults.pop('refwidth', None)
    sharex = True if 'lat' in sizes else 'labels'
    sharey = True if 'plev' in sizes else 'labels'
    kw_fig = {'sharex': sharex, 'sharey': sharey, 'span': False, 'refwidth': width}
    kw_axs = {'title': title, **defaults}
    kw_grd = {}  # no defaults currently
    if len(sizes) == 0 and len(inputs) > 1:  # offset is used here
        inputs, kwargs = _merge_commands(dataset, inputs, kwargs)
    for kw_plt in kwargs[::-1]:
        kw_fig.update(kw_plt.figure)
    for kw_plt in kwargs[::-1]:
        kw_axs.update(kw_plt.axes)
    for kw_plt in kwargs[::-1]:
        kw_grd.update(kw_plt.gridspec)
    if fig is None:  # also support updating? or too slow?
        fig = pplt.figure(**kw_fig)
    if gs is None:
        gs = pplt.GridSpec(*geom[:2], **kw_plt.gridspec)
    if max(geom[:2]) == 1:
        kw_axs.pop('abc', None)

    # Infer commands
    # TODO: Support *stacked* scatter plots and *grouped* bar plots with 2D arrays
    # for non-project multiple selections? Not too difficult... but maybe not worth it.
    contours = []  # contour keywords
    contours.append({'color': 'gray8', 'linestyle': None})
    contours.append({'color': 'gray3', 'linestyle': ':'})
    shade = set(kw_plt.other.get('shade', None) for kw_plt in kwargs)
    cycle = set(kw_plt.other.get('cycle', None) for kw_plt in kwargs)
    cycle = _get_colors(cycle=cycle.pop(), shade=shade.pop())
    ax = iax = fig.add_subplot(gs[geom[2]], **kw_axs)
    results, colors, units = [], {}, {}
    for i, (args, kw_plt) in enumerate(zip(inputs, kwargs)):
        iunits = args[-1].attrs.get('units', None)  # independent variable units
        icolor = kw_plt.command.setdefault('color', cycle[i % len(cycle)])
        iax = units.get(iunits, ax)
        if units and iunits not in units:
            value = colors.get(ax, ())  # number of colors used so far
            value = value.pop() if len(value) == 1 else 'k'
            axis = 'y' if 'plev' in sizes else 'x'
            ax.format(**{f'{axis}color': value})  # line color or simply black
            iax = getattr(ax, f'alt{axis}')(**{f'{axis}color': icolor})
        contour = contours[min(i, len(contours) - 1)]
        shading = (i == 0)  # shade first plot only
        command, kw_cmd = _get_command(args, kw_plt, shading=shading, contour=contour)
        kw_plt.command.update(kw_cmd)
        results.append((iax, command, args, kw_plt))
        colors.setdefault(ax, set()).add(icolor)  # colors plotted per axes
        units.update({iunits: iax})  # axes indexed by units
    axs, commands, inputs, kwargs = zip(*results)
    return axs, commands, inputs, kwargs


def _setup_axes(ax, command, *args):
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
    # Initial stuff and add zero line
    # NOTE: Want to disable autoscaling based on zero line but not currently
    # possible with convenience functions axvline and axhline. Instead use
    # manual plot(). See: https://github.com/matplotlib/matplotlib/issues/14651
    top = ax._get_topmost_axes()
    fig = top.figure
    nrows, ncols, *_ = top.get_subplotspec()._get_geometry()  # custom geometry function
    if command in ('barh', 'boxh', 'violinh'):
        y = None
    elif command in ('bar', 'box', 'violin', 'line', 'scatter'):
        y = args[-1]
    else:
        y = args[-1].coords[args[-1].dims[0]]
    if command in ('bar', 'box', 'violin'):
        x = None
    elif command in ('barh', 'boxh', 'violinh'):
        x = args[-1]
    elif command in ('linex', 'scatter'):
        x = args[0]
    else:
        x = args[-1].coords[args[-1].dims[-1]]  # e.g. contour() is y by x
    if ax == top:
        if command in ('bar', 'box', 'violin', 'line', 'vlines'):
            transform = ax.get_yaxis_transform()
            ax.plot([0, 1], [0, 0], transform=transform, **KWARGS_ZERO)
        if command in ('barh', 'boxh', 'violinh', 'linex', 'hlines'):
            transform = ax.get_xaxis_transform()
            ax.plot([0, 0], [0, 1], transform=transform, **KWARGS_ZERO)

    # Handle x and y axis labels
    # TODO: Handle this automatically with climopy and proplot autoformatting
    units = []
    rows, cols = top._range_subplotspec('y'), top._range_subplotspec('x')
    xbool = ax == top or not fig._sharex or min(rows) == 0
    ybool = ax == top or not fig._sharey or max(cols) == ncols - 1
    refscale = 1 if ax._name == 'cartesian' else 0.7  # applied for y-axis labels only
    for axis, data, bool_, scale in zip('xy', (x, y), (xbool, ybool), (1, refscale)):
        if data is None:
            unit = None
            locator = getattr(ax, f'{axis}axis').get_major_locator()
            if isinstance(locator, (pplt.AutoLocator, pplt.MaxNLocator)):
                ax.format(**{f'{axis}locator': 'null'})  # do not overwrite labels
        else:
            data = data.copy()
            prefix = data.attrs.get(f'{axis}label_prefix', '')
            if command == 'scatter' and len(prefix) < 20:
                data.attrs['short_prefix'] = prefix  # use other prefix
            else:
                data.attrs.pop('short_prefix', None)  # ignore legend indicator
            if 'short_name' not in data.attrs:
                data.attrs['short_name'] = ''
            if (unit := data.attrs.get('units', None)) is None:
                continue  # e.g. skip 'facets' coordinate
            if not bool_ or getattr(ax, f'get_{axis}label')():
                continue
            label = _wrap_label(data.climo.short_label, refwidth=fig._refwidth)
            getattr(ax, f'set_{axis}label')(label)
        units.append(unit)
    return units


def _setup_guide(*axs, horizontal='bottom', vertical='bottom', default='bottom'):
    """
    Set up the guide

    Parameters
    ----------
    *axs : pplt.Axes
        The input axes.
    horizontal, vertical, default : str, optional
        The default horizontal and vertical stuff.

    Returns
    -------
    src : object
        The figure or axes to use.
    loc : str
        The guide location to use.
    span : 2-tuple
        The span for figure-edge labels.
    """
    fig = axs[0].figure
    rows = set(n for ax in axs for n in ax._get_topmost_axes()._range_subplotspec('y'))
    cols = set(n for ax in axs for n in ax._get_topmost_axes()._range_subplotspec('x'))
    nrows, ncols, *_ = fig.gridspec.get_geometry()
    if not (len(rows) == 1) ^ (len(cols) == 1):
        src = fig
        loc = default
    elif len(rows) == 1:  # single row
        loc = horizontal
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
        loc = vertical
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
        span = None
    elif loc[0] in 'lr':
        span = (min(rows) + 1, max(rows) + 1)
    else:
        span = (min(cols) + 1, max(cols) + 1)
    return src, loc, span


def _setup_bars(ax, data, bardata=None, horizontal=False, annotate=False):
    """
    Adjust and optionally add content to bar plots.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes.
    data : xarray.DataArray
        The original data.
    bardata : xarray.Dataarray
        The error bar data.
    horizontal : bool, optional
        Whether the bars were plotted horizontally.
    annotate : bool, optional
        Whether to add annotations to the bars.
    """
    # NOTE: Using set_in_layout False significantly improves appearance since
    # generally don't mind overlapping with tick labels for bar plots and
    # improves draw time since tight bounding box calculation is expensive.
    if not annotate:
        return
    if data.ndim != 1:
        raise RuntimeError(f'Unexpected bar data dimensionality {data.ndim}.')
    kw_annotate = {'fontsize': 'x-small', 'textcoords': 'offset points'}
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    if not horizontal:
        height, _ = ax._get_size_inches()
        diff = (pplt.rc.fontsize / 72) * (max(ylim) - min(ylim)) / height
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
    index = data.indexes[data.dims[0]]
    if 'label' in index.names:
        labels, rotation = data.label.values, 0  # assume small so do not rotate these
    elif 'model' in index.names:
        labels, rotation = data.model.values, 90
    else:
        labels, rotation = list(map(str, data.coords[data.dims[0]].values)), 90
    for i, label in enumerate(labels.values):
        value = data.values[i]
        lower, upper = (None, None) if bardata is None else bardata.values[:, i]
        if not horizontal:
            va = 'bottom' if value > 0 else 'top'
            kw = {'ha': 'center', 'va': va, 'rotation': rotation}
        else:
            ha = 'left' if value > 0 else 'right'
            kw = {'ha': ha, 'va': 'center', 'rotation': 0}
        slice_ = slice(None) if not horizontal else slice(None, None, -1)
        point = value if bardata is None else (lower, upper)[value > 0]
        xydata = (i, point)[slice_]  # point is x coordinate if horizontal
        xytext = (0, 2 if value > 0 else -2)[slice_]  # as above for offset
        res = ax.annotate(label, xydata, xytext, **kw, **kw_annotate)
        res.set_in_layout(False)


def _setup_scatter(ax, data0, data1, oneone=False, linefit=False, annotate=False):
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
    # NOTE: Here climopy automatically reapplies dataarray coordinates to fit line
    # and lower and upper bounds so do not explicitly need sorted x coordinates.
    # NOTE: Using set_in_layout False significantly improves appearance since
    # generally don't mind overlapping with tick labels for scatter plots and
    # improves draw time since tight bounding box calculation is expensive.
    kw_annotate = {'fontsize': 'x-small', 'textcoords': 'offset points'}
    if linefit:  # https://en.wikipedia.org/wiki/Simple_linear_regression
        slope, _, _, rsquare, fit, fit_lower, fit_upper = _components_slope(data0, data1)  # noqa: E501
        sign = '(\N{MINUS SIGN})' if slope < 0 else ''  # point out negative r-squared
        rsquare = ureg.Quantity(rsquare.item(), '').to('percent')
        ax.format(ultitle=f'$R^2 = {sign}{rsquare:~L.1f}$'.replace('%', r'\%'))
        ax.plot(fit, color='r', ls='-', lw=1.5 * pplt.rc.metawidth)
        ax.area(fit_lower, fit_upper, color='r', alpha=0.5 ** 2, lw=0)
    if oneone:
        lim = (*ax.get_xlim(), *ax.get_ylim())
        lim = (min(lim), max(lim))
        avg = 0.5 * (lim[0] + lim[1])
        span = lim[1] - lim[0]
        ones = (avg - 1e3 * span, avg + 1e3 * span)
        ax.format(xlim=lim, ylim=lim)  # autoscale disabled
        ax.plot(ones, ones, ls='--', lw=1.5 * pplt.rc.metawidth, color='k')
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


def breakdown_transport(breakdown=None, component=None, transport=None, maxcols=None):
    """
    Return the names and colors associated with transport components.

    Parameters
    ----------
    component : str, optional
        The individual component to use.
    breakdown : str, optional
        The transport components to show.
    transport : str, optional
        The transport type to show.
    maxcols : int, optional
        The maximum number of columns (unused currently).

    Returns
    -------
    names : list of str
        The transport components.
    maxcols : int
        The number of columns.
    colors : list of str
        The associated colors to use.
    """
    # TODO: Expand to support mean-stationary-eddy decompositions and whatnot. Perhaps
    # indicate those with dashes and dots and/or with different shades. And/or allow
    # individual components per subplot with logical arrangement as with feedbacks.
    breakdown = breakdown or 'all'
    shading = 7
    colors = {
        'total': 'gray',
        'ocean': 'cyan',
        'mse': 'yellow',
        'lse': 'blue',
        'dse': 'red'
    }
    if transport is None:
        raise ValueError('Transport component must be explicitly passed.')
    if component is not None:
        names = (component,)
    elif breakdown == 'all':
        names = ('total', 'ocean', 'mse', 'lse', 'dse')
    elif breakdown == 'total':
        names = ('total', 'ocean', 'mse')
    elif breakdown == 'atmos':
        names = ('mse', 'lse', 'dse')
    else:
        raise RuntimeError
    colors = [colors[name] + str(shading) for name in names]
    names = [name + transport for name in names]
    return names, maxcols, colors


def breakdown_feedbacks(
    component=None,
    breakdown=None,
    feedbacks=True,
    adjusts=False,
    forcing=False,
    sensitivity=False,
    maxcols=None,
):
    """
    Return the feedback, forcing, and sensitivity parameter names sensibly
    organized depending on the number of columns in the plot.

    Parameters
    ----------
    component : str, optional
        The individual component to use.
    breakdown : str, default: 'all'
        The breakdown preset to use.
    feedbacks, adjusts, forcing, sensitivity : bool, optional
        Whether to include various components.
    maxcols : int, default: 4
        The maximum number of columns (influences order of the specs).

    Returns
    -------
    specs : list of str
        The variables.
    maxcols : int
        The possibly adjusted number of columns.
    gridskip : list of int
        The gridspec entries to skip.
    """
    # Initial stuff
    # NOTE: Idea is to automatically remove feedbacks, filter out all-none
    # rows and columns at the end, and then infer the 'gridskip' from them.
    # NOTE: Options include 'wav', 'atm', 'alb', 'res', 'all', 'atm_wav', 'alb_wav'
    # with the 'wav' suffixes including longwave and shortwave cloud components
    # instead of a total cloud feedback. Strings denote successively adding atmospheric,
    # albedo, residual, and remaining temperature/humidity feedbacks with options.
    original = maxcols = maxcols or 4
    init_names = lambda ncols: ((names := np.array([[None] * ncols] * 25)), names.flat)
    if breakdown is None:
        component = component or 'net'
    if component is None:
        breakdown = breakdown or 'all'
    if not component and not feedbacks and not forcing and not sensitivity:
        raise RuntimeError

    # Three variable breakdowns
    if component is not None:
        names, iflat = init_names(maxcols)
        iflat[0] = component
        gridskip = None
    elif breakdown in ('net', 'atm', 'cld', 'wav'):  # shortwave longwave
        if breakdown == 'net':  # net lw/sw
            lams = ['net', 'sw', 'lw']
            erfs = ['erf', 'rsnt_erf', 'rlnt_erf']
        elif breakdown == 'atm':  # net cloud, atmosphere
            lams = ['net', 'cld', 'atm']
            erfs = ['erf', 'cl_rfnt_erf', 'atm_rfnt_erf']
        elif breakdown == 'cld':  # cloud lw/sw
            lams = ['cld', 'swcld', 'lwcld']
            erfs = ['cl_rfnt_erf', 'cl_rsnt_erf', 'cl_rlnt_erf']
        elif breakdown == 'wav':  # cloud lw/sw
            lams = ['net', 'swcld', 'lwcld']
            erfs = ['erf', 'cl_rsnt_erf', 'cl_rlnt_erf']
        else:
            raise RuntimeError
        if maxcols == 2:
            names, iflat = init_names(maxcols)
            if sensitivity:
                iflat[0] = 'ecs'
            if feedbacks and adjusts:
                names[1:4, 0] = lams
                names[1:4, 1] = erfs
            elif feedbacks and forcing:
                names[1, :] = lams[:1] + erfs[:1]
                names[2, :] = lams[1:]
            elif feedbacks:
                iflat[1] = lams[0]
                names[1, :] = lams[1:]
            elif adjusts:
                iflat[1] = erfs[0]
                names[1, :] = erfs[1:]
        else:
            offset = 0
            maxcols = 1 if maxcols == 1 else 3
            names, iflat = init_names(maxcols)
            if sensitivity:
                iflat[offset] = 'ecs'
            if forcing:
                iflat[offset + 1] = 'erf'
            if feedbacks:
                idx = 2 * maxcols
                iflat[idx:idx + 3] = lams
            if adjusts:
                idx = 3 * maxcols
                iflat[idx:idx + 3] = erfs

    # Four variable breakdowns
    elif breakdown in ('cld_wav', 'atm_wav', 'atm_res', 'alb'):
        if 'cld' in breakdown:  # net cloud, cloud lw/sw
            lams = ['net', 'cld', 'swcld', 'lwcld']
            erfs = ['erf', 'cl_rfnt_erf', 'cl_rsnt_erf', 'cl_rlnt_erf']
        elif 'res' in breakdown:  # net cloud, residual, atmosphere
            lams = ['net', 'cld', 'resid', 'atm']
            erfs = ['erf', 'cl_rfnt_erf', 'resid_rfnt_erf', 'atm_rfnt_erf']  # noqa: E501
        elif 'atm' in breakdown:  # cloud lw/sw, atmosphere
            lams = ['net', 'swcld', 'lwcld', 'atm']
            erfs = ['erf', 'cl_rsnt_erf', 'cl_rlnt_erf', 'atm_rfnt_erf']
        elif breakdown == 'alb':  # net cloud, atmosphere, albedo
            lams = ['net', 'cld', 'alb', 'atm']
            erfs = ['erf', 'cl_rfnt_erf', 'alb_rfnt_erf', 'atm_rfnt_erf']
        else:
            raise RuntimeError
        if maxcols == 2:
            names, iflat = init_names(maxcols)
            if sensitivity:
                iflat[0] = 'ecs'
            if feedbacks and adjusts:
                names[1:5, 0] = lams
                names[1:5, 1] = erfs
            elif feedbacks:
                iflat[1] = 'erf' if forcing else None
                names[1, :] = lams[::3]
                names[2, :] = lams[1:3]
            elif adjusts:
                names[1, :] = erfs[::3]
                names[2, :] = erfs[1:3]
        else:
            offset = 0 if maxcols == 1 else 1
            maxcols = 1 if maxcols == 1 else 3
            names, iflat = init_names(maxcols)
            if sensitivity:
                iflat[-offset % 3] = 'ecs'
            if forcing or adjusts:
                iflat[2 - offset] = 'erf'
            if feedbacks:
                idx = 3
                iflat[1 - offset] = lams[0]
                iflat[idx:idx + 3] = lams[1:]
            if adjusts:
                idx = 3 + 3
                iflat[idx:idx + 3] = erfs[1:]

    # Five variable breakdowns
    # NOTE: Currently this is the only breakdown with both clouds
    elif breakdown in ('atm_cld', 'alb_wav', 'res', 'res_wav'):
        if 'atm' in breakdown:  # net cloud, cloud lw/sw, atmosphere
            lams = ['net', 'cld', 'swcld', 'lwcld', 'atm']
            erfs = ['erf', 'cl_rfnt_erf', 'cl_rsnt_erf', 'cl_rlnt_erf', 'atm_rfnt_erf']
        elif 'alb' in breakdown:  # cloud lw/sw, atmosphere, albedo
            lams = ['net', 'swcld', 'lwcld', 'alb', 'atm']
            erfs = ['erf', 'cl_rsnt_erf', 'cl_rlnt_erf', 'alb_rfnt_erf', 'atm_rfnt_erf']
        elif breakdown == 'res':  # net cloud, atmosphere, albedo, residual
            lams = ['net', 'cld', 'alb', 'resid', 'atm']
            erfs = ['erf', 'cl_rfnt_erf', 'alb_rfnt_erf', 'resid_rfnt_erf', 'atm_rfnt_erf']  # noqa: E501
        elif 'res' in breakdown:  # cloud lw/sw, atmosphere, residual
            lams = ['net', 'swcld', 'lwcld', 'resid', 'atm']
            erfs = ['erf', 'cs_rlnt_erf', 'cl_rlnt_erf', 'resid_rfnt_erf', 'atm_rfnt_erf']  # noqa: E501
        else:
            raise RuntimeError
        if maxcols == 2:
            names, iflat = init_names(maxcols)
            if sensitivity:
                iflat[0] = 'ecs'
            if feedbacks and adjusts:
                names[1:5, 0] = lams
                names[1:5, 1] = erfs  # noqa: E501
            elif feedbacks and forcing:
                names[1, :] = lams[:1] + erfs[:1]
                names[2, :] = lams[1:3]
                names[3, :] = lams[3:5]
            elif feedbacks:
                names[0, 1] = lams[0]
                names[1, :] = lams[1:3]
                names[2, :] = lams[3:5]
            elif adjusts:
                names[0, 1] = erfs[0]
                names[1, :] = erfs[1:3]
                names[2, :] = erfs[3:5]
        else:
            offset = 0 if maxcols == 1 else 1
            maxcols = 1 if maxcols == 1 else 4  # disallow 3 columns
            names, iflat = init_names(maxcols)
            if sensitivity:
                iflat[-offset % 3] = 'ecs'  # either before or after net and erf
            if forcing or adjusts:
                iflat[2 - offset] = erfs[0]
            if feedbacks:
                idx = 4  # could leave empty single-column row
                iflat[1 - offset] = lams[0]
                iflat[idx:idx + 4] = lams[1:]
            if adjusts:
                idx = 4 + 4
                iflat[idx:idx + 4] = erfs[1:]

    # Full breakdown
    elif breakdown == 'all':
        lams = ['net', 'cld', 'swcld', 'lwcld', 'atm', 'alb', 'resid']
        erfs = ['erf', 'cl_rfnt_erf', 'cl_rsnt_erf', 'cl_rlnt_erf', 'atm_rfnt_erf', 'alb_rfnt_erf', 'resid_rfnt_erf']  # noqa: E501
        hums = ['wv', 'rh', 'lr', 'lr*', 'pl', 'pl*']
        if maxcols == 2:
            names, iflat = init_names(maxcols)
            if sensitivity:
                iflat[0] = 'ecs'
            if feedbacks and adjusts:
                names[1:8, 0] = lams
                names[1:8, 1] = erfs
                iflat[16:22] = hums
            elif feedbacks:
                iflat[0] = lams[0]
                iflat[1] = 'erf' if forcing else None
                names[2, :] = lams[1:3]
                names[3, :] = lams[3:5]
            elif adjusts:
                iflat[0] = erfs[0]
                names[1, :] = erfs[1:3]
                names[2, :] = erfs[3:5]
        else:
            offset = 0 if maxcols == 1 else 1
            maxcols = 1 if maxcols == 1 else 4  # disallow 3 columns
            names, iflat = init_names(maxcols)
            if sensitivity:
                iflat[-offset % 3] = 'ecs'  # either before or after net and erf
            if forcing or adjusts:
                iflat[2 - offset] = erfs[0]
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
            if adjusts:
                idx = 4 + 3 * 4
                iflat[idx:idx + 6] = erfs[1:7]
    else:
        raise RuntimeError(f'Invalid breakdown key {breakdown!r}.')

    # Remove all-none segments and determine gridskip
    # NOTE: This also automatically flattens the arangement if there are
    # fewer than names than the originally requested maximum column count.
    idx, = np.where(np.any(names != None, axis=0))  # noqa: E711
    names = np.take(names, idx, axis=1)
    idx, = np.where(np.any(names != None, axis=1))  # noqa: E711
    names = np.take(names, idx, axis=0)
    idxs = np.where(names == None)  # noqa: E711
    gridskip = np.ravel_multi_index(idxs, names.shape)
    names = names.ravel().tolist()
    names = [spec for spec in names if spec is not None]
    if len(names) <= original and maxcols != 1:  # then keep it simple!
        maxcols = len(names)
        gridskip = np.array([], dtype=int)
    return names, maxcols, gridskip


def create_plot(
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
    hlegend='bottom',
    vcolorbar='bottom',
    vlegend='bottom',
    dcolorbar='right',
    dlegend='bottom',
    standardize=False,
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
        used to generate data in rows and columns. See `parse_specs` for details.
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
    dcolorbar, dlegend : {'bottom', 'right', 'top', 'left'}
        The default location for colorbars or legends annotating a mix of
        axes. Placed with the figure colorbar or legend method.
    standardize : bool, optional
        Whether to standardize axis limits to span the same range for all
        plotted content with the same units.
    save : path-like, optional
        The save folder base location. Stored inside a `figures` subfolder.
    **kw_specs
        Passed to `parse_specs`.
    **kw_method
        Passed to `apply_method`.

    Returns
    -------
    fig : pplt.Figure
        The figure.
    axs : pplt.Subplotgrid
        The subplots.

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
    dataspecs, plotspecs, figlabel, pathlabel, gridlabels = parse_specs(
        dataset, rowspecs, colspecs, **kwargs
    )
    nrows, ncols = map(len, gridlabels)
    nrows, ncols = max(nrows, 1), max(ncols, 1)
    titles = (None,) * nrows * ncols
    print('All:', repr(figlabel))
    print('Rows:', ', '.join(map(repr, gridlabels[0])))
    print('Columns:', ', '.join(map(repr, gridlabels[1])))
    print('Path:', pathlabel)
    figtitle = figtitle or figlabel
    figprefix, figsuffix = figprefix or '', figsuffix or ''
    if figprefix:
        figprefix = figprefix if figprefix[:1].isupper() else figprefix[0].upper() + figprefix[1:]  # noqa: E501
        figtitle = figtitle if figtitle[:2].isupper() else figtitle[0].lower() + figtitle[1:]  # noqa: E501
    figparts = (figprefix, figtitle, figsuffix)
    figtitle = ' '.join(filter(None, figparts))
    if nrows == 1 or ncols == 1:
        naxes = gridskip.size + max(nrows, ncols)
        ncols = min(naxes, maxcols or 4)
        nrows = 1 + (naxes - 1) // ncols
        titles = max(gridlabels, key=lambda labels: len(labels))
        titles = max(nrows, ncols) > 1 and titles or (None,) * nrows * ncols
        gridlabels = (None, None)

    # Generate data arrays and queued plotting commands
    # NOTE: Critical to disable 'grouping' so that e.g. colorbars or legends that
    # extend into other panel slots are not considered in the tight layout algorithm.
    # NOTE: This will automatically allocate separate colorbars for
    # variables with different declared level-restricting arguments.
    fig = gs = None  # delay instantiation
    count = 0
    tuples = list(zip(titles, dataspecs, plotspecs))
    queue, methods, commands = {}, [], []
    print('Getting data...', end=' ')
    for num in range(nrows * ncols):
        if num in gridskip:
            continue
        count += 1
        inputs, kwargs = [], []
        title, dspecs, pspecs = tuples[count - 1]
        print(f'{num + 1}/{nrows * ncols}', end=' ')
        for kws_dat, kw_plt in zip(dspecs, pspecs):
            args, default = get_data(dataset, *kws_dat, attrs=kw_plt.attrs.copy())
            for key, value in default.items():  # also adds 'method' key
                kw_plt.command.setdefault(key, value)
            inputs.append(args)
            kwargs.append(kw_plt)
        infer = dict(fig=fig, gs=gs, geom=(num, nrows, ncols), title=title)
        fig, gs, axs, inputs, kwargs = _infer_commands(dataset, inputs, kwargs, **infer)
        for ax, args, kw_plt in zip(axs, inputs, kwargs):
            args = tuple(args)
            name = '_'.join(arg.name for arg in args if isinstance(arg, xr.DataArray))
            kw = kw_plt.command
            method = kw.pop('method')
            command = kw.pop('command')
            cmap = kw.get('cmap', None)
            cmap = tuple(cmap) if isinstance(cmap, list) else cmap
            color = kw.get('color', None)
            color = tuple(color) if isinstance(color, list) else color
            size = kw.get('sizes', None)
            size = tuple(size) if isinstance(size, list) else size
            key = (name, method, command, cmap, color, size)
            tups = queue.setdefault(key, [])
            tups.append((ax, args, kw_plt))
            if method not in methods:
                methods.append(method)
            if command not in commands:
                commands.append(command)

    # Carry out the plotting commands
    # NOTE: Axes are always added top-to-bottom and left-to-right so leverage
    # this fact below when selecting axes for legends and colorbars.
    print('\nPlotting data...', end=' ')
    axs_objs = {}
    axs_units = {}  # axes grouped by units
    for num, (key, values) in enumerate(queue.items()):
        # Get guide and plotting arguments
        # NOTE: Here 'argskip' is isued to skip arguments with vastly different
        # ranges when generating levels that annotate multiple different subplots.
        print(f'{num + 1}/{len(queue)}', end=' ')
        name, method, command, cmap, color, *_ = key
        axs, args, kws_plt = zip(*values)
        kw_cba = {key: val for kw_plt in kws_plt for key, val in kw_plt.colorbar.items()}  # noqa: E501
        kw_leg = {key: val for kw_plt in kws_plt for key, val in kw_plt.legend.items()}
        kw_oth = {key: val for kw_plt in kws_plt for key, val in kw_plt.other.items()}
        kws_cmd = [kw_plt.command.copy() for kw_plt in kws_plt]

        # TODO: Move all of this stuff to _get_command!
        # TODO: Move all of this stuff to _get_command!
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
            cartesian = axs[0]._name == 'cartesian'
            refwidth = axs[0].figure._refwidth or pplt.rc['subplots.refwidth']
            refscale = 1.3 if cartesian else 0.8  # WARNING: vertical colorbars only
            extendsize = 1.2 if cartesian else 2.2  # WARNING: vertical colorbars only
            guide, kw_guide = 'colorbar', kw_cba
            label = args[0][-1].climo.short_label
            label = _wrap_label(label, refwidth=refwidth, refscale=refscale)
            locator = pplt.DiscreteLocator(levels, nbins=7)
            minorlocator = pplt.DiscreteLocator(levels, nbins=7, minor=True)
            kw_guide.setdefault('locator', locator)
            kw_guide.setdefault('minorlocator', minorlocator)  # scaled internally
            kw_guide.setdefault('extendsize', extendsize)
        else:  # TODO: permit short *or* long
            guide, kw_guide = 'legend', kw_leg
            cfvar = args[0][-1].climo.cfvariable
            label = cfvar.short_label if command == 'contour' else cfvar.short_name  # noqa: E501
            label = None if hatches else label
            keys = ('cmap', 'norm', 'norm_kw')
            maps = ('robust', 'symmetric', 'diverging', 'levels', 'locator', 'extend')
            if 'contour' not in command and 'pcolor' not in command:
                keys += maps
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
        obj = result = None
        for l, (ax, dats, kw_cmd) in enumerate(zip(axs, args, kws_cmd)):
            cmd = getattr(ax, command)
            # vals = dats
            # if command in ('bar', 'barh', 'box', 'boxh', 'violin', 'violinh'):
            #     vals = [da.values if isinstance(da, xr.DataArray) else da for da in dats]  # noqa: E501
            with warnings.catch_warnings():  # ignore 'masked to nan'
                warnings.simplefilter('ignore', UserWarning)
                result = cmd(*dats, **kw_cmd)
            if command == 'contour' and result.collections:
                obj = result.collections[-1]
            elif command in ('contourf', 'pcolormesh'):
                obj = result
            elif command in ('line', 'linex'):
                obj = result[0]  # get line or (shade, line) tuple from singleton list
            elif command in ('bar', 'barh'):
                obj = result[1] if isinstance(result, tuple) else result
            else:
                obj = None
            if ax._name == 'cartesian':
                xunits, yunits = _setup_axes(ax, command, *dats)
                axs_units.setdefault(('x', xunits), []).append(ax)
                axs_units.setdefault(('y', yunits), []).append(ax)
            if 'bar' in command:  # ensure padding around zero level and bar edges
                for patch in obj:
                    patch.sticky_edges.x.clear()
                    patch.sticky_edges.y.clear()
            if 'bar' in command or 'lines' in command:
                keys = ('horizontal', 'annotate')
                kw = {key: val for key, val in kw_oth.items() if key in keys}
                _setup_bars(ax, *dats, bardata=kw_cmd.get('bardata'), **kw)
            if command == 'scatter':
                keys = ('oneone', 'linefit', 'annotate')
                kw = {key: val for key, val in kw_oth.items() if key in keys}
                _setup_scatter(ax, *dats, **kw)  # 'dats' is 2-tuple
        if not obj or not label:
            continue
        if command in ('contourf', 'pcolormesh'):
            key = (name, method, command, cmap, guide, label)
        else:
            key = (command, color, guide, label)
        objs = axs_objs.setdefault(key, [])
        objs.append((axs, obj, kw_guide))

    # Add shared colorbars, legends, and axis limits
    # TODO: Should support colorbars spanning multiple columns or rows in the
    # center of the gridspec in addition to figure edges.
    print('\nAdding guides...')
    handles = {}
    for key, objs in axs_objs.items():
        *_, guide, label = key
        axs, objs, kws_guide = zip(*objs)
        kw_guide = {key: val for kw in kws_guide for key, val in kw.items()}
        if guide == 'colorbar':
            hori, vert, def_ = hcolorbar, vcolorbar, dcolorbar
        else:
            hori, vert, def_ = hlegend, vlegend, dlegend
        axs = [ax for iaxs in axs for ax in iaxs]
        src, loc, span = _setup_guide(*axs, horizontal=hori, vertical=vert, default=def_)  # noqa: E501
        if guide == 'colorbar':
            kw_guide.update({} if span is None else {'span': span})
            src.colorbar(objs[0], label=label, loc=loc, **kw_guide)
        else:
            tups = handles.setdefault((src, span), [])
            tups.append((objs[0], label, loc, kw_guide))
    for (src, span), tups in handles.items():  # support 'queue' for figure legends
        objs, labels, locs, kws = zip(*tups)
        kw = {key: val for kw in kws for key, val in kw.items()}
        if span is not None:  # figure row or column span
            kw['span'] = span
        src.legend(list(objs), list(labels), loc=locs[0], **kw)
    if standardize:
        for (axis, _), axs in axs_units.items():
            lims = []
            for ax in axs:
                if getattr(ax, f'get_autoscale{axis}_on')():
                    ax.autoscale(axis=axis)  # trigger scaling for shading
                lims.append(getattr(ax, f'get_{axis}lim')())
            span = max(abs(lim[1] - lim[0]) for lim in lims)
            for ax, lim in zip(axs, lims):
                avg = 0.5 * (lim[0] + lim[1])
                lim = (avg - 0.5 * span, avg + 0.5 * span)
                getattr(ax, f'set_{axis}lim')(lim)

    # Format the axes and optionally save
    # NOTE: Here default labels are overwritten with non-none 'rowlabels' or
    # 'collabels', and the file name can be overwritten with 'save'.
    custom = {'rowlabels': rowlabels, 'collabels': collabels}
    default = {'rowlabels': gridlabels[0], 'collabels': gridlabels[1]}
    for num in gridskip:  # kludge to center super title above empty slots
        ax = fig.add_subplot(gs[num])
        for obj in (ax.xaxis, ax.yaxis, ax.patch, *ax.spines.values()):
            obj.set_visible(False)
    for key, clabels, dlabels in zip(custom, custom.values(), default.values()):
        nlabels = nrows if key == 'rowlabels' else ncols
        clabels = clabels or [None] * nlabels
        dlabels = dlabels or [None] * nlabels
        if len(dlabels) != nlabels or len(clabels) != nlabels:
            raise RuntimeError(f'Expected {nlabels} labels but got {len(dlabels)} and {len(clabels)}.')  # noqa: E501
        fig.format(figtitle=figtitle)
        fig.format(**{key: [clab or dlab for clab, dlab in zip(clabels, dlabels)]})
    if save:
        if save is True:
            path = Path()
        else:
            path = Path(save).expanduser()
        figs = path / 'figures'
        if figs.is_dir():
            path = figs
        if path.is_dir():
            path = path / '_'.join(('-'.join(methods), '-'.join(commands), pathlabel))
        print(f'Saving {path.parent}/{path.name}...')
        fig.save(path)  # optional extension
    return fig, fig.subplotgrid
