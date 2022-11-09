#!/usr/bin/env python3
"""
Templates for figures detailing coupled model output.
"""
import warnings
from pathlib import Path

import numpy as np
import xarray as xr
from icecream import ic  # noqa: F401

import proplot as pplt
from climopy import decode_units, format_units, ureg, var, vreg  # noqa: F401
from .process import _apply_method, _parse_institute, _parse_project, get_data
from .internals import _wrap_label, parse_specs

__all__ = ['breakdown_feedbacks', 'breakdown_transport', 'plot_general']

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
    cartesian = ax._name == 'cartesian'
    refwidth = ax.figure._refwidth
    refscale = 1.3 if cartesian else 0.8  # applied for y-axis labels only
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
        xlabel = ''
        xunits = x.attrs.get('units', None)
        xprefix = x.attrs.get('xlabel_prefix', '')
        x.attrs.pop('short_prefix', None)  # ignore legend indicators
        if command == 'scatter' and len(xprefix) < 20:
            x.attrs['short_prefix'] = xprefix
        if 'short_name' not in x.attrs:
            x.attrs['short_name'] = ''
        if xunits is not None:
            xlabel = x.climo.short_label
        if ncols > 1:
            xlabel = _wrap_label(xlabel, refwidth=refwidth)
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
        ylabel = ''
        yunits = y.attrs.get('units', None)
        yprefix = y.attrs.get('ylabel_prefix', '')
        y.attrs.pop('short_prefix', None)  # ignore legend indicators
        if command == 'scatter' and len(yprefix) < 20:
            y.attrs['short_prefix'] = yprefix
        if 'short_name' not in y.attrs:
            y.attrs['short_name'] = ''
        if yunits is not None:
            ylabel = y.climo.short_label
        if nrows > 1:
            ylabel = _wrap_label(ylabel, refwidth=refwidth, refscale=refscale)
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
    args = list(args)
    cycle = pplt.Cycle(cycle or ['blue7', 'red7', 'yellow7', 'gray7'])
    colors = pplt.get_colors(cycle)
    flier = {'markersize': 2, 'marker': 'x'}
    kw_hash = {'colors': 'none', 'edgecolor': 'k'}
    kw_contour = {'robust': 96, 'nozero': True, 'linewidth': pplt.rc.metawidth}
    kw_contourf = {'robust': 98, 'levels': 20, 'extend': 'both'}
    kws_contour = []
    kws_contour.append({'color': 'gray8', 'linestyle': None, **kw_contour})
    kws_contour.append({'color': 'gray3', 'linestyle': ':', **kw_contour})
    kw_bar = {'linewidth': pplt.rc.metawidth, 'edgecolor': 'black', 'width': 1.0}
    kw_line = {'linestyle': '-', 'linewidth': 1.5 * pplt.rc.metawidth}
    kw_scatter = {'color': 'gray7', 'linewidth': 1.5 * pplt.rc.metawidth}
    kw_scatter.update({'marker': 'x', 'markersize': 0.1 * pplt.rc.fontsize ** 2})
    kw_box = {'means': True, 'whis': (5, 95), 'widths': 0.7, 'flierprops': flier}
    sizes = args[-1].sizes.keys() - {'facets', 'version', 'period'}

    # Helper functions
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

    # Infer commands
    # TODO: Support hist and hist2d perlots in addition to scatter and barh plots
    # (or just hist since, hist2d usually looks ugly with so little data)
    num = num or 0
    offset = offset or 0
    defaults = {}
    if len(sizes) == 2:
        if 'hatches' in kwargs:
            command = 'contourf'
            defaults = kw_hash.copy()
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
            if alternate:  # TODO: handle parent axis color inside plot_general()
                ax.format(xcolor=color0)
                ax = ax.altx(color=color1)
        else:
            command = 'line'
            color0 = colors[max(num - 1, 0) % len(colors)]
            color1 = colors[num % len(colors)]
            defaults = {'color': color1, **kw_line}
            if alternate:  # TODO: handle parent axis color inside plot_general()
                ax.format(ycolor=color0)
                ax = ax.alty(color=color1)
    elif len(sizes) == 0 and multiple:
        if len(args) == 2:  # compare correlations
            command = 'barh' if horizontal else 'bar'
            project = args[0].project.values[0]
            (slope,), *_ = _apply_method(*args, method='slope')
            short_prefix = project
            if hasattr(args[0], 'short_prefix') and hasattr(args[1], 'short_prefix'):
                short_prefix += f' {args[1].short_prefix} vs. {args[0].short_prefix}'
            slope.attrs['short_prefix'] = short_prefix
            args = (np.array([num + offset]), slope)  # TODO: add error bars
            color = colors[num % len(colors)]
            defaults = {'color': color, **kw_bar}
        else:  # compare distributions
            # command = 'violinh' if horizontal else 'violin'
            command = 'boxh' if horizontal else 'box'
            args = (np.array([num + offset]), args[-1].expand_dims('num', axis=1))
            color = colors[num % len(colors)]
            defaults = {'facecolor': color, **kw_box}
            if 'color' in kwargs:  # only color needs to be translated (alpha works)
                kwargs['facecolor'] = kwargs.pop('color')
        data = args[-1].copy()
        data.attrs['units'] = ''
        string = data.attrs.get('short_prefix', data.attrs.get('short_name', ''))
        string = _wrap_label(string, 20 * pplt.rc.fontsize / 72, nmax=1)
        if label:
            if horizontal:
                trans = ax.get_yaxis_transform()
                align = {'ha': 'right', 'va': 'center', 'rotation': 0}
                ax.text(-0.05, num + offset, string, transform=trans, **align)
            else:
                trans = ax.get_xaxis_transform()
                align = {'ha': 'center', 'va': 'top', 'rotation': 90}
                ax.text(num + offset, -0.05, string, transform=trans, **align)
    elif len(sizes) == 0 and not multiple:
        if len(args) == 2:  # plot correlation
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
            color = ['gray8' if name[:3] in ('ecs', 'erf') else c for c in color]
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


def breakdown_transport(breakdown=None, component=None, transport=None):
    """
    Return the names and colors associated with transport components.

    Parameters
    ----------
    breakdown : str, optional
        The transport components to show.
    transport : str, optional
        The transport type to show.

    Returns
    -------
    names : list of str
        The transport components.
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
    return names, colors


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
    # Interpret keyword arguments from suffixes
    # NOTE: Idea is to automatically remove feedbacks, filter out all-none
    # rows and columns at the end, and then infer the 'gridskip' from them.
    original = maxcols = maxcols or 4
    if breakdown is None:
        component = component or 'net'
    if component is None:
        breakdown = breakdown or 'all'
    if breakdown and '_lam' in breakdown:
        breakdown, *_ = breakdown.split('_lam')
        feedbacks = True
    if breakdown and '_adj' in breakdown:
        breakdown, *_ = breakdown.split('_erf')
        adjusts, forcing = True, False  # effective forcing *and* rapid adjustments
    if breakdown and '_erf' in breakdown:
        breakdown, *_ = breakdown.split('_erf')
        forcing, adjusts = True, False  # effective forcing *without* rapid adjustments
    if breakdown and '_ecs' in breakdown:
        breakdown, *_ = breakdown.split('_ecs')
        sensitivity = True
    if not component and not feedbacks and not forcing and not sensitivity:
        raise RuntimeError

    # Three variable breakdowns
    # NOTE: Options include 'wav', 'atm', 'alb', 'res', 'all', 'atm_wav', 'alb_wav'
    # with the 'wav' suffixes including longwave and shortwave cloud components
    # instead of a total cloud feedback. Strings denote successively adding atmospheric,
    # albedo, residual, and remaining temperature/humidity feedbacks with options.
    def _get_arrayfeedback_breakdown(cols):
        names = np.array([[None] * cols] * 25)
        iflat = names.flat
        return names, iflat
    if component is not None:
        names, iflat = _get_array(maxcols)
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
            names, iflat = _get_array(maxcols)
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
            names, iflat = _get_array(maxcols)
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
            names, iflat = _get_array(maxcols)
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
            names, iflat = _get_array(maxcols)
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
            names, iflat = _get_array(maxcols)
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
            names, iflat = _get_array(maxcols)
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
            names, iflat = _get_array(maxcols)
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
            names, iflat = _get_array(maxcols)
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
        gridskip = np.array([])
    return names, maxcols, gridskip


def plot_general(
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
    annotate=False,
    linefit=False,
    oneone=False,
    proj='hammer',
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
        The cartopy projection for longitude-latitude type plots. Default is the
        area-conserving and polar distortion-minimizing ``'hammer'`` projection.
    proj_kw : dict, optional
        The cartopy projection keyword arguments for longitude-latitude
        type plots. Default is ``{'lon_0': 180}``.
    save : path-like, optional
        The save folder base location. Stored inside a `figures` subfolder.
    cycle, pcolor, horizontal : optional
        Passed to `_infer_command`.
    **kw_specs
        Passed to `parse_specs`.
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

    # Iterate over axes and plots
    # NOTE: Critical to disable 'grouping' so that e.g. colorbars or legends that
    # extend into other panel slots are not considered in the tight layout algorithm.
    fig = gs = None  # delay instantiation
    proj = pplt.Proj(proj, **(proj_kw or {'lon_0': 180}))
    queue, methods, commands = {}, [], []
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
            if max(nrows, ncols) == 1:
                kw_axs.pop('abc', None)
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
            tups = queue.setdefault(key, [])
            tups.append((jax, args, kw_plt))
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
    for k, (key, values) in enumerate(queue.items()):
        # Get guide and plotting arguments
        # NOTE: Here 'argskip' is isued to skip arguments with vastly different
        # ranges when generating levels that annotate multiple different subplots.
        print(f'{k + 1}/{len(queue)}', end=' ')
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
            cartesian = axs[0]._name == 'cartesian'
            refwidth = axs[0].figure._refwidth or pplt.rc['subplots.refwidth']
            refscale = 1.3 if cartesian else 0.8  # WARNING: vertical colorbars only
            extendsize = 1.2 if cartesian else 1.8  # WARNING: vertical colorbars only
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
            variable = args[0][-1].climo.cfvariable
            label = variable.short_label if command == 'contour' else variable.short_name  # noqa: E501
            label = None if hatches else label
            keys = ['cmap', 'norm', 'norm_kw']
            maps = ['robust', 'symmetric', 'diverging', 'levels', 'locator', 'extend']
            if 'contour' not in command and 'pcolor' not in command:
                keys.extend(maps)
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
        if not len(rows) == 1 ^ len(cols) == 1:
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
            span = None
        elif loc[0] in 'lr':
            span = (min(rows) + 1, max(rows) + 1)
        else:
            span = (min(cols) + 1, max(cols) + 1)
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
