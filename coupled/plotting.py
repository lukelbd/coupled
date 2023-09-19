#!/usr/bin/env python3
"""
Plotting functions for showing coupled model output.
"""
import copy
import itertools
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from icecream import ic  # noqa: F401

import proplot as pplt
import matplotlib.legend_handler as mhandler
import matplotlib.collections as mcollections
import matplotlib.container as mcontainer
import matplotlib.lines as mlines
import seaborn as sns
from climopy import ureg, vreg  # noqa: F401
from .internals import _capitalize_label, _fit_label, _infer_labels, parse_specs
from .process import _constrain_response, process_data
from .reduce import _components_corr, _components_slope, _parse_institute, _parse_project  # noqa: E501

__all__ = ['generate_plot']

# Default configuration settings
# See: https://agu.org/Publish-with-AGU/Publish/Author-Resources/Graphic-Requirements
CONFIG_SETTINGS = {
    'fontname': 'Helvetica',
    'autoformat': False,
    'inlineformat': 'png',  # switch between png and retina
    'fontsize': 8.0,  #
    'negcolor': 'cyan7',  # differentiate from colors used for variables
    'poscolor': 'pink7',
    'unitformat': '~L',
    'axes.inbounds': False,  # ignore for proper shading bounds
    'axes.margin': 0.03,
    'bottomlabel.pad': 6,
    'cmap.inbounds': False,
    'colorbar.extend': 1.0,
    'colorbar.width': 0.18,  # inches not font-sizes
    'grid.alpha': 0.06,  # gridline opacity
    'hatch.linewidth': 0.3,
    'leftlabel.pad': 6,
    'legend.handleheight': 1.2,
    'rightlabel.pad': 8,
    'subplots.refwidth': 2,
    'toplabel.pad': 8,
}
pplt.rc.update(CONFIG_SETTINGS)

# Default color cycle
CYCLE_DEFAULT = [
    f'{color}6' if color == 'gray' else f'{color}7'
    for color in (
        'gray',
        'blue',
        'yellow',
        'red',
        'cyan',
        'pink',
        'indigo',
        'orange',
    )
]

# Properties to preserve when grouping legend handles
# TODO: Permit both project and flagship labels
PROPS_LEGEND = (
    'alpha',
    'color',
    'cmap',
    'edgecolor',
    'facecolor',
    'hatch',
    'marker',
    'markersize',
    'markeredgewidth',
    'markeredgecolor',
    'markerfacecolor',
    'linestyle',
    'linewidth',
    # 'sizes',  # NOTE: ignore flagship indicator
)

# Properties to ignore for grouping like commands
PROPS_PLURAL = (
    'width',
    'mean', 'median',
    'showmean', 'showmedian',
    'shadestd', 'shadepctile',
    'fadestd', 'fadepctile',
    'barstd', 'barpctile',
    'boxstd', 'boxpctile',
)
PROPS_IGNORE = (
    'absolute_width',
    'absolute_size',
    'negpos',
    'negcolor',
    'poscolor',
    'color',  # join separate bar plots
    'hatch',  # join separate bar plots
    'sym',
    'whis',
    'flierprops',
    *(f'{props}{suffix}' for suffix in ('', 's') for props in PROPS_PLURAL),
)

# Default format keyword arguments
KWARGS_FIG = {
    'refwidth': 1.3,
    'abcloc': 'l',
    'abc': 'A.',
}
KWARGS_GEO = {
    'proj': 'hammer',
    'proj_kw': {'lon_0': 210},
    'lonlines': 30,
    'latlines': 30,
    'refwidth': 2.0,
    'titlepad': 3,  # default is 5 points
    'abcloc': 'ul',
    'coast': True,
}
KWARGS_LAT = {
    'xlabel': 'latitude',
    'xformatter': 'deg',
    'xlocator': 30,
    'xscale': 'sine',  # either 'sine' or 'linear'
    'xlim': (-89, 89),
}
KWARGS_PLEV = {
    'ylim': (1000, 70),
    'yminorlocator': 100,
    'ylocator': 200,
    'ylabel': 'pressure (hPa)',
    # 'yreverse': True,
}

# Default extra plotting keyword arguments
KWARGS_ANNOTATE = {  # annotation text
    'color': 'gray8',
    'alpha': 1.0,
    'textcoords': 'offset points',
}
KWARGS_BACKGROUND = {
    'alpha': 0.05,
    'color': 'black',
    'zorder': 0.1,
    'linewidth': 0.0,
}
KWARGS_CENTER = {  # violinplot markers
    'color': 'w',
    'marker': 'o',
    'markersize': (5 * pplt.rc.metawidth) ** 2,
    'markeredgecolor': 'k',
    'markeredgewidth': pplt.rc.metawidth,
    'absolute_size': True,
}
KWARGS_ERRBAR = {  # thin error whiskers
    'capsize': 0,
    'barcolor': 'gray8',
    'barlinewidth': 1.5 * pplt.rc.metawidth,
}
KWARGS_ERRBOX = {  # thick error whiskers
    'capsize': 0,
    'boxcolor': 'gray8',
    'boxlinewidth': 3.5 * pplt.rc.metawidth,
}
KWARGS_REFERENCE = {  # reference zero or one line
    # 'alpha': 0.15,
    'alpha': 0.50,
    'color': 'black',
    'scalex': False,
    'scaley': False,
    'zorder': 0.50,
    'linestyle': ':',  # differentiate from violin lines
    'linewidth': 1.2 * pplt.rc.metawidth,
}

# Default main plotting keyword arguments
# NOTE: If expect the shortest violin plots have length 1 W m-2 K-1 and are roughly
# triangle shaped then maximum density sample in units 1 / (W m-2 K-1)-1 is 2. Idea
# should be to make the maximum around 1 hence scaling violin widths by 0.5.
KWARGS_BAR = {
    'color': 'gray8',
    'edgecolor': 'black',
    'width': 1.0,
    'absolute_width': False,
}
KWARGS_BOX = {
    'whis': (2.5, 97.5),  # consistent with default 'pctiles' ranges
    'means': False,
    'widths': 1.0,  # previously 0.85
    'color': 'gray8',
    'linewidth': pplt.rc.metawidth,
    'flierprops': {'markersize': 2, 'marker': 'x'},
    'manage_ticks': False,
}
KWARGS_CONTOUR = {
    'globe': True,
    'color': 'gray9',
    'robust': 96,
    'nozero': True,
    'labels': True,
    'levels': 3,
    'linewidth': pplt.rc.metawidth,
}
KWARGS_HATCH = {
    'globe': True,
    'color': 'none',
    'linewidth': 0,
    'edgecolor': 'black'
}
KWARGS_LINE = {
    'color': 'gray8',
    'linestyle': '-',
    'linewidth': 2.5 * pplt.rc.metawidth,
}
KWARGS_SCATTER = {
    'color': 'gray8',
    'marker': 'x',
    'markersize': 0.1 * pplt.rc.fontsize ** 2,
    'linewidth': 1.5 * pplt.rc.metawidth,  # for marker thickness
    'absolute_size': True,
}
KWARGS_SHADING = {
    'globe': True,
    'robust': 98,
    'levels': 20,
    'extend': 'both',
}
KWARGS_VIOLIN = {
    'color': 'gray8',
    'inner': 'stick',  # options are None, 'stick', or 'box' for 1.5 * interquartile
    'scale': 'width',  # options are 'width', 'count', or 'area' (default)
    'width': 0.5,  # kde pdf scale applied to native units '1 / W m-2 K-1'
    'bw': 0.3,  # in feedback or temperature units
    'cut': 0.3,  # no more than this fraction of bandwidth (still want to see lines)
    'saturation': 1.0,  # exactly reproduce input colors (not the default! weird!)
    'linewidth': pplt.rc.metawidth,
}


def _get_coords(*args, units=True, tuples=False):
    """
    Get the scalar coordinates for labels and groups.

    Parameters
    ----------
    *args : xarray.DataArray
        The input arrays.
    units : bool, optional
        Whether to include units attributes.
    tuples : bool, optional
        Whether to include tuple index values.

    Returns
    -------
    result : dict of list
        The resulting coordinates.
    """
    # NOTE: This is used in both _combine_commands and in generate_plot for assigning
    # groups of shared colormaps and legends based on coordinates.
    attrs = ('name', 'units') if units else ('name',)
    keys = list(attrs)
    keys += sorted(set(key for arg in args for key in getattr(arg, 'coords', ())))
    result = {}
    for key in keys:
        values = []
        for arg in args:
            if not isinstance(arg, xr.DataArray):
                continue
            if key in attrs:
                value = np.array(getattr(arg, key, None))
            else:
                value = np.array(arg.coords.get(key))
            if value.size != 1:
                continue
            if np.issubdtype(value.dtype, float) and np.isnan(value):
                continue
            if isinstance(value := value.item(), tuple) and not tuples:
                continue
            values.append(value)
        if len(args) == len(values):
            result[key] = tuple(values)
    return result


def _get_flagships(data):
    """
    Get boolean flagship status for data facets.

    Parameters
    ----------
    data : xarray.DataArray
        The input array.

    Returns
    -------
    numpy.ndarray
        The booleans.
    """
    if 'facets' not in data.coords:
        raise ValueError('Facets coordinate is missing.')
    elif 'institute' in data.coords:
        bools = [data.coords['institute'].item() == 'flagship'] * data.facets.size
    else:
        filt = _parse_institute('flagship')
        bools = []
        names = data.indexes['facets'].names
        if 'project' not in names and 'project' not in data.coords:
            raise ValueError('Project version is missing.')
        for facet in data.facets.values:
            if 'project' not in names:
                facet = (data.coords['project'].item().upper(), *facet)
            b = filt(tuple(facet))
            bools.append(b)
    return np.array(bools)


def _get_projects(data):
    """
    Get string sub-project name for data facets.

    Parameters
    ----------
    data : xarray.DataArray
        The input array.

    Returns
    -------
    numpy.ndarray
        The projects.
    """
    if 'project' in data.coords:  # __contains__ excludes index levels
        projects = [data.coords['project'].item()] * data.sizes.get('facets', 1)
    elif 'facets' not in data.coords:
        raise ValueError('Facets coordinate is missing.')
    else:
        filt65 = _parse_project(data.facets.values, 'cmip65')
        filt66 = _parse_project(data.facets.values, 'cmip66')
        projects = []
        for facet in data.facets.values:
            if filt66(facet):
                project = 'cmip66'
            elif filt65(facet):
                project = 'cmip65'
            else:
                project = 'cmip5'
            projects.append(project)
        projects = np.array(projects)
    return projects


def _get_handles(args, handles, pad=None, size=None, sort=True):
    """
    Return multiple handles and labels from a single plot.

    Parameters
    ----------
    args : list of xarray.DataArray
        The input data arrays.
    handles : list of iterable
        The input plot handle lists.
    sort : bool, optional
        Whether to sort input values.

    Other Parameters
    ----------------
    pad : optional
        Total padding between items.
    size : optional
        Total length and height of each item.

    Returns
    -------
    handles : list of proplot.Artist
        The handles.
    labels : list of str
        The labels.
    kw_legend : dict
        Additional legend keyword arguments.
    """
    # Helper function
    # NOTE: A 'luminance' feature was originally needed for sorting violins for which
    # we apply alpha manually but now just get handles with insertion order.
    # values = [1] * len(colors)
    # if luminance:  # include for e.g. violins with manual opacity applied
    #     values = [100 - pplt.to_xyz(value, space='hsv')[2] for value in colors]
    # keys = list(zip(alphas, hatches, widths, range(len(alphas))))
    def _sort_idxs(handles, sort=True):
        none = lambda: None
        idxs = list(range(len(handles)))
        if sort and handles and hasattr(handles[0], 'get_facecolor'):
            colors = [getattr(handle, 'get_facecolor', none)() for handle in handles]
            colors = ['none' if value is None else value for value in colors]
            colors = [value.squeeze() if hasattr(value, 'squeeze') else value for value in colors]  # noqa: E501
            alphas = [pplt.to_rgba(value)[3] for value in colors]
            hatches = [bool(getattr(handle, 'get_hatch', none)()) for handle in handles]
            widths = [handle.get_linewidth() for handle in handles]
            keys = list(zip(alphas, hatches, widths, range(len(alphas))))
            idxs = [idx for idx, _ in sorted(enumerate(keys), key=lambda pair: pair[1])]
        return idxs

    # Iterate jointly over arguments handle components
    # NOTE: Identify handles that have not yet been drawn by explicit list
    # of possible artist properties. Can add to this list.
    # WARNING: Iterate in reverse order for special case where effective climate
    # sensitivity places on first column but has empty violin slots.
    tuples = {}
    properties = set()
    for num, (arg, ihandles) in enumerate(zip(args, handles)):
        labels = arg.coords.get('label', None)
        if labels is None:
            raise ValueError('Input data has no label coordinate.')
        labels = [label for label in labels.values if label is not None]  # for seaborn
        if len(ihandles) > len(labels):  # allow less than for e.g. sensitivity
            raise ValueError(f'Number of labels ({len(labels)}) differs from handles ({len(ihandles)}).')  # noqa: E501
        for idx in _sort_idxs(ihandles, sort=sort)[::-1]:  # e.g. cmip5 then cmip6
            label, handle = labels[idx], ihandles[idx]
            props = {}
            if not handle:  # empty violin slot due to e.g. all-zero array
                continue
            for key, value in handle.properties().items():
                if key not in PROPS_LEGEND:
                    continue
                if hasattr(value, 'name'):  # e.g. PathCollection, LineCollection cmap
                    value = value.name
                if isinstance(value, list):
                    value = tuple(value)
                if getattr(value, 'ndim', None) == 1:
                    value = tuple(value.tolist())
                if np.issubdtype(getattr(value, 'dtype', object), float):
                    value = np.round(value, 3)
                if getattr(value, 'ndim', None) == 2:
                    value = tuple(map(tuple, value.tolist()))
                props[key] = value
            tups = tuples.setdefault(label, [])
            props = tuple(props.items())
            if props not in properties:  # only unique handles
                tups.append((num, idx, handle))  # sort by original index below
                properties.add(props)

    # Get legend args and kwargs
    # NOTE: Try to make each box in the handle-tuple approximately square and use
    # same 'ndivide' for all entries so that final handles have the same length.
    pad = pad or 0  # make items flush by default
    size = size or pplt.rc['legend.handleheight']
    result = {}
    for label, handles in tuple(tuples.items())[::-1]:  # restore *original* order
        handles = [handle for *_, handle in sorted(handles, key=lambda tup: tup[:2])]
        handle = tuple(handles[idx] for idx in _sort_idxs(handles, sort=False))
        result[label] = handle  # combined with tupler handle
    ndivide = max(map(len, result.values()))
    handler = mhandler.HandlerTuple(pad=pad, ndivide=ndivide)
    kw_legend = {
        'handleheight': size,
        'handlelength': size * max(map(len, result.values())),
        'handler_map': {tuple: handler},  # or use tuple coords as map keys
    }
    labels, handles = map(list, zip(*result.items()))
    return handles, labels, kw_legend


def _auto_guide(*axs, loc=None, figurespan=False, cbarlength=None, cbarwrap=None):
    """
    Generate a source, location, and span for the guide based on input axes.

    Parameters
    ----------
    *axs : proplot.Axes
        The input axes.
    loc : str, optional
        The input guide location.
    figurespan : bool, optional
        Whether to make colorbars and legends span the entire figure.
    cbarlength : float, default: auto
        The relative colorbar length based on input.
    cbarwrap : float, default: 1.2
        Scaling to apply to size used to wrap colorbar labels.

    Returns
    -------
    src : object
        The figure or axes to use.
    loc : str
        The guide location to use.
    span : 2-tuple
        The span for figure-edge labels.
    bbox : 2-tuple
        The bounding box for legends.
    length : float
        The relative length for colorbars.
    size : float
        The size in inches.
    """
    # Infer gridspec location of axes
    # WARNING: This all must come after axes are drawn and content is plotted inside.
    # Otherwise position will be wrong.
    fig = axs[0].figure
    axs = pplt.SubplotGrid(axs)  # then select
    rows = tuple(zip(*(ax._get_topmost_axes()._range_subplotspec('y') for ax in axs)))
    cols = tuple(zip(*(ax._get_topmost_axes()._range_subplotspec('x') for ax in axs)))
    nrows, ncols = axs.shape  # shape of underlying gridspec
    rows = (min(rows[0]), max(rows[1]))
    cols = (min(cols[0]), max(cols[1]))
    row = (rows[0] + rows[1] + 1) // 2  # prefer bottom i.e. higher index
    col = (cols[0] + cols[1]) // 2  # prefer left i.e. lower index
    if loc is None:
        pass
    elif not isinstance(loc, str) or loc[0] not in 'lrtb':
        raise ValueError(f'Invalid location {loc!r}.')
    if figurespan:
        src = fig
        if loc is not None:
            pass
        elif max(cols) == axs.shape[1] - 1 and max(rows) != axs.shape[0] - 1:
            loc = 'right'
        elif min(cols) == 0 and max(rows) != axs.shape[0] - 1:
            loc = 'left'
        else:
            loc = 'bottom'
    else:
        if loc is None:
            loc = 'right' if len(rows) == 1 else 'bottom'
        if loc[0] == 't':
            src = fig if min(rows) == 0 else axs[rows[0], col]
        elif loc[0] == 'l':
            src = fig if min(cols) == 0 else axs[row, cols[0]]
        elif loc[0] == 'b':
            src = fig if max(rows) == nrows - 1 else axs[rows[1], col]
        elif loc[0] == 'r':
            src = fig if max(cols) == ncols - 1 else axs[row, cols[1]]
        else:
            src = axs[row, col]  # fallback but should be impossible
    if isinstance(src, pplt.SubplotGrid):
        src = axs[0]  # fallback in case empty list

    # Infer settings
    # TODO: Here 'extendsize' is always accurate since we call .auto_layout(tight=False)
    # whenever axes are created, but in current proplot branch this is wrong. Should
    # consider applying .auto_layout(tight=False) after drawing each axes.
    if loc[0] in 'lr':
        nspan = max(rows) - min(rows) + 1
        span = (min(rows) + 1, max(rows) + 1) if src is fig else None
    else:
        nspan = max(cols) - min(cols) + 1
        span = (min(cols) + 1, max(cols) + 1) if src is fig else None
    shrink = 0.5 if nspan > 2 else 0.66  # shorter for e.g. 3 or 4 columns
    adjust = 1.3 if src is fig and nspan > 1 else 1.0
    offset = 1.10 if axs[0]._name == 'cartesian' else 1.02  # for colorbar and legend
    factor = cbarwrap or 1.2  # additional scaling
    xoffset = loc[0] in 'lr' and (rows[1] - rows[0]) % 2
    yoffset = loc[0] in 'tb' and (cols[1] - cols[0]) % 2
    xcoords = {'l': 1, 'b': offset, 'r': 0, 't': offset}
    ycoords = {'l': offset, 'b': 1, 'r': offset, 't': 0}
    width, height = axs[0]._get_size_inches()
    if cbarlength:
        length = cbarlength
    elif nspan == 1:
        length = 1
    elif src is fig:  # shrink along figure
        length = shrink
    elif src is not fig:  # extend beyond axes
        length = adjust * nspan * shrink
    if src is fig or not xoffset and not yoffset:
        bbox = None
        length = (0.5 - length / 2, 0.5 + length / 2)
    else:
        bbox = (xcoords[loc[0]], ycoords[loc[0]])
        length = (offset - length / 2, offset + length / 2)
    if loc[0] in 'lr':  # get size used for wrapping
        size = nspan * height
    else:
        size = nspan * width
    if src is fig:  # include space between subplots
        size += (adjust - 1) * (nspan - 1)
    if factor:  # NOTE: currently always true
        size *= factor
    return src, loc, span, bbox, length, size


def _auto_props(data, kw_collection):
    """
    Automatically infer scalar plot properties.

    Parameters
    ----------
    data : xarray.DataArray
        Input data argument.
    kw_collection : namedtuple
        Inpute keyword arguments.

    Other Parameters
    ----------------
    multicolor : bool, optional
        Whether to use colors instead of alpha for projects (taken from `.other`).
    cycle : cycle-spec, optional
        The manual color cycle (taken from `.other`).

    Returns
    -------
    colors : list
        The default colors.
    kw_patch : dict
        The default patch properties.
    kw_scatter : list
        The default marker properties.
    """
    # Get derived coordinates
    # WARNING: Previously tried np.repeat([(0, 0, 150)], data.size) for periods
    # but expanded the values and resulted in N x 3. Stick to lists instead.
    # NOTE: This preserves project/institute styling across bar/box/regression
    # plots by default, but respecting user-input alpha/color/hatch etc.
    def _get_lists(values, options, default=None):  # noqa: E306
        values = list(values)  # from arbitrary iterable
        if len(values) == 1:
            values = [values[0] for _ in range(data.size)]
        result = [options.get(value, default) for value in values]
        if len(set(result)) <= 1:
            result = [default] * len(values)
        return result
    cycle = kw_collection.other.get('cycle', None)
    others = []  # other coordinates for color cycle
    linewidth = 0.6 * pplt.rc.metawidth  # default reference bar line width
    markersize = 0.1 * pplt.rc.fontsize ** 2  # default reference scatter marker size
    multicolor = kw_collection.other.get('multicolor', False)
    keepcolor = ('project', 'institute')  # no multicolor colors
    keepcolor += (() if multicolor else ('experiment', 'start', 'stop'))
    for key in (idx := data.indexes[data.dims[0]]):  # 'facets' or 'components'
        other = tuple(key for name, key in zip(idx.names, key) if name not in keepcolor)
        others.append(other)
    if 'facets' in data.sizes:  # others == 'model', do not want unique colors!
        flagship = _get_flagships(data)
        project = _get_projects(data)
    else:
        flagship = np.atleast_1d(data.coords.get('institute', None))
        project = np.atleast_1d(data.coords.get('project', None))

    # Standardize coordinates for inferring default properties
    # WARNING: Coordinates created by _combine_commands might have NaN in place
    # of None values in combination with other floats. Also np.isnan raises float
    # casting errors and 'is np.nan' can fail for some reason. So take advantage of
    # the property that np.nan != np.nan evaluates to true unlike all other objects.
    experiment = np.atleast_1d(data.coords.get('experiment', None))
    start = np.atleast_1d(data.coords.get('start', None))
    stop = np.atleast_1d(data.coords.get('stop', None))
    if stop.size == 1:  # TODO: figure out issue and remove kludge
        stop = np.repeat(stop, start.size)
    if start.size == 1:  # TODO: figure out issue and remove kludge
        start = np.repeat(start, stop.size)
    mask = np.array([value is None or value != value for value in experiment])
    mask1 = np.array([value is None or value != value for value in start])
    mask2 = np.array([value is None or value != value for value in stop])
    experiment = np.where(mask, 'abrupt4xco2', experiment)
    start, stop = np.where(mask1, 0, start), np.where(mask2, 150, stop)
    perturbs = {'picontrol': 0, 'abrupt4xco2': 1, 'abrupt4xco2-picontrol': 1}
    perturbs = [perturbs.get(opt, 2) for opt in experiment]
    periods = list(zip(start.tolist(), stop.tolist()))
    projs = [int(opt[4] if opt and len(opt) in (5, 6) else 0) for opt in project]
    groups = [int(opt[4:] if opt and len(opt) in (5, 6) else 0) for opt in project]

    # General settings associated with coordinates
    # NOTE: Here 'cycle' used for groups without other settings by default.
    # periods = periods * (len(perturbs) if len(periods) == 1 else 1)
    perturbs = perturbs * (len(periods) if len(perturbs) == 1 else 1)
    tuples = list(zip(perturbs, *zip(*periods)))
    cycle = CYCLE_DEFAULT if cycle is None else pplt.get_colors(cycle)
    delta = 0 if perturbs[0] == 0 and len(set(perturbs)) == 1 else 1
    control = [(0, 0, 150)]
    early = [(delta, 0, 20), (delta, 0, 50), (delta, 2, 20), (delta, 2, 50)]
    late = [(delta, 20, 150), (delta, 100, 150)]  # label as 'late' but is just control
    full = [(delta, 0, 150), (delta, 2, 150)]
    sizes = {False: 0.50 * markersize, True: 1.5 * markersize}  # institute
    edge1 = {key: 'gray3' for key in early}  # early groups
    edge2 = {key: 'gray9' for key in (*late, *full)}
    hatch0 = {key: 'xxxxxx' for key in control}  # unperturbed
    hatch1 = {key: 'ooo' for key in early}
    hatch1 = {} if ('20-0', '150-20') in periods else hatch1
    hatch2 = {key: '...' for key in (*late, *full)}
    edge, edges = groups, {66: 'gray3'}  # default edges
    hatch, hatches = groups, {66: 'xxxxxx'}  # default hatches
    alpha, alphas = projs, {5: 0.3}  # default alpha
    fade, fades = flagship, {False: 0.85, True: 1}  # flagship status
    # hatch0 = {key: 'xxxxxx' for key in control}  # alternative
    # hatch1 = {key: 'xxxxxx' for key in early}
    # hatch2 = {key: 'xxx' for key in (*late, *full)}

    # Additional settings dependent on input coordinates
    # NOTE: Here ignore input 'cycle' if multicolor was passed. Careful to support
    # both _combine_commands() regression bars/violins and distribution bar plots.
    if not multicolor:
        color = others
        # seen = set(order)  # previously used for 'multicolor'
        # colors = [key for key in order if key in color]
        # colors.extend((key for key in color if key not in seen and not seen.add(key)))
        seen = set()  # record auto-generated color names
        colors = [key for key in others if key not in seen and not seen.add(key)]
        colors = {key: cycle[i % len(cycle)] for i, key in enumerate(colors)}
        if len(set(tuples)) > 1:  # use hatching for early/late or experiment instead
            edge, hatch = tuples, tuples
            alpha, alphas = groups, {5: 0.3, 56: 0.3, 65: 0.6}
            edges = {} if len(set(periods)) == 1 else {**edge1, **edge2}
            hatches = hatch0 if len(set(periods)) == 1 else hatch1  # early hatching
            hatches = hatches if len(set(periods)) == 1 else {**hatch1, **hatch2}
    else:
        color = tuples
        # order = [*carly, *late, *full, *control]
        edge, hatch = groups, groups
        colors = {}
        if len(set(tuples)) == 1:  # e.g. cmip5 vs. cmip6 regressions
            colors[tuples[0]] = 'gray6'
            colors.update({key: 'pink7' for key in (*early, *late, *full)})
        elif not set(tuples) & set(early):  # no early-late partition
            colors.update({key: 'cyan7' for key in control})
            colors.update({key: 'pink7' for key in (*late, *full)})
        else:  # early-late partition
            colors.update({key: 'gray6' for key in control})
            colors.update({key: 'cyan7' for key in early})
            colors.update({key: 'pink7' for key in (*late, *full)})

    # Infer default properties from standardized coordinates
    # NOTE: Alternative settings are commented out below. Also tried varying
    # edge widths but really really impossible to be useful for bar plots.
    colors = _get_lists(color, colors, default=None)
    edges = _get_lists(edge, edges, default='black')
    hatches = _get_lists(hatch, hatches, default=None)
    sizes = _get_lists(fade, sizes, default=markersize)
    alphas = _get_lists(alpha, alphas, default=1.0)
    fades = _get_lists(fade, fades, default=1.0)
    alphas = [a - (1 - f) * a ** 0.1 for a, f in zip(alphas, fades)]
    kw_patch = {'alpha': alphas, 'hatch': hatches, 'edgecolor': edges, 'linewidth': linewidth}  # noqa: E501
    kw_scatter = {'alpha': alphas, 'sizes': sizes}
    return colors, kw_patch, kw_scatter


def _auto_command(args, kw_collection, violin=True, shading=True, contour=None):
    """
    Generate a plotting command and keyword arguments from the input arguments.

    Parameters
    ----------
    args : xarray.DataArray
        Input arguments.
    kw_collection : namedtuple
        Input keyword arguments.
    violin : bool, optional
        Whether to use violins for several 1D array inputs.
    shading : bool, optional
        Whether to use shading or contours for 2D array inputs.
    contour : dict, optional
        Additional contour properties for 2D array inputs.

    Other Parameters
    ----------------
    horizontal : bool, optional
        Whether to use vertical or horizontal orientation (taken from `.other`).
    pcolor : bool, optional
        Whether to use `pcolormesh` for 2D shaded plot (taken from `.other`).

    Results
    -------
    command : str
        The command to use.
    guide : str
        The guide to use for the command.
    args : tuple
        The possibly updated positional arguments.
    kw_collection : namedtuple
        The possibly updated keyword arguments.
    """
    # Infer commands
    # defaults = KWARGS_VIOLIN.copy(); kw_collection.other.update(kw_patch)
    ignore = {'facets', 'version', 'components'}
    *_, data = args  # possibly ignore coordinate arrays
    if dims := sorted(data.sizes.keys() - ignore):
        # Two-dimensional plots
        # TODO: Support hist and hist2d plots in addition to scatter and barh
        # plots (or just hist since hist2d usually looks ugly with so little data).
        pcolor = kw_collection.other.get('pcolor', False)
        hatching = any(val for key, val in kw_collection.command.items() if 'hatch' in key)  # noqa: E501
        if len(dims) == 1 and dims[0] == 'plev':
            command = 'linex'
            defaults = KWARGS_LINE.copy()
        elif len(dims) == 1:
            command = 'line'
            defaults = KWARGS_LINE.copy()
        elif hatching:
            command = 'contourf'
            defaults = KWARGS_HATCH.copy()
        elif shading:
            command = 'pcolormesh' if pcolor else 'contourf'
            defaults = KWARGS_SHADING.copy()
        else:
            command = 'contour'
            defaults = {**KWARGS_CONTOUR, **(contour or {})}
    else:
        # Scalar plots with advanced default properties
        # NOTE: _violin_data applies custom grouping and spacing of violin plots by
        # inserting nan columns then _setup_violins imposes properties on resluts.
        nargs = sum(isinstance(arg, xr.DataArray) for arg in args)
        ragged = data.ndim == 2 or data.ndim == 1 and data.dtype == 'O'
        facets = 'facets' in data.sizes  # auto settings
        if nargs == 1 and facets and not ragged:  # sort facets
            args = (*args[:-1], data := data.sortby(data, ascending=True))
        colors, kw_patch, kw_scatter = _auto_props(data, kw_collection)
        if nargs == 1 and violin and ragged:
            *args, kw_collection = _violin_data(*args, kw_collection, color=colors, **kw_patch)  # noqa: E501
        horizontal = kw_collection.other.get('horizontal', False)
        if nargs == 2:
            command = 'scatter'
            defaults = {**KWARGS_SCATTER, **kw_scatter}
        elif not ragged:  # bar plot
            command = 'barh' if horizontal else 'bar'
            defaults = {**KWARGS_BAR, **kw_patch}
        elif not violin:
            command = 'boxh' if horizontal else 'box'
            defaults = {**KWARGS_BOX, **kw_patch}
        else:
            command = 'violinh' if horizontal else 'violin'  # used in filenames only
            defaults = {**KWARGS_VIOLIN}  # ignore patch
        if facets and 'bar' in command:
            if 'color' not in kw_collection.command:  # no manual color/auto multicolor
                defaults['negpos'] = data.name[-3:] not in ('ecs',)
        if not facets and 'violin' not in command and 'scatter' not in command:
            if all(c is not None for c in colors):
                defaults['color'] = colors

    # Update guide keywords
    # NOTE: This enforces legend handles grouped only for parameters with identical
    # units e.g. separate legends for sensitivity and feedback bar plots.
    # WARNING: Critical to delay wrapping the colorbar label until content is
    # drawn or else the reference width and height cannot be known.
    prefix = data.attrs.pop('short_prefix', None)
    kw_collection = copy.deepcopy(kw_collection)
    for key, value in defaults.items():  # including e.g. width=None
        if kw_collection.command.get(key, None) is None:
            kw_collection.command[key] = value
    if command in ('contourf', 'pcolormesh') and not hatching:
        guide = 'colorbar'
        label = data.climo.short_label
        kw_collection.colorbar.setdefault('label', label)
    elif command == 'contour':
        guide = 'legend'
        label = data.climo.short_label  # including units and always ignoring prefix
        kw_collection.legend.setdefault('label', label)
        if 'color' in kw_collection.command:  # cmap for queue
            kw_collection.command['cmap'] = (kw_collection.command.pop('color'),)
    else:  # TODO: allow full label outside of contour?
        guide = 'legend'
        short = data.attrs.get('short_name', '')
        long = data.attrs.get('long_name', '')
        label = prefix or long.replace(short, '').strip()
        if 'label' in data.coords or 'contour' in command and hatching:
            label = None
        kw_collection.legend.setdefault('label', label)
        if not dims and command[-1] == 'h':
            kw_collection.axes.setdefault('yreverse', True)
    return command, guide, args, kw_collection


def _combine_commands(dataset, arguments, kws_collection):
    """
    Combine several distribution plots into a single box or bar plot instruction.

    Parameters
    ----------
    dataset : xarray.Dataset
        The source dataset.
    arguments : list of tuple
        The input arguments.
    kws_collection : list of namedtuple
        The keyword arguments.

    Other Parameters
    ----------------
    labels : list of str, optional
        Optional overrides for outer labels (taken from `.command`).
    horizontal : bool, optional
        Whether to plot horizontally (taken from `.other`).
    offset : float, optional
        Offset between outer labels (taken from `.other`).
    intersect : bool, optional
        Whether to intersect coordinates on outer labels (taken from `.other`).
    correlation : bool, optional
        Whether to show correlation coefficients (taken from `.other`).

    Returns
    -------
    args : tuple
        The merged arguments.
    kw_collection : namedtuple
        The merged keyword arguments.
    """
    # Merge scatter plot instructions into array
    # NOTE: Currently this does regressions of differences instead of differences
    # of regressions (and correlations, covariances, etc.) when using multidimensional
    # reductions. Although they are the same if predictor is the same.
    # NOTE: Considered removing this and having get_spec support 'lists of lists
    # of specs' per subplot but this is complicated, and would still have to split
    # then re-combine scalar coordinates e.g. 'project', so not much more elegant.
    nargs = set(map(len, arguments))
    if len(nargs) != 1:
        raise RuntimeError(f'Unexpected combination of argument counts {nargs}.')
    if (nargs := nargs.pop()) not in (1, 2):
        raise RuntimeError(f'Unexpected number of arguments {nargs}.')
    non_feedback = (ureg.K, ureg.K / ureg.K, ureg.W / ureg.m ** 2)
    kw_collection = copy.deepcopy(kws_collection[0])  # deep copy of namedtuple
    for kw, field in itertools.product(kws_collection, kws_collection[0]._fields):
        getattr(kw_collection, field).update(getattr(kw, field))
    if nargs == 1:
        # Get violin data. To facillitate putting feedbacks and sensitivity in
        # same panel, scale 4xCO2 values into 2xCO2 values. Also detect e.g.
        # unperturbed sensitivitye stimates and set all values to zero.
        boxdata = bardata = annotations = None
        args = [arg for (arg,) in arguments]
        units = [arg.climo.units for arg in args]
        for i, iunits in enumerate(units):
            arg = args[i]
            if iunits in non_feedback and arg.name not in ('ts', 'tstd', 'tdev'):
                arg = 0.5 * arg  # halve the spread
                arg.attrs.clear()  # defer to feedback labels
                if np.all(np.abs(args[i]) < 0.1):  # pre-industrial forcing/sensitivity
                    arg = xr.zeros_like(arg)
                elif ureg.parse_units('W m^-2 K^-1') in units:
                    arg = arg - 4  # adjust abrupt4xco2 sensitivity or forcing to 2xco2
                args[i] = arg
    else:
        # Get regression data. To facillitate putting feedback and sensitivity
        # regression coefficients in same panel, scale 4xCO2 values into 2xCO2 values.
        # TODO: support *difference between slopes* instead of slopes of difference?
        args, boxdata, bardata, annotations = [], [], [], []
        units = [args[-1].climo.units for args in arguments]
        correlation = kw_collection.other.get('correlation', False)  # correlation?
        kw_collection.command['width'] = 1.0  # ignore staggered bars
        kw_collection.command['absolute_width'] = True
        for iargs, iunits in zip(arguments, units):  # merge into slope estimators
            keys = sorted(set(key for arg in iargs for key in arg.coords))
            dim = iargs[0].dims[0]
            pct = [50, 95] if correlation else [50, 95]  # very certain normally
            cmd = _components_corr if correlation else _components_slope
            data, data_lower, data_upper, rsquare, *_ = cmd(*iargs, dim=dim, pctile=pct)  # noqa: E501
            if correlation:
                data.attrs['units'] = ''
                data.attrs['short_name'] = 'correlation coefficient'
            elif iunits in non_feedback:  # adjust to 2xco2 scaling
                data.attrs.clear()  # NOTE: defer to feedback labels!
                data, data_lower, data_upper = 0.5 * data, 0.5 * data_lower, 0.5 * data_upper  # noqa: E501
            else:
                data.attrs['units'] = f'{iargs[1].units} / ({iargs[0].units})'
                data.attrs['short_name'] = 'regression coefficient'
            for key in keys:
                if any(key in arg.sizes for arg in iargs):  # non-scalar
                    continue
                if key in ('facets', 'version'):  # multi-index
                    continue
                coords = [arg.coords[key].item() for arg in iargs if key in arg.coords]
                dtype = np.asarray(coords[0]).dtype  # see process_data()
                isnan = np.issubdtype(dtype, float) and np.isnan(coords[0])
                if isnan or len(coords) == 1 or coords[0] == coords[1]:
                    value = coords[0]
                elif key in ('start', 'stop'):
                    value = coords[1]  # kludge for constraints
                else:  # WARNING: currently dropped by _get_coords() below
                    value = np.array(None, dtype=object)  # see process_data()
                    value[...] = tuple(coords)
                data.coords[key] = value
            data.name = '|'.join((arg.name for arg in iargs))
            data_lower1, data_lower2 = data_lower.values.flat
            data_upper1, data_upper2 = data_upper.values.flat
            rsquare = ureg.Quantity(rsquare.item(), '').to('percent')
            annotation = f'${rsquare:~L.0f}$'.replace(r'\ ', '')
            # annotation = f'${rsquare:~L.0f}$'.replace('%', r'\%').replace(r'\ ', '')
            # rvalue = np.sign(data.item()) * rsquare.item() ** 0.5
            # annotation = f'${rvalue:.2f}$'  # latex for long dash minus sign
            # annotation = '' if correlation else annotation
            args.append(data)
            boxdata.append([data_lower1, data_upper1])
            bardata.append([data_lower2, data_upper2])
            annotations.append(annotation)

    # Infer inner and outer coordinates
    # TODO: Switch to tuples=True and support tuple coords in _auto_command
    # NOTE: The below code for kws_outer supports both intersections of outer labels
    # and intersections of inner labels. Default behavior is to intersect inner
    # labels on the legend and only denote "outer-most" coordinates with outer labels.
    kws_outer, kws_inner = {}, {}  # for label inference
    kw_scalar, kw_vector, kw_outer, kw_inner = {}, {}, {}, {},
    kw_coords = _get_coords(*args, units=False, tuples=False)  # drop multi-index
    kw_counts = {key: len(list(itertools.groupby(value))) for key, value in kw_coords.items()}  # noqa: E501
    intersect = kw_collection.other.get('intersect', False)  # intersect on outer?
    for key, values, count in zip(kw_coords, kw_coords.values(), kw_counts.values()):
        if key == 'units':  # used for queue identifiers
            continue
        if count == 1:
            kw_scalar[key] = values[0]
        elif intersect or count == min(n for n in kw_counts.values() if n > 1):
            kw_vector[key] = kw_outer[key] = values
        else:
            kw_vector[key] = kw_inner[key] = values
    if not kw_vector:
        raise RuntimeError('No coordinates found for concatenation.')
    num, base, locs, ticks, kw_groups = 0, 0, [], [], []
    kws_inner = [dict(zip(kw_inner, vals)) for vals in zip(*kw_inner.values())]
    kws_outer = [dict(zip(kw_outer, vals)) for vals in zip(*kw_outer.values())]
    for kw_inner in kws_inner:  # use e.g. 'perturbed' and 'unperturbed'
        kw_inner.setdefault('name', 'net')

    # Infer object and tick locations
    # WARNING: For some reason naked len(itertools.groupby()) fails. Note this finds
    # consecutive groups in a list of hashables and we want the fewest groups.
    offset = kw_collection.other.get('offset', 0.8)  # additional offset coordinate
    groups = list(itertools.groupby(kws_outer))
    for group, items in itertools.groupby(kws_outer):
        items = list(items)
        count = len(list(items))  # convert itertools._group object
        ilocs = np.arange(base, base + count - 0.5)
        ikws = kws_inner[num:num + count]
        keys = [('name',), ('project',), ('experiment', 'start', 'stop')]
        for key in keys:  # TODO: generalize the above 'keys' groups
            values = [tuple(kw.get(_) for _ in key) for kw in ikws]
            lengths = [len(list(items)) for _, items in itertools.groupby(values)]
            if 1 < len(lengths) < len(values):  # i.e. additional groups are present
                for idx in np.cumsum(lengths[:-1]):
                    ilocs[idx:] += 0.5 * offset  # TODO: make this configurable?
        tick = 0.5 * (ilocs[0] + ilocs[-1])
        base += offset + 1 + (ilocs[-1] - ilocs[0])  # float coordinate
        num += count  # integer index
        locs.extend(ilocs)
        ticks.append(tick)
        kw_groups.append(group)  # unique keyword argument groups

    # Get inner and outer labels
    # NOTE: This also tries to set up appropriate automatic line breaks.
    # WARNING: For now use 'skip_names' for inner labels to prevent unnecessary
    horizontal = kw_collection.other.get('horizontal', False)
    axis = 'y' if horizontal else 'x'
    key = 'refheight' if horizontal else 'refwidth'
    refwidth = kw_collection.figure.get(key, None)
    refwidth = pplt.units(refwidth or pplt.rc['subplots.refwidth'], 'in')
    refwidth *= 1.0 / len(groups)  # scale spacing by total number of groups
    kw_infer = dict(identical=False, long_names=True, title_case=False)
    labels_inner = _infer_labels(
        dataset, *kws_inner, refwidth=np.inf, skip_names=True, **kw_infer,
    )
    labels_outer = kw_collection.command.pop('labels', None) or _infer_labels(
        dataset, *kw_groups, refwidth=refwidth, skip_names=False, **kw_infer,
    )
    if len(labels_outer) != len(kw_groups):
        raise ValueError(f'Mismatch between {len(labels_outer)} labels and {len(kw_groups)}.')  # noqa: E501
    if labels_inner or labels_outer:
        seen = set()  # seen labels
        labels_print = [l for l in labels_inner if l not in seen and not seen.add(l)]
        print()  # end previous line
        print('Inner labels:', ', '.join(map(repr, labels_print)), end=' ')
        print('Outer labels:', ', '.join(map(repr, labels_outer)), end=' ')
    if labels_outer:
        kw_axes = {
            f'{axis}ticks': ticks,
            f'{axis}ticklabels': labels_outer,
            f'{axis}grid': False,
            f'{axis}ticklen': 0,
            f'{axis}tickmajorpad': 5,
            f'{axis}rotation': 90 if axis == 'y' else 0,
        }
        kw_collection.axes.update(kw_axes)

    # Concatenate arrays
    # WARNING: np.array([(1, 2), (3, 4)], dtype=object) will create 2D array with
    # tuples along rows instead of ragged arrays. Use proplot workaround below.
    # NOTE: Only 'project' and 'institute' levels of 'components' are used elsewhere
    keys = ('short_prefix', 'short_suffix', 'short_name', 'units')
    attrs = {key: val for arg in args for key, val in arg.attrs.items() if key in keys}
    locs = np.array(locs or np.arange(len(kws_inner)), dtype=float)
    if nargs == 1:  # violin plot arguments
        values = np.empty(len(args), dtype=object)
        values[...] = [tuple(arg.values) for arg in args]
    else:
        values = [arg.item() for arg in args]
        values = np.array(values)
    name = args[0].name
    index = pd.MultiIndex.from_arrays(list(kw_vector.values()), names=list(kw_vector))
    coords = {'components': index, **kw_scalar}
    values = xr.DataArray(values, name=name, dims='components', attrs=attrs, coords=coords)  # noqa: E501
    if boxdata:
        kw_collection.command.update(boxdata=np.array(boxdata).T, **KWARGS_ERRBOX)
    if bardata:
        kw_collection.command.update(bardata=np.array(bardata).T, **KWARGS_ERRBAR)
    if labels_inner:  # used for legend entries
        values.coords['label'] = ('components', labels_inner)
    if annotations:  # used for _setup_bars annotations
        values.coords['annotation'] = ('components', annotations)
    args = (locs, values)
    return args, kw_collection


def _infer_commands(
    dataset, arguments, kws_collection, fig=None, gs=None, ax=None, geom=None, title=None,  # noqa: E501
):
    """
    Infer the suitable plotting commands from input arguments and apply settings.

    Parameters
    ----------
    dataset : xarray.Dataset
        The input dataset.
    arguments : list of tuple of xarray.DataArray
        The plotting arguments.
    kws_collection : list of namedtuple of dict
        The keyword arguments for each command.
    fig : proplot.Figure, optional
        The existing figure.
    gs : proplot.GridSpec, optional
        The existing gridspec.
    ax : proplot.Axes, optional
        The existing axes.
    geom : tuple, optional
        The ``(nrows, ncols, index)`` geometry.
    title : str, optional
        The default title of the axes.

    Other Parameters
    ----------------
    cycle : cycle-spec, optional
        The manual color cycle (taken from `.other`).
    colorbar : str, optional
        The colorbar location (translated to `loc` by `get_spec`)
    legend : str, optional
        The legend location (translated to `loc` by `get_spec`)

    Returns
    -------
    fig : proplot.Figure
        The source figure.
    gs : proplot.GridSpec
        The source gridspec.
    axs : proplot.Axes
        The plotting axes.
    commands : list of str
        The plotting commands.
    guides : list of str
        The plotting guide types.
    arguments : tuple
        The possibly modified plotting arguments.
    kws_collection : dict
        The possibly modified plotting keyword arguments.
    """
    # Initial stuff
    # NOTE: Delay object creation until this block so we can pass arbitrary
    # loose keyword arguments and simply parse them in _parse_item.
    ignore = {'facets', 'version', 'period'}  # additional keywords
    sizes = [tuple(sorted(args[-1].sizes.keys() - ignore)) for args in arguments]
    if len(set(sizes)) != 1:
        raise RuntimeError(f'Conflicting dimensionalities in single subplot: {sizes}')
    sizes = set(sizes[0])
    sharex = sharey = 'labels'
    kw_default = KWARGS_FIG.copy()
    if title is None:  # avoid only a-b-c above title
        kw_default.update(abcloc='ul')
    if 'lon' in sizes and 'lat' in sizes:
        kw_default.update(KWARGS_GEO)
    if 'lat' in sizes and 'lon' not in sizes:
        sharex = True  # no duplicate latitude labels
        kw_default.update(KWARGS_LAT)
    if 'plev' in sizes:
        sharey = True  # no duplicate pressure labels
        kw_default.update(KWARGS_PLEV)
    refwidth = kw_default.pop('refwidth', None)
    kw_figure = {'sharex': sharex, 'sharey': sharey, 'span': False, 'refwidth': refwidth}  # noqa: E501
    kw_gridspec = {}  # no defaults currently
    kw_axes = {'title': title, **kw_default}

    # Merge commands and initialize figure and axes
    # TODO: Support *stacked* scatter plots and *grouped* bar plots with 2D arrays
    # for non-project multiple selections? Not difficult... but maybe not worth it.
    iax = ax
    geom = geom or (1, 1, 0)  # total subplotspec geometry
    if arguments and not sizes:
        if len(arguments) > 1:  # concatenate and add label coordinate
            args, kw_collection = _combine_commands(dataset, arguments, kws_collection)
            arguments, kws_collection = (args,), (kw_collection,)
        elif 'facets' in arguments[0][-1].dims:
            arg = arguments[0][-1]
            labels = [{'project': value} for value in _get_projects(arg)]
            labels = _infer_labels(dataset, *labels, identical=False)
            arg.coords['annotation'] = ('facets', arg.model.values)
            if all(bool(label) for label in labels):
                arg.coords['label'] = ('facets', labels)
    if fig is None:  # also support updating? or too slow?
        if any('share' in kw_collection.figure for kw_collection in kws_collection):
            kw_figure.pop('sharex', None); kw_figure.pop('sharey', None)  # noqa: E702
        for kw_collection in kws_collection:
            kw_figure.update(kw_collection.figure)
        fig = pplt.figure(**kw_figure)
    if gs is None:
        for kw_collection in kws_collection:
            kw_gridspec.update(kw_collection.gridspec)
        gs = pplt.GridSpec(*geom[:2], **kw_collection.gridspec)
    if ax is None:
        for kw_collection in kws_collection:
            kw_axes.update(kw_collection.axes)
        if max(geom[:2]) == 1:  # single subplot
            kw_axes.pop('abc', None)
        ax = iax = fig.add_subplot(gs[geom[2]], **kw_axes)
        fig.auto_layout(tight=False)  # TODO: commit on master

    # Get commands and default keyword args
    # NOTE: This applies fixed properties for project and institute selections. Also
    # applies color cycle to remaining attributes not otherwise assigned property.
    kw_other = {}
    for kw_collection in kws_collection:
        kw_other.update(kw_collection.other)
    cycle = kw_other.get('cycle')
    cycle = pplt.get_colors(cycle) if cycle else CYCLE_DEFAULT
    lineplot = len(sizes) < 2 and len(arguments) > 1  # multicolor line plots
    contours = []  # contour keywords
    contours.append({'color': 'gray8'})  # WARNING: ls=None disables negative dash
    contours.append({'color': 'gray3', 'linestyle': ':'})
    results, colors, units = [], {}, {}
    for idx, (args, kw_collection) in enumerate(zip(arguments, kws_collection)):
        iunits = args[-1].attrs.get('units', None)  # independent variable units
        icolor = kw_collection.command.get('color', None)
        if lineplot:  # scalar plot cycle colors
            icolor = kw_collection.command.setdefault('color', cycle[idx % len(cycle)])
        if lineplot and idx > 0 and iunits not in units:
            value = colors.get(ax, ())  # number of colors used so far
            value = value.pop() if len(value) == 1 else 'k'
            axis = 'y' if 'plev' in sizes else 'x'
            ax.format(**{f'{axis}color': value})  # line color or simply black
            iax = getattr(ax, f'alt{axis}')(**{f'{axis}color': icolor})
        command, guide, args, kw_collection = _auto_command(
            args, kw_collection,  # keyword arg colleciton
            shading=(idx == 0),  # shade first plot only
            contour=contours[max(0, min(idx - 1, len(contours) - 1))],
        )
        results.append((iax, command, guide, args, kw_collection))
        colors.setdefault(ax, set()).add(icolor)  # colors plotted per axes
        units.update({iunits: iax})  # axes indexed by units
    axs, commands, guides, arguments, kws_collection = zip(*results)
    return fig, gs, axs, commands, guides, arguments, kws_collection


def _setup_axes(ax, *args, command=None):
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
    **kwargs
        Additional format arguments.

    Returns
    -------
    xunits, yunits : str
        The associated x and y-axis units.
    """
    # Initial stuff
    # TODO: Handle labels automatically with climopy and proplot autoformatting...
    # for some reason share=True seems to have no effect but not sure why.
    top = ax._get_topmost_axes()
    fig = top.figure
    cmds = ('scatter', 'contour', 'contourf', 'pcolor', 'pcolormesh')
    if command is None:
        raise ValueError('Input command is required.')
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

    # Handle x and y axis settings
    # NOTE: Want to disable axhline()/axvline() autoscaling but not currently possible
    # so use plot(). See: https://github.com/matplotlib/matplotlib/issues/14651
    units = []
    nrows, ncols, *_ = top.get_subplotspec()._get_geometry()  # custom geometry function
    rows, cols = top._range_subplotspec('y'), top._range_subplotspec('x')
    edgex, edgey = max(rows) == nrows - 1, min(cols) == 0
    for s, data, other, edge in zip('xy', (x, y), (y, x), (edgex, edgey)):
        unit = getattr(data, 'units', None)
        unit = ureg.parse_units(unit) if unit is not None else None
        share = getattr(fig, f'_share{s}', None)
        label = edge or not share or ax != top
        locator = getattr(ax, f'{s}axis').get_major_locator()
        dlabel = getattr(ax, f'{s}axis').isDefault_label
        dlocator = getattr(ax, f'{s}axis').isDefault_majloc
        transform = getattr(ax, f'get_{s}axis_transform')()
        if data is None and dlocator:  # hide tick labels
            kw = {f'{s}locator': 'null'}
            ax.format(**kw)
        if other is not None and 'components' in other.sizes and hasattr(locator, 'locs'):  # shading # noqa: E501
            cmd = ax.area if s == 'x' else ax.areax
            kw_bg = {**KWARGS_BACKGROUND, 'transform': transform}
            coords = pplt.edges(locator.locs)
            for x0, x1 in zip(coords[::2], coords[1::2]):
                cmd([x0, x1], [0, 0], [1, 1], **kw_bg)
        if ax == top and data is not None and command not in cmds:  # reference lines
            cmd = ax.linex if s == 'x' else ax.line
            kw_ref = {**KWARGS_REFERENCE, 'transform': transform}
            coords = []
            if unit is None or unit != ureg.degrees_north:
                if command:  # add for bar, violin, and line plots
                    coords.append(0)
            if unit is not None and unit == ureg.dimensionless:
                if command in ('bar', 'barh'):  # add only for bar plots
                    coords.extend((1, -1))
            for coord in coords:
                h, = cmd([0, 1], [coord, coord], **kw_ref)
        if label and dlabel and unit is not None:  # axis label
            data = data.copy()
            for key in ('short_prefix', 'short_suffix'):
                data.attrs.pop(key, None)  # avoid e.g. 'anomaly' for non-anomaly data
            data.attrs.setdefault('short_name', '')   # avoid cfvariable inheritence
            data.attrs['short_prefix'] = data.attrs.get(f'{s}label_prefix', '')
            if getattr(fig, f'_span{s}'):
                width, height = fig.get_size_inches()
            else:
                width, height = ax._get_size_inches()  # all axes present by now
            size = width if s == 'x' else height
            label = _fit_label(data.climo.cfvariable.short_label, refwidth=size)
            ax.format(**{f'{s}label': label})  # include share settings
        units.append(unit)
    return units


def _setup_bars(ax, args, errdata=None, handle=None, horizontal=False, annotate=False):
    """
    Adjust and optionally add content to bar plots.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes.
    args : xarray.DataArray
        The input arguments.
    errdata : xarray.Dataarray
        The error bar data.
    handle : matplotlib.containers.BarContainer
        The bar container.

    Other Parameters
    ----------------
    horizontal : bool, optional
        Whether the bars were plotted horizontally.
    annotate : bool, optional
        Whether to add annotations to the bars.
    """
    # Plot thick outline bars on top (flagship models) and remove alpha from edge
    # NOTE: Skip applying opacity to edges because faded outline appears conflicting
    # in combination with white hatching and outline used for non-matching CMIP6.
    axis = 'y' if horizontal else 'x'
    handle = handle or ()
    for obj in handle:
        getattr(obj.sticky_edges, axis).clear()  # offset from zero and edge
        alpha = obj.get_alpha()
        color = pplt.to_xyz(obj.get_edgecolor(), space='hsv')
        zorder = 1 + 0.001 * (100 - color[2]) + 0.01 * obj.get_linewidth()
        obj.set_zorder(zorder)
        if alpha is not None and alpha < 1:  # restore edge color
            color = pplt.set_alpha(obj.get_facecolor(), alpha)
            obj.set_alpha(None)
            obj.set_facecolor(color)

    # Figure out space occupied by text
    # WARNING: Here 1.5 is used when orientation is in direction of bars but
    # also use 1.5 for R^2 tex annotations because gives best size after testing.
    # NOTE: Matplotlib cannot include text labels in autoscaling so have
    # to adjust manually. See: https://stackoverflow.com/a/32637550/4970632
    locs, data = (np.arange(args[-1].size), *args) if len(args) == 1 else args
    labels = data.coords.get('annotation', data.coords[data.dims[0]]).values
    labels = [' '.join(lab) if isinstance(lab, tuple) else str(lab) for lab in labels]
    nwidth = max(2.2 if '$' in label else 1 for label in labels)
    width, height = ax._get_size_inches()  # axes size
    if not horizontal:
        s, width, height, slice_ = 'y', width, height, slice(None)
    else:
        s, width, height, slice_ = 'x', height, width, slice(None, None, -1)
    space = width / (max(locs) - min(locs) + 2)  # +2 accounts for padding on ends
    space = pplt.units(space, 'in', 'pt')
    scales = pplt.arange(0.75, 1.0, 0.025)  # automatically scale in this range
    sizes = nwidth * pplt.rc.fontsize * scales
    fontscale = scales[np.argmin(np.abs(sizes - space))]
    fontsize = pplt.rc.fontsize * fontscale

    # Adjust axes limits
    # NOTE: This also asserts that error bars without labels are excluded from the
    # default data limits (e.g. when extending below zero).
    lower, upper = (data, data) if errdata is None else errdata
    data = getattr(data, 'values', data)
    lower = getattr(lower, 'values', lower)
    upper = getattr(upper, 'values', upper)
    above = np.mean(data) >= 0  # average bar position
    # above = np.sum(data >= 0) >= data.size // 2  # average bar sign
    points = np.array(upper if above else lower)  # copy for the labels
    points = np.clip(points, 0 if above else None, None if above else 0)
    lower = data if above else lower  # used for auto scaling
    upper = upper if above else data  # used for auto scaling
    min_, max_ = min(np.min(lower), 0), max(np.max(upper), 0)
    margin = pplt.rc[f'axes.{s}margin'] * (max_ - min_)
    offsets = [2.8 if '$' in label else len(label) for label in labels]
    offsets = (fontsize / 72) * np.array(offsets)  # approx inches
    offsets *= 0.8 * (max_ - min_) / height  # approx data units
    min_ = np.min(lower - (not above) * annotate * offsets)
    max_ = np.max(upper + above * annotate * offsets)
    min_, max_ = min(min_ - margin, 0), max(max_ + margin, 0)
    if getattr(ax, f'get_autoscale{s}_on')():
        ax.format(**{f'{s}lim': (min_, max_)})

    # Add annotations
    # NOTE: Using set_in_layout False significantly improves speed since tightbbox is
    # faster and looks nicer to allow overlap into margin without affecting the space.
    if annotate:
        for loc, point, label in zip(locs, points, labels):
            # rotation = 90
            rotation = 0 if '$' in label else 90  # assume math does not need rotation
            kw_annotate = {'fontsize': fontsize, **KWARGS_ANNOTATE}
            if not horizontal:
                va = 'bottom' if above else 'top'
                kw_annotate.update({'ha': 'center', 'va': va, 'rotation': rotation})
            else:
                ha = 'left' if above else 'right'
                kw_annotate.update({'ha': ha, 'va': 'center', 'rotation': 0})
            # offset = 0.2 * fontscale * pplt.rc.fontsize
            offset = 0.4 * fontscale * pplt.rc.fontsize
            xydata = (loc, point)[slice_]  # position is 'y' if horizontal
            xytext = (0, offset * (1 if above else -1))[slice_]  # as above
            res = ax.annotate(label, xydata, xytext, **kw_annotate)
            res.set_in_layout(False)


def _setup_scatter(
    ax, data0, data1, collection=None, zeros=False, oneone=False, linefit=False, annotate=False, constraint=None, graphical=False,  # noqa: E501
):
    """
    Adjust and optionally add content to scatter plots.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes.
    data0, data1 : xarray.DataArray
        The input arguments.
    collection : matplotlib.collection.Collection
        The scatter collection.

    Other Parameters
    ----------------
    zeros : bool, optional
        Whether to add dotted lines on zeros.
    oneone : bool, optional
        Whether to add a one-one dotted line.
    linefit : bool, optional
        Whether to add a least-squares fit line.
    annotate : bool, optional
        Whether to add annotations to the scatter markers.
    constraint : float, optional
        Whether to illustrate a constraint. If ``True`` uses a default value.
    graphical : bool, optional
        Whether to use bootstrapped or graphical estimates of the constraint.
    """
    # Add reference one:one line
    # NOTE: This also disables autoscaling so that line always looks like a diagonal
    # drawn right across the axes. Also requires same units on x and y axis.
    if zeros:
        style = dict(color='k', lw=1 * pplt.rc.metawidth)
        ax.axhline(0, alpha=0.1, zorder=0, **style)
        ax.axvline(0, alpha=0.1, zorder=0, **style)
    if oneone:
        style = dict(color='k', dashes=(1, 3), lw=pplt.rc.metawidth)
        units0 = data0.climo.units
        units1 = data1.climo.units
        if units0 == units1:
            lim = (*ax.get_xlim(), *ax.get_ylim())
            lim = (min(lim), max(lim))
            avg = 0.5 * (lim[0] + lim[1])
            span = lim[1] - lim[0]
            ones = (avg - 1e3 * span, avg + 1e3 * span)
            # ax.format(xlim=lim, ylim=lim)  # autoscale disabled
            # ax.plot(ones, ones, color='k', dashes=(1, 3), lw=pplt.rc.metawidth, scalex=0, scaley=0)  # noqa: E501
            ax.add_artist(mlines.Line2D(ones, ones, **style))  # noqa: E501

    # Add manual regression line
    # NOTE: Here climopy automatically reapplies dataarray coordinates to fit line
    # and lower and upper bounds so do not explicitly need sorted x coordinates.
    if linefit:  # https://en.wikipedia.org/wiki/Simple_linear_regression
        dim = data0.dims[0]  # generally facets dimension
        slope, _, _, rsquare, fit, fit_lower, fit_upper = _components_slope(
            data0, data1, dim=dim, adjust=False, pctile=None,  # use default of 95
        )
        # sign = '\N{MINUS SIGN}' if slope < 0 else ''  # negative r-squared
        # rvalue = rsquare.item() ** 0.5
        # annotation = rf'r$=${sign}${rvalue:.2f}$'
        sign = '(\N{MINUS SIGN})' if slope < 0 else ''  # negative r-squared
        rsquare = ureg.Quantity(rsquare.item(), '').to('percent')
        label = rf'$r^2={sign}{rsquare:~L.0f}$'
        label = re.sub(r'(?<!\\)%', r'\%', label)
        label = label.replace(r'\ ', '')
        datax = np.sort(data0, axis=0)  # linefit returns result for sorted data
        c = collection[0].get_color()
        c = 'red' if not constraint else c  # line fit color
        ax.plot(datax, fit, c=c, ls='-', lw=1.5 * pplt.rc.metawidth)
        ax.area(datax, fit_lower.squeeze(), fit_upper.squeeze(), c=c, a=0.3, lw=0)
        ax.format(lltitle=label)  # add annotation
        if constraint is not None:
            constraint = 'cld' if constraint is True else constraint
            xcolor, ycolor = 'cyan7', 'pink7'  # TODO: fix colors?
            xs, ys1, ys2 = _constrain_response(
                data0, data1, graphical=graphical, constraint=constraint,
            )
            xmin, xmean, xmax = xs
            ymin1, ymean, ymax1 = ys1
            ymin2, ymean, ymax2 = ys2
            xmin0, xmax0 = np.percentile(data0, [2.5, 97.5])
            ymin0, ymax0 = np.percentile(data1, [2.5, 97.5])
            args = ([xmean, xmean], [ymin2 - 20, ymean])
            # for x0 in (xmin0, xmax0):  # unconstrained bounds
            #     ax.axvline(x0, color=xcolor, alpha=0.5, ls='--', lw=1)
            ax.axvspan(xmin, xmax, color=xcolor, alpha=0.2, lw=0)
            ax.add_artist(mlines.Line2D(*args, color=xcolor, lw=1))
            args = ([xmin - 20, xmean], [ymean, ymean])
            # for y0 in (ymin0, ymax0):  # unconstrained bounds
            #     ax.axhline(y0, color=ycolor, alpha=0.5, ls='--', lw=1)
            ax.axhspan(ymin2, ymax2, color=ycolor, alpha=0.1, lw=0)
            ax.axhspan(ymin1, ymax1, color=ycolor, alpha=0.2, lw=0)
            ax.add_artist(mlines.Line2D(*args, color=ycolor, lw=1))

    # Add annotations
    # NOTE: Using set_in_layout False significantly improves speed since tight bounding
    # box is faster and looks nicer to allow slight overlap with axes edge.
    if annotate:
        kw_annotate = {'fontsize': 'x-small', 'textcoords': 'offset points'}
        labels = data1.coords.get('annotation', data1.coords[data1.dims[0]]).values
        labels = [' '.join(lab) if isinstance(lab, tuple) else str(lab) for lab in labels]  # noqa: E501
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        width, _ = ax._get_size_inches()
        diff = (pplt.rc.fontsize / 72) * (max(xlim) - min(xlim)) / width
        xmax = xlim[1] + 5 * diff if ax.get_autoscalex_on() else None
        ymin = ylim[0] - 1 * diff if ax.get_autoscaley_on() else None
        ax.format(xmax=xmax, ymin=ymin)  # skip if overridden by user
        for x, y, label in zip(data0.values, data1.values, labels):
            xy = (x, y)  # data coordaintes
            xytext = (2, -2)  # offset coordaintes
            res = ax.annotate(label, xy, xytext, ha='left', va='top', **kw_annotate)
            res.set_in_layout(False)


def _setup_violins(
    ax, data, handle, line=None, width=None, median=False, horizontal=False, **kwargs
):
    """
    Adjust and optionally add content to violin plots.

    Parameters
    ----------
    ax : matplotlib.axes.axes
        The original axes.
    data : array-like
        The original violin data.
    handle : list of matplotlib.patches.Path
        The violin patches.
    line : list of matplotlib.lines.Line2D
        The violin lines.

    Other Parameters
    ----------------
    width : float, optional
        The violin width in data units.
    median : bool, optional
        Whether to show the median instead of mean.
    horizontal : bool, optional
        Whether the violins were plotted horizontally.
    **kwargs
        Additional (optionally vector) properties to apply to violins.
    """
    # Apply violin styling
    # NOTE: Skip applying opacity to edges because faded outline appears conflicting
    # in combination with white hatching and outline used for non-matching CMIP6.
    # NOTE: Convert opacity to actual solid color so that overlapping violin shapes
    # are not transparent and line opacities are not combined with background.
    # NOTE: Seaborn ignores opacity channel on input color palette so must be
    # manually applied here. Also apply to lines denoting individual observations
    # and use manual alpha blending so that violin shapes can overlap without issues.
    # See: https://github.com/mwaskom/seaborn/issues/622
    # See: https://stackoverflow.com/q/68731566/4970632
    # See: https://matplotlib.org/3.1.0/tutorials/colors/colors.html
    data = getattr(data, 'values', data)
    hbase, lbase, handles, lines, locs = 0, 0, [], [], []
    for i in range(data.shape[1]):
        num = np.sum(np.isfinite(np.unique(data[:, i])))
        if num > 0:
            locs.append(i)
            handles.append(handle[hbase] if handle and num > 1 else None)
            lines.append(line[lbase:lbase + num] if line else ())
        hbase += int(num > 1)
        lbase += num
    for key in ('alpha', 'hatch', 'linewidth', 'edgecolor'):
        values = kwargs.get(key, None)
        if values is None or np.isscalar(values):
            values = [values] * len(handles)
        for ihandle, ilines, value in zip(handles, lines, values):
            if value is None:
                continue
            if ihandle:
                if key != 'alpha':  # update property
                    ihandle.update({key: value})
                else:  # manually impose opacity
                    color = ihandle.get_facecolor().squeeze()[:3]  # singleton array
                    color = color * value + np.array([1, 1, 1]) * (1 - value)
                    ihandle.set_facecolor(color)
            for line in ilines:
                line_color = line_alpha = False
                if line_color and key == 'edgecolor':  # apply violin edge color
                    line.set_color(value)
                if line_alpha and key == 'alpha':  # apply violin fill opacity
                    color = np.array(pplt.to_rgb(line.get_color()))
                    color = color * value + np.array([1, 1, 1]) * (1 - value)
                    line.set_color(color)

    # Scale widths and add median
    # TODO: Move this into custom proplot function. Matplotlib and seaborn both
    # proihbit skipping width scaling, matplotlib seems to use weaker custom gaussian
    # kde estimator instead of scipy version, and seaborn has major design limitations.
    # NOTE: Seaborn violin plot 'width' and 'count' both scale by maximum density per
    # violin and 'area' by maximum density across all violins. Otherwise the default
    # scipy kde gaussian result evaluates to one when integrated over its domain in
    # data units. To restore that behavior, and thus have equal area violins across
    # many subplots, we undo the scaling by performing the integral ourselves.
    adjust = data.shape[1] / len(handles)  # treat violins as if separated by 1 x-step
    axis = 'y' if horizontal else 'x'
    width = width or KWARGS_VIOLIN.get('width', 1.0)  # scaled width preserving area
    scatter = 'scatterx' if horizontal else 'scatter'
    for i, (ihandle, ilines, iloc) in enumerate(zip(handles, lines, locs)):
        # zorder = 2 - (i + 1) / len(handles)
        zorder = 1 + (i + 1) / len(handles)
        if not ihandle:
            continue
        polys = ihandle.get_paths()[0].to_polygons()[0]
        grids = polys[:, 1 - int(horizontal)]  # gridpoints for gaussian kde sample
        kdefit = polys[:, int(horizontal)]  # symmetric gaussian kde sample points
        span = np.max(grids) - np.min(grids)
        center = np.mean(kdefit)  # central coordinate of symmetric gaussian kde
        scale = span * np.mean(np.abs(kdefit - center))  # integral of pdf over shape
        polys[:, int(horizontal)] = center + adjust * width * (kdefit - center) / scale
        ihandle.set_verts([polys])
        ihandle.set_zorder(zorder)
        for line in ilines:
            points = getattr(line, f'get_{axis}data')()
            points = center + adjust * width * (points - center) / scale
            getattr(line, f'set_{axis}data')(points)
            line.set_solid_capstyle('butt')  # prevent overlapping on violin edges
            line.set_zorder(zorder)
        cmd = np.nanmedian if median else np.nanmean
        point = cmd(data[:, iloc])
        getattr(ax, scatter)(center, point, zorder=zorder + 0.001, **KWARGS_CENTER)

    return handles  # possibly adjusted handles


def _violin_data(locs, data, kw_collection, color=None, **kwargs):
    """
    Get data array and keyword arguments suitable for seaborn `violinplot`.

    Parameters
    ----------
    locs : array-like
        The coordinates to apply.
    data : array-like
        The ragged array of 1D arrays.
    kw_collection : namedtuple
        The keyword arguments.
    color : optional
        Optional color array.
    **kwargs : optional
        Optional additional property arrays.

    Other Parameters
    ----------------
    horizontal : bool, optional
        Whether to use horizontal orientation (taken from `.other`)

    Returns
    -------
    data : xarray.DataArray
        The in-filled and merged 2D array.
    kw_collection : namedtuple
        The updated keyword arguments.
    """
    # Infer labels and colors
    # TODO: Replace this with proplot violin plot once refactor is finished.
    # NOTE: Seaborn cannot handle custom violin positions. So use fake data to achieve
    # the same effective spacing. See https://stackoverflow.com/a/52729348/4970632
    kw_collection = copy.deepcopy(kw_collection)
    horizontal = kw_collection.other.get('horizontal', False)
    if not isinstance(data, xr.DataArray) or data.dtype != object or data.ndim != 1:
        raise ValueError('Unexpected input array for violin plot formatting.')
    scale = 100  # highest precision of 'offset' used in _combine_commands
    locs = locs - np.min(locs)  # ensure starts at zero
    locs = np.round(scale * locs).astype(int)
    step = np.gcd.reduce(locs)  # e.g. 100 if locs were integer, 50 if were [1, 2.5, 4]
    locs = (locs / step).astype(int)  # e.g. from [0, 1, 2.5, 4] to [0, 2, 5, 8]
    color = np.atleast_1d(color)  # optional multiple colors
    color = KWARGS_VIOLIN['color'] if any(c is None for c in color.flat) else color
    color = [np.array(color).item()] * len(locs) if np.isscalar(color) else color
    orient = 'h' if horizontal else 'v'  # different convention
    axis = 'y' if horizontal else 'x'
    ticks = np.array(kw_collection.axes.get(f'{axis}ticks', 1))  # see _combine_commands

    # Concatenate array and update keyword args
    # WARNING: Critical to keep 'components' because formatting in
    # _setup_axes despends on whether this coordinate is present.
    # NOTE: This keeps requires to_pandas() on output, and tick locator has to be
    # changed since violin always draws violins at increasing integers from zero...
    # if dataframe columns are float adds string labels! Seaborn is just plain weird.
    index = range(max(len(vals) for vals in data.values))  # ragged array values
    columns = np.arange(0, np.max(locs) + 1)  # violinplot always uses 0 to N points
    palette = np.full(columns.size, '#000000', dtype=object)
    labels = np.empty(columns.size, dtype=object)
    merged = pd.DataFrame(index=index, columns=columns, dtype=float)
    for col, lab, loc, vals in zip(color, data.label.values, locs, data.values):
        palette[loc] = col
        labels[loc] = lab  # unfilled slots are None
        merged[loc].iloc[:len(vals)] = np.array(vals)  # unfilled slots are np.nan
    data = xr.DataArray(
        merged,
        name=data.name,
        dims=('index', 'components'),  # expand singleton data.dims
        coords={'label': ('components', np.array(labels))},  # possibly non-unique
        attrs=data.attrs,
    )
    kw_collection.command.update({'orient': orient, 'palette': palette.tolist()})
    kw_collection.other.update(kwargs)
    kw_collection.axes.update({f'{axis}ticks': ticks * scale / step})
    return data, kw_collection


def generate_plot(
    dataset,
    rowspecs=None,
    colspecs=None,
    figtitle=None,
    figprefix=None,
    figsuffix=None,
    rowlabels=None,
    collabels=None,
    titles=None,
    labelbottom=False,
    labelright=False,
    labelparams=False,
    standardize=False,
    identical=False,
    groupnames=True,
    figurespan=False,
    gridskip=None,
    argskip=None,
    cbarlength=None,
    cbarwrap=None,
    cbarpad=None,
    cbarspace=None,
    leggroup=None,
    legcols=None,
    legpad=None,
    legspace=None,
    ncols=None,
    nrows=None,
    rxlim=None,
    rylim=None,
    save=None,
    suffix=None,
    **kwargs
):
    """
    Plot any combination of variables across rows and columns.

    Parameters
    ----------
    dataset : xarray.Dataset
        A dataset generated by `open_bulk`.
    *args : 2-tuple or list of 2-tuple
        Tuples containing the ``(name, kwargs)`` passed to ``ClimoAccessor.get``
        used to generate data in rows and columns. See `parse_specs` for details.
    figtitle, rowlabels, collabels, titles : optional
        The figure settings. The labels are determined automatically from
        the specs but can be overridden in a pinch.
    figprefix, figsuffix : str, optional
        Optional modifications to the default figure title determined
        from shared reduction instructions.
    labelbottom, labelright : bool, optional
        Whether to label column labels on the bottom and row labels
        on the right. Otherwise they are on the left and top.
    labelparams : bool, optional
        Whether to change axis labels of scatter plots with multiple forcing-feedback
        from the terms 'forcing/feedback/sensitivity' to the term 'parameter'.
    standardize : bool, optional
        Whether to make axis limits span the same range for all axes with same
        units. See also `rxlim` and `rylim`.
    identical : bool, optional
        Whether to make axis limits span the exact same values for all axes with same
        units. Stricter condition.
    groupnames : bool, str, or sequence of str, optional
        If boolean, whether to group mappable scaling and colorbars by unique array
        ``name`` or by all other scalar coordinates plus ``'units'`` attributes. If
        iterable, indicates the specific coordinates to use for grouping (optionally
        including ``'name'`` and ``'units'``). Also if ``True``, unique legend entries
        are grouped only by ``label``; otherwise they are grouped as with colorbars.

    Returns
    -------
    fig : proplot.Figure
        The figure.
    axs : proplot.Subplotgrid
        The subplots.

    Other Parameters
    ----------------
    gridskip : int or sequence, optional
        The integer gridspec slots to skip.
    argskip : int or sequence, optional
        The axes indices to omit from auto scaling in each group of axes.
    figurespan : bool, optional
        Whether to make colorbars and legends span the entire figure.
    cbarlength : float, optional
        Length of colorbar.
    cbarwrap : float, optional
        Scaling to apply to size used to wrap colorbar labels.
    leggroup : bool, optional
        Whether to group legends. Default is ``groupnames is not True``.
    legcols : int, optional
        Number of legend entry columns. Standard keyword conflicts with `ncols`.
    cbarpad, legpad : float, optional
        Padding for colorbar and legend entries.
    cbarspace, legspace : float, optional
        Space for colorbar and legend entries.
    nrows, ncols : float, optional
        Number of rows or columns when either of the plot specs are singleton.
    rxlim, rylim : float or 2-tuple, optional
        Relative x and y axis limits to apply to groups of shared or standardized axes.
    save : path-like, optional
        Save folder base location. Stored inside a `figures` subfolder.
    suffix : str, optional
        Optional suffix to append to the default path label.
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
    # Initital stuff
    # NOTE: Input geometry not respected if both row and column specs non singleton.
    # TODO: Support e.g. passing 2D arrays to line plotting methods with built-in
    # shadestd, shadepctile, etc. methods instead of using map. See apply_method.
    argskip = np.atleast_1d(() if argskip is None else argskip)
    gridskip = np.atleast_1d(() if gridskip is None else gridskip)
    kws_process, kws_collection, figlabel, pathlabel, gridlabels = parse_specs(
        dataset, rowspecs, colspecs, **kwargs  # parse input specs
    )
    if isinstance(gridlabels, tuple):  # both row and column specs non singleton
        grows, gcols = map(len, gridlabels)
        labels_default = (*gridlabels, [None] * grows * gcols)
    else:  # either row or column spec was singleton
        naxes = len(gridlabels) if gridlabels else 1
        naxes += gridskip.size
        labels_default = (None, None, gridlabels)
        if nrows is not None:
            grows = min(naxes, nrows)
            gcols = 1 + (naxes - 1) // grows
        else:
            gcols = min(naxes, ncols or 4)
            grows = 1 + (naxes - 1) // gcols

    # Label overrides
    # NOTE: Here 'figsuffix' is for the figure and 'suffix' is for the path.
    # NOTE: This supports selective overrides e.g. rowlabels=['custom', None, None]
    figtitle = figtitle if figtitle is not None else figlabel
    figtitle = _capitalize_label(figtitle, prefix=figprefix, suffix=figsuffix)
    geometry = (grows, gcols, grows * gcols)
    labels_input = (rowlabels, collabels, titles)
    labels_output = []
    for nlabels, ilabels, dlabels in zip(geometry, labels_input, labels_default):
        if ilabels is None or isinstance(ilabels, str):
            ilabels = [ilabels] * nlabels
        if dlabels is None or isinstance(dlabels, str):  # so far not used...
            dlabels = [dlabels] * nlabels
        if len(ilabels) > nlabels:
            raise RuntimeError(f'Expected {nlabels} labels but got {len(ilabels)}.')
        labels = [  # permite overriding with e.g. title=['', 'title', '']
            ilabel if ilabel is not None else dlabel
            for ilabel, dlabel in zip(ilabels, dlabels)
        ]
        labels_output.append(labels)
    rowlabels, collabels, titles = labels_output
    nprocess, mprocess = len(kws_process), max(map(len, kws_process))
    indicator = f'{grows}x{gcols}-{nprocess}x{mprocess}'
    print('Figure:', repr(figlabel))
    if isinstance(gridlabels, tuple):  # default grid labels
        print('Rows:', ', '.join(map(repr, gridlabels[0])))
        print('Columns:', ', '.join(map(repr, gridlabels[1])))
    else:  # default axes titles
        print('Axes:', ', '.join(map(repr, gridlabels or ())))
    if suffix:
        pathlabel = f'{pathlabel}_{suffix}'
    if save:  # default figure path
        print('Path:', repr(pathlabel), repr(indicator))

    # Generate data arrays and queued plotting commands
    # NOTE: Critical to disable 'grouping' so that e.g. colorbars or legends that
    # extend into other panel slots are not considered in the tight layout algorithm.
    # NOTE: This will automatically allocate separate colorbars for
    # variables with different declared level-restricting arguments.
    fig = gs = None  # delay instantiation
    leggroup = groupnames is not True if leggroup is None else leggroup
    methods, commands, groups_commands = [], [], {}
    count, count_items = 0, list(zip(kws_process, kws_collection, titles))
    print('Getting data:', end=' ')
    for num in range(grows * gcols):
        # Retrieve data
        if num in gridskip:
            continue
        if count >= nprocess:
            continue
        print(f'{num + 1}/{grows * gcols}', end=' ')
        count += 1  # position accounting for gridskip
        geometry = (grows, gcols, num)
        ikws_process, ikws_collection, title = count_items[count - 1]
        imethods, icommands, arguments, kws_collection = [], [], [], []
        for kw_process, kw_collection in zip(ikws_process, ikws_collection):
            args, method, default = process_data(
                dataset, *kw_process, attrs=kw_collection.attrs.copy()
            )
            for key, value in default.items():  # also adds 'method' key
                kw_collection.command.setdefault(key, value)
            imethods.append(method)
            arguments.append(args)
            kws_collection.append(kw_collection)

        # Infer commands
        kwargs = dict(fig=fig, gs=gs, geom=geometry, title=title)
        iterables = _infer_commands(dataset, arguments, kws_collection, **kwargs)
        fig, gs, axs, icommands, guides, arguments, kws_collection = iterables
        iterables = (axs, imethods, icommands, guides, arguments, kws_collection)
        for ax, method, command, guide, args, kw_collection in zip(*iterables):
            types = (dict, list, np.ndarray, xr.DataArray)  # e.g. flierprops, hatches
            coords = _get_coords(*args, units=True, tuples=True)  # include multi-index
            props = {key: getattr(val, 'name', val) for key, val in kw_collection.command.items()}  # noqa: E501
            if groupnames is True:
                keys = ('name',)
            elif groupnames is False:
                keys = coords.keys() - {'name'}
            else:
                keys = np.atleast_1d(groupnames).tolist()
            coords = tuple((key, val) for key, val in coords.items() if key in keys)
            props = tuple((key, val) for key, val in props.items() if key not in PROPS_IGNORE and not isinstance(val, types))  # noqa: E501
            identifier = (coords, props, method, command, guide)
            tuples = groups_commands.setdefault(identifier, [])
            tuples.append((ax, args, kw_collection))
            if method not in methods:
                methods.append(method)
            if command not in commands:
                commands.append(command)

    # Carry out the plotting commands
    # NOTE: Separate command and handle groups are necessary here because e.g. want to
    # group contours by identical processing keyword arguments so their levels can be
    # shared but then have a colorbar referencing several identical-label variables.
    print('\nPlotting data:', end=' ')
    oneone = False  # whether to scale x and y separately
    groups_xunits = {}
    groups_yunits = {}  # groupings of units across axes
    groups_handles = {}  # groupings of handles across axes
    for num, (identifier, values) in enumerate(groups_commands.items()):
        # Combine plotting and guide arguments
        # WARNING: Use 'step' for determining default colorbar levels and 'locator'
        # for assigning colorbar ticks. Avoids keyword conflicts.
        # NOTE: Here 'colorbar' and 'legend' keywords are automatically added to
        # plotting command keyword arguments by _auto_command.
        # NOTE: Here 'argskip' is isued to skip arguments with vastly different
        # ranges when generating levels that annotate multiple different subplots.
        print(f'{num + 1}/{len(groups_commands)}', end=' ')
        coords, props, method, command, guide = identifier
        axs, arguments, kws_collection = zip(*values)
        kws_command = [kw_collection.command.copy() for kw_collection in kws_collection]
        kws_guide = [getattr(kw_collection, guide).copy() for kw_collection in kws_collection]  # noqa: E501
        kws_axes = [kw_collection.axes.copy() for kw_collection in kws_collection]
        kws_other = [kw_collection.other.copy() for kw_collection in kws_collection]
        if command in ('contour', 'contourf', 'pcolormesh'):
            # Combine command arguments and keywords
            # NOTE: Possibly skip arguments based on index in group list.
            args = list(arguments[0][-1].coords[dim] for dim in arguments[0][-1].dims)
            args.extend(arg for i, args in enumerate(arguments) for arg in args if i not in argskip)  # noqa: E501
            if command == 'contour':  # reserve input 'vmin' and 'vmax' for shading
                min_levels, keys_skip = 1, ('symmetric', 'diverging')
            else:
                min_levels, keys_skip = 2, ()
            kw_levels = {
                key: val for kw_collection in kws_collection
                for key, val in kw_collection.command.items() if key not in keys_skip
            }
            kw_levels.update(min_levels=min_levels, norm_kw={})
            pattern = any(arg.name == 'tpat' for arg in args[2:])
            anomaly = any(arg.attrs.get('long_suffix') == 'anomaly' for arg in args[2:])
            vcenter = kw_levels.pop('vcenter', None)  # normally handled in _parse_cmap
            step = kw_levels.pop('step', None)
            key = 'locator' if np.isscalar(step) else 'levels'
            if pattern and not anomaly:
                vcenter = 1 if vcenter is None else vcenter
            if vcenter is not None:  # update norm_kw as _parse_cmap does
                kw_levels.update({'norm': 'div', 'norm_kw': {'vcenter': vcenter}})
            if step is not None:  # update locator
                kw_levels.update({key: step})
            if step is not None and not np.isscalar(step):
                kw_levels.update(levels=step)
            # Infer color levels
            # NOTE: This ensures consistent levels across gridspec slots
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')  # proplot and cmap runtime warnings
                levels, vmin, vmax, norm, norm_kw, _ = axs[0]._parse_level_vals(*args, **kw_levels)  # noqa: E501
            ls = np.where(levels < 0, '--', '-')  # cmap=['k'] disables -ve linestyle
            locator = pplt.DiscreteLocator(levels, nbins=7)
            minorlocator = pplt.DiscreteLocator(levels, nbins=7, minor=True)
            for kw_command in kws_command:
                kw_command.update(levels=levels, vmin=vmin, vmax=vmax, norm=norm, norm_kw=norm_kw)  # noqa: E501
                if command == 'contour':
                    kw_command.update(linestyles=ls)
                for key in ('step', 'robust'):
                    kw_command.pop(key, None)
            for kw_guide in kws_guide:
                if command != 'contour':
                    kw_guide.setdefault('locator', locator)
                    kw_guide.setdefault('minorlocator', minorlocator)

        # Add plotted content and queue guide instructions
        # NOTE: Commands are grouped so that levels can be synchronized between axes
        # and referenced with a single colorbar... but for contour and other legend
        # entries only the unique labels and handle properties matter. So re-group
        # here into objects with unique labels by the rows and columns they span.
        for ax, args, kw_axes, kw_command, kw_other, kw_guide in zip(
            axs, arguments, kws_axes, kws_command, kws_other, kws_guide
        ):
            # Call plotting command and infer handles
            # NOTE: Still keep axes visible so super title is centered above empty
            # slots and so row and column labels can exist above empty slots.
            print('.', end=' ' if ax is axs[-1] else '')
            prevlines = list(ax.lines)
            if 'contour' in command:  # TODO: remove this group? no longer needed
                if np.allclose(args[-1], 0.0) or np.allclose(args[-1], 1.0):
                    ax._invisible = True
                if getattr(ax, '_invisible', None):
                    continue
            with warnings.catch_warnings():  # ignore 'masked to nan'
                warnings.simplefilter('ignore', (UserWarning, RuntimeWarning))
                if 'violin' in command:
                    handle = result = None  # seaborn violinplot returns ax! ugly!
                    sns.violinplot(*args, ax=ax, **kw_command)
                else:
                    cmd = getattr(ax, command)
                    handle = result = cmd(*args, autoformat=False, **kw_command)
            if command == 'contour':  # pick last contour to avoid negative dashes
                handle = result.collections[-1] if result.collections else None
            if command == 'scatter':  # sizes empty if array not passed
                handle = result.legend_elements('sizes')[0] or result
            if 'bar' in command:  # either container or (errorbar, container)
                handle = result[-1] if type(result) is tuple else result
            if 'line' in command:  # either [line] or [(shade, line)]
                handle = result[0]
            if 'box' in command:  # silent list of PathPatch or Line2D
                handle = result['boxes']
            if 'violin' in command:
                handle = [obj for obj in ax.collections if isinstance(obj, mcollections.PolyCollection)]  # noqa: E501

            # Update and setup plots
            # NOTE: Must be called for each group so violin widths scale across subplots
            # WARNING: Critical to call format before _setup_axes
            kw_fmt = {key: val for key, val in kw_axes.items() if 'proj' not in key}
            ax.format(**kw_fmt)  # apply formatting
            if ax._name == 'cartesian':
                xunits, yunits = _setup_axes(ax, *args, command=command)
                groups_xunits.setdefault(xunits, []).append(ax)
                groups_yunits.setdefault(yunits, []).append(ax)
            if 'scatter' in command:
                ax.use_sticky_edges = False  # show end of line fit shading
                keys = ('zeros', 'oneone', 'linefit', 'annotate', 'constraint', 'graphical')  # noqa: E501
                oneone = oneone or kw_other.get('oneone', False)
                kw_other = {key: val for key, val in kw_other.items() if key in keys}
                _setup_scatter(ax, *args, handle, **kw_other)  # 'args' is 2-tuple
            if 'bar' in command:  # ensure padding around bar edges
                keys = ('horizontal', 'annotate')
                errdata = kw_command.get('bardata', None)
                kw_other = {key: val for key, val in kw_other.items() if key in keys}
                _setup_bars(ax, args, errdata, handle, **kw_other)
            if 'violin' in command:
                keys = ('horizontal', 'alpha', 'hatch', 'linewidth', 'edgecolor')
                line = [line for line in ax.lines if line not in prevlines]
                width = kw_command.get('width', None)  # optionally override
                kw_other = {key: val for key, val in kw_other.items() if key in keys}
                handle = _setup_violins(ax, args[-1], handle, line, width=width, **kw_other)  # noqa: E501

            # Group individual legend or colorbar labels
            # TODO: Optionally do *automatic* grouping of colorbars when they were
            # not normalized together with 'groupnames' (i.e. exclude 'coords' tuple).
            # NOTE: If 'label' is in args[-1].coords it will be used for legend but
            # still want to segregate based on default short_name label to help reader
            # differentiate between sensitivity, forcing, and feedbacks.
            items = props if guide == 'legend' and not leggroup else (*coords, *props)
            label = kw_guide.pop('label', None)
            if 'label' in args[-1].coords:  # TODO: optionally disable
                label = None
            if command == 'scatter':  # TODO: possibly remove this
                handle = label = None
            if command in ('contourf', 'pcolormesh') and (norm := handle.norm):
                items += (('vmin', norm.vmin), ('vmax', norm.vmax), ('N', norm.N))
            identifier = (items, method, command, guide, label)
            tuples = groups_handles.setdefault(identifier, [])
            tuples.append((ax, args[-1], handle, kw_guide, kw_other))

    # Queue legends or colorbars in distinct locations
    # NOTE: This enforces legend handles grouped only for parameters with identical
    # units e.g. separate legends for sensitivity and feedback bar plots.
    # WARNING: Critical to delay wrapping the colorbar label until content is
    # drawn or else the reference width and height cannot be known.
    print('\nAdding guides:', end=' ')
    groups_colorbars, groups_legends = {}, {}
    for identifier, tuples in groups_handles.items():
        handles = []
        *_, guide, label = identifier
        axs, args, handles, kws_guide, _ = zip(*tuples)
        sort = any(arg.sizes.get('facets', None) for arg in args)  # only for 'version'
        if all(handles) and all('label' in arg.coords for arg in args):
            handles, labels, kw_update = _get_handles(args, handles, sort=sort)
        else:  # e.g. scatter
            handles, labels, kw_update = handles[:1], [label], {}
        kw_guide = {key: val for kw_guide in kws_guide for key, val in kw_guide.items()}
        kw_guide.update(kw_update)
        loc = kw_guide.pop('loc', None)
        if loc is False:
            continue
        kw = dict(cbarwrap=cbarwrap, cbarlength=cbarlength, figurespan=figurespan)
        identifier = _auto_guide(*axs, loc=loc, **kw)
        groups = groups_colorbars if guide == 'colorbar' else groups_legends
        for ihandle, ilabel in zip(handles, labels):
            if ihandle is None or ilabel is None:  # e.g. scatter plots
                continue
            if isinstance(ihandle, (list, mcontainer.Container)):
                ihandle = ihandle[0]  # legend_elements list BarContainer container
            tuples = groups.setdefault(identifier, [])
            tuples.append((ihandle, ilabel, kw_guide))

    # Add shared legends and colorbar for each location
    # WARNING: For some reason extendsize adjustment is still incorrect
    # even though axes are already drawn here. Not sure why.
    for guide, groups in zip(('colorbar', 'legend'), (groups_colorbars, groups_legends)):  # noqa: E501
        print('.', end='')
        for identifier, tuples in groups.items():
            src, loc, span, bbox, length, size = identifier
            handles, labels, kws_guide = zip(*tuples)
            kw_guide = {key: val for kw_guide in kws_guide for key, val in kw_guide.items()}  # noqa: E501
            kw_guide.update({} if span is None else {'span': span})
            if guide == 'legend':
                kw_guide.setdefault('frame', False)
                kw_guide.setdefault('ncols', legcols or 1)
                kw_guide.setdefault('order', 'F')
                kw_guide.setdefault('bbox_to_anchor', bbox)
                kw_guide.setdefault('pad', legpad)
                kw_guide.setdefault('space', legspace)
                src.legend(list(handles), list(labels), loc=loc, **kw_guide)
            else:  # TODO: explicitly support colorbars spanning multiple subplots
                kw_guide.setdefault('length', length)
                kw_guide.setdefault('pad', cbarpad)
                kw_guide.setdefault('space', cbarspace)
                for ihandle, ilabel in zip(handles, labels):
                    ilabel = _fit_label(ilabel, refwidth=size)
                    src.colorbar(ihandle, loc=loc, label=ilabel, **kw_guide)

    # Adjust shared units
    # WARNING: This is a kludge. Should consider comparing shared axes instead of
    # just comparing labels across many axes. Consider revising.
    # TODO: Implement this kludge for single axes plots of quantities with different
    # units, e.g. climate sensitivity and climate feedback violins or bars.
    print('.')
    regex = ('effective', 'climate', 'sensitivity', 'radiative', 'forcing', 'feedback')
    regex = re.compile('(' + '|'.join(regex) + r').*\Z', re.DOTALL)
    units = ('K', 'W m^-2', 'W m^-2 K^-1')
    units = tuple(ureg.parse_units(s) for s in units)
    if labelparams:  # multi-unit kludge for e.g. scatter plots
        if set(groups_xunits) <= set(units) and len(groups_xunits) > 1:
            xlabels = [lab for ax in fig.subplotgrid if (lab := ax.get_xlabel())]
            if xlabels and fig._sharex > 0 and fig.gridspec.nrows > 1:
                xlabel = regex.sub('parameter', xlabels[0])
                fig.format(xlabel=xlabel)
        if set(groups_yunits) <= set(units) and len(groups_yunits) > 1:
            ylabels = [lab for ax in fig.subplotgrid if (lab := ax.get_ylabel())]
            if ylabels and fig._sharey > 0 and fig.gridspec.ncols > 1:
                ylabel = regex.sub('parameter', ylabels[0])
                fig.format(ylabel=ylabel)

    # Standardize relative axes limits and impose relative units
    # NOTE: Previously permitted e.g. rxlim=[(0, 1), (0, 0.5)] but these would
    # be applied *implicitly* based on drawing order so too confusing. Use
    # 'outer' from constraints '_build_specs()' function instead.
    if not standardize and not identical:  # auto-search shared axes
        ref = fig.subplotgrid[0]
        groups_xunits, groups_yunits = {}, {}
        if hasattr(ref, '_shared_axes'):
            groups_shared = {'x': list(ref._shared_axes['x']), 'y': list(ref._shared_axes['y'])}  # noqa: E501
        else:
            groups_shared = {'x': list(ref._shared_x_axes), 'y': list(ref._shared_y_axes)}  # noqa: E501
        for axis, groups in tuple(groups_shared.items()):
            for idx, axs in enumerate(groups):
                if all(ax in fig.subplotgrid for ax in axs):
                    if axis == 'x':
                        groups_xunits[idx] = groups[idx]
                    else:
                        groups_yunits[idx] = groups[idx]
    for axis1, axis2, groups1, groups2, rlim in zip(
        'xy', 'yx', (groups_xunits, groups_yunits), (groups_yunits, groups_xunits), (rxlim, rylim),  # noqa: E501
    ):
        rlim = rlim or (0, 1)  # disallow *implicit* application of multiple options
        for i, (unit, axs) in enumerate(groups1.items()):
            pairs = [(axis1, ax) for ax in axs]
            if oneone:  # additionally scale by other axis
                pairs.extend((axis2, ax) for ax in groups2.get(unit, ()))
            lims = [getattr(ax, f'get_{axis}lim')() for axis, ax in pairs]
            span = max((lim[1] - lim[0] for lim in lims), key=abs)  # preserve sign
            amin = min(lim[0] for lim in lims)
            amax = max(lim[1] for lim in lims)
            for (axis, ax), lim in zip(pairs, lims):
                if identical:
                    getattr(ax, f'set_{axis}lim')((amin, amax))
                else:
                    average = 0.5 * (lim[0] + lim[1])
                    min_ = average + span * (rlim[0] - 0.5)
                    max_ = average + span * (rlim[1] - 0.5)
                    getattr(ax, f'set_{axis}lim')((min_, max_))

    # Optionally save the figure
    # NOTE: Still add empty axes so super title is centered above empty
    # slots and so row and column labels can exist above empty slots.
    for num in gridskip:
        ax = fig.add_subplot(gs[num])
        ax._invisible = True
    for ax in fig.axes:
        if getattr(ax, '_invisible', None):
            ax.format(grid=False)  # needed for cartopy axes
            for obj in ax.get_children():
                obj.set_visible(False)
    rowkey = 'rightlabels' if labelright else 'leftlabels'
    colkey = 'bottomlabels' if labelbottom else 'toplabels'
    fig.format(figtitle=figtitle, **{rowkey: rowlabels, colkey: collabels})
    if save:  # save path
        path = Path('' if save is True else save)
        path = path.expanduser().resolve()
        if path.name in ('notebooks', 'meetings', 'manuscripts'):
            path = path.parent  # parent project directory
        path = path / 'figures'
        if not path.is_dir():
            raise ValueError(f'Path {str(path)!r} does not exist.')
        methods = '-'.join(methods)  # prepend methods
        commands = '-'.join(commands)  # append commands
        if path.is_dir():
            path = path / '_'.join((methods, pathlabel, commands, indicator))
        figwidth, figheight = fig.get_size_inches()
        figsize = f'{figwidth:.1f}x{figheight:.1f}in'  # figure size
        print(f'Saving ({figsize}): ~/{path.relative_to(Path.home())}')
        fig.save(path)  # optional extension
    return fig, fig.subplotgrid
