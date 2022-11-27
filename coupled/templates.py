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
import matplotlib.legend_handler as mhandler
import matplotlib.container as mcontainer
import matplotlib.font_manager as mfonts
from climopy import decode_units, format_units, ureg, vreg  # noqa: F401
from .process import _components_slope, _parse_institute, _parse_project, get_data
from .internals import _infer_labels, _wrap_label, parse_specs

__all__ = ['create_plot']

# Default configuration settings
CONFIG_SETTINGS = {
    'unitformat': '~L',
    'inlineformat': 'retina',  # switch between png and retina
    'subplots.refwidth': 2,
    'legend.handleheight': 1.2,
    'hatch.linewidth': 0.3,
    'axes.inbounds': False,  # ignore for proper shading bounds
    'axes.margin': 0.04,
    'toplabel.pad': 8,
    'bottomlabel.pad': 6,
    'leftlabel.pad': 6,
    'rightlabel.pad': 8,
    'negcolor': 'cyan7',  # differentiate from colors used for variables
    'poscolor': 'pink7',
}
pplt.rc.update(CONFIG_SETTINGS)

# Properties to preserve when grouping legend handles
# TODO: Permit both project and flagship labels
PROPS_LEGEND = (
    'alpha',
    'color',
    'cmap',
    'edgecolor',
    'facecolor',
    'marker',
    'markersize',
    'markeredgewidth',
    'markeredgecolor',
    'markerfacecolor',
    'linestyle',
    'linewidth',
    # 'sizes',  # NOTE: ignore flagship indicator
    # 'hatch',  # NOTE: ignore flagship indicator
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
    'hatch',
    'sym',
    'whis',
    'flierprops',
    *(props + suffix for suffix in ('', 's') for props in PROPS_PLURAL),
)

# Default format keyword arguments
KWARGS_FIG = {
    'refwidth': 1.5,
    'abc': 'A.',
}
KWARGS_GEO = {
    'proj': 'hammer',
    'proj_kw': {'lon_0': 180},
    'lonlines': 30,
    'latlines': 30,
    'refwidth': 2.3,
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
    'yminorlocator': 100,
    'ylocator': 200,
    'yreverse': True,
    'ylabel': 'pressure (hPa)',
}

# Default plotting keyword arguments
KWARGS_ANNOTATE = {
    'color': 'gray8',
    'alpha': 1.0,
    'textcoords': 'offset points',
}
KWARGS_BAR = {
    'color': 'gray8',
    'edgecolor': 'black',
    'width': 1.0,
    'absolute_width': False,
}
KWARGS_BOX = {
    'color': 'gray8',
    'means': False,
    'whis': (2.5, 97.5),  # consistent with default 'pctiles' ranges
    'widths': 0.85,
    'manage_ticks': False,
    'flierprops': {'markersize': 2, 'marker': 'x'},
}
KWARGS_CONTOUR = {
    'color': 'gray6',
    'globe': True,
    'robust': 96,
    'nozero': True,
    'labels': True,
    'levels': 3,
    'linewidth': pplt.rc.metawidth,
}
KWARGS_ERRBAR = {
    'capsize': 0,
    'barcolor': 'gray8',
    'barlinewidth': 1.5 * pplt.rc.metawidth,
}
KWARGS_ERRBOX = {
    'capsize': 0,
    'boxcolor': 'gray8',
    'boxlinewidth': 3.5 * pplt.rc.metawidth,
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
KWARGS_ZERO = {
    'color': 'gray9',
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
    colors = [f'{color}{shade}' for color in colors]
    colors = pplt.get_colors(cycle or colors)
    return colors


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
        filt = _parse_institute(data, 'flagship')
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
    if 'facets' not in data.coords:
        raise ValueError('Facets coordinate is missing.')
    elif 'project' in data.coords:  # __contains__ excludes index levels
        projects = [data.coords['project'].item()] * data.facets.size
    else:
        filt65 = _parse_project(data, 'cmip65')
        filt66 = _parse_project(data, 'cmip66')
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


def _get_properties(values, mapping, count, default=None):
    """
    Get lists of properties from input values.

    Parameters
    ----------
    values : list
        The coordinate values.
    mapping : dict
        Mapping of coordinates to properties.
    count : int
        The default number of items.
    default : optional
        The default property.

    Returns
    -------
    result : list or None
        The resulting properties.
    """
    values = np.repeat(values, count) if len(values) == 1 else values
    result = [mapping.get(value, default) for value in values]
    result = default if len(set(result)) <= 1 else result
    return result


def _get_handles(args, objects, pad=None, size=None):
    """
    Return multiple handles and labels from a single plot.

    Parameters
    ----------
    args : list of xarray.DataArray
        The input data arrays.
    objects : list of iterable
        The input plot objects.

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
    # NOTE: Try to make each box in the handle-tuple approximately square and use
    # same 'ndivide' for all entries so that final handles have the same length.
    handles = {}
    properties = set()
    for arg, obj in zip(args, objects):
        labels = arg.coords.get('label', None)
        if labels is None:
            raise ValueError('Input data has no label coordinate.')
        if len(labels) != len(obj):
            raise ValueError(f'Number of labels ({len(labels)}) differs from handles ({len(obj)}).')  # noqa: E501
        idxs = np.arange(len(obj))
        if hasattr(obj[0], 'get_alpha'):  # always label by increasing alpha
            alpha = [handle.get_alpha() for handle in obj]
            idxs = np.argsort([1.0 if a is None else a for a in alpha])
        for idx in idxs:
            label, handle = labels.values[idx], obj[idx]
            props = {}
            for key, value in handle.properties().items():
                if key not in PROPS_LEGEND:
                    continue
                if hasattr(value, 'name'):  # e.g. PathCollection, LineCollection cmap
                    value = value.name
                if isinstance(value, list):
                    value = tuple(value)
                if getattr(value, 'ndim', None) == 1:
                    value = tuple(value.tolist())
                if getattr(value, 'ndim', None) == 2:
                    value = tuple(map(tuple, value.tolist()))
                props[key] = value
            props = tuple(props.items())
            ihandles = handles.setdefault(label, [])
            if hasattr(handle, 'get_hatch') and handle.get_hatch():
                continue  # do not use hatching in handles!
            if props not in properties:
                ihandles.append(handle)
                properties.add(props)
    pad = pad or 0  # make items flush by default
    size = size or pplt.rc['legend.handleheight']
    handles = {label: tuple(ihandles) for label, ihandles in handles.items()}
    handler = mhandler.HandlerTuple(pad=pad, ndivide=max(map(len, handles.values())))
    kw_legend = {
        'handleheight': size,
        'handlelength': size * max(map(len, handles.values())),
        'handler_map': {tuple: handler},  # or use tuple coords as map keys
    }
    labels, handles = map(list, zip(*handles.items()))
    return handles, labels, kw_legend


def _auto_guide(*axs, horizontal='bottom', vertical='bottom', default='bottom'):
    """
    Generate a source, location, and span for the guide based on input axes.

    Parameters
    ----------
    *axs : proplot.Axes
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


def _auto_command(ax, args, kw_collection, shading=True, contour=None):
    """
    Generate a plotting command and keyword arguments from the input arguments.

    Parameters
    ----------
    ax : proplot.Axes
        The input axes.
    args : xarray.DataArray
        Input arguments.
    kw_collection : namedtuple
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
        The command to use.
    guide : str
        The guide to use for the command.
    args : tuple
        The possibly updated positional arguments.
    kw_collection : namedtuple
        The possibly updated keyword arguments.
    """
    # Initial stuff
    # TODO: Support hist and hist2d plots in addition to scatter and barh plots
    # (or just hist since, hist2d usually looks ugly with so little data)
    kw_collection = copy.deepcopy(kw_collection)
    markersize = 0.1 * pplt.rc.fontsize ** 2  # default reference scatter marker size
    linewidth = pplt.rc.metawidth  # default reference bar line width
    hatching = any(val for key, val in kw_collection.command.items() if 'hatch' in key)
    cycle = kw_collection.other.get('cycle', None)
    shade = kw_collection.other.get('shade', None)
    colors = _get_colors(cycle=cycle, shade=shade)
    contour = contour or {}  # count-specific properties
    horizontal = kw_collection.other.get('horizontal', False)
    pcolor = kw_collection.other.get('pcolor', False)
    ignore = {'facets', 'version', 'period', 'components'}  # additional keywords
    args = list(args)  # support assignment below
    dims = tuple(sorted(args[-1].sizes.keys() - ignore))

    # Get default plotting commands
    # NOTE: This preserves project/institute styling across bar/box/regression
    # plots by default, but respecting user-input alpha/color/hatch etc.
    if dims:
        # Plots with longitude, latitude, and/or pressure dimension
        if dims == ('plev',):
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
            defaults = {**KWARGS_CONTOUR, **contour}
    else:
        # Plots with 'facets' or 'components' dimension
        seen, tuples, skip = set(), [], ('project', 'instititute')
        index = args[-1].indexes[args[-1].dims[0]]  # usually 'facets' or 'components'
        ndim = args[-1].size  # in case array is missing
        nargs = sum(isinstance(arg, xr.DataArray) for arg in args)
        multi = args[-1].ndim == 2 or args[-1].ndim == 1 and args[-1].dtype == 'O'
        if nargs == 1 and not multi and 'facets' in args[-1].sizes:
            args[-1] = args[-1].sortby(args[-1], ascending=True)
        for tup in index:
            tuples.append(tuple(val for key, val in zip(index.names, tup) if key not in skip))  # noqa: E501
        if 'facets' in args[-1].sizes:  # tuples == 'model', do not want unique colors!
            use_colors, negpos = False, args[-1].name[-3:] not in ('ecs', 'erf')
            flagship = _get_flagships(args[-1])
            project = _get_projects(args[-1])
        else:
            use_colors, negpos = True, False  # different colors per bar or box
            flagship = np.atleast_1d(args[-1].coords.get('institute', None))
            project = np.atleast_1d(args[-1].coords.get('project', None))
        # Infer default properties
        options = ['cmip5', 'cmip56', 'cmip55', 'cmip6', 'cmip65', 'cmip66']
        iproject = [key for key in options if key in project]
        iproject += [None] if any(proj not in options for proj in project) else []
        ituples = [tup for tup in tuples if tup not in seen and not seen.add(tup)]
        hatches = {False: None, True: '//////'}  # institute
        sizes = {False: 0.5 * markersize, True: 1.5 * markersize}  # institute
        widths = {key: 0.5 * linewidth for key in options[:3]}
        alphas = np.linspace(1 / len(iproject), 1, len(iproject)) ** 1.2
        alphas = dict(zip(iproject, alphas))
        colors = {key: colors[i % len(colors)] for i, key in enumerate(ituples)}
        # Infer property lists and apply to commands
        # NOTE: This just returns 'default' if all items in list are identical
        hatches = _get_properties(flagship, hatches, ndim, default=None)
        sizes = _get_properties(flagship, sizes, ndim, default=markersize)
        alphas = _get_properties(project, alphas, ndim, default=None)
        widths = _get_properties(project, widths, ndim, default=pplt.rc.metawidth)
        colors = _get_properties(tuples, colors, ndim, default=None)
        if nargs == 2:
            command = 'scatter'
            defaults = {**KWARGS_SCATTER, 'alpha': alphas, 'sizes': sizes}
        elif multi:  # box plot
            command = 'boxh' if horizontal else 'box'
            defaults = {**KWARGS_BOX, 'alpha': alphas, 'hatch': hatches}
        else:  # bar plot
            command = 'barh' if horizontal else 'bar'
            defaults = {**KWARGS_BAR, 'alpha': alphas, 'hatch': hatches, 'linewidth': widths}  # noqa: E501
        if use_colors:
            defaults['color'] = colors
        if negpos and 'bar' in command and 'color' not in kw_collection.command:
            defaults['negpos'] = True
    # Update command keywords
    for key, value in defaults.items():
        kw_collection.command.setdefault(key, value)
    if 'contour' in command and 'color' in kw_collection.command:  # cmap for queue keys
        kw_collection.command['cmap'] = (kw_collection.command.pop('color'),)
    if not dims and command[-1] == 'h':  # reverse direction
        kw_collection.axes.setdefault('yreverse', True)

    # Infer guide settings
    # NOTE: This enforces legend handles grouped only for parameters with identical
    # units e.g. separate legends for sensitivity and feedback bar plots.
    if not hatching and command in ('contourf', 'pcolormesh'):
        cartesian = ax._name == 'cartesian'
        refwidth = pplt.units(ax.figure._refwidth or pplt.rc['subplots.refwidth'], 'in')
        refwidth *= 0.7 if not cartesian else 1.0  # see also parse_specs
        extendsize = 1.2 if cartesian else 2.2  # WARNING: vertical colorbars only
        guide = 'colorbar'
        label = args[-1].climo.short_label
        label = _wrap_label(label, refwidth=refwidth)
        kw_collection.colorbar.setdefault('label', label)
        kw_collection.colorbar.setdefault('extendsize', extendsize)
    else:
        keys = ('cmap', 'norm', 'norm_kw')
        guide = 'legend'
        if 'label' in args[-1].coords:  # for legend grouping keys
            label = args[-1].climo.short_name
        elif hatching and 'contour' in command:
            label = None
        elif command == 'contour':  # use the full label
            label = args[-1].climo.short_label
        else:
            label = args[-1].attrs.get('short_prefix', None)
        if 'contour' not in command and 'pcolor' not in command:
            keys += ('robust', 'symmetric', 'diverging', 'levels', 'locator', 'extend')
        kw_collection.legend.setdefault('label', label)
        kw_collection.legend.setdefault('frame', False)
        kw_collection.legend.setdefault('ncols', 1)
    return command, guide, args, kw_collection


def _merge_commands(dataset, arguments, kws_collection):
    """
    Merge several distribution plots into a single box or bar plot instruction.

    Parameters
    ----------
    dataset : xarray.Dataset
        The source dataset.
    arguments : list of tuple
        The input arguments.
    kws_collection : list of namedtuple
        The keyword arguments.

    Returns
    -------
    args : tuple
        The merged arguments.
    kw_collection : namedtuple
        The merged keyword arguments.
    """
    # Combine keyword arguments and coordinates
    # WARNING: Critical to compare including the pairs or else e.g. regressions
    # of early/late abrupt feedbacks on full pre-industrial cannot be labeled.
    # NOTE: Considered removing this and having get_spec support 'lists of lists
    # of specs' per subplot but this is complicated, and would still have to split
    # then re-combine scalar coordinates e.g. 'project', so not much more elegant.
    nargs = set(map(len, arguments))
    if len(nargs) != 1:
        raise RuntimeError(f'Unexpected combination of argument counts {nargs}.')
    if (nargs := nargs.pop()) not in (1, 2):
        raise RuntimeError(f'Unexpected number of arguments {nargs}.')
    keys = ['name', *sorted(set(key for args in arguments for arg in args for key in arg.coords))]  # noqa: E501
    kws_outer, kws_inner = {}, {}  # for label inference
    scalars, vectors, outers, inners = {}, {}, {}, {}
    for key in keys:
        values = []
        for args in arguments:
            vals = []
            for arg in args:
                value = np.array(arg.name if key == 'name' else arg.coords.get(key, ''))
                if np.issubdtype(value.dtype, float) and np.isnan(value):
                    continue
                if value.size != 1:
                    continue
                if isinstance(value := value.item(), tuple):
                    continue
                vals.append(value)
            if len(vals) != len(args):  # ignore if skipped any
                pass
            else:
                values.append(tuple(vals))
        if len(values) != len(arguments):
            pass
        elif all(value == values[0] for value in values):
            scalars[key] = values[0]  # individual tuple
        elif any(value1 == value2 for value1, value2 in zip(values[1:], values[:-1])):
            vectors[key] = outers[key] = values
        else:
            vectors[key] = inners[key] = values
    if not vectors:
        raise RuntimeError('No coordinates found for concatenation.')
    kw_scalar = {key: tup[-1] for key, tup in scalars.items()}
    kw_vector = {key: [tup[-1] for tup in vals] for key, vals in vectors.items()}
    kws_outer = [
        tuple(dict(zip(outers, vals)) for vals in zip(*tups))
        for tups in zip(*outers.values())
    ]
    kws_inner = [
        tuple(dict(zip(inners, vals)) for vals in zip(*tups))
        for tups in zip(*inners.values())
    ]

    # Get tick locations and labels for outer group
    # TODO: Also support colors along *inner* group instead of *outer*
    kw_collection = copy.deepcopy(kws_collection[0])  # deep copy of namedtuple
    for kw in kws_collection:
        for field in kw._fields:
            getattr(kw_collection, field).update(getattr(kw, field))
    horizontal = kw_collection.other.get('horizontal', False)
    offset = kw_collection.other.get('offset', 0.5)  # additional offset coordinate
    axis = 'y' if horizontal else 'x'
    base, group, locs, ticks, kwargs = 0, 0, [], [], []
    for group, (kw_outer, items) in enumerate(itertools.groupby(kws_outer)):
        count = len(items := list(items))  # convert itertools._group object
        ilocs = np.arange(base, base + count)
        itick = 0.5 * (ilocs[0] + ilocs[-1])
        base = base + count + offset
        locs.extend(ilocs)
        ticks.append(itick)
        kwargs.append(kw_outer)  # unique keyword argument groups
    kw_infer = dict(identical=False, long_names=True, title_case=False)
    key1, key2 = ('refheight', 'refwidth') if horizontal else ('refheight', 'refwidth')
    refwidth = kw_collection.figure.get(key1, kw_collection.figure.get(key2))
    refwidth = pplt.units(refwidth or pplt.rc['subplots.refwidth'], 'in')
    refwidth *= 1 / (group + 1)  # scale spacing by number of groups
    labels_inner = _infer_labels(dataset, *kws_inner, **kw_infer)
    labels_outer = _infer_labels(dataset, *kwargs, refwidth=refwidth, **kw_infer)
    if ticks and labels_outer:
        print()  # end previous line
        print('Outer labels:', ', '.join(map(repr, labels_outer)), end=' ')
        kw_axes = {
            f'{axis}ticks': ticks,
            f'{axis}ticklabels': labels_outer,
            f'{axis}grid': False,
            f'{axis}ticklen': 0,
            f'{axis}tickmajorpad': 5,
            f'{axis}rotation': 90 if axis == 'y' else 0,
        }
        kw_collection.axes.update(kw_axes)

    # Merge arrays and infer slope products
    # TODO: Somehow combine regression component coordinates
    locs = np.array(locs, dtype=float)
    keys = ('short_prefix', 'short_suffix', 'short_name', 'units')
    attrs = {
        key: value for (*_, arg) in arguments
        for key, value in arg.attrs.items() if key in keys
    }
    if nargs == 1:  # box plot arguments
        annotations = boxdata = bardata = None
        values = np.array([tuple(arg.values) for (arg,) in arguments], dtype=object)
    else:
        values, annotations, boxdata, bardata = [], [], [], []
        for args in arguments:
            slope, slope_lower, slope_upper, rsquare, _, _, _ = _components_slope(
                *args, adjust=False, pctile=[50, 95],  # percentile ranges
            )
            slope_lower1, slope_lower2 = slope_lower.values.flat
            slope_upper1, slope_upper2 = slope_upper.values.flat
            rsquare = ureg.Quantity(rsquare.item(), '').to('percent')
            annotation = f'${rsquare:~L.0f}$'
            annotation = annotation.replace('%', r'\%').replace(r'\ ', '')
            values.append(slope.item())
            boxdata.append([slope_lower1, slope_upper1])
            bardata.append([slope_lower2, slope_upper2])
            annotations.append(annotation)
        attrs['units'] = slope.attrs['units']  # automatic
        attrs['short_name'] = 'regression coefficient'
        kw_collection.command['width'] = 1.0  # ignore staggered bars
        kw_collection.command['absolute_width'] = True
    index = pd.MultiIndex.from_arrays(list(kw_vector.values()), names=list(kw_vector))
    values = xr.DataArray(
        values,
        name='_'.join(arg.name for arg in arguments[0]),
        dims='components',
        attrs=attrs,
        coords={'components': index, **kw_scalar}
    )
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
    dataset, arguments, kws_collection,
    fig=None, gs=None, ax=None, geom=None, title=None, merge=True,
):
    """
    Infer the plotting command from the input arguments and apply settings.

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
    merge : bool, optional
        Whether to merge commands.

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
    abcloc = 'ul' if all(value > 1 for value in geom[:2]) else 'l'  # avoid offset
    kw_default = KWARGS_FIG.copy()
    if 'lon' in sizes and 'lat' in sizes:
        abcloc = 'ul'  # never necessary
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
    kw_axes = {'title': title, 'abcloc': abcloc, **kw_default}

    # Merge commands and initialize figure and axes
    # TODO: Support *stacked* scatter plots and *grouped* bar plots with 2D arrays
    # for non-project multiple selections? Not too difficult... but maybe not worth it.
    iax = ax
    geom = geom or (1, 1, 0)  # total subplotspec geometry
    if sizes:
        pass
    elif merge and len(arguments) > 1:  # concatenate and add label coordinates
        args, kw_collection = _merge_commands(dataset, arguments, kws_collection)
        arguments, kws_collection = (args,), (kw_collection,)
    else:
        for (*_, arg) in arguments:
            if 'facets' not in arg.dims:
                continue
            labels = [{'project': value} for value in _get_projects(arg)]
            labels = _infer_labels(dataset, *labels, identical=False)
            arg.coords['annotation'] = ('facets', arg.model.values)
            if any(bool(label) for label in labels):
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

    # Get commands and default keyword args
    # NOTE: This implements automatic settings for projects and institutes
    contours = []  # contour keywords
    contours.append({'color': 'gray8', 'linestyle': None})
    contours.append({'color': 'gray3', 'linestyle': ':'})
    kw_other = {
        key: value for kw_collection in kws_collection
        for key, value in kw_collection.other.items()
    }
    cycle = _get_colors(cycle=kw_other.get('cycle'), shade=kw_other.get('shade'))
    results, colors, units = [], {}, {}
    for num, (args, kw_collection) in enumerate(zip(arguments, kws_collection)):
        iunits = args[-1].attrs.get('units', None)  # independent variable units
        if len(sizes) < 2 and len(arguments) > 1:  # else use default bar/line/scatter
            kw_collection.command.setdefault('color', cycle[num % len(cycle)])
        icolor = kw_collection.command.get('color', None)
        if len(sizes) < 2 and units and iunits not in units:
            value = colors.get(ax, ())  # number of colors used so far
            value = value.pop() if len(value) == 1 else 'k'
            axis = 'y' if 'plev' in sizes else 'x'
            ax.format(**{f'{axis}color': value})  # line color or simply black
            iax = getattr(ax, f'alt{axis}')(**{f'{axis}color': icolor})
        command, guide, args, kw_collection = _auto_command(
            iax, args, kw_collection,
            shading=(num == 0),  # shade first plot only
            contour=contours[max(0, min(num - 1, len(contours) - 1))],
        )
        results.append((iax, command, guide, args, kw_collection))
        colors.setdefault(ax, set()).add(icolor)  # colors plotted per axes
        units.update({iunits: iax})  # axes indexed by units
    axs, commands, guides, arguments, kws_collection = zip(*results)
    return fig, gs, axs, commands, guides, arguments, kws_collection


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
    **kwargs
        Additional format arguments.

    Returns
    -------
    xunits, yunits : str
        The associated x and y-axis units.
    """
    # Initial stuff
    # NOTE: Want to disable axhline()/axvline() autoscaling but not currently possible
    # so use plot(). See: https://github.com/matplotlib/matplotlib/issues/14651
    top = ax._get_topmost_axes()
    fig = top.figure
    if command in ('barh', 'boxh'):
        y = None
    elif command in ('bar', 'box', 'line', 'scatter'):
        y = args[-1]
    else:
        y = args[-1].coords[args[-1].dims[0]]
    if command in ('bar', 'box'):
        x = None
    elif command in ('barh', 'boxh'):
        x = args[-1]
    elif command in ('linex', 'scatter'):
        x = args[0]
    else:
        x = args[-1].coords[args[-1].dims[-1]]  # e.g. contour() is y by x
    if ax == top:
        if command in ('bar', 'box', 'line'):
            transform = ax.get_yaxis_transform()
            ax.plot([0, 1], [0, 0], transform=transform, **KWARGS_ZERO)
        if command in ('barh', 'boxh', 'linex'):
            transform = ax.get_xaxis_transform()
            ax.plot([0, 0], [0, 1], transform=transform, **KWARGS_ZERO)

    # Handle x and y axis labels
    # TODO: Handle this automatically with climopy and proplot autoformatting
    units = []
    nrows, ncols, *_ = top.get_subplotspec()._get_geometry()  # custom geometry function
    rows, cols = top._range_subplotspec('y'), top._range_subplotspec('x')
    labelx = ax != top or not fig._sharex or max(rows) == nrows - 1
    labely = ax != top or not fig._sharey or min(cols) == 0
    for s, data, label in zip('xy', (x, y), (labelx, labely)):
        axis = getattr(ax, f'{s}axis')
        unit = getattr(data, 'units', None)
        if data is None and axis.isDefault_majloc:
            ax.format(**{f'{s}locator': 'null'})  # do not overwrite labels
        if label and unit is not None and axis.isDefault_label:
            data = data.copy()
            for key in ('short_prefix', 'short_suffix'):
                data.attrs.pop(key, None)  # avoid e.g. 'anomaly' for non-anomaly data
            data.attrs.setdefault('short_name', '')   # avoid cfvariable inheritence
            data.attrs['short_prefix'] = data.attrs.get(f'{s}label_prefix', '')
            if getattr(fig, f'_span{s}'):
                width, height = fig.get_size_inches()
            else:
                width, height = ax._get_size_inches()  # all axes present by now
            if unit is not None and axis.isDefault_label:
                refwidth = width if s == 'x' else height
                string = _wrap_label(data.climo.short_label, refwidth=refwidth)
                ax.format(**{f'{s}label': string})  # include share settings
        units.append(unit)
    return units


def _setup_bars(ax, *args, bardata=None, horizontal=False, annotate=False):
    """
    Adjust and optionally add content to bar plots.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes.
    *args : xarray.DataArray
        The input arguments.
    bardata : xarray.Dataarray
        The error bar data.

    Other Parameters
    ----------------
    horizontal : bool, optional
        Whether the bars were plotted horizontally.
    annotate : bool, optional
        Whether to add annotations to the bars.
    """
    # Adjust axis bounds
    if not annotate:
        return
    if len(args) == 1:
        positions, data = np.arange(args[-1].size), args[-1]
    else:
        positions, data = args
    labels = data.coords.get('annotation', data.coords[data.dims[0]])
    labels = [' '.join(lab) if isinstance(lab, tuple) else str(lab) for lab in labels.values]  # noqa: E501
    xlim, ylim, width, height = ax.get_xlim(), ax.get_ylim(), *ax._get_size_inches()
    if not horizontal:
        s, lim, width, height, slice_ = 'y', ylim, width, height, slice(None)
    else:
        s, lim, width, height, slice_ = 'x', xlim, height, width, slice(None, None, -1)
    space = 0.6 * pplt.units(width / data.size, 'in', 'pt')
    sizes = [scale * pplt.rc.fontsize for scale in mfonts.font_scalings.values()]
    names, sizes = list(mfonts.font_scalings), np.array(sizes)
    fontsize = names[np.argmin(np.abs(sizes - space))]
    fontscale = mfonts.font_scalings[fontsize]
    fontsize, fontscale = ('medium', 1) if fontscale > 1 else (fontsize, fontscale)
    nchars = max(5 if '$' in label else len(label) for label in labels)
    lower, upper = (data, data) if bardata is None else bardata
    diff = fontscale * pplt.units(nchars, 'em', 'in')
    diff = diff * (max(lim) - min(lim)) / height  # convert to data units
    min_ = lim[0] - diff * np.any(upper < 0)
    max_ = lim[1] + diff * np.any(upper >= 0)
    if getattr(ax, f'get_autoscale{s}_on')():
        ax.format(**{f'{s}lim': (min_, max_)})

    # Add annotations
    # NOTE: Using set_in_layout False significantly improves speed since tight bounding
    # box is faster and looks nicer to allow slight overlap with axes edge.
    for i, label in enumerate(labels):
        rotation = 0 if '$' in label else 90  # assume math does not need rotation
        kw_annotate = {'fontsize': fontsize, **KWARGS_ANNOTATE}
        point = upper[i] if upper[i] >= 0 else lower[i]
        if not horizontal:
            va = 'bottom' if point > 0 else 'top'
            kw_annotate.update({'ha': 'center', 'va': va, 'rotation': rotation})
        else:
            ha = 'left' if point > 0 else 'right'
            kw_annotate.update({'ha': ha, 'va': 'center', 'rotation': 0})
        offset = 0.2 * fontscale * pplt.rc.fontsize
        xydata = (positions[i], point)[slice_]  # position is 'y' if horizontal
        xytext = (0, offset * (1 if point >= 0 else -1))[slice_]  # as above for offset
        res = ax.annotate(label, xydata, xytext, **kw_annotate)
        res.set_in_layout(False)


def _setup_scatter(ax, *args, oneone=False, linefit=False, annotate=False):
    """
    Adjust and optionally add content to scatter plots.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes.
    *args : xarray.DataArray
        The input arguments.
    collection : matplotlib.collection.BarCollection
        The collection.

    Other Parameters
    ----------------
    oneone : bool, optional
        Whether to add a one-one line.
    linefit : bool, optional
        Whether to add a least-squares fit line.
    annotate : bool, optional
        Whether to add annotations to the bars.
    """
    # Add reference lines
    # NOTE: Here climopy automatically reapplies dataarray coordinates to fit line
    # and lower and upper bounds so do not explicitly need sorted x coordinates.
    data0, data1 = args  # exactly two required
    if oneone:
        lim = (*ax.get_xlim(), *ax.get_ylim())
        lim = (min(lim), max(lim))
        avg = 0.5 * (lim[0] + lim[1])
        span = lim[1] - lim[0]
        ones = (avg - 1e3 * span, avg + 1e3 * span)
        ax.format(xlim=lim, ylim=lim)  # autoscale disabled
        ax.plot(ones, ones, color='k', linestyle='--', linewidth=1.5 * pplt.rc.metawidth)  # noqa: E501
    if linefit:  # https://en.wikipedia.org/wiki/Simple_linear_regression
        dim = data0.dims[0]  # generally facets dimension
        slope, _, _, rsquare, fit, fit_lower, fit_upper = _components_slope(
            data0, data1, dim=dim, adjust=False, pctile=None,  # use default of 95
        )
        sign = '(\N{MINUS SIGN})' if slope < 0 else ''  # point out negative r-squared
        rsquare = ureg.Quantity(rsquare.item(), '').to('percent')
        title = f'$R^2 = {sign}{rsquare:~L.1f}$'
        ax.format(lrtitle=title.replace('%', r'\%').replace(r'\ ', ''))
        xdata = np.sort(data0, axis=0)  # linefit sorts this stuff first
        args = (fit_lower.squeeze(), fit_upper.squeeze())  # remove facets dimension
        ax.plot(
            xdata, fit,
            color='r', linestyle='-', linewidth=1.5 * pplt.rc.metawidth, autoformat=False,  # noqa: E501
        )
        ax.area(
            xdata, *args,
            color='r', alpha=0.5 ** 2, linewidth=0, autoformat=False,
        )

    # Add annotations
    # NOTE: Using set_in_layout False significantly improves speed since tight bounding
    # box is faster and looks nicer to allow slight overlap with axes edge.
    if not annotate:
        return
    kw_annotate = {'fontsize': 'x-small', 'textcoords': 'offset points'}
    labels = data1.coords.get('annotation', data1.coords[data1.dims[0]])
    labels = [' '.join(lab) if isinstance(lab, tuple) else str(lab) for lab in labels.values]  # noqa: E501
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    width, _ = ax._get_size_inches()
    diff = (pplt.rc.fontsize / 72) * (max(xlim) - min(xlim)) / width
    xmax = xlim[1] + 5 * diff if ax.get_autoscalex_on() else None
    ymin = ylim[0] - 1 * diff if ax.get_autoscaley_on() else None
    ax.format(xmax=xmax, ymin=ymin)  # skip if overridden by user
    for x, y, label in zip(data0.values, data1.values, labels.values):
        xy = (x, y)  # data coordaintes
        xytext = (2, -2)  # offset coordaintes
        res = ax.annotate(label, xy, xytext, ha='left', va='top', **kw_annotate)
        res.set_in_layout(False)


def create_plot(
    dataset,
    rowspecs=None,
    colspecs=None,
    figtitle=None,
    figprefix=None,
    figsuffix=None,
    rowlabels=None,
    collabels=None,
    labelbottom=False,
    labelright=False,
    maxcols=None,
    argskip=None,
    gridskip=None,
    dcolorbar='right',
    dlegend='bottom',
    hcolorbar='right',
    hlegend='bottom',
    vcolorbar='bottom',
    vlegend='bottom',
    standardize=False,
    relxlim=None,
    relylim=None,
    merge=True,
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
    labelbottom, labelright : bool, optional
        Whether to label column labels on the bottom and row labels
        on the right. Otherwise they are on the left and top.
    maxcols : int, optional
        The default number of columns. Used only if one of the row or variable
        specs is scalar (in which case there are no row or column labels).
    argskip : int or sequence, optional
        The axes indices to omit from auto color scaling in each group of axes
        that shares the same colorbar. Can be used to let columns saturate.
    gridskip : int of sequence, optional
        The gridspec slots to skip. Can be useful for uneven grids when we
        wish to leave earlier slots empty.
    dcolorbar, dlegend : {'bottom', 'right', 'top', 'left'}
        The default location for colorbars or legends annotating a mix of
        axes. Placed with the figure colorbar or legend method.
    hcolorbar, hlegend : {'right', 'left', 'bottom', 'top'}
        The location for colorbars or legends annotating horizontal rows.
        Automatically placed along the relevant axes.
    vcolorbar, vlegend : {'bottom', 'top', 'right', 'left'}
        The location for colorbars or legends annotating vertical columns.
        Automatically placed along the relevant axes.
    standardize : bool, optional
        Whether to standardize axis limits to span the same range for all
        plotted content with the same units.
    relxlim, relylim : float or 2-tuple, optional
        Relative x-limit and y-limit scaling to apply to the standardized range(s).
        Values should lie between 0 and 1.
    merge : bool, optional
        Whether to merge multiple bar or scatter plots or plot on top of each other.
    save : path-like, optional
        The save folder base location. Stored inside a `figures` subfolder.
    **kw_specs
        Passed to `parse_specs`.
    **kw_method
        Passed to `apply_method`.

    Returns
    -------
    fig : proplot.Figure
        The figure.
    axs : proplot.Subplotgrid
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
    kws_process, kws_collection, figlabel, pathlabel, gridlabels = parse_specs(
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
    fig, gs, count = None, None, 0  # delay instantiation
    methods, commands = [], []
    groups_commands = {}
    iterator = tuple(zip(titles, kws_process, kws_collection))
    print('Getting data:', end=' ')
    for num in range(nrows * ncols):
        if num in gridskip:
            continue
        if count > len(iterator):
            continue
        print(f'{num + 1}/{nrows * ncols}', end=' ')
        # Retrieve data
        ititle, ikws_process, ikws_collection = iterator[(count := count + 1) - 1]
        imethods, icommands, arguments, kws_collection = [], [], [], []
        for kw_process, kw_collection in zip(ikws_process, ikws_collection):
            args, method, default = get_data(
                dataset, *kw_process, attrs=kw_collection.attrs.copy()
            )
            for key, value in default.items():  # also adds 'method' key
                kw_collection.command.setdefault(key, value)
            imethods.append(method)
            arguments.append(args)
            kws_collection.append(kw_collection)
        # Infer commands
        kwargs = dict(fig=fig, gs=gs, geom=(nrows, ncols, num), title=ititle, merge=merge)  # noqa: E501
        fig, gs, axs, icommands, guides, arguments, kws_collection = _infer_commands(
            dataset, arguments, kws_collection, **kwargs
        )
        for ax, method, command, guide, args, kw_collection in zip(
            axs, imethods, icommands, guides, arguments, kws_collection
        ):
            keys = []
            name = '_'.join(arg.name for arg in args if isinstance(arg, xr.DataArray))
            types = (dict, list, np.ndarray, xr.DataArray)  # e.g. flierprops, hatches
            props = tuple(
                (key, value) for key, value in kw_collection.command.items()
                if key not in PROPS_IGNORE and not isinstance(value, types)
            )
            identifier = (name, props, method, command, guide)
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
    groups_units, groups_handles = {}, {}  # groupings of units and handles across axes
    for num, (identifier, values) in enumerate(groups_commands.items()):
        # Combine plotting and guide arguments
        # NOTE: Here 'colorbar' and 'legend' keywords are automatically added to
        # plotting command keyword arguments by _auto_command.
        # NOTE: Here 'argskip' is isued to skip arguments with vastly different
        # ranges when generating levels that annotate multiple different subplots.
        print(f'{num + 1}/{len(groups_commands)}', end=' ')
        name, props, method, command, guide = identifier
        axs, arguments, kws_collection = zip(*values)
        kws_command = [kw_collection.command.copy() for kw_collection in kws_collection]
        kws_guide = [getattr(kw_collection, guide).copy() for kw_collection in kws_collection]  # noqa: E501
        kws_axes = [kw_collection.axes.copy() for kw_collection in kws_collection]
        kw_other = {
            key: val for kw_collection in kws_collection
            for key, val in kw_collection.other.items()
        }
        if command in ('contour', 'contourf', 'pcolormesh'):
            xy = tuple(arguments[0][-1].coords[dim] for dim in arguments[0][-1].dims)
            zs = tuple(
                arg for count, args in enumerate(arguments) for arg in args
                if count not in argskip
            )
            kw_levels = {
                key: val for kw_collection in kws_collection
                for key, val in kw_collection.command.items()
            }
            kw_levels.update(norm_kw={}, min_levels=1 if command == 'contour' else 2)
            keys_keep = () if command == 'contour' else ('extend',)
            kw_keep = {key: kw_levels[key] for key in keys_keep if key in kw_levels}
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', pplt.internals.ProplotWarning)
                levels, *_, kw_levels = axs[0]._parse_level_vals(*xy, *zs, **kw_levels)
            locator = pplt.DiscreteLocator(levels, nbins=7)
            minorlocator = pplt.DiscreteLocator(levels, nbins=7, minor=True)
            for kw_command in kws_command:
                kw_command.pop('robust', None)
                kw_command.update({**kw_keep, 'levels': levels})
            if command != 'contour':
                for kw_guide in kws_guide:
                    kw_guide.setdefault('locator', locator)
                    kw_guide.setdefault('minorlocator', minorlocator)

        # Add plotted content and queue guide instructions
        # NOTE: Matplotlib boxplot() manually overrides tick locators and formatters...
        # could work around this in future but for now just re-apply kw_axes each time.
        # NOTE: Commands are grouped so that levels can be synchronized between axes
        # and referenced with a single colorbar... but for contour and other legend
        # entries only the unique labels and handle properties matter. So re-group
        # here into objects with unique labels by the rows and columns they span.
        for ax, args, kw_axes, kw_command, kw_guide in zip(
            axs, arguments, kws_axes, kws_command, kws_guide
        ):
            # Call plotting command
            print('.', end=' ' if ax is axs[-1] else '')
            cmd = getattr(ax, command)
            with warnings.catch_warnings():  # ignore 'masked to nan'
                warnings.simplefilter('ignore', UserWarning)
                handle = result = cmd(*args, autoformat=False, **kw_command)
            if 'bar' in command:  # either container or (errorbar, container)
                handle = result[-1] if type(result) is tuple else handle
            if 'line' in command:  # either [line] or [(shade, line)]
                handle = result[0]
            if 'box' in command:  # silent list of PathPatch or Line2D
                handle = result['boxes']
            if command == 'contour':  # pick last contour to avoid negative dashes
                handle = result.collections[-1] if result.collections else None
            if command == 'scatter':  # sizes empty if array not passed
                handle = result.legend_elements('sizes')[0] or result
            # Update and setup plots
            if 'bar' in command:  # ensure padding around bar edges
                axis = 'y' if kw_other.get('horizontal') else 'x'
                for patch in handle:
                    getattr(patch.sticky_edges, axis).clear()
            if 'bar' in command:
                keys = ('horizontal', 'annotate')
                kw_other = {key: val for key, val in kw_other.items() if key in keys}
                _setup_bars(ax, *args, bardata=kw_command.get('bardata'), **kw_other)
            if command == 'scatter':
                keys = ('oneone', 'linefit', 'annotate')
                kw_other = {key: val for key, val in kw_other.items() if key in keys}
                _setup_scatter(ax, *args, **kw_other)  # 'args' is 2-tuple
            if ax._name == 'cartesian':
                xunits, yunits = _setup_axes(ax, command, *args)
                ax.format(**{key: val for key, val in kw_axes.items() if 'proj' not in key})  # noqa: E501
                groups_units.setdefault(('x', xunits), []).append(ax)
                groups_units.setdefault(('y', yunits), []).append(ax)
            # Group guide handles
            # NOTE: If 'label' is in args[-1].coords it will be used for legend but
            # still want to segregate based on default short_name label to help reader
            # differentiate between sensitivity, forcing, and feedbacks.
            label = kw_guide.pop('label', None)
            if command == 'scatter':
                handle = label = None  # TODO: possibly remove this
            if command in ('contourf', 'pcolormesh'):
                identifier = (name, props, method, command, guide, label)
            else:
                identifier = (props, method, command, guide, label)
            tuples = groups_handles.setdefault(identifier, [])
            tuples.append((axs, args[-1], handle, kw_guide, kw_command))

    # Add legend and colorbar queue
    # WARNING: Must determine spans before drawing or they are overwritten.
    print('\nAdding guides:', end=' ')
    groups_colorbars, groups_legends = {}, {}
    for identifier, tuples in groups_handles.items():
        handles = []
        *_, guide, label = identifier
        axs, args, handles, kws_guide, kws_command = zip(*tuples)
        axs = [ax for subaxs in axs for ax in subaxs]
        kw_update = {}
        if all(handles) and all('label' in arg.coords for arg in args):
            handles, labels, kw_update = _get_handles(args, handles)
        else:  # e.g. scatter
            handles, labels = handles[:1], [label]
        if 'CMIP5' in labels and (idx := labels.index('CMIP5')) is not None:
            handles = [handles[idx], *handles[:idx], *handles[idx + 1:]]
            labels = [labels[idx], *labels[:idx], *labels[idx + 1:]]
        kw_guide = {key: val for kw_guide in kws_guide for key, val in kw_guide.items()}
        kw_guide.update(kw_update)
        if guide == 'colorbar':
            groups, def_, hori, vert = groups_colorbars, dcolorbar, hcolorbar, vcolorbar
        else:
            groups, def_, hori, vert = groups_legends, dlegend, hlegend, vlegend
        src, loc, span = _auto_guide(*axs, default=def_, horizontal=hori, vertical=vert)  # noqa: E501
        for handle, label in zip(handles, labels):
            if handle is not None and label is not None:  # e.g. scatter plots
                if isinstance(handle, (list, mcontainer.Container)):
                    handle = handle[0]  # legend_elements list BarContainer container
                tuples = groups.setdefault((src, loc, span), [])
                tuples.append((handle, label, kw_guide))

    # Add shared colorbars, legends, and axis limits
    # TODO: Should support colorbars spanning multiple columns or
    # rows in the center of the gridspec in addition to figure edges.
    for guide, groups in zip(('colorbar', 'legend'), (groups_colorbars, groups_legends)):  # noqa: E501
        print('.', end='')
        for (src, loc, span), tuples in groups.items():
            handles, labels, kws_guide = zip(*tuples)
            kw_guide = {key: val for kw_guide in kws_guide for key, val in kw_guide.items()}  # noqa: E501
            if span is not None:  # figure row or column span
                kw_guide['span'] = span
            if guide == 'legend':
                src.legend(list(handles), list(labels), loc=loc, **kw_guide)
            else:
                for handle, label in zip(handles, labels):
                    src.colorbar(handle, label=label, loc=loc, **kw_guide)
    print('.')
    ref = fig.subplotgrid[0]
    if not standardize:
        groups_units = {}
        if hasattr(ref, '_shared_axes'):
            groups_shared = {'x': list(ref._shared_axes['x']), 'y': list(ref._shared_axes['y'])}  # noqa: E501
        else:
            groups_shared = {'x': list(ref._shared_x_axes), 'y': list(ref._shared_y_axes)}  # noqa: E501
        for axis, groups in tuple(groups_shared.items()):
            for idx, axs in enumerate(groups):
                if all(ax in fig.subplotgrid for ax in axs):
                    groups_units[axis, idx] = groups[idx]
    if groups_units:  # adjust limits
        for (axis, _), axs in groups_units.items():
            lims = [getattr(ax, f'get_{axis}lim')() for ax in axs]
            span = max((lim[1] - lim[0] for lim in lims), key=abs)  # preserve sign
            rellim = relxlim if axis == 'x' else relylim
            rellim = [0, 1] if rellim is None else rellim
            for ax, lim in zip(axs, lims):
                average = 0.5 * (lim[0] + lim[1])
                offsets = [(-0.5 + rellim[0]) * span, (-0.5 + rellim[1]) * span]
                lim = (average + offsets[0], average + offsets[1])
                getattr(ax, f'set_{axis}lim')(lim)

    # Format the axes and optionally save
    # NOTE: Here default labels are overwritten with non-none 'rowlabels' or
    # 'collabels', and the file name can be overwritten with 'save'.
    rowkey = 'rightlabels' if labelright else 'leftlabels'
    colkey = 'bottomlabels' if labelbottom else 'toplabels'
    custom = {rowkey: rowlabels, colkey: collabels}
    default = {rowkey: gridlabels[0], colkey: gridlabels[1]}
    for num in gridskip:  # kludge to center super title above empty slots
        ax = fig.add_subplot(gs[num])
        for obj in (ax.xaxis, ax.yaxis, ax.patch, *ax.spines.values()):
            obj.set_visible(False)
    for key, clabels, dlabels in zip(custom, custom.values(), default.values()):
        nlabels = nrows if key == rowkey else ncols
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
        print(f'Saving: {path.parent.parent.name}/{path.parent.name}/{path.name}')
        fig.save(path)  # optional extension
    return fig, fig.subplotgrid
