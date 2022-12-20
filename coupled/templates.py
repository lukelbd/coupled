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
import matplotlib.collections as mcollections
import matplotlib.container as mcontainer
import matplotlib.font_manager as mfonts
import seaborn as sns
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
    'colorbar.extend': 1.0,
    'hatch.linewidth': 0.3,
    'axes.inbounds': False,  # ignore for proper shading bounds
    'axes.margin': 0.04,
    'toplabel.pad': 8,
    'bottomlabel.pad': 6,
    'leftlabel.pad': 6,
    'rightlabel.pad': 8,
    'negcolor': 'cyan7',  # differentiate from colors used for variables
    'poscolor': 'pink7',
    'autoformat': False,
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
    'color',  # join separate bar plots
    'hatch',  # join separate bar plots
    'sym',
    'whis',
    'flierprops',
    *(f'{props}{suffix}' for suffix in ('', 's') for props in PROPS_PLURAL),
)

# Default format keyword arguments
KWARGS_FIG = {
    'refwidth': 1.5,
    'abc': 'A.',
}
KWARGS_GEO = {
    'proj': 'hammer',
    'proj_kw': {'lon_0': 210},
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
    'ylim': (1000, 70),
    'yminorlocator': 100,
    'ylocator': 200,
    'ylabel': 'pressure (hPa)',
    # 'yreverse': True,
}

# Default extra plotting keyword arguments
KWARGS_ANNOTATE = {
    'color': 'gray8',
    'alpha': 1.0,
    'textcoords': 'offset points',
}
KWARGS_CENTER = {  # violinplot marker
    'color': 'w',
    'marker': 'o',
    'markersize': (5 * pplt.rc.metawidth) ** 2,
    'markeredgecolor': 'k',
    'markeredgewidth': pplt.rc.metawidth,
    'absolute_size': True,
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
KWARGS_ZERO = {
    'color': 'gray9',
    'scalex': False,
    'scaley': False,
    'zorder': 0.5,
    'alpha': 0.5,
    'linestyle': '-',
    'linewidth': 1.0 * pplt.rc.metawidth,
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
    'color': 'gray6',
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
    'width': 0.7,  # kde pdf scale applied to native units '1 / W m-2 K-1'
    'bw': 0.3,  # in feedback or temperature units
    'cut': 0.3,  # no more than this fraction of bandwidth (still want to see lines)
    'saturation': 1.0,  # exactly reproduce input colors (not the default! weird!)
    'linewidth': pplt.rc.metawidth,
}


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
    if 'project' in data.coords:  # __contains__ excludes index levels
        projects = [data.coords['project'].item()] * data.sizes.get('facets', 1)
    elif 'facets' not in data.coords:
        raise ValueError('Facets coordinate is missing.')
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


def _get_handles(args, handles, pad=None, size=None):
    """
    Return multiple handles and labels from a single plot.

    Parameters
    ----------
    args : list of xarray.DataArray
        The input data arrays.
    handles : list of iterable
        The input plot handles.

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
    def _sort_idxs(handles):
        idxs = list(range(len(handles)))
        if handles and hasattr(handles[0], 'get_facecolor'):  # use increasing alpha
            null = lambda: None
            cols = [getattr(handle, 'get_facecolor', null)() for handle in handles]
            cols = ['none' if val is None else val for val in cols]
            cols = [val.squeeze() if hasattr(val, 'squeeze') else val for val in cols]
            alphas = [pplt.to_xyza(val, space='hsv')[3] for val in cols]
            hatches = [bool(getattr(handle, 'get_hatch', null)()) for handle in handles]
            widths = [handle.get_linewidth() for handle in handles]
            keys = list(zip(alphas, hatches, widths, range(len(alphas))))
            idxs = [idx for idx, _ in sorted(enumerate(keys), key=lambda tup: tup[1])]
        return idxs
    # Iterate jointly over arguments handle components
    tuples = {}
    properties = set()
    for arg, ihandles in zip(args, handles):
        labels = arg.coords.get('label', None)
        if labels is None:
            raise ValueError('Input data has no label coordinate.')
        labels = [label for label in labels.values if label is not None]  # for seaborn
        if len(labels) != len(ihandles):
            raise ValueError(f'Number of labels ({len(labels)}) differs from handles ({len(ihandles)}).')  # noqa: E501
        for idx in _sort_idxs(ihandles):  # infer relevant artist properties
            label, handle = labels[idx], ihandles[idx]
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
            tups = tuples.setdefault(label, [])
            props = tuple(props.items())
            if props not in properties:
                tups.append(handle)
                properties.add(props)
    # Get legend args and kwargs
    # NOTE: Try to make each box in the handle-tuple approximately square and use
    # same 'ndivide' for all entries so that final handles have the same length.
    pad = pad or 0  # make items flush by default
    size = size or pplt.rc['legend.handleheight']
    tuples = {
        label: tuple(handles[idx] for idx in _sort_idxs(handles))
        for label, handles in tuples.items()
    }
    ndivide = max(map(len, tuples.values()))
    handler = mhandler.HandlerTuple(pad=pad, ndivide=ndivide)
    kw_legend = {
        'handleheight': size,
        'handlelength': size * max(map(len, tuples.values())),
        'handler_map': {tuple: handler},  # or use tuple coords as map keys
    }
    labels, handles = map(list, zip(*tuples.items()))
    return handles, labels, kw_legend


def _auto_guide(
    *axs, horizontal='bottom', vertical='bottom', default='bottom', figurespan=False
):
    """
    Generate a source, location, and span for the guide based on input axes.

    Parameters
    ----------
    *axs : proplot.Axes
        The input axes.
    horizontal, vertical, default : str, optional
        The default horizontal and vertical stuff.
    figurespan : bool, optional
        Whether to span the whole figure by default.

    Returns
    -------
    src : object
        The figure or axes to use.
    loc : str
        The guide location to use.
    span : 2-tuple
        The span for figure-edge labels.
    """
    # Infer gridspec location of axes
    fig = axs[0].figure
    rows = set(n for ax in axs for n in ax._get_topmost_axes()._range_subplotspec('y'))
    cols = set(n for ax in axs for n in ax._get_topmost_axes()._range_subplotspec('x'))
    nrows, ncols, *_ = fig.gridspec.get_geometry()
    if figurespan or not (len(rows) == 1) ^ (len(cols) == 1):
        src = fig
        loc = default
        # if loc[0] == 'r' and min(cols) == 0 and max(cols) != ncols - 1:
        #     loc = 'left'  # so far do *not* auto adjust top and bottom
        # if loc[0] == 'l' and min(cols) != 0 and max(cols) == ncols - 1:
        #     loc = 'right'
    # Single-row legend or colorbar
    elif len(rows) == 1:
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
    # Single-column legend or colorbar
    else:
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


def _auto_command(
    args, kw_collection, violin=True, shading=True, contour=None, multicolor=False,
):
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
    multicolor : bool, optional
        Whether to use colors instead of alpha for projects.

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
    *locs, data = args  # note coords could be empty
    dims = sorted(data.sizes.keys() - {'facets', 'version', 'components'})
    contour = contour or {}  # count-specific properties
    horizontal = kw_collection.other.get('horizontal', False)
    pcolor = kw_collection.other.get('pcolor', False)
    cycle = kw_collection.other.get('cycle', None)
    markersize = 0.1 * pplt.rc.fontsize ** 2  # default reference scatter marker size
    linewidth = 0.6 * pplt.rc.metawidth  # default reference bar line width
    hatching = any(val for key, val in kw_collection.command.items() if 'hatch' in key)
    if cycle is not None:
        cycle = pplt.get_colors(cycle)
    elif multicolor:  # TODO: improve
        cycle = ['cyan7', 'pink7', 'pink3']
    else:
        cycle = CYCLE_DEFAULT
    def _get_lists(values, options, default=None):  # noqa: E301
        if len(values) == 1:
            values = np.repeat(values, data.size)
        result = [options.get(value, default) for value in values]
        if len(set(result)) <= 1:
            result = [default] * len(values)
        return result

    # Get default plotting commands
    # NOTE: This preserves project/institute styling across bar/box/regression
    # plots by default, but respecting user-input alpha/color/hatch etc.
    if dims:
        # Plots with longitude, latitude, and/or pressure dimension
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
            defaults = {**KWARGS_CONTOUR, **contour}
    else:
        # Plots with 'facets' or 'components' dimension
        skip, seen, index = ('project', 'instititute'), set(), []
        nargs = sum(isinstance(arg, xr.DataArray) for arg in args)
        ragged = data.ndim == 2 or data.ndim == 1 and data.dtype == 'O'
        if nargs == 1 and not ragged and 'facets' in data.sizes:
            data = data.sortby(data, ascending=True)
        for key in (idx := data.indexes[data.dims[0]]):  # 'facets' or 'components'
            index.append(tuple(s for n, s in zip(idx.names, key) if n not in skip))
        if 'facets' in data.sizes:  # others == 'model', do not want unique colors!
            use_colors = multicolor
            flagship = _get_flagships(data)
            project = _get_projects(data)
        else:
            use_colors = True  # always different colors per bar or box
            flagship = np.atleast_1d(data.coords.get('institute', None))
            project = np.atleast_1d(data.coords.get('project', None))
        # Infer default properties
        # widths = {False: 0.1 * linewidth, True: 1.5 * linewidth}  # institute
        # widths = _get_lists(flagship, widths, default=linewidth)
        order = ['cmip5', 'cmip56', 'cmip55', 'cmip6', 'cmip65', 'cmip66']
        projs = [key for key in order if key in project]
        projs += [None] if any(proj not in order for proj in project) else []
        keys = [key for key in index if key not in seen and not seen.add(key)]
        dict_ = {False: 0.50 * markersize, True: 1.5 * markersize}  # institute
        sizes = _get_lists(flagship, dict_, default=markersize)
        dict_ = {False: 0.85, True: 1}  # subtle distinction
        alphas = _get_lists(flagship, dict_, default=1.0)
        if multicolor:
            dict_ = {key: cycle[i % len(cycle)] for i, key in enumerate(projs)}
            colors = _get_lists(project, dict_, default=None)
        else:
            nproj = len(set(proj[4] for proj in projs if proj))  # both cmip5 and cmip6
            dict_ = {'cmip66': 'xxxxxx'}
            hatches = _get_lists(project, dict_, default=None)
            dict_ = {'cmip66': 'gray3'}
            edges = _get_lists(project, dict_, default='black')
            dict_ = {'cmip5': 0.3, 'cmip56': 0.3, 'cmip55': 0.3} if nproj > 1 else {}
            fades = _get_lists(project, dict_, default=1.0)
            alphas = [f - f ** 0.1 * (1 - a) for a, f in zip(alphas, fades)]
            dict_ = {key: cycle[i % len(cycle)] for i, key in enumerate(keys)}
            colors = _get_lists(index, dict_, default=None)
        # Apply properties to commands
        kw_vector = {
            'alpha': alphas,
            'hatch': hatches,
            'edgecolor': edges,
            'linewidth': linewidth,  # TODO: return to thinner lines for cmip5?
        }
        if nargs == 2:
            command = 'scatter'
            defaults = {**KWARGS_SCATTER, 'alpha': alphas, 'sizes': sizes}
        elif not ragged:  # bar plot
            command = 'barh' if horizontal else 'bar'
            defaults = {**KWARGS_BAR, **kw_vector}
        elif not violin:
            command = 'boxh' if horizontal else 'box'
            defaults = {**KWARGS_BOX, **kw_vector}
        else:
            command = 'violinh' if horizontal else 'violin'  # used in filenames only
            kwargs = dict(color=colors, horizontal=horizontal)
            data, kw_collection = _seaborn_data(*locs, data, kw_collection, **kwargs)
            locs, defaults = (), KWARGS_VIOLIN.copy()
            kw_collection.other.update(kw_vector)
        if not use_colors and 'bar' in command and 'color' not in kw_collection.command:
            defaults['negpos'] = data.name[-3:] not in ('ecs',)
        if use_colors and 'violin' not in command and colors is not None:
            defaults['color'] = colors

    # Update command and guide keywords
    # NOTE: This enforces legend handles grouped only for parameters with identical
    # units e.g. separate legends for sensitivity and feedback bar plots.
    # WARNING: Critical to delay wrapping the colorbar label until content is
    # drawn or else the reference width and height cannot be known.
    kw_collection = copy.deepcopy(kw_collection)
    for key, value in defaults.items():
        kw_collection.command.setdefault(key, value)
    if 'contour' in command and 'color' in kw_collection.command:  # cmap for queue keys
        kw_collection.command['cmap'] = (kw_collection.command.pop('color'),)
    if not dims and command[-1] == 'h':  # reverse direction
        kw_collection.axes.setdefault('yreverse', True)
    if not hatching and command in ('contourf', 'pcolormesh'):
        guide = 'colorbar'
        label = data.climo.short_label
        kw_collection.colorbar.setdefault('label', label)
    else:
        guide = 'legend'
        if 'label' in data.coords:  # for legend grouping keys
            label = data.climo.short_name
        elif hatching and 'contour' in command:
            label = None
        elif command == 'contour':  # use the full label
            label = data.climo.short_label
        else:
            label = data.attrs.get('short_prefix', None)
        kw_collection.legend.setdefault('label', label)
    return command, guide, (*locs, data), kw_collection


def _combine_commands(dataset, arguments, kws_collection, labels=None):
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
    labels : list of str, optional
        Optional overrides for outer labels.

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
                value = np.array(arg.name if key == 'name' else arg.coords.get(key))
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

    # Get tick locations for outer group
    # TODO: Also support colors along *inner* group instead of *outer*
    kws_outer = [
        tuple(dict(zip(outers, vals)) for vals in zip(*tups))
        for tups in zip(*outers.values())
    ]
    kw_collection = copy.deepcopy(kws_collection[0])  # deep copy of namedtuple
    for kw in kws_collection:
        for field in kw._fields:
            getattr(kw_collection, field).update(getattr(kw, field))
    offset = kw_collection.other.get('offset', 0.5)  # additional offset coordinate
    base, group, locs, ticks, kw_groups = 0, 0, [], [], []
    for group, (kw_outer, items) in enumerate(itertools.groupby(kws_outer)):
        count = len(items := list(items))  # convert itertools._group object
        ilocs = np.arange(base, base + count)
        itick = 0.5 * (ilocs[0] + ilocs[-1])
        base = base + count + offset
        locs.extend(ilocs)
        ticks.append(itick)
        kw_groups.append(kw_outer)  # unique keyword argument groups

    # Assign inner and outer labels
    # NOTE: This also tries to set up appropriate automatic line breaks
    kws_inner = [
        tuple(dict(zip(inners, vals)) for vals in zip(*tups))
        for tups in zip(*inners.values())
    ]
    hori = kw_collection.other.get('horizontal', False)
    axis = 'y' if hori else 'x'
    kw_infer = dict(identical=False, long_names=True, title_case=False)
    key1, key2 = ('refheight', 'refwidth') if hori else ('refheight', 'refwidth')
    refwidth = kw_collection.figure.get(key1, kw_collection.figure.get(key2))
    refwidth = pplt.units(refwidth or pplt.rc['subplots.refwidth'], 'in')
    refwidth *= 1.2 / (group + 1)  # scale spacing by number of groups
    labels_inner = _infer_labels(dataset, *kws_inner, **kw_infer)
    labels_outer = _infer_labels(dataset, *kw_groups, refwidth=refwidth, **kw_infer)
    if labels is None:
        pass
    elif len(labels) == len(labels_outer):
        labels_outer = labels
    else:
        raise ValueError(f'Mismatch between {len(labels)} and {len(labels_outer)}.')
    if labels_outer:
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

    # Merge into ragged array or array of coefficients
    # TODO: Somehow combine regression component coordinates
    locs = np.array(locs or np.arange(len(kws_inner)), dtype=float)
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

    # Concatenate arrays
    # NOTE: For now only 'project' and 'institute' levels of 'components' are used
    kw_scalar = {key: tup[-1] for key, tup in scalars.items()}
    kw_vector = {key: [tup[-1] for tup in vals] for key, vals in vectors.items()}
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
    fig=None, gs=None, ax=None, geom=None, title=None,
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
    colorbar : str, optional
        The default colorbar location.

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
    abcloc = 'ul' if title is None else 'l'  # avoid offset
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
    # for non-project multiple selections? Not difficult... but maybe not worth
    # it... also grouped bar plots would need to be merged into 2D array.
    iax = ax
    geom = geom or (1, 1, 0)  # total subplotspec geometry
    if sizes:
        pass
    elif len(arguments) > 1:  # concatenate and add label coordinates
        args, kw_collection = _combine_commands(dataset, arguments, kws_collection)
        arguments, kws_collection = (args,), (kw_collection,)
    else:
        for *_, arg in arguments:
            labels = [{'project': value} for value in _get_projects(arg)]
            if 'facets' not in arg.dims:
                continue
            labels = _infer_labels(dataset, *labels, identical=False)
            arg.coords['annotation'] = ('facets', arg.model.values)
            if not all(bool(label) for label in labels):
                continue
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
    kw_other = {}
    for kw_collection in kws_collection:
        kw_other.update(kw_collection.other)
    cycle = kw_other.get('cycle')
    cycle = pplt.get_colors(cycle) if cycle else CYCLE_DEFAULT
    contours = []  # contour keywords
    contours.append({'color': 'gray8', 'linestyle': None})
    contours.append({'color': 'gray3', 'linestyle': ':'})
    results, colors, units = [], {}, {}
    for num, (args, kw_collection) in enumerate(zip(arguments, kws_collection)):
        iunits = args[-1].attrs.get('units', None)  # independent variable units
        if len(sizes) < 2 and len(arguments) > 1:  # line plot cycle colors
            kw_collection.command.setdefault('color', cycle[num % len(cycle)])
        icolor = kw_collection.command.get('color', None)
        if len(sizes) < 2 and units and iunits not in units:
            value = colors.get(ax, ())  # number of colors used so far
            value = value.pop() if len(value) == 1 else 'k'
            axis = 'y' if 'plev' in sizes else 'x'
            ax.format(**{f'{axis}color': value})  # line color or simply black
            iax = getattr(ax, f'alt{axis}')(**{f'{axis}color': icolor})
        command, guide, args, kw_collection = _auto_command(
            args,
            kw_collection,  # keyword arg colleciton
            shading=(num == 0),  # shade first plot only
            contour=contours[max(0, min(num - 1, len(contours) - 1))],
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
    # NOTE: Want to disable axhline()/axvline() autoscaling but not currently possible
    # so use plot(). See: https://github.com/matplotlib/matplotlib/issues/14651
    top = ax._get_topmost_axes()
    fig = top.figure
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
    if ax == top:
        if command in ('bar', 'box', 'violin', 'line'):
            transform = ax.get_yaxis_transform()
            ax.plot([0, 1], [0, 0], transform=transform, **KWARGS_ZERO)
        if command in ('barh', 'boxh', 'violinh', 'linex'):
            transform = ax.get_xaxis_transform()
            ax.plot([0, 0], [0, 1], transform=transform, **KWARGS_ZERO)

    # Handle x and y axis labels
    # TODO: Handle this automatically with climopy and proplot autoformatting...
    # for some reason share=True seems to have no effect but not sure why.
    units = []
    nrows, ncols, *_ = top.get_subplotspec()._get_geometry()  # custom geometry function
    rows, cols = top._range_subplotspec('y'), top._range_subplotspec('x')
    edgex, edgey = max(rows) == nrows - 1, min(cols) == 0
    for s, data, edge in zip('xy', (x, y), (edgex, edgey)):
        share = getattr(fig, f'_share{s}')
        label = edge or not share or ax != top
        axis = getattr(ax, f'{s}axis')
        unit = getattr(data, 'units', None)
        if data is None and axis.isDefault_majloc:
            kw = {f'{s}locator': 'null'}
            ax.format(**kw)  # do not overwrite labels
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
                label = _wrap_label(data.climo.short_label, refwidth=refwidth)
                kw = {f'{s}label': label}
                ax.format(**kw)  # include share settings
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
    # NOTE: Matplotlib cannot include text labels in autoscaling so have
    # to adjust manually. See: https://stackoverflow.com/a/32637550/4970632
    locs, data = (np.arange(args[-1].size), args[-1]) if len(args) == 1 else args
    labels = data.coords.get('annotation', data.coords[data.dims[0]]).values
    labels = [' '.join(lab) if isinstance(lab, tuple) else str(lab) for lab in labels]
    nchars = [3 if '$' in label else len(label) for label in labels]
    width, height = ax._get_size_inches()
    if not horizontal:
        s, width, height, slice_ = 'y', width, height, slice(None)
    else:
        s, width, height, slice_ = 'x', height, width, slice(None, None, -1)
    space = 0.6 * pplt.units(width / data.size, 'in', 'pt')
    sizes = [scale * pplt.rc.fontsize for scale in mfonts.font_scalings.values()]
    names, sizes = list(mfonts.font_scalings), np.array(sizes)
    fontsize = names[np.argmin(np.abs(sizes - space))]
    fontscale = mfonts.font_scalings[fontsize]
    fontsize, fontscale = ('medium', 1) if fontscale > 1 else (fontsize, fontscale)

    # Adjust axes limits
    # NOTE: This also asserts that error bars without labels are excluded from the
    # default data limits (e.g. when extending below zero).
    lower, upper = (data, data) if errdata is None else errdata
    data = getattr(data, 'values', data)
    lower = getattr(lower, 'values', lower)
    upper = getattr(upper, 'values', upper)
    above = np.sum(upper >= 0) >= upper.size // 2
    points = np.array(upper if above else lower)  # copy for the labels
    points = np.clip(points, 0 if above else None, None if above else 0)
    lower = data if above else lower  # used for auto scaling
    upper = upper if above else data  # used for auto scaling
    min_, max_ = min(np.min(lower), 0), max(np.max(upper), 0)
    margin = pplt.rc[f'axes.{s}margin'] * (max_ - min_)
    offsets = np.array(pplt.units(nchars, 'em', 'in'))  # approx inches
    offsets *= 0.8 * fontscale * (max_ - min_) / height  # approx data units
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
            rotation = 0 if '$' in label else 90  # assume math does not need rotation
            kw_annotate = {'fontsize': fontsize, **KWARGS_ANNOTATE}
            if not horizontal:
                va = 'bottom' if above else 'top'
                kw_annotate.update({'ha': 'center', 'va': va, 'rotation': rotation})
            else:
                ha = 'left' if above else 'right'
                kw_annotate.update({'ha': ha, 'va': 'center', 'rotation': 0})
            offset = 0.2 * fontscale * pplt.rc.fontsize
            xydata = (loc, point)[slice_]  # position is 'y' if horizontal
            xytext = (0, offset * (1 if above else -1))[slice_]  # as above
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
    # Add reference one:one line
    # NOTE: This also disables autoscaling so that line always looks like a diagonal
    # drawn right across the axes. Also requires same units on x and y axis.
    data0, data1 = args  # exactly two required
    if oneone:
        lim = (*ax.get_xlim(), *ax.get_ylim())
        lim = (min(lim), max(lim))
        avg = 0.5 * (lim[0] + lim[1])
        span = lim[1] - lim[0]
        ones = (avg - 1e3 * span, avg + 1e3 * span)
        ax.format(xlim=lim, ylim=lim)  # autoscale disabled
        ax.plot(ones, ones, color='k', linestyle='--', linewidth=1.5 * pplt.rc.metawidth)  # noqa: E501

    # Add manual line fit
    # NOTE: Here climopy automatically reapplies dataarray coordinates to fit line
    # and lower and upper bounds so do not explicitly need sorted x coordinates.
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
        ax.plot(xdata, fit, color='r', linestyle='-', linewidth=1.5 * pplt.rc.metawidth)
        ax.area(xdata, *args, color='r', alpha=0.5 ** 2, linewidth=0)

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


def _setup_violins(ax, data, handle, lines=None, width=None, horizontal=False, **kwargs):  # noqa: E501
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
    lines : list of matplotlib.lines.Line2D
        The violin lines.
    width : float, optional
        The violin width in data units.

    Other Parameters
    ----------------
    horizontal : bool, optional
        Whether the violins were plotted horizontally.
    **kwargs
        Additional (optionally vector) properties to apply to violins.
    """
    # Apply violin styling
    # NOTE: Seaborn ignores opacity channel on input color palette so must be
    # manually applied here. Also apply to lines denoting individual observations
    # and use manual alpha blending so that violin shapes can overlap without issues.
    # See: https://github.com/mwaskom/seaborn/issues/622
    # See: https://stackoverflow.com/q/68731566/4970632
    # See: https://matplotlib.org/3.1.0/tutorials/colors/colors.html
    # NOTE: Skip applying opacity to edges because faded outline appears conflicting
    # in combination with white hatching and outline used for non-matching CMIP6.
    data = getattr(data, 'values', data)
    nums = [np.sum(np.isfinite(data[:, i])) for i in range(data.shape[1])]
    idxs = np.append(0, np.cumsum([num for num in nums if num > 0]))
    assert not lines or len(lines) == max(idxs)
    locs = [i for i, num in enumerate(nums) if num > 0]
    lines = [lines[i:j] if lines else () for i, j in zip(idxs[:-1], idxs[1:])]
    assert not lines or len(lines) == len(locs)
    for key in ('alpha', 'hatch', 'linewidth', 'edgecolor'):
        values = kwargs.get(key, None)
        if np.isscalar(values):
            values = [values] * len(handle)
        for ihandle, ilines, value in zip(handle, lines, values, strict=True):
            if value is None:
                continue
            if key != 'alpha':
                ihandle.update({key: value})
            else:
                facecolor = ihandle.get_facecolor().squeeze()[:3]  # singleton array
                facecolor = facecolor * value + np.array([1, 1, 1]) * (1 - value)
                ihandle.set_facecolor(facecolor)
                # for line in ilines:  # WARNING: too much going on... keep line colors
                #     linecolor = np.array(pplt.to_rgb(line.get_color()))
                #     linecolor = linecolor * value + np.array([1, 1, 1]) * (1 - value)
                #     line.set_color(linecolor)

    # Scale widths and add median
    # TODO: Move this into custom proplot function. Matplotlib and seaborn both
    # proihbit skipping width scaling, matplotlib seems to use weaker custom gaussian
    # kde estimator instead of scipy version, and seaborn has major design limitations.
    # NOTE: Seaborn violin plot 'width' and 'count' both scale by maximum density per
    # violin and 'area' by maximum density across all violins. Otherwise the default
    # scipy kde gaussian result evaluates to one when integrated over its domain in
    # data units. To restore that behavior, and thus have equal area violins across
    # many subplots, we undo the scaling by performing the integral ourselves.
    axis = 'y' if horizontal else 'x'
    width = width or KWARGS_VIOLIN.get('width', 1.0)  # scaled width preserving area
    scatter = 'scatterx' if horizontal else 'scatter'
    for i, (ihandle, ilines, iloc) in enumerate(zip(handle, lines, locs, strict=True)):
        zorder = 2 - (i + 1) / len(handle)
        polys = ihandle.get_paths()[0].to_polygons()[0]
        grids = polys[:, 1 - int(horizontal)]  # gridpoints for gaussian kde sample
        kdefit = polys[:, int(horizontal)]  # gaussian kde sample points
        span = np.max(grids) - np.min(grids)
        center = np.mean(kdefit)  # centered position of violin shape
        scale = span * np.mean(np.abs(kdefit - center))  # integral of pdf over shape
        polys[:, int(horizontal)] = center + width * (kdefit - center) / scale
        ihandle.set_verts([polys])
        ihandle.set_zorder(zorder)
        for line in ilines:
            points = getattr(line, f'get_{axis}data')()
            points = center + width * (points - center) / scale
            getattr(line, f'set_{axis}data')(points)
            line.set_solid_capstyle('butt')  # prevent overlapping on violin edges
            line.set_zorder(zorder)
        point = np.nanmedian(data[:, iloc])
        getattr(ax, scatter)(center, point, zorder=zorder + 0.001, **KWARGS_CENTER)


def _seaborn_data(locs, data, kw_collection, color=None, horizontal=False):
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
        Optional color

    Returns
    -------
    data : xarray.DataArray
        The in-filled and merged 2D array.
    kw_collection : namedtuple
        The updated keyword arguments.
    """
    # Infer labels and colors
    # NOTE: Seaborn cannot handle custom violin positions. So use fake data to achieve
    # the same effective spacing. See https://stackoverflow.com/a/52729348/4970632
    kw_collection = copy.deepcopy(kw_collection)
    if not isinstance(data, xr.DataArray) or data.dtype != object or data.ndim != 1:
        raise ValueError('Unexpected input array for violin plot formatting.')
    scale = 100  # highest precision of 'offset' used in _combine_commands()
    locs = locs - np.min(locs)  # ensure starts at zero
    locs = np.round(scale * locs).astype(int)
    step = np.gcd.reduce(locs)  # e.g. 100 if lcos were integer, 50 if were [1, 2.5, 4]
    locs = (locs / step).astype(int)  # e.g. from [0, 1, 2.5, 4] to [0, 2, 5, 8]
    color = KWARGS_VIOLIN['color'] if color is None else color
    color = [color] * len(locs) if np.isscalar(color) else color
    orient = 'h' if horizontal else 'v'  # different convention
    axis = 'y' if horizontal else 'x'
    ticks = np.array(kw_collection.axes.get(f'{axis}ticks', 1))  # see _combine_commands

    # Concatenate array and update keyword args
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
        dims=('index', 'label'),  # expand singleton data.dims
        coords={'label': np.array(labels)},  # possibly non-unique
        attrs=data.attrs,
    )
    kw_collection.axes.update({f'{axis}ticks': ticks * scale / step})  # overwrite
    kw_collection.command.update({'orient': orient, 'palette': palette.tolist()})
    return data, kw_collection


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
    argskip=None,
    gridskip=None,
    dcolorbar='right',
    dlegend='bottom',
    hcolorbar='right',
    hlegend='bottom',
    vcolorbar='bottom',
    vlegend='bottom',
    figurespan=False,
    standardize=False,
    ncols=None,
    nrows=None,
    rxlim=None,
    rylim=None,
    save=None,
    **kwargs
):
    """
    Plot any combination of variables across rows and columns.

    Parameters
    ----------
    dataset : xarray.Dataset
        A dataset generated by `open_bulk`.
    *args : list of 2-tuple
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
    figurespan : bool, optional
        Whether to make colobar and legend labels span the entire figure
        by default instead of matching to rows and columns.
    standardize : bool, optional
        Whether to standardize axis limits to span the same range for all
        plotted content with the same units.
    nrows, ncols : float, optional
        The number of rows or columns to use when either of the row
        or column plotting specifiers are singleton.
    rxlim, rylim : float or 2-tuple, optional
        Relative x and y axis limits to apply to groups of shared or standardized axes.
        Values should lie between 0 and 1. Note `xlim` and `ylim` can also be used.
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
    args = (dataset, rowspecs, colspecs)
    argskip = np.atleast_1d(() if argskip is None else argskip)
    gridskip = np.atleast_1d(() if gridskip is None else gridskip)
    kws_process, kws_collection, figlabel, pathlabel, gridlabels = parse_specs(*args, **kwargs)  # noqa: E501
    srows, scols = map(len, gridlabels)
    srows, scols = max(srows, 1), max(scols, 1)
    titles = (None,) * srows * scols
    # Title overrides
    # NOTE: This supports optional selections e.g. rowlabels=[None, 'override'].
    rowkey = 'rightlabels' if labelright else 'leftlabels'
    colkey = 'bottomlabels' if labelbottom else 'toplabels'
    figtitle = figtitle or figlabel
    figprefix, figsuffix = figprefix or '', figsuffix or ''
    if figprefix:
        figtitle = figtitle if figtitle[:2].isupper() else figtitle[0].lower() + figtitle[1:]  # noqa: E501
        figprefix = figprefix if figprefix[:1].isupper() else figprefix[0].upper() + figprefix[1:]  # noqa: E501
    figparts = (figprefix, figtitle, figsuffix)
    figtitle = ' '.join(filter(None, figparts))
    # Grid label overrides
    kw_gridlabels = {}
    for key, clabels, dlabels in zip((rowkey, colkey), (rowlabels, collabels), gridlabels):  # noqa: E501
        nlabels = srows if key == rowkey else scols
        clabels = clabels or [None] * nlabels
        dlabels = dlabels or [None] * nlabels
        if len(dlabels) != nlabels or len(clabels) != nlabels:
            raise RuntimeError(f'Expected {nlabels} labels but got {len(dlabels)} and {len(clabels)}.')  # noqa: E501
        kw_gridlabels[key] = [clab or dlab for clab, dlab in zip(clabels, dlabels)]
    if srows == 1 or scols == 1:
        naxes = gridskip.size + max(srows, scols)
        if nrows is not None:
            srows = min(naxes, nrows)
            scols = 1 + (naxes - 1) // srows
        else:
            scols = min(naxes, ncols or 4)
            srows = 1 + (naxes - 1) // scols
        titles = max(kw_gridlabels.values(), key=lambda labels: len(labels))
        titles = max(srows, scols) > 1 and titles or (None,) * srows * scols
        kw_gridlabels = {rowkey: None, colkey: None}
    # Print message
    print('All:', repr(figlabel))
    print('Rows:', ', '.join(map(repr, gridlabels[0])))
    print('Columns:', ', '.join(map(repr, gridlabels[1])))
    print('Path:', pathlabel)

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
    for num in range(srows * scols):
        # Retrieve data
        if num in gridskip:
            continue
        if count > len(iterator):
            continue
        print(f'{num + 1}/{srows * scols}', end=' ')
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
        kwargs = dict(fig=fig, gs=gs, geom=(srows, scols, num), title=ititle)
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
                (key, getattr(value, 'name', value))  # e.g. colormap name
                for key, value in kw_collection.command.items()
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
    groups_xunits = {}
    groups_yunits = {}  # groupings of units across axes
    groups_handles = {}  # groupings of handles across axes
    for num, (identifier, values) in enumerate(groups_commands.items()):
        # Combine plotting and guide arguments
        # NOTE: Here 'colorbar' and 'legend' keywords are automatically added to
        # plotting command keyword arguments by _auto_command.
        # NOTE: Here 'argskip' is isued to skip arguments with vastly different
        # ranges when generating levels that annotate multiple different subplots.
        print(f'{num + 1}/{len(groups_commands)}', end=' ')
        *_, command, guide = identifier
        axs, arguments, kws_collection = zip(*values)
        kws_command = [kw_collection.command.copy() for kw_collection in kws_collection]
        kws_guide = [getattr(kw_collection, guide).copy() for kw_collection in kws_collection]  # noqa: E501
        kws_axes = [kw_collection.axes.copy() for kw_collection in kws_collection]
        kws_other = [kw_collection.other.copy() for kw_collection in kws_collection]
        if command in ('contour', 'contourf', 'pcolormesh'):
            # Combine command arguments and keywords
            xy = [arguments[0][-1].coords[dim] for dim in arguments[0][-1].dims]
            zs = [arg for i, args in enumerate(arguments) for arg in args if i not in argskip]  # noqa: E501
            kw_levels = {
                key: val for kw_collection in kws_collection
                for key, val in kw_collection.command.items()
            }
            locator = kw_levels.pop('step', None)
            min_levels = 1 if command == 'contour' else 2
            kw_levels.update(locator=locator, min_levels=min_levels, norm_kw={})
            keys_keep = () if command == 'contour' else ('extend',)
            kw_keep = {key: kw_levels[key] for key in keys_keep if key in kw_levels}
            # Infer color levels
            # print('Args:', command, argskip, len(arguments), len(zs), zs[0].name)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')  # proplot and cmap runtime warnings
                levels, *_ = axs[0]._parse_level_vals(*xy, *zs, **kw_levels)
            locator = pplt.DiscreteLocator(levels, nbins=7)
            minorlocator = pplt.DiscreteLocator(levels, nbins=7, minor=True)
            for kw_command in kws_command:
                kw_command.pop('step', None)  # stand-in for level-selection locator
                kw_command.pop('robust', None)
                kw_command.update({**kw_keep, 'levels': levels})
            if command != 'contour':
                for kw_guide in kws_guide:
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
            print('.', end=' ' if ax is axs[-1] else '')
            prevlines = list(ax.lines)
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
            # NOTE: This must be called for each group so that violin widths can
            # be scaled across multiple subplots.
            if 'scatter' in command:
                keys = ('oneone', 'linefit', 'annotate')
                kw_other = {key: val for key, val in kw_other.items() if key in keys}
                _setup_scatter(ax, *args, **kw_other)  # 'args' is 2-tuple
            if 'bar' in command:  # ensure padding around bar edges
                keys = ('horizontal', 'annotate')
                errdata = kw_command.get('bardata', None)
                kw_other = {key: val for key, val in kw_other.items() if key in keys}
                _setup_bars(ax, args, errdata, handle, **kw_other)
            if 'violin' in command:
                keys = ('lines', 'horizontal', 'alpha', 'hatch', 'linewidth', 'edgecolor')  # noqa: E501
                lines = [line for line in ax.lines if line not in prevlines]
                width = kw_command.get('width', None)  # optionally override
                kw_other = {key: val for key, val in kw_other.items() if key in keys}
                _setup_violins(ax, args[-1], handle, lines, width=width, **kw_other)

            # Group units and guide handles
            # NOTE: If 'label' is in args[-1].coords it will be used for legend but
            # still want to segregate based on default short_name label to help reader
            # differentiate between sensitivity, forcing, and feedbacks.
            xunits = yunits = None
            if ax._name == 'cartesian':  # re-apply for boxplot and violinplot
                xunits, yunits = _setup_axes(ax, *args, command=command)
                ax.format(**{key: val for key, val in kw_axes.items() if 'proj' not in key})  # noqa: E501
            if ax._name == 'cartesian' and standardize:
                groups_xunits.setdefault(xunits, []).append(ax)
                groups_yunits.setdefault(yunits, []).append(ax)
            label = kw_guide.pop('label', None)
            if 'label' in args[-1].coords:  # TODO: optionally disable
                label = None
            if command == 'scatter':
                handle = label = None  # TODO: possibly remove this
            key = identifier[int(command not in ('contourf', 'pcolormesh')):]
            tuples = groups_handles.setdefault((*key, label), [])
            tuples.append((axs, args[-1], handle, kw_guide, kw_other))

    # Queue shared legends and colorbars
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
        axs = [ax for subaxs in axs for ax in subaxs]
        kw_update = {}
        if all(handles) and all('label' in arg.coords for arg in args):
            handles, labels, kw_update = _get_handles(args, handles)
        else:  # e.g. scatter
            handles, labels = handles[:1], [label]
        kw_guide = {key: val for kw_guide in kws_guide for key, val in kw_guide.items()}
        kw_guide.update(kw_update)
        if guide == 'colorbar':
            groups, def_, hori, vert = groups_colorbars, dcolorbar, hcolorbar, vcolorbar
        else:
            groups, def_, hori, vert = groups_legends, dlegend, hlegend, vlegend
        kw = dict(default=def_, horizontal=hori, vertical=vert, figurespan=figurespan)
        src, loc, span = _auto_guide(*axs, **kw)
        for handle, label in zip(handles, labels):
            if handle is not None and label is not None:  # e.g. scatter plots
                if isinstance(handle, (list, mcontainer.Container)):
                    handle = handle[0]  # legend_elements list BarContainer container
                tuples = groups.setdefault((axs[0], src, loc, span), [])
                tuples.append((handle, label, kw_guide))

    # Add shared legends and colorbar
    # TODO: Should support colorbars spanning multiple columns or
    # rows in the center of the gridspec in addition to figure edges.
    # WARNING: For some reason extendsize adjustment is still incorrect
    # even though axes are already drawn here. Not sure why.
    for guide, groups in zip(('colorbar', 'legend'), (groups_colorbars, groups_legends)):  # noqa: E501
        print('.', end='')
        for (ax, src, loc, span), tuples in groups.items():
            handles, labels, kws_guide = zip(*tuples)
            kw_guide = {key: val for kw_guide in kws_guide for key, val in kw_guide.items()}  # noqa: E501
            kw_guide.update({} if span is None else {'span': span})
            if guide == 'legend':
                kw_guide.setdefault('frame', False)
                kw_guide.setdefault('ncols', 1)
                src.legend(list(handles), list(labels), loc=loc, **kw_guide)
            else:
                width, height = ax._get_size_inches()  # sample axes
                multi = src is fig and (span is None or span[1] - span[0] > 0)
                size = height if loc[0] in 'lr' else width
                size *= span[1] - span[0] + 1 if multi else 1
                ratio = width / height if width > height else 1
                kw_guide['length'] = 0.66 if multi else 1.0
                kw_guide['extendsize'] = ratio * pplt.rc['colorbar.extend']
                for handle, label in zip(handles, labels):
                    label = _wrap_label(label, refwidth=1.2 * size)
                    src.colorbar(handle, label=label, loc=loc, **kw_guide)

    # Standardize relative axes limits and impose relative units
    # NOTE: Previously permitted e.g. rxlim=[(0, 1), (0, 0.5)] but these would
    # be applied *implicitly* based on drawing order so too confusing. Use
    # 'outer' from constraints '_build_specs()' function instead.
    print('.')
    if not standardize:  # auto-search for shared axes
        ref = fig.subplotgrid[0]
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
    for axis, rlim, groups in zip('xy', (rxlim, rylim), (groups_xunits, groups_yunits)):
        rlim = rlim or (0, 1)  # disallow *implicit* application of multiple options
        for i, (unit, axs) in enumerate(groups.items()):
            lims = [getattr(ax, f'get_{axis}lim')() for ax in axs]
            span = max((lim[1] - lim[0] for lim in lims), key=abs)  # preserve sign
            for ax, lim in zip(axs, lims):
                average = 0.5 * (lim[0] + lim[1])
                min_ = average + span * (rlim[0] - 0.5)
                max_ = average + span * (rlim[1] - 0.5)
                getattr(ax, f'set_{axis}lim')((min_, max_))

    # Optionally save the figure
    # NOTE: Here default labels are overwritten with non-none 'rowlabels' or
    # 'collabels', and the file name can be overwritten with 'save'.
    fig.format(figtitle=figtitle, **kw_gridlabels)
    for num in gridskip:  # kludge to center super title above empty slots
        ax = fig.add_subplot(gs[num])
        for obj in (ax.xaxis, ax.yaxis, ax.patch, *ax.spines.values()):
            obj.set_visible(False)
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
