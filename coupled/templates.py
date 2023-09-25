#!/usr/bin/env python3
"""
Template functions for generating complex plots.
"""
import itertools
from pathlib import Path

import climopy as climo  # noqa: F401
import numpy as np
import xarray as xr
from climopy import ureg, vreg  # noqa: F401
from coupled import plotting, _warn_coupled
from icecream import ic  # noqa: F401

from .plotting import CYCLE_DEFAULT
from .results import ALIAS_FEEDBACKS

__all__ = [
    'divide_specs',
    'generate_specs',
    'feedback_specs',
    'transport_specs',
    'scalar_grid',
    'pattern_rows',
    'constraint_rows',
    'relationship_rows',
]

# Variable and selection keywords
# NOTE: See also specs.py key lists
KEYS_PLEV = (
    'ta', 'zg', 'ua', 'va', 'hus', 'hur', 'cl', 'clw', 'cll', 'cli', 'clp', 'pt', 'slope',  # noqa: E501
)
KEYS_CONTOUR = (  # plurals not required
    'levels', 'linewidth', 'linestyle', 'edgecolor',
)
KEYS_SHADING = (
    'cmap', 'cmap_kw', 'norm', 'norm_kw', 'vmin', 'vmax', 'step', 'locator', 'symmetric', 'diverging', 'sequantial',  # noqa: E501
)
KEYS_EITHER = (  # used in constraint_rows()
    *KEYS_CONTOUR, *KEYS_SHADING
)

# Keywords for generating dictionary specs from kwargs
# NOTE: These can be passed as vectors to generate_specs for combining along rows/cols.
# TODO: Remove 'base' kludge and permit vectors specifications on each side of pair.
KEYS_BREAK = (
    'breakdown', 'component', 'feedbacks', 'adjusts', 'forcing', 'sensitivity', 'transport'  # noqa: E501
)
KEYS_PLOT = (
    'method', 'pctile', 'std', 'normalize', 'hemisphere', 'hemi',
    'name', 'period', 'spatial', 'volume', 'area', 'plev', 'lat', 'lon',
    'project', 'institute', 'experiment', 'ensemble', 'model', 'base',  # TODO: remove
    'source', 'style', 'region', 'start', 'stop',
    'alpha', 'edgecolor', 'linewidth', 'linestyle', 'color', 'facecolor',
    'xmin', 'xmax', 'ymin', 'ymax', 'xlabel', 'ylabel',
    'vmin', 'vmax', 'cmap', 'cmap_kw', 'norm', 'norm_kw', 'robust', 'extend', 'extendsize',  # noqa: E501
    'step', 'locator', 'levels', 'diverging', 'symmetric', 'sequential',
    'loc', 'label', 'align', 'colorbar', 'legend', 'length', 'shrink',
)


# Variable name specs for feedbakc and transport breakdowns
# NOTE: Lists below are parsed by feedback_specs() into either single columns,
# two columns, or some fixed number of columns compatible with the decomposition.
# NOTE: The 'wav' suffixes including longwave and shortwave cloud components
# instead of net cloud feedback. Strings denote successively adding atmospheric,
# albedo, residual, and remaining temperature/humidity feedbacks with options.
SPECS_TRANSPORT = {
    'atmos': ('mse', 'lse', 'dse'),
    'total': ('total', 'ocean', 'mse'),
    'all': ('total', 'ocean', 'mse', 'lse', 'dse'),
}
SPECS_FEEDBACK = {
    # Three-variable
    'ecs': ('net', 'erf', 'ecs'),  # shorthand for sensitivity=True forcing=True
    'erf': ('net', 'ecs', 'erf'),  # shorthand for sensitivity=True forcing=True
    'net': ('net', 'sw', 'lw'),
    'atm': ('net', 'cld', 'atm'),
    'ncl': ('net', 'cld', 'ncl'),
    'cld': ('cld', 'swcld', 'lwcld'),  # special case
    'cre': ('cre', 'swcre', 'lwcre'),  # special case
    'wav_cld': ('net', 'swcld', 'lwcld'),
    'wav_cre': ('net', 'swcre', 'lwcre'),
    # Four-variable
    'atm_resid': ('net', 'cld', 'resid', 'atm'),
    'atm_alb': ('net', 'cld', 'alb', 'atm'),
    'cld_net': ('net', 'cld', 'swcld', 'lwcld'),
    'cre_net': ('net', 'cld', 'swcld', 'lwcld'),
    'wav_atm': ('net', 'swcld', 'lwcld', 'atm'),
    'wav_ncl': ('net', 'swcld', 'lwcld', 'ncl'),
    'wav_cs': ('net', 'swcre', 'lwcre', 'cs'),
    'tstd_net': ('tstd', 'net', 'cld', 'ncl'),
    'tpat_net': ('tpat', 'net', 'cld', 'ncl'),
    'ecs_net': ('ecs', 'net', 'cld', 'ncl'),
    # Five-variable
    'resid': ('net', 'cld', 'alb', 'atm', 'resid'),
    'cld_atm': ('net', 'cld', 'swcld', 'lwcld', 'atm'),
    'cld_ncl': ('net', 'cld', 'swcld', 'lwcld', 'ncl'),
    'cre_cs': ('net', 'cre', 'swcre', 'lwcre', 'cs'),
    'wav_resid': ('net', 'swcld', 'lwcld', 'resid', 'atm'),
    'wav_alb': ('net', 'swcld', 'lwcld', 'alb', 'atm'),
    'atm_pl': ('atm', 'pl*', 'lr*', 'rh', 'alb'),
    'ncl_pl': ('ncl', 'pl*', 'lr*', 'rh', 'resid'),
    'ncl_alb': ('ncl', 'lr*', 'rh', 'alb', 'resid'),
    # Sensitivity components
    'tstd_ncl': ('tstd', 'net', 'swcld', 'lwcld', 'ncl'),
    'tpat_ncl': ('tpat', 'net', 'swcld', 'lwcld', 'ncl'),
    'tstd_cs': ('tstd', 'net', 'swcre', 'lwcre', 'cs'),
    'tpat_cs': ('tpat', 'net', 'swcre', 'lwcre', 'cs'),
    'tstd_cld': ('tstd', 'net', 'cld', 'swcld', 'lwcld'),
    'tpat_cld': ('tpat', 'net', 'cld', 'swcld', 'lwcld'),
    'ecs_cld': ('ecs', 'net', 'cld', 'swcld', 'lwcld'),
    'tstd_cre': ('tstd', 'net', 'cre', 'swcre', 'lwcre'),
    'tpat_cre': ('tpat', 'net', 'cre', 'swcre', 'lwcre'),
    'ecs_cre': ('ecs', 'net', 'cre', 'swcre', 'lwcre'),
    # Additional variables
    'hus': ('net', 'resid', 'cld', 'swcld', 'lwcld', 'alb', 'atm', 'wv', 'lr', 'pl'),
    'hur': ('net', 'resid', 'cld', 'swcld', 'lwcld', 'alb', 'atm', 'rh', 'lr*', 'pl*'),
    'all': ('net', 'cld', 'swcld', 'lwcld', 'atm', 'alb', 'resid', 'wv', 'rh', 'lr', 'lr*', 'pl', 'pl*'),  # noqa: E501
    'cld_alb': ('net', 'cld', 'swcld', 'lwcld', 'atm', 'alb'),
    'cld_resid': ('net', 'cld', 'swcld', 'lwcld', 'atm', 'resid'),
    'ncl_cld': ('swcld', 'lwcld', 'lr*', 'rh', 'alb', 'resid'),
    'ncl_resid': ('ncl', 'pl*', 'lr*', 'rh', 'alb', 'resid'),
}


def _adjust_warming(dataset, source='~/scratch/cmip-processed'):
    """
    Update with ad hoc scaled warming projection terms. In future should
    be stored in feedback files with regression warming patterns.

    Parameters
    ----------
    dataset : xarray.Dataset
        The input dataset.

    Returns
    -------
    dataset : xarray.Dataset
        The updated dataset.
    """
    # NOTE: Tried scaling with feedback to get 'absolute' climate sensitivity implied
    # by feedback but unperturbed can be very small! So now use vector projections.
    # tpar = (1 + dataset['tpat']) / dataset['rfnt_lam'].climo.average('area')
    from cmip_data.utils import average_periods
    if 'tstd' in dataset and 'tdev' not in dataset:  # added by feedback_datasets()
        _warn_coupled("Variable 'tstd' present but 'tdev' not present.")
    if 'tstd' in dataset:
        return dataset
    if 'tpat' not in dataset:
        raise ValueError('Could not find temperature pattern data.')
    data = xr.full_like(dataset.tpat, np.nan)
    data.data[:] = np.nan  # reset all values
    attrs = dict(units='K', standard_units='K', short_name='warming', long_name='regional warming')  # noqa: E501
    data.attrs.update(attrs)
    sel = dict(source='eraint', style='slope', region='globe')
    if 'period' in data.sizes:  # select annual
        sel['period'] = 'ann'
    source = Path(source).expanduser()
    periods = sorted(set(
        (start, stop) for _, style, _, start, stop in dataset.version.values
        if style == 'slope'
    ))
    print('Getting scaled temperature pattern.')
    print('Model:', end=' ')
    for i, facets in enumerate(dataset.facets.values):
        project, model, experiment, ensemble = facets
        folder = source / f'{project.lower()}-{experiment}-amon'
        if ensemble != 'flagship':
            continue
        if not folder.is_dir():
            raise RuntimeError(f'Missing folder {folder}.')
        files = list(folder.glob(f'ts_Amon_{model}_*_0000-0150-series.nc'))
        if len(files) > 1:
            print(f'Found {len(files)} files for project {project} model {model}.')
            continue
        if not files:  # no error
            print(f'No files for project {project} model {model}.')
            continue
        print(f'{model}_{experiment}', end=' ')
        ranges = periods if experiment == 'abrupt4xco2' else [(0, 150)]
        base = (project, model, 'picontrol', ensemble)  # facets selection
        base = dataset.ts.sel(facets=base)
        if 'period' in base.sizes:
            base = base.sel(period='ann')
        base = base.climo.add_cell_measures().climo.average('area')
        temp = xr.open_dataset(files[0], use_cftime=True)['ts']
        temp = average_periods(temp, seasonal=False, monthly=False)
        temp = temp.climo.add_cell_measures().climo.average('area')
        temp = temp.sel(period='ann', drop=True) - base  # pre-industrial anomaly
        for start, stop in ranges:
            if (start, stop) != (0, 150) and experiment == 'picontrol':
                continue
            isel = dict(facets=facets, start=start, stop=stop, **sel)
            scale = temp.isel(year=slice(start, stop))  # select years
            scale = np.sqrt(((scale - scale.mean('year')) ** 2).sum('year'))
            pattern = dataset.tpat.loc[isel]  # add back global-mean warming
            print(f'{scale.item():.1f}', end=' ')
            data.loc[isel] = scale * pattern
    anom = data - data.climo.add_cell_measures().climo.average('area')
    attrs = dict(units='K', standard_units='K', short_name='warming', long_name='relative warming')  # noqa: E501
    anom.attrs.update(attrs)
    dataset['tstd'] = data
    dataset['tdev'] = anom  # NOTE: implemented in feedback_datasets()
    return dataset


def _constraint_props(method=None):
    """
    Generate hatches and levels suitable for indicating significant
    correlation coefficients or variance explained proportions.

    Parameters
    ----------
    method : bool
        The method to use for shading.

    Returns
    -------
    label : str
        The constraint label.
    methods : tuple
        The filled contour and hatching methods.
    hatches, levels : list
        The hatching instructions and associated level boundaries.
    """
    method = method or 'slope'
    check, *options = method.split('_')
    if check in ('proj', 'norm'):  # TODO: merge with constraint 'pairs'
        label = 'Projection'
        methods = (method, 'corr')
        # label = 'Correlation'
        # methods = ('corr', 'corr')
        hatches = ['.....', '...', None, '...', '.....']
        levels = [-10, -0.8, -0.5, 0.5, 0.8, 10]
        # levels = [-10, -0.66, -0.33, 0.33, 0.66, 10]
    elif check in ('corr', 'rsq', 'slope', 'var', 'std'):
        label = 'Regression'
        methods = (method, 'rsq')
        # label = 'Correlation'
        # methods = ('corr', 'rsq')
        hatches = [None, '...', '.....']
        levels = [0, 33.333, 66.666, 1000]
    else:
        raise RuntimeError
    return label, methods, hatches, levels


def _default_size(refsize=None, project=None, institute=None, **kwargs):
    """
    Generate appropriate scale for bar-type feedback subplots so that
    annotations can be shown without overlapping.

    Parameters
    ----------
    refsize : float
        The reference size.
    project : optional
        The project string.
    institute : optional
        The institute string.

    Returns
    -------
    refsize, altsize : float
        The two sizes for axes.
    """
    # TODO: Add similar defaults for violin and multiple regression plots. Currently
    # logic is hardcoded into notebooks. Size depends on number of variables.
    refsize = refsize or 1.7  # narrow for bar plots
    if project in (None, 'cmip'):  # 90 models 45 institutes
        scale = 1.5 if institute else 3
    elif project in ('cmip6',):  # 60 models 30 institutes
        scale = 1 if institute else 2
    elif project in ('cmip65',):  # 40 models ~15 institutes
        scale = 0.6 if institute else 1.4
    elif project in ('cmip5', 'cmip56'):  # 30 models ~15 institutes
        scale = 0.6 if institute else 1.0
    elif project in ('cmip66',):  # 20 models ~15 institutes
        scale = 0.6 if institute else 0.8
    elif project in ('cmip6-cmip5', 'cmip65-cmip56'):  # ~15 matching institutes
        scale = 0.6
    else:
        raise RuntimeError(f'Unknown project {project}.')
    # altsize = 1.3 * scale * refsize
    altsize = 1.4 * scale * refsize
    if kwargs.get('horizontal', False):
        refsize, altsize = altsize, refsize
    return refsize, altsize


def _generate_dicts(*kws, expand=True, check=True):
    """
    Helper function to generate dictionaries from lists.

    Parameters
    ----------
    *kws : dict
        The input dictionaries.
    expand : bool, optional
        Whether to expand lists into separate dictionaries.
    check : bool, optional
        Whether to check lists have equivalent lengths.
    """
    # NOTE: This allows e.g. list applied to one side of correlation pair for
    # multiple items in a single subplot. See notebooks for details.
    kwargs = {
        f'{key}{i + 1}': value
        for i, kw in enumerate(kws) for key, value in kw.items()
    }
    lengths = {
        key: len(value)  # allow e.g. pair of different names
        for key, value in kwargs.items()
        if 'color' not in key and 'cycle' not in key  # e.g. 'color1'
    }
    length = set(lengths.values()) - {1}
    if check and len(length) > 1:  # or e.g. allow e.g. ('name', 'cycle') truncation
        values = {key: kwargs[key] for key in lengths}
        raise ValueError(f'Unexpected mixed lengths {lengths} from {values}.')
    length = length.pop() if length else 1
    results = []
    for kw in kws:
        result = {key: value * length if len(value) == 1 else value for key, value in kw.items()}  # noqa: E501
        if expand:
            result = [dict(zip(result, vals)) for vals in zip(*result.values())]
            result = result or [{}]
        results.append(result)
    return results[0] if len(results) == 1 else results


def divide_specs(key, specs, **kwargs):
    """
    Divide feedback and variable specification lists.

    Parameters
    ----------
    key : {'rows', 'cols'}
        Whether to split rows or columns.
    specs : list of dict
        The feedback or variable specifications.
    rowsplit, colsplit : bool or list of int, optional
        The split instruction. Lists of integers indicate individual rows to select.
        Sub-lists indicates multiple selections. Can optionally use trailing dictionary.
    **kwargs : optional
        Additional keywords passed to plotting function. Any subplot geometry vector
        arguments e.g. ``hspace``, ``hratios``, ``rowlabels`` will be sub-selected.

    Returns
    -------
    specs : list of list of dict
        The specification iteration groups.
    """
    # Initial stuff
    # NOTE: This is used to take arbitrary cross-sections of more
    # complex figures with many panels for use in presentations.
    split = kwargs.pop(f'{key}split', None)
    nosplit = split is False or split is None
    pre = 'w' if key == 'row' else 'h'
    geom = 'nrows' if key == 'row' else 'ncols'
    count = kwargs.get(geom, None)
    titles = kwargs.get('titles', None)
    labels = kwargs.get(f'{key}labels', None)
    space = kwargs.get(f'{pre}space', None)
    ratios = kwargs.get(f'{pre}ratios', None)
    if nosplit:  # split nothing
        split = ([*range(0, len(specs))],)
    if split is True:  # split everything
        split = [*range(0, len(specs))]
    if not np.iterable(split):
        split = (split,)

    # Iterate over split indices
    # NOTE: This comes *after* generating specs with _parse_specs. Simply yield
    # subselections of resulting dictionary lists and update figure settings.
    for idxs in split:
        kwargs = kwargs.copy()
        if not np.iterable(idxs):
            idxs = (idxs,)
        if isinstance(idxs[-1], dict):
            kwargs = {**kwargs, **idxs[-1]}
            idxs = idxs[:-1]
        subs = [specs[idx] for idx in idxs]
        noskip = np.all(np.diff(idxs) == 1)
        kwargs[geom] = count if nosplit else None
        kwargs.setdefault('horizontal', key == 'row')
        if not nosplit and np.iterable(ratios):
            kwargs[f'{pre}ratios'] = [ratios[idx] for idx in idxs]
        if not nosplit and np.iterable(space):
            kwargs[f'{pre}space'] = space[idxs[0]:idxs[-1]] or None if noskip else None
        if not nosplit and np.iterable(labels) and not isinstance(labels, str):
            if len(labels) == len(specs):  # TODO: re-address (need for tiled plots?)
                kwargs[f'{key}labels'] = [labels[idx] for idx in idxs]
        if not nosplit and np.iterable(titles) and not isinstance(titles, str):
            if len(titles) == len(specs) and key == 'row':  # TODO: re-address
                kwargs['titles'] = [titles[idx] for idx in idxs]
        yield subs, kwargs


def generate_specs(pairs=None, product=None, outer='breakdown', maxcols=None, **kwargs):
    """
    Generate feedback and variable specifications based on input keywords.

    Parameters
    ----------
    pairs : str or list of str, optional
        The coordinate name(s) to be used for feedback constraints. These can
        also be generated using combination of ``'name'`` and ``'breakdown'``.
    product : tuple of str or list of str, optional
        The list-like kwargs to combine with `itertools.product`. Can include multiple
        keys per multiplicand e.g. ``product=(('experiment', 'color'), 'project')``.
    outer : str or list of str, optional
        The kwargs for the outer plotting specs. Can include multiple keys per
        multiplicand e.g. ``outer=(('breakdown', 'color'), ('project', 'experiment'))``.
    **kwargs : item or list of item, optional
        The reduce specifications. These can be scalar strings or tuples for generating
        comparisons across columns or within subplots. Can also append ``1`` or ``2``
        to apply to one side of a subplot spec correlation pair.

    Returns
    -------
    *kws_outer : lists of dict
        List of plot specs for rows and/or columns.
    *kws_pair : list of dict
        List of plot specs suitable for each pair item.
    kwargs : dict
        The remaining keywords with auto-updated `maxcols` and `gridskip`.
    """
    # Retrieve breakdown variable names
    # NOTE: Critical to keep 'maxcols' separate so it can be passed on if we
    # are not using a maxcols-modifying 'breakdown' function.
    # NOTE: Here permit generating lists of names and default keyword arguments from
    # shorthand 'breakdown' keys. Default is to generate feedback components with
    # e.g. generate_specs(breakdown='all') but generate transport components if
    # 'transport' is passed with e.g. generate_specs(breakdown='all', transport='t')
    outer = (outer,) if isinstance(outer, str) else outer or ()
    outer = [[keys] if isinstance(keys, str) else list(keys) for keys in outer]
    ncols = kwargs.get('ncols', None)
    maxcols = 1 if ncols else maxcols  # disable special arrangements
    kw_break = {key: kwargs.pop(key) for key in tuple(kwargs) if key in KEYS_BREAK}
    if len(outer) > 2:  # only ever used for rows and columns
        raise ValueError('Too many outer variables specified.')
    if 'transport' in kw_break:  # transport breakdown
        breakdown, kw_default = transport_specs(maxcols=maxcols, **kw_break)
        kw_default = {**kw_default, 'proj_kw': {'lon_0': 0}}
    elif kw_break:  # non-empty breakdown
        breakdown, kw_default = feedback_specs(maxcols=maxcols, **kw_break)
        kw_default = {**kw_default, 'proj_kw': {'lon_0': 210}}
    elif any(f'name{s}' in kwargs for s in ('', 1, 2)):  # explicit name
        breakdown = None
        kw_default = {'ncols': ncols or maxcols}
    else:
        raise ValueError('Neither variable name nor breakdown was specified.')
    idxs = [i for i, keys in enumerate(outer) if 'breakdown' in keys or 'component' in keys]  # noqa: E501
    kws_pair = [{}, {}]
    kws_outer = [{} for i in range(len(outer))]
    for key, value in kw_default.items():
        kwargs.setdefault(key, value)
    if not breakdown:
        pass
    elif not idxs:
        kws_pair[0]['name'] = kws_pair[1]['name'] = breakdown
    else:
        kws_outer[idxs[0]]['name'] = breakdown

    # Assign separate plot spec dictionaries
    # WARNING: Numpy arrays should be used for e.g. scalar level lists or color values
    # and tuples or lists should be used for vectors of settings across subplots.
    # NOTE: Here 'outer' is used to specify different reduce instructions across rows
    # or columns. For example: correlate with feedback parts in rows, generate_specs(
    # name='psl', breakdown='net'); or correlate with parts (plus others) in subplots,
    # generate_specs(name='psl', breakdown='net', outer=None, experiment=('picontrol', 'abrupt4xco2')).  # noqa: E501
    pairs = pairs or ()
    pairs = (pairs,) if isinstance(pairs, str) else tuple(pairs)
    kw_plot = {
        key: list(item) if isinstance(item := kwargs.pop(key), (list, tuple)) else (item,)  # noqa: E501
        for key in tuple(kwargs) if key in KEYS_PLOT
    }
    for keys, kw in zip(outer, kws_outer):
        for key in keys:
            if key in pairs:
                raise ValueError(f'Keyword {key} cannot be in both outer and pairs.')
            if key == 'color':  # generally joined with e.g. 'breakdown'
                kw['color'] = kwargs.pop(key, CYCLE_DEFAULT)  # ignore defaults
            elif key in kw_plot:
                kw[key] = kw_plot.pop(key)
    for key in pairs:  # specifications for pairs
        if values := kw_plot.get(key):
            if not np.iterable(values) or len(values) > 2:
                raise ValueError(f'Coordinate pair {key}={values!r} is not a 2-tuple.')
            del kw_plot[key]
            values = values * 2 if len(values) == 1 else values
            for kw, value in zip(kws_pair, values):
                kw[key] = list(value) if isinstance(value, (tuple, list)) else [value]
    sentinel = object()  # see below
    spatial = ('lon', 'lat', 'plev', 'area')
    kws_pair[0].update(kw_plot)  # remaining scalars or vectors
    kws_pair[1].update(kw_plot)
    for key in KEYS_PLOT:  # add scalar versions
        value = kwargs.pop(f'{key}1', sentinel)  # ignore none iteration placeholders
        is_vector = isinstance(value, (tuple, list))
        if value is not sentinel and (key in spatial or value is not None):
            if len(outer) > 0 and key in outer[0] and is_vector:
                kws_outer[0].update({key: value})  # e.g. outer=('name', 'name')
            else:
                kws_pair[0].update({key: value if is_vector else (value,)})
        value = kwargs.pop(f'{key}2', sentinel)
        is_vector = isinstance(value, (tuple, list))
        if value is not sentinel and (key in spatial or value is not None):
            if len(outer) > 1 and key in outer[1] and is_vector:
                kws_outer[1].update({key: value})
            else:
                kws_pair[1].update({key: value if is_vector else (value,)})

    # Convert reduce vector iterables to dictionaries
    # NOTE: This also builds optional Cartesian products between lists of keywords
    # specified by 'product' keyword. Others are enforced to have same length.
    # NOTE: Above, scalar 'kwargs' are kept in place for simplicity, but still need
    # to repeat scalar (name, breakdown) or manual scalar correlation pair items.
    product = product or ()
    product = [[keys] if isinstance(keys, str) else list(keys) for keys in product]
    for dict_ in (*kws_outer, *kws_pair):
        kws = []
        for keys in product:  # then skip if absent
            if any(key in dict_ for key in keys):
                kw = {key: dict_.pop(key) for key in keys if key in dict_}
                kws.append(_generate_dicts(kw, expand=False))
        keys = [key for kw in kws for key in kw]
        values = itertools.product(*(zip(*kw.values()) for kw in kws))
        values = [[v for val in vals for v in val] for vals in values]
        dict_.update({key: vals for key, vals in zip(keys, zip(*values))})
    kws_outer = tuple(map(_generate_dicts, kws_outer))
    kws_pair = _generate_dicts(*kws_pair)
    return *kws_outer, *kws_pair, kwargs


def feedback_specs(
    breakdown=None,
    component=None,
    sensitivity=False,
    forcing=False,
    feedbacks=True,
    adjusts=False,
    maxcols=None,
    ncols=None,
):
    """
    Return the feedback, forcing, and sensitivity parameter names sensibly
    organized depending on the number of columns in the plot.

    Parameters
    ----------
    breakdown : str, default: 'all'
        The breakdown preset to use.
    component : str, optional
        The individual component to use.
    sensitivity, forcing : bool, optional
        Whether to include net sensitivity and forcing.
    feedbacks, adjusts : bool, optional
        Whether to include feedback and forcing adjustment components.
    maxcols : int, default: 4
        The maximum number of columns (influences order of the specs).
    ncols : int
        The user input fixed column size. Then `maxcols` will be ignored.

    Returns
    -------
    specs : list of str
        The variables.
    kwargs : dict
        The keyword args to pass to `create_plots`.
    """
    # Initial stuff
    # NOTE: User input 'ncols' will *cancel* auto gridspec arrangement and instead
    # impose same ordering used e.g. when breakdowns are placed along rows.
    # NOTE: Idea is to automatically remove feedbacks, filter out all-none rows
    # and columns at the end, and then infer the 'gridskip' from them.
    maxcols = inputcols = 1 if ncols is not None else maxcols or 4
    options = ', '.join(map(repr, SPECS_FEEDBACK))
    init_names = lambda ncols: ((names := np.array([[None] * ncols] * 25)), names.flat)
    if not component and not feedbacks and not forcing and not sensitivity:
        raise ValueError('Invalid keyword argument combination.')
    if breakdown is None:
        components = component or 'net'
    elif SPECS_FEEDBACK:
        components = SPECS_FEEDBACK[breakdown]
    else:
        raise ValueError(f'Invalid breakdown {breakdown!r}. Options are: {options}')
    components = (components,) if isinstance(components, str) else tuple(components)

    # Generate plot layouts
    # NOTE: This is relevant for general_subplots() style figures when we wrap have
    # wrapped rows or columns of components but not as useful for e.g. summary_rows()
    lams = [ALIAS_FEEDBACKS.get(name, name) for name in components]
    erfs = [name.replace('_lam', '_erf') for name in lams]
    if len(lams) == 1 or len(lams) == 2:  # user input breakdowns
        gridskip = None
        names, iflat = init_names(maxcols)
        for i, name in enumerate(components):  # assign possibly correlation tuples!
            iflat[i] = name

    # Three variable
    # NOTE: Includes net-lw-sw and cld-swcld-lwcld
    elif len(lams) == 3:
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

    # Four variable
    # NOTE: Includes net-cld-swcld-lwcld and net-cld-atm-resid
    elif len(lams) == 4:
        if maxcols == 2:
            names, iflat = init_names(maxcols)
            if sensitivity:
                iflat[0] = 'ecs'
            if feedbacks and adjusts:
                names[1:5, 0] = lams
                names[1:5, 1] = erfs
            elif feedbacks:
                iflat[1] = 'erf' if forcing else None
                names[1:3, :] = lams[::3], lams[1:3]
            elif adjusts:
                names[1:3, :] = erfs[::3], erfs[1:3]
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
                iflat[idx:idx + len(lams[1:])] = lams[1:]
            if adjusts:
                idx = 3 + 3
                iflat[idx:idx + len(erfs[1:])] = erfs[1:]

    # Five variable
    # NOTE: Includes net-cld-alb-atm-resid and net-swcld-lwcld-atm-resid
    elif len(lams) in (5, 6):
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
                iflat[idx:idx + len(lams[1:])] = lams[1:]
            if adjusts:
                idx = 4 + 4
                iflat[idx:idx + len(erfs[1:])] = erfs[1:]

    # Ten variables
    # NOTE: These are same as 'all' but with only one style non-cloud breakdown
    elif len(lams) == 10:
        if maxcols == 2:
            names, iflat = init_names(maxcols)
            if sensitivity:
                iflat[0] = 'ecs'
            if feedbacks and adjusts:
                names[1:len(lams) + 1, 0] = lams
                names[1:len(erfs) + 1, 1] = erfs
            elif feedbacks:
                iflat[0] = 'ecs' if sensitivity else None
                iflat[1] = 'erf' if forcing else None
                iflat[2:4] = [lams[0], lams[1]]
                iflat[4:6] = [lams[2], lams[5]]
                iflat[6:8] = [lams[3], lams[4]]
                iflat[8:8 + 4] = lams[6:]
            elif adjusts:
                iflat[1] = 'ecs' if sensitivity else None
                iflat[2:4] = [erfs[0], erfs[1]]
                iflat[4:6] = [erfs[2], erfs[5]]
                iflat[6:8] = [erfs[3], erfs[4]]
                iflat[8:8 + 4] = erfs[6:]
        else:
            offset = 0 if maxcols == 1 else 1
            maxcols = 1 if maxcols == 1 else 4  # disallow 3 columns
            names, iflat = init_names(maxcols)
            iflat[2] = 'erf' if forcing and not adjusts else None
            iflat[3] = 'ecs' if sensitivity else None
            for b, arr, idx in zip((feedbacks, adjusts), (lams, erfs), (0, 12)):
                if not b:
                    continue
                iflat[idx:idx + 2] = arr[:2]
                idx = 4 + 4
                iflat[idx:idx + 8] = arr[2:]

    # Every variable
    # NOTE: Includes both relative and absolute-style breakdowns of non-cloud feedbacks
    elif len(lams) == 13:
        if maxcols == 2:
            names, iflat = init_names(maxcols)
            if feedbacks and adjusts:
                iflat[0] = 'ecs' if sensitivity else None
                names[1:len(lams) + 1, 0] = lams
                names[1:len(erfs) + 1, 1] = erfs
            elif feedbacks:
                iflat[0] = 'ecs' if forcing and sensitivity else None
                iflat[2] = lams[0]
                iflat[3] = 'erf' if forcing else 'ecs' if sensitivity else None
                iflat[4:6] = [lams[1], lams[4]]
                iflat[6:8] = [lams[2], lams[3]]
                iflat[8:8 + 8] = lams[5:]
            elif adjusts:
                iflat[1] = 'ecs' if sensitivity else None
                iflat[2:4] = [erfs[1], erfs[4]]
                iflat[4:6] = [erfs[2], erfs[3]]
                names[6:6 + 8] = erfs[5:]
        else:
            offset = 0 if maxcols == 1 else 1
            maxcols = 1 if maxcols == 1 else 4  # disallow 3 columns
            names, iflat = init_names(maxcols)
            if sensitivity:
                iflat[-offset % 3] = 'ecs'  # either before or after net and erf
            if forcing and not adjusts:
                iflat[2 - offset] = erfs[0]
            for b, arr, hum, idx in zip((feedbacks, adjusts), (lams, erfs), (4, 16)):
                if not b:
                    continue
                iflat[1 - offset] = arr[0]
                iflat[idx:idx + 4] = arr[1:5]
                if maxcols == 1:
                    idx = 4 + 4
                    iflat[idx:idx + 8] = arr[5:7] + arr[7:]
                else:
                    idx = 4 + 4
                    iflat[idx:idx + 4] = arr[5:6] + arr[7::2]
                    idx = 4 + 2 * 4
                    iflat[idx:idx + 4] = arr[6:7] + arr[8::2]
    else:
        raise NotImplementedError(
            f'Invalid variable count {len(lams)} for breakdown {breakdown!r}.'
        )

    # Remove all-none segments and determine gridskip
    # NOTE: This flattens if there are fewer names than originally requested maxcols
    # or if maxcols == 1 was passed e.g. for summary_rows() feedbacks along rows.
    # TODO: Consider re-implementing automatic application of color cycle for
    # specific variable names as with transport.
    # colors = {'ecs': 'gray', 'erf': 'gray', 'lam': 'gray'}
    # names = [FEEDBACK_TRANSLATIONS.get(name, (name,))[0] for name in names]
    # cycle = [colors[name[-3:]] + '7' for name in names]
    # kwargs = {'ncols': maxcols, 'gridskip': gridskip, 'cycle': cycle}
    idx, = np.where(np.any(names != None, axis=0))  # noqa: E711
    names = np.take(names, idx, axis=1)
    idx, = np.where(np.any(names != None, axis=1))  # noqa: E711
    names = np.take(names, idx, axis=0)
    idxs = np.where(names == None)  # noqa: E711
    gridskip = np.ravel_multi_index(idxs, names.shape)
    names = [name for name in names.ravel().tolist() if name is not None]
    if maxcols == 1:  # e.g. single row
        kwargs = {}
    elif len(names) <= inputcols:  # simpler default
        kwargs = {'gridskip': None, 'ncols': len(names)}
    else:  # custom arrangement
        kwargs = {'gridskip': gridskip, 'ncols': maxcols}
    return names, kwargs


def transport_specs(breakdown=None, component=None, transport=None, maxcols=None):
    """
    Return the names and colors associated with transport components.

    Parameters
    ----------
    breakdown : str, optional
        The transport components to show.
    component : str, optional
        The individual component to use.
    transport : str, optional
        The transport type to show.
    maxcols : int, optional
        The maximum number of columns (unused currently).

    Returns
    -------
    names : list of str
        The transport components.
    kwargs : dict
        The keyword args to pass to `create_plots`.
    """
    # TODO: Expand to support mean-stationary-eddy decompositions and whatnot. Perhaps
    # indicate those with dashes and dots and/or with different shades. And/or allow
    # individual components per subplot with logical arrangement as with feedbacks.
    options = ', '.join(map(repr, SPECS_TRANSPORT))
    breakdown = breakdown or 'all'
    kwargs = {'ncols': maxcols}
    colors = {  # currently for transport only
        'total': 'gray',
        'ocean': 'cyan',
        'mse': 'yellow',
        'lse': 'blue',
        'dse': 'red'
    }
    if transport is None:
        raise ValueError('Transport component must be explicitly passed.')
    if transport not in SPECS_TRANSPORT:
        raise ValueError(f'Invalid breakdown {transport!r}. Options are: {options}')
    if component is not None:
        names = (component,)
    else:
        names = SPECS_TRANSPORT[transport]
        shading = 7  # open color shading level
        kwargs['cycle'] = [colors[name] + str(shading) for name in names]
        names = [name + transport for name in names]
    return names, kwargs


def scalar_grid(data, forward=True, **kwargs):
    """
    Plot scalar results in each grid slot (e.g. bars, boxes, lines).

    Parameters
    ----------
    data : xarray.Dataset
        The source dataset.
    forward : bool, optional
        Whether to apply name pair forward or backward.
    rowsplit, colsplit : optional
        Passed to `divide_specs`.
    **kwargs
        Passed to `generate_specs`.
    """
    # NOTE: In constraint_rows() support e.g. name=('ts', None) combined with
    # breakdown='cld' or component=('swcld', 'lwcld') because the latter vector
    # is placed in outer specs while the former is placed in subspecs. However here
    # often need to vectorize breakdown inside subspecs (e.g. bar plots with many
    # feedback components) so approach is e.g. name='ts' and forward=True or False.
    names = ('name', 'breakdown', 'component')
    compare = sum(bool(kwargs.get(key)) for key in names) > 1  # noqa: E501
    spread = kwargs.get('method', None) in ('std', 'var', 'cov', 'slope')
    globe = kwargs.get('area', None) is not None
    if 'rowspecs' in kwargs and 'colspecs' in kwargs:
        rowspecs = kwargs.pop('rowspecs', None) or [{}]
        colspecs = kwargs.pop('colspecs', None) or [{}]
        figspecs = [rowspecs, colspecs]
        subspecs1 = subspecs2 = {}
    else:
        *figspecs, subspecs1, subspecs2, kwargs = generate_specs(**kwargs)
        rowspecs = figspecs[0] if len(figspecs) > 0 else [{}]
        colspecs = figspecs[1] if len(figspecs) > 1 else [{}]
    name = kwargs.pop('name', None)
    results = []
    for rowspecs, kwargs in divide_specs('row', rowspecs, **kwargs):
        defaults = {}
        if globe and not compare and not spread and subspecs1 == subspecs2:
            kws = (*subspecs1, *subspecs2, kwargs)
            projects = set(kw['project'] for kw in kws if 'project' in kw)
            institutes = set(kw['institute'] for kw in kws if 'institute' in kw)
            project = projects.pop() if len(projects) == 1 else None
            institute = institutes.pop() if len(institutes) == 1 else None
            kw_size = {**kwargs, 'project': project, 'institute': institute}
            refsize, altsize = _default_size(**kw_size)
            defaults.update(refheight=refsize, refwidth=altsize, annotate=True)
        rspecs = []
        kwargs = {**defaults, **kwargs}
        for rspec in rowspecs:
            ispecs = []
            for spec1, spec2 in zip(subspecs1, subspecs2):
                spec1, spec2 = spec1.copy(), spec2.copy()
                if spec1 == spec2:
                    spec = {'name': name, **rspec, **spec1}
                else:
                    ispec, jspec = (spec1, spec2) if forward else (spec2, spec1)
                    if name is not None:  # e.g. 'ts' or 'tpat'
                        ispec['name'] = name
                        jspec.setdefault('name', name)  # e.g. no 'breakdown'
                    spec = ({**rspec, **spec1}, {**rspec, **spec2})
                ispecs.append(spec)
            rspecs.append(ispecs or [rspec])
        for cspecs, kwargs in divide_specs('col', colspecs, **kwargs):
            result = plotting.generate_plot(data, rspecs, cspecs, **kwargs)
            results.append(result)
    result = results[0] if len(results) == 1 else results
    return result


def pattern_rows(data, method=None, shading=True, contours=True, **kwargs):
    """
    Plot averages and standard deviations per row (e.g. maps).

    Parameters
    ----------
    data : xarray.Dataset
        The source dataset.
    method : bool, optional
        The single-variable method to use.
    shading : bool, optional
        Whether to include shading.
    contours : bool, optional
        Whether to include reference contours.
    rowsplit, colsplit : optional
        Passed to `divide_specs`.
    **kwargs
        Passed to `generate_specs`.
    """
    # NOTE: For simplicity pass scalar 'outer' and other vectors are used in columns.
    if not isinstance(method := method or 'avg', str):
        method1, method2 = method
    elif method in ('avg', 'med'):
        method1, method2 = method, 'std'
    elif method in ('std', 'var', 'pctile'):
        method1, method2 = method, 'avg'
    else:
        raise ValueError(f'Invalid pattern_rows() method {method!r}.')
    if 'breakdown' not in kwargs and 'component' not in kwargs and 'outer' not in kwargs:  # noqa: E501
        raise RuntimeError
    rowspecs, colspecs, *_, kwargs = generate_specs(maxcols=1, **kwargs)
    kw_shading = {'method': method1} if shading is True else dict(shading or {})
    kw_shading.update({key: kwargs.pop(key) for key in KEYS_SHADING if key in kwargs})
    kw_contour = {'method': method2} if contours is True else dict(contours or {})
    kw_contour.update({key: kwargs.pop(key) for key in KEYS_CONTOUR if key in kwargs})
    results = []
    for rowspecs, kwargs in divide_specs('row', rowspecs, **kwargs):
        rspecs = []
        for rspec in rowspecs:
            kw = {key: val for key, val in rspec.items() if key not in KEYS_CONTOUR}
            spec = [{**kw, **kw_shading}]
            if contours:
                kw = {key: val for key, val in rspec.items() if key not in KEYS_SHADING}
                spec.append({**kw, **kw_contour})
            rspecs.append(spec)
        cspecs = []  # NOTE: irrelevant keywords for non-cmap figures are ignored
        for cspec in colspecs:
            kw = {key: val for key, val in cspec.items() if key not in KEYS_CONTOUR}
            spec = [{**kw, **kw_shading}]  # noqa: E501
            if contours:
                kw = {key: val for key, val in cspec.items() if key not in KEYS_SHADING}
                spec.append({**kw, **kw_contour})
            cspecs.append(spec)
        for cspecs, kwargs in divide_specs('col', cspecs, **kwargs):
            # ic(rspecs, cspecs)
            result = plotting.generate_plot(data, rspecs, cspecs, **kwargs)
            results.append(result)
    result = results[0] if len(results) == 1 else results
    return result


def constraint_rows(data, method=None, contours=True, hatching=True, **kwargs):
    """
    Plot two quantifications of constraint relationship per row (e.g. maps).

    Parameters
    ----------
    data : xarray.DataArray
        The data.
    method : str, optional
        The two-variable method to use.
    contours : bool, optional
        Whether to include reference averages.
    hatching : bool optional
        Whether to include correlation hatching.
    rowsplit, colsplit : optional
        Passed to `divide_specs`.
    **kwargs
        Passed to `generate_specs`.

    Other Parameters
    ----------------
    base : str, optional
        Kludge. As with `experiment1` but applies over 'outer' arrays.
    pattern : str, optional
        Kludge. The method to use for temperature pattern variables.
    """
    # TODO: Remove 'base' keyword kludge and support two pairs of outer arrays
    if 'breakdown' not in kwargs and 'component' not in kwargs and 'outer' not in kwargs:  # noqa: E501
        raise RuntimeError
    rowspecs, *colspecs, subspecs1, subspecs2, kwargs = generate_specs(maxcols=1, **kwargs)  # noqa: E501
    if len(subspecs1) != 1 or len(subspecs2) != 1:
        raise ValueError(f'Too many constraints {subspecs1} and {subspecs2}. Check outer argument.')  # noqa: E501
    (subspec1,), (subspec2,) = subspecs1, subspecs2
    pattern = kwargs.pop('pattern', None)  # TODO: remove kludge
    kw_shading = {key: kwargs.pop(key) for key in KEYS_SHADING if key in kwargs}
    kw_contour = {key: kwargs.pop(key) for key in KEYS_CONTOUR if key in kwargs}
    _, methods, hatches, levels = _constraint_props(method=method)
    if usefirst := subspec2.get('area'):
        subspec = subspec1
    elif usefirst := subspec1.get('name') in KEYS_PLEV and not subspec1.get('plev'):
        subspec = subspec1  # TODO: more complex rules?
    else:
        subspec = subspec2
    # ic(kwargs, subspec, subspec1, subspec2)
    if colspecs:  # i.e. single row-column plot
        colspecs, = colspecs
    else:
        colspecs = [{}]
    results = []
    warming = ('tpat', 'tstd', 'tdev', 'tabs', 'ecs')
    for rowspecs, kwargs in divide_specs('row', rowspecs, **kwargs):
        rspecs = []
        for rspec in rowspecs:  # WARNING: critical to put overrides in row specs
            kw1, kw2 = {**rspec, **subspec1}, {**rspec, **subspec2}
            base, _ = kw1.pop('base', None), kw2.pop('base', 0)  # TODO: remove kludge
            if base is not None:  # apply possibly vector base experiment
                kw1['experiment'] = base
            ikw1 = {key: val for key, val in kw1.items() if key not in KEYS_CONTOUR}
            ikw2 = {key: val for key, val in kw2.items() if key not in KEYS_CONTOUR}
            if base is not None:  # always use slope
                ikw1['method'] = ikw2['method'] = 'slope'
            if pattern and any(kw.get('name') in warming for kw in (kw1, kw2)):
                ikw1['method'] = ikw2['method'] = pattern
            spec = [(ikw1, ikw2)]  # takes precedence over columns
            if contours:
                ikw = kw1 if usefirst else kw2
                ikw = {key: val for key, val in ikw.items() if key not in KEYS_SHADING}
                if name := ikw.get('name', None):  # TODO: remove kludge
                    ikw['name'] = 'tstd' if name == 'tdev' else name
                spec.append(ikw)  # possibly feedbacks
            if hatching:
                ikw1 = {key: val for key, val in kw1.items() if key not in KEYS_EITHER}
                ikw2 = {key: val for key, val in kw2.items() if key not in KEYS_EITHER}
                spec.append((ikw1, ikw2))
            rspecs.append(spec)
        cspecs = []
        for cspec in colspecs:
            kw = {'method': methods[0], 'symmetric': True, 'cmap': 'Balance', **kw_shading}  # noqa: E501
            ikw1, ikw2 = {**cspec, **subspec1}, {**cspec, **subspec2}
            ikw1 = {key: val for key, val in ikw1.items() if key not in KEYS_CONTOUR}
            ikw2 = {key: val for key, val in ikw2.items() if key not in KEYS_CONTOUR}
            spec = [({**kw, **ikw1}, {**kw, **ikw2})]
            if contours:
                kw = {'method': 'avg', 'symmetric': False, **kw_contour}
                ikw = {**cspec, **subspec}
                ikw = {key: val for key, val in ikw.items() if key not in KEYS_SHADING}
                spec.append({**kw, **ikw})
            if hatching:
                kw = {'method': methods[1], 'levels': levels, 'hatches': hatches}
                ikw1, ikw2 = {**cspec, **subspec1}, {**cspec, **subspec2}
                ikw1 = {key: val for key, val in ikw1.items() if key not in KEYS_EITHER}
                ikw2 = {key: val for key, val in ikw2.items() if key not in KEYS_EITHER}
                spec.append(({**kw, **ikw1}, {**kw, **ikw2}))
            cspecs.append(spec)
        # ic(cspecs, rspecs, kwargs)
        for cspecs, kwargs in divide_specs('col', cspecs, **kwargs):
            result = plotting.generate_plot(data, rspecs, cspecs, **kwargs)
            results.append(result)
    result = results[0] if len(results) == 1 else results
    return result


def relationship_rows(data, method=None, **kwargs):
    """
    Plot average of constraint components and their relationship per row (e.g. maps).

    Parameters
    ----------
    data : xarray.Dataset
        The source dataset.
    method : bool, optional
        The two-variable method to use.
    rowsplit, colsplit : optional
        Passed to `divide_specs`.
    **kwargs
        Passed to `generate_specs`.
    """
    # TODO: Update this. It is out of date with constraint_rows.
    if 'breakdown' not in kwargs and 'component' not in kwargs and 'outer' not in kwargs:  # noqa: E501
        raise RuntimeError
    rowspecs, colspecs1, colspecs2, kwargs = generate_specs(maxcols=1, **kwargs)
    if len(colspecs1) != 1 or len(colspecs2) != 1:
        raise ValueError(f'Too many constraints {colspecs1} and {colspecs2}. Check outer argument.')  # noqa: E501
    label, methods, hatches, levels = _constraint_props(method=method)
    kwargs['collabels'] = [None, None, label]
    rplots = (
        {'method': 'avg', 'symmetric': True, 'cmap': 'ColdHot'},
        {'method': 'std', 'symmetric': False, 'cmap': 'ColdHot'},
    )
    cplots = (
        {'method': methods[0], 'symmetric': True, 'cmap': 'Balance'},
        {'method': methods[1], 'levels': levels, 'hatches': hatches, 'colors': 'none'},
    )
    results = []
    for rowspecs, kwargs in divide_specs('row', rowspecs, **kwargs):
        rspecs = rowspecs  # possibly none
        colspecs = [
            [{**colspecs1[0], **spec} for spec in rplots],
            [{**colspecs2[0], **spec} for spec in rplots],
            [({**colspecs1[0], **spec}, {**colspecs2[1], **spec}) for spec in cplots]
        ]
        for cspecs, kwargs in divide_specs('col', colspecs, **kwargs):
            # ic(rspecs, cspecs)
            result = plotting.generate_plot(data, rspecs, cspecs, **kwargs)
            results.append(result)
    result = results[0] if len(results) == 1 else results
    return result
