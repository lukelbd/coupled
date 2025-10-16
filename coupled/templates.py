#!/usr/bin/env python3
"""
Helper functions for plotting coupled model output.
"""
import itertools

import climopy as climo  # noqa: F401
import numpy as np
from climopy import var, ureg, vreg  # noqa: F401
from icecream import ic  # noqa: F401

from .general import CYCLE_DEFAULT
from .general import general_plot
from .specs import _expand_lists

__all__ = [
    'build_specs',
    'divide_specs',
    'feedback_specs',
    'transport_specs',
    'scalar_plot',
    'pattern_plot',
    'regression_plot',
    'regression_parts',
]

# Keywords for specific commands or variables
# NOTE: See also specs.py key lists
KEYS_CONTOUR = (
    'levels',  # restricted to contours
    'linewidth', 'linestyle', 'edgecolor',
)
KEYS_SHADING = (
    'vmin', 'vmax', 'cmap', 'cmap_kw', 'norm', 'norm_kw',
    'robust', 'step', 'locator', 'symmetric', 'diverging', 'sequantial',
)
KEYS_VERTICAL = (
    'ta', 'zg', 'ua', 'va', 'hus', 'hur', 'pt', 'slope',
    *(f'cl{s}' for s in ('', 'w', 'l', 'i', 'p')),
)

# Keywords for generating dictionary specs from kwargs
# NOTE: These can be passed as vectors to build_specs for combining along rows/cols.
# TODO: Remove 'base' kludge and permit vectors specifications on each side of pair.
KEYS_SPECS = (
    'breakdown', 'component', 'feedbacks', 'adjusts', 'forcing', 'sensitivity', 'transport'  # noqa: E501
)
KEYS_ITER = (
    'name', 'method', 'time', 'season', 'month', 'spatial', 'observed',
    'period', 'initial', 'volume', 'area', 'plev', 'lat', 'lon',  # TODO: remove below
    'project', 'institute', 'experiment', 'ensemble', 'model', 'bootstrap', 'base',
    'source', 'style', 'start', 'stop', 'remove', 'detrend', 'error', 'correct',
    'region', 'alpha', 'edgecolor', 'linewidth', 'linestyle', 'color', 'facecolor',
    'xmin', 'xmax', 'ymin', 'ymax', 'xlabel', 'ylabel', 'xlocator', 'ylocator',
    'loc', 'label', 'align', 'colorbar', 'legend', 'length', 'shrink',
    'extend', 'extendsize', *KEYS_SHADING, *KEYS_CONTOUR
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
    'cre_net': ('net', 'cre', 'swcre', 'lwcre'),
    'wav_atm': ('net', 'swcld', 'lwcld', 'atm'),
    'wav_ncl': ('net', 'swcld', 'lwcld', 'ncl'),
    'wav_cs': ('net', 'swcre', 'lwcre', 'cs'),
    'ncl_wav': ('net', 'ncl', 'swcld', 'lwcld'),  # clear-sky first
    'cs_wav': ('net', 'cs', 'swcre', 'lwcre'),
    'tstd_net': ('tstd', 'net', 'cld', 'ncl'),
    'tpat_net': ('tpat', 'net', 'cld', 'ncl'),
    'ecs_net': ('ecs', 'net', 'cld', 'ncl'),
    # Five-variable
    'resid': ('net', 'cld', 'alb', 'atm', 'resid'),
    'cld_atm': ('net', 'cld', 'swcld', 'lwcld', 'atm'),
    'cld_ncl': ('net', 'cld', 'swcld', 'lwcld', 'ncl'),
    'cre_cs': ('net', 'cre', 'swcre', 'lwcre', 'cs'),
    'ncl_cld': ('net', 'ncl', 'cld', 'swcld', 'lwcld'),  # clear-sky first
    'cs_cre': ('net', 'cs', 'cre', 'swcre', 'lwcre'),
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
    'ncl_hur': ('ncl', 'pl*', 'lr*', 'rh', 'alb', 'resid'),
    'wav_hur': ('swcld', 'lwcld', 'lr*', 'rh', 'alb', 'resid'),
}


def _get_props(method=None):
    """
    Helper function to get contour and hatching properties from default reduction.

    Parameters
    ----------
    method : bool
        The method to use for regression shading.

    Returns
    -------
    stipple, hatches, levels : list
        The stipple method and associated hatching style and level boundaries.
    """
    methods = method or 'slope'
    methods = (methods,) if isinstance(methods, str) else tuple(methods or (None,))
    corr = all(method and method.split('_')[0] in ('proj', 'rsq') for method in methods)
    if corr:  # correlation instead of slope
        stipple = 'corr'
        hatches = ['.....', '...', None, '...', '.....']
        levels = [-10, -0.8, -0.5, 0.5, 0.8, 10]
        # levels = [-10, -0.66, -0.33, 0.33, 0.66, 10]
    else:
        stipple = 'rsq'
        hatches = [None, '...', '.....']
        levels = [0, 33.333, 66.666, 1000]
        # levels = [0, 50, 80, 1000]
    return stipple, hatches, levels


def _get_dicts(*kws, scalar=True, splat=True, check=True):
    """
    Helper function to get dictionaries from lists of dictionaries.

    Parameters
    ----------
    *kws : dict
        The input dictionaries.
    splat : bool, optional
        Whether to splat lists into separate dictionaries.
    check : bool, optional
        Whether to check lists have equivalent lengths.
    """
    # NOTE: This allows e.g. list applied to one side of correlation pair
    # for multiple items in a single subplot. See notebooks for details.
    lengths, results = [], []
    for i, kw in enumerate(kws):
        ilengths = {}
        for key, value in tuple(kw.items()):  # TODO: revisit
            if not isinstance(value, (list, tuple)):
                kw[key] = value = (value,)
            if 'color' in key or 'cycle' in key:
                continue
            ilengths[key] = len(value)
        ilength = set(ilengths.values()) - {1}
        if check and len(ilength) > 1:
            msg = ', '.join(f'{key}={value}' for key, value in ilengths.items())
            raise ValueError(f'Unexpected inner keyword mixed lengths: {msg}.')
        ilength = ilength and ilength.pop() or 1
        iresult = {
            key: ilength * value if len(value) == 1
            else value for key, value in kw.items()
        }
        if splat:
            result = [
                _get_dicts(dict(zip(iresult, vals)), splat=False)
                for vals in zip(*iresult.values())
            ]
        else:
            result = {
                key: vals[0] if scalar and len(vals) == 1 else vals
                for key, vals in iresult.items()
            }
        results.append(result)
        lengths.append({f'{key}{i + 1}': value for key, value in ilengths.items()})
    lens = {key: value for lens in lengths for key, value in lens.items()}
    jlengths = set(lens.values()) - {1}
    if check and len(jlengths) > 1:
        msg = ', '.join(f'{key}={value}' for key, value in lens.items())
        raise ValueError(f'Unexpected inner keyword mixed lengths: {msg}.')
    return results[0] if len(results) == 1 else results


def build_specs(outer='breakdown', pairs=None, product=None, maxcols=None, **kwargs):
    """
    Generate feedback and variable specifications based on input keywords.

    Parameters
    ----------
    outer : str or list of str, optional
        The kwargs for the outer plotting specs. Can include multiple keys per
        multiplicand e.g. ``outer=(('breakdown', 'color'), ('project', 'experiment'))``.
    pairs : str or list of str, optional
        The coordinate name(s) to be used for feedback constraints. These can
        also be generated using combination of ``'name'`` and ``'breakdown'``.
    product : tuple of str or list of str, optional
        The list-like kwargs to combine with `itertools.product`. Can include multiple
        keys per multiplicand e.g. ``product=(('experiment', 'color'), 'project')``.
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
    # e.g. build_specs(breakdown='all') but generate transport components if
    # 'transport' is passed with e.g. build_specs(breakdown='all', transport='t')
    outer = [[outer]] if isinstance(outer, str) else outer or []
    outer = [[keys] if isinstance(keys, str) else list(keys) for keys in outer]
    pairs = pairs or ()
    pairs = (pairs,) if isinstance(pairs, str) else tuple(pairs)
    ncols = kwargs.get('ncols', None)
    maxcols = 1 if ncols else maxcols  # disable special arrangements
    kw_break = {key: kwargs.pop(key) for key in tuple(kwargs) if key in KEYS_SPECS}
    if len(outer) > 2:  # only ever used for rows and columns
        raise ValueError('Too many outer variables specified.')
    if 'transport' in kw_break:  # transport breakdown
        breakdown, kw_default = transport_specs(maxcols=maxcols, **kw_break)
        kw_default = {**kw_default, 'proj_kw': {'lon_0': 0}}
    elif kw_break:  # non-empty breakdown
        breakdown, kw_default = feedback_specs(maxcols=maxcols, **kw_break)
        kw_default = {**kw_default, 'proj_kw': {'lon_0': 210}}
    else:  # note scalar_plot() pops the name first so permit zero arguents
        breakdown = None
        kw_default = {'ncols': ncols or maxcols}
    idxs = [i for i, keys in enumerate(outer) if 'breakdown' in keys]
    jdxs = [i for i, keys in enumerate(outer) if 'name' in keys]  # if breakdown not passed  # noqa: E501
    kws_inner = [{}, {}]  # ensure always paired
    kws_outer = [{} for i in range(len(outer))]
    for key, value in kw_default.items():
        kwargs.setdefault(key, value)
    if not breakdown:
        pass
    elif not idxs and not jdxs:
        kws_inner[0]['name'] = kws_inner[1]['name'] = breakdown
    else:  # permit e.g. feedback breakdown rows and variable name columns
        kws_outer[idxs[0] if idxs else jdxs[0]]['name'] = breakdown

    # Assign outer and inner dictionaries
    # NOTE: Numpy arrays should be used for e.g. scalar level lists or color values
    # and tuples or lists should be used for vectors of settings across subplots.
    # NOTE: Here 'outer' is used to specify different reduce instructions across rows
    # or columns. For example: correlate with feedback parts in rows, build_specs(
    # name='psl', breakdown='net'); or correlate with parts (plus others) in subplots,
    # build_specs(name='psl', breakdown='net', outer=None, experiment=('picontrol', 'abrupt4xco2')).  # noqa: E501
    for keys, kw in zip(outer, kws_outer):
        for key in keys:
            if key in pairs:
                raise ValueError(f'Keyword {key} cannot be in both outer and pairs.')
            if key == 'color':  # generally joined with e.g. 'breakdown'
                value = kwargs.get(key, CYCLE_DEFAULT)  # ignore defaults
            else:
                value = kwargs.get(key, None)
            if isinstance(value, (tuple, list)):
                kwargs.pop(key, None)
                kw[key] = value
    for key in pairs:  # specifications for pairs
        values = kwargs.pop(key, None)
        if values is None:
            continue
        if not np.iterable(values) or len(values) > 2:
            raise ValueError(f'Coordinate pair {key}={values!r} is not a 2-tuple.')
        values = values * 2 if len(values) == 1 else values
        for kw, value in zip(kws_inner, values):
            kw[key] = list(value) if isinstance(value, (tuple, list)) else [value]
    kw_inner = {
        key: list(kwargs.pop(key)) for key in tuple(kwargs)
        if key in KEYS_ITER and isinstance(kwargs[key], (list, tuple))
    }
    kws_inner[0].update(kw_inner)  # remaining scalars or vectors
    kws_inner[1].update(kw_inner)

    # Convert reduce vector iterables to dictionaries
    # NOTE: Here use 'force' to ensure concatenated distributions have non-non labels
    # e.g. 'full' 'early' 'late' instead of '' 'early' 'late' for violin plots.
    # NOTE: This also builds optional Cartesian products between lists of keywords
    # specified by 'product' keyword. Others are enforced to have same length.
    # NOTE: This can fail if user passes feedbacks along rows and requests
    # e.g. 'name1' so should only assign to outer if vector is passed.
    sentinel = object()  # see below
    spatial = ('lon', 'lat', 'plev', 'area')  # include user input none
    product = product or ()
    product = [[keys] if isinstance(keys, str) else list(keys) for keys in product]
    restrict = tuple(key for keys in product for key in keys)
    restrict += ('startstop',) if 'start' in restrict and 'stop' in restrict else ()
    kwargs.setdefault('restrict', restrict)
    for key in KEYS_ITER:  # add scalar versions
        value = kwargs.pop(f'{key}1', sentinel)  # ignore none iteration placeholders
        if value is not sentinel and (key in spatial or value is not None):
            value = list(value) if isinstance(value, (tuple, list)) else [value]
            if len(outer) > 0 and key in outer[0] and len(value) > 1:
                kws_outer[0].update({key: value})
            else:
                kws_inner[0].update({key: value})
        value = kwargs.pop(f'{key}2', sentinel)  # ignore none iteration placeholders
        if value is not sentinel and (key in spatial or value is not None):
            value = list(value) if isinstance(value, (tuple, list)) else [value]
            if len(outer) > 1 and key in outer[1] and len(value) > 1:
                kws_outer[1].update({key: value})
            else:
                kws_inner[1].update({key: value})
    for idx, groups in enumerate((kws_outer, kws_inner)):
        for group in groups:
            kws, inners = [], []
            check = tuple(key for keys in product for key in keys)
            for key in check:
                value = group.get(key, None)
                if idx or not isinstance(value, (list, tuple)):
                    continue
                if any(isinstance(val, (list, tuple)) for val in value):  # TODO: all?
                    inners.append(key)  # TODO: revisit or make configurable
            for keys in product:  # then skip if absent
                keys = set(keys) - set(inners)
                if not any(key in group for key in keys):
                    continue
                kw = {key: group.pop(key) for key in keys if key in group}
                res = _get_dicts(kw, scalar=False, splat=False)
                kws.append(res)
            keys = [key for kw in kws for key in kw]
            values = itertools.product(*(zip(*kw.values()) for kw in kws))
            values = [[v for val in vals for v in val] for vals in values]
            group.update({key: vals for key, vals in zip(keys, zip(*values))})
    kws_outer = tuple(map(_get_dicts, kws_outer))  # rows and columns
    kws_outer = [kw for kw in kws_outer if kw]  # single gridspec
    kws_inner = _get_dicts(*kws_inner)  # inner reduce pairs
    kws_inner = [kw or [{}] for kw in kws_inner]
    cnts, cmax = list(map(len, kws_inner)), max(map(len, kws_inner))
    kws_inner[0] *= cmax if cnts[0] == 1 else 1  # match lengths
    kws_inner[1] *= cmax if cnts[1] == 1 else 1
    return *kws_outer, *kws_inner, kwargs


def divide_specs(name, specs, **kwargs):
    """
    Divide feedback and variable specification lists.

    Parameters
    ----------
    name : {'rows', 'cols'}
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
    split = kwargs.pop(f'{name}split', None)
    nosplit = split is False or split is None
    side = 'h' if name == 'row' else 'w'
    geom = 'nrows' if name == 'row' else 'ncols'
    count = kwargs.get(geom, None)
    titles = kwargs.get('titles', None)
    labels = kwargs.get(f'{name}labels', None)
    pad = kwargs.get(f'{side}pad', None)
    space = kwargs.get(f'{side}space', None)
    ratios = kwargs.get(f'{side}ratios', None)
    if nosplit:  # split nothing
        split = ([*range(0, len(specs))],)
    if split is True:  # split everything
        split = [*range(0, len(specs))]
    if not np.iterable(split):
        split = (split,)

    # Iterate over split indices
    # NOTE: This comes *after* generating specs with _parse_specs. Simply yield
    # subselections of resulting dictionary lists and update figure settings.
    for idxs in split:  # split indexes
        kwargs = kwargs.copy()
        if not np.iterable(idxs):
            idxs = (idxs,)
        if isinstance(idxs[-1], dict):
            kwargs = {**kwargs, **idxs[-1]}
            idxs = idxs[:-1]
        subs = [specs[idx] for idx in idxs]
        noskip = np.all(np.diff(idxs) == 1)
        kwargs[geom] = count if nosplit else None
        kwargs.setdefault('horizontal', name == 'row')
        if not nosplit and np.iterable(ratios):
            kwargs[f'{side}ratios'] = [ratios[idx] for idx in idxs]
        if not nosplit and np.iterable(space):
            kwargs[f'{side}space'] = space[idxs[0]:idxs[-1]] or None if noskip else None
        if not nosplit and np.iterable(pad):
            kwargs[f'{side}pad'] = pad[idxs[0]:idxs[-1]] or None if noskip else None
        if not nosplit and np.iterable(labels) and not isinstance(labels, str):
            if len(labels) == len(specs):  # TODO: re-address (need for tiled plots?)
                kwargs[f'{name}labels'] = [labels[idx] for idx in idxs]
        if not nosplit and np.iterable(titles) and not isinstance(titles, str):
            if len(titles) == len(specs) and name == 'row':  # TODO: re-address
                kwargs['titles'] = [titles[idx] for idx in idxs]
        yield subs, kwargs


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
        The variable specifications.
    kwargs : dict
        The keyword args to pass to `general_plot`.
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
    forcing = forcing or breakdown and '_erf' in breakdown
    sensitivity = sensitivity or breakdown and '_ecs' in breakdown
    if breakdown is not None:
        breakdown = breakdown.split('_erf')[0].split('_ecs')[0]
    if breakdown is None or component is not None:
        components = component or 'net'  # 'component' input overrides 'breakdown'
    elif SPECS_FEEDBACK:
        components = SPECS_FEEDBACK[breakdown]  # pre-configured breakdown
    else:
        raise ValueError(f'Invalid breakdown {breakdown!r}. Options are: {options}')
    components = (components,) if isinstance(components, str) else tuple(components)

    # Generate plot layouts
    # NOTE: This is relevant for scalar_plot() style figures when we wrap have
    # wrapped rows or columns of components but not as useful for e.g. pattern_plot()
    from .feedbacks import ALIAS_VARIABLES as aliases
    lams = [aliases.get(name, name) for name in components]
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
            for b, arr, idx in zip((feedbacks, adjusts), (lams, erfs), (4, 16)):
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
    # or if maxcols == 1 was passed e.g. for pattern_plot() feedbacks along rows.
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
    if maxcols == 1 or len(names) == 1:  # e.g. single row
        kwargs = {}
    elif len(names) <= inputcols:  # simpler default
        kwargs = {'ncols': len(names)}
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
        The keyword args to pass to `general_plot`.
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


def scalar_plot(data, forward=True, **kwargs):
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
        Passed to `build_specs`.
    """
    # NOTE: In regression_plot() support e.g. name=('ts', None) combined with
    # breakdown='cld' or component=('swcld', 'lwcld') because the latter vector
    # is placed in outer specs while the former is placed in subspecs. However here
    # often need to vectorize breakdown inside subspecs (e.g. bar plots with many
    # feedback components) which can cause issues matching lengths of regression pair
    # objects so approach is to instead use e.g. name='ts' and forward=True or False.
    name = kwargs.pop('name', None)  # WARNING: critial to show here first
    results = []
    if 'rowspecs' in kwargs and 'colspecs' in kwargs:
        rowspecs = kwargs.pop('rowspecs', None) or [{}]
        colspecs = kwargs.pop('colspecs', None) or [{}]
        figspecs, subspecs1, subspecs2 = [rowspecs, colspecs], {}, {}
    else:
        *figspecs, subspecs1, subspecs2, kwargs = build_specs(**kwargs)
        rowspecs = figspecs[0] if len(figspecs) > 0 else [{}]
        colspecs = figspecs[1] if len(figspecs) > 1 else [{}]
    for rowspecs, kwargs in divide_specs('row', rowspecs, **kwargs):
        defaults = {}
        rspecs = []
        kwargs = {**defaults, **kwargs}
        for rspec in rowspecs:
            ispecs = []
            for spec1, spec2 in zip(subspecs1, subspecs2):
                spec1, spec2 = spec1.copy(), spec2.copy()
                if spec1 == spec2:
                    spec = {'name': name, **rspec, **spec1}
                    spec = _get_dicts(spec)
                else:
                    ispec, jspec = (spec1, spec2) if forward else (spec2, spec1)
                    if name is not None:  # e.g. 'ts' or 'tpat'
                        ispec['name'] = name
                        jspec.setdefault('name', None)  # e.g. no 'breakdown'
                    spec = ({**rspec, **spec1}, {**rspec, **spec2})
                    spec = tuple(zip(*map(_get_dicts, spec)))
                ispecs.extend(spec)
            rspecs.append(ispecs)
        for cspecs, kwargs in divide_specs('col', colspecs, **kwargs):
            result = general_plot(data, rspecs, cspecs, **kwargs)
            results.append(result)
    result = results[0] if len(results) == 1 else results
    return result


def pattern_plot(data, method=None, shading=True, contours=True, **kwargs):
    """
    Plot averages and standard deviations per row (e.g. maps).

    Parameters
    ----------
    data : xarray.Dataset
        The source dataset.
    method : str or 2-tuple, optional
        The single-variable method(s) to use.
    shading : dict, optional
        Additional shading-only keyword arguments.
    contours : bool or dict, optional
        Whether to include contours or contours-only keyword arguments.
    rowsplit, colsplit : optional
        Passed to `divide_specs`.
    **kwargs
        Passed to `build_specs`.
    """
    # NOTE: Pass e.g. method=('avg', 'var') for custom shading and contour assignment
    # NOTE: For simplicity pass scalar 'outer' and other vectors are used in columns
    if 'breakdown' not in kwargs and 'component' not in kwargs and 'outer' not in kwargs:  # noqa: E501
        raise ValueError('Feedback breakdown not specified.')
    if not isinstance(method := method or 'avg', str):
        method1, *methods = method
    elif method in ('avg', 'med'):
        method1, *methods = method, 'std'
    elif method in ('std', 'var', 'pctile'):
        method1, *methods = method, 'avg'
    else:
        raise ValueError(f'Invalid pattern_plot() method {method!r}.')
    kwargs.pop('base', None)  # base not used
    kwargs.update(maxcols=1)  # use custom method assignment
    rowspecs, colspecs, *_, kwargs = build_specs(**kwargs)
    kw_shading = dict(shading) if isinstance(shading, dict) else {}
    kw_shading.update({key: kwargs.pop(key) for key in KEYS_SHADING if key in kwargs})
    kw_contour = dict(contours) if isinstance(contours, dict) else {}
    kw_contour.update({key: kwargs.pop(key) for key in KEYS_CONTOUR if key in kwargs})
    results = []
    for rowspecs, kwargs in divide_specs('row', rowspecs, **kwargs):
        rspecs = []
        for rspec in rowspecs:  # kwargs take precedence
            kw = {key: val for key, val in rspec.items() if key not in KEYS_CONTOUR}
            spec = [{**kw, 'method': method1, **kw_shading}]
            if contours and methods:
                kw = {key: val for key, val in rspec.items() if key not in KEYS_SHADING}
                spec.append({**kw, 'method': methods[0], **kw_contour})
            rspecs.append(spec)
        cspecs = []  # NOTE: irrelevant keywords for non-cmap figures are ignored
        for cspec in colspecs:
            kw = {key: val for key, val in cspec.items() if key not in KEYS_CONTOUR}
            spec = [kw]
            if contours:
                kw = {key: val for key, val in cspec.items() if key not in KEYS_SHADING}
                spec.append(kw)
            cspecs.append(spec)
        for cspecs, kwargs in divide_specs('col', cspecs, **kwargs):
            # ic(rspecs, cspecs)
            result = general_plot(data, rspecs, cspecs, **kwargs)
            results.append(result)
    result = results[0] if len(results) == 1 else results
    return result


def regression_plot(data, method=None, contours=True, hatching=True, **kwargs):
    """
    Plot two quantifications of constraint relationship per row (e.g. maps).

    Parameters
    ----------
    data : xarray.DataArray
        The data.
    method : str or sequence, optional
        The double-variable method(s) to use.
    contours : bool, optional
        Whether to include contours or contour-only keyword arguments.
    hatching : bool optional
        Whether to include hatching or hatching-only keyword arguments.
    rowsplit, colsplit : optional
        Passed to `divide_specs`.
    base : str, optional
        Similar to `experiment1` but applies over 'outer' arrays.
    **kwargs
        Passed to `build_specs`.
    """
    # NOTE: This permits passing vector 'method' with 'outer' unlike other templates
    # NOTE: This forbids e.g. tuple of levels to specify separate shading and contours
    # TODO: Remove 'base' keyword kludge now that vector 'experiment1' supported
    if 'breakdown' not in kwargs and 'component' not in kwargs and 'outer' not in kwargs:  # noqa: E501
        raise ValueError('Feedback breakdown not specified.')
    method = method or 'slope'
    kwargs.update(maxcols=1, method=method)
    stipple, hatches, levels = _get_props(method=method)
    rowspecs, *colspecs, subspecs1, subspecs2, kwargs = build_specs(**kwargs)
    colspecs = colspecs and colspecs[0] or [{}]
    if len(subspecs1) != 1 or len(subspecs2) != 1:
        raise ValueError(f'Too many constraints {subspecs1} and {subspecs2}. Check outer argument.')  # noqa: E501
    (subspec1,), (subspec2,) = subspecs1, subspecs2
    if subspec2.get('area'):
        subspec = subspec1
    elif subspec1.get('name') in KEYS_VERTICAL and not subspec1.get('plev'):
        subspec = subspec1  # TODO: more complex rules?
    else:
        subspec = subspec2
    default = kwargs.pop('base', None)
    keys_both = (*KEYS_SHADING, *KEYS_CONTOUR)
    kw_shading = {key: kwargs.pop(key) for key in KEYS_SHADING if key in kwargs}
    kw_contour = dict(contours) if isinstance(contours, dict) else {}
    kw_contour = {key: kwargs.pop(key) for key in KEYS_CONTOUR if key in kwargs}
    kw_hatching = dict(hatching) if isinstance(hatching, dict) else {}
    results = []
    for rowspecs, kwargs in divide_specs('row', rowspecs, **kwargs):
        rspecs = []
        for rspec in rowspecs:  # WARNING: critical to put overrides in row specs
            ispec = {key: val for key, val in rspec.items() if key not in KEYS_CONTOUR}
            ispec.update(kw_shading)
            base = default or ispec.pop('base', None)
            kw1, kw2 = {**ispec, **subspec1}, {**ispec, **subspec2}
            kw1.update({'experiment': base} if base else {})
            spec = [(kw1, kw2)]  # rows take precedence over columns
            if contours:
                ispec = {key: val for key, val in rspec.items() if key not in KEYS_SHADING}  # noqa: E501
                ispec.update(kw_contour)
                name = ispec.get('name') or subspec.get('name')
                kw = {**ispec, **subspec, 'method': 'avg'}
                kw.update({'name': 'tstd' if name == 'tdev' else name} if name else {})
                spec.append(kw)
            if hatching:
                ispec = {key: val for key, val in rspec.items() if key not in keys_both}
                ispec.update({**kw_hatching, 'levels': levels, 'hatches': hatches})
                kw1 = {**ispec, **subspec1, 'method': stipple}
                kw2 = {**ispec, **subspec2, 'method': stipple}
                spec.append((kw1, kw2))
            rspecs.append(spec)
        cspecs = []
        for cspec in colspecs:
            ispec = {key: val for key, val in cspec.items() if key not in KEYS_CONTOUR}
            base = default or ispec.pop('base', None)
            kw1, kw2 = {**ispec, **subspec1}, {**ispec, **subspec2}
            kw1.update({'experiment': base} if base else {})
            spec = [(kw1, kw2)]
            if contours:  # contour
                ispec = {key: val for key, val in cspec.items() if key not in KEYS_SHADING}  # noqa: E501
                spec.append({**ispec, **subspec})
            if hatching:  # stipple
                ispec = {key: val for key, val in cspec.items() if key not in keys_both}
                spec.append(({**ispec, **subspec1}, {**ispec, **subspec2}))
            cspecs.append(spec)
        for cspecs, kwargs in divide_specs('col', cspecs, **kwargs):
            result = general_plot(data, rspecs, cspecs, **kwargs)
            results.append(result)
    result = results[0] if len(results) == 1 else results
    return result


def regression_parts(data, method=None, **kwargs):
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
        Passed to `build_specs`.
    """
    # NOTE: This is used for general circulation constraints on feedbacks
    # TODO: Update this. It is out of date with regression_plot.
    if 'breakdown' not in kwargs and 'component' not in kwargs and 'outer' not in kwargs:  # noqa: E501
        raise ValueError('Feedback breakdown not specified.')
    kwargs.update(maxcols=1, method=method)
    rowspecs, colspecs1, colspecs2, kwargs = build_specs(**kwargs)
    if len(colspecs1) != 1 or len(colspecs2) != 1:
        raise ValueError(f'Too many constraints {colspecs1} and {colspecs2}. Check outer argument.')  # noqa: E501
    stipple, hatches, levels = _get_props(method=method)
    kwargs.update(collabels=[None, None, 'Inter-model\nrelationship'])
    rplots = (
        {'method': 'avg', 'symmetric': True, 'cmap': 'ColdHot'},
        {'method': 'std', 'symmetric': False, 'cmap': 'ColdHot'},
    )
    cplots = (
        {'method': method, 'symmetric': True, 'cmap': 'Balance'},
        {'method': stipple, 'levels': levels, 'hatches': hatches, 'colors': 'none'},
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
            result = general_plot(data, rspecs, cspecs, **kwargs)
            results.append(result)
    result = results[0] if len(results) == 1 else results
    return result
