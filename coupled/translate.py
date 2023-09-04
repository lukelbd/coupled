#!/usr/bin/env python3
"""
Helper functions for translating figure specifications.
"""
import itertools

import climopy as climo  # noqa: F401
import numpy as np
from climopy import ureg, vreg  # noqa: F401
from coupled import plotting
from icecream import ic  # noqa: F401

from coupled.results import ALIAS_FEEDBACKS

__all__ = [
    'create_specs',
    'split_specs',
    'feedback_breakdown',
    'transport_breakdown',
]


# Variable and selection keywords
# NOTE: These can be passed as vectors to create_specs for combining along rows or cols
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

# Breakdown definitions
# NOTE: Lists below are parsed by feedback_breakdown() into either single columns,
# two columns, or some fixed number of columns compatible with the decomposition.
# NOTE: The 'wav' suffixes including longwave and shortwave cloud components
# instead of net cloud feedback. Strings denote successively adding atmospheric,
# albedo, residual, and remaining temperature/humidity feedbacks with options.
TRANSPORT_BREAKDOWNS = {
    'atmos': ('mse', 'lse', 'dse'),
    'total': ('total', 'ocean', 'mse'),
    'all': ('total', 'ocean', 'mse', 'lse', 'dse'),
}
FEEDBACK_BREAKDOWNS = {
    # Three-variable
    'net': ('net', 'sw', 'lw'),
    'atm': ('net', 'cld', 'atm'),
    'ncl': ('net', 'cld', 'ncl'),
    'wav': ('net', 'swcld', 'lwcld'),
    'cld': ('cld', 'swcld', 'lwcld'),  # special case!
    # Four-variable
    'alb': ('net', 'cld', 'alb', 'atm'),
    'cld_wav': ('net', 'cld', 'swcld', 'lwcld'),
    'atm_res': ('net', 'cld', 'resid', 'atm'),
    'atm_wav': ('net', 'swcld', 'lwcld', 'atm'),
    'ncl_wav': ('net', 'swcld', 'lwcld', 'ncl'),
    # Five-variable
    'res': ('net', 'cld', 'alb', 'atm', 'resid'),
    'res_wav': ('net', 'swcld', 'lwcld', 'resid', 'atm'),
    'alb_wav': ('net', 'swcld', 'lwcld', 'alb', 'atm'),
    'atm_cld': ('net', 'cld', 'swcld', 'lwcld', 'atm'),
    'ncl_cld': ('net', 'cld', 'swcld', 'lwcld', 'ncl'),
    # Additional variables
    'hus': ('net', 'resid', 'cld', 'swcld', 'lwcld', 'alb', 'atm', 'wv', 'lr', 'pl'),
    'hur': ('net', 'resid', 'cld', 'swcld', 'lwcld', 'alb', 'atm', 'rh', 'lr*', 'pl*'),
    'all': ('net', 'cld', 'swcld', 'lwcld', 'atm', 'alb', 'resid', 'wv', 'rh', 'lr', 'lr*', 'pl', 'pl*'),  # noqa: E501
    'alb_cld': ('net', 'cld', 'swcld', 'lwcld', 'atm', 'alb'),  # TODO support
    'res_cld': ('net', 'cld', 'swcld', 'lwcld', 'atm', 'resid'),  # TODO: support
}


def _create_dicts(*kws, expand=True, check=True):
    """
    Helper function to create dictionaries.

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


def create_specs(pairs=None, product=None, outer='breakdown', maxcols=None, **kwargs):
    """
    Create feedback and variable specifications based on input keywords.

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
        multiplicand e.g. ``outer=(('breakdown', 'cycle'), ('project', 'experiment'))``.
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
    # e.g. create_specs(breakdown='all') but generate transport components if
    # 'transport' is passed with e.g. create_specs(breakdown='all', transport='t')
    outer = (outer,) if isinstance(outer, str) else outer or ()
    outer = [[keys] if isinstance(keys, str) else list(keys) for keys in outer]
    ncols = kwargs.get('ncols', None)
    maxcols = 1 if ncols else maxcols  # disable special arrangements
    kw_break = {key: kwargs.pop(key) for key in tuple(kwargs) if key in KEYS_BREAK}
    if len(outer) > 2:  # only ever used for rows and columns
        raise ValueError('Too many outer variables specified.')
    if 'transport' in kw_break:  # transport breakdown
        breakdown, kw_default = transport_breakdown(maxcols=maxcols, **kw_break)
        kw_default = {**kw_default, 'proj_kw': {'lon_0': 0}}
    elif kw_break:  # non-empty breakdown
        breakdown, kw_default = feedback_breakdown(maxcols=maxcols, **kw_break)
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
    # or columns. For example: correlate with feedback parts in rows, create_specs(
    # name='psl', breakdown='net'); or correlate with parts (plus others) in subplots,
    # create_specs(name='psl', breakdown='net', outer=None, experiment=('picontrol', 'abrupt4xco2')).  # noqa: E501
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
            if key == 'cycle':  # generally joined with e.g. 'breakdown'
                kw['color'] = kwargs.pop(key, plotting.CYCLE_DEFAULT)  # ignore defaults
            elif key in kw_plot:
                kw[key] = kw_plot.pop(key)
    for key in pairs:  # specifications for pairs
        if values := kw_plot.get(key):
            if not np.iterable(values) or len(values) != 2:
                raise ValueError(f'Coordinate pair {key} must be a length-2 tuple.')
            for kw, value in zip(kws_pair, kw_plot.pop(key)):
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
        for keys in product:
            if any(key in dict_ for key in keys):
                kw = {key: dict_.pop(key) for key in keys if key in dict_}
                kws.append(_create_dicts(kw, expand=False))
        keys = [key for kw in kws for key in kw]
        values = itertools.product(*(zip(*kw.values()) for kw in kws))
        values = [[v for val in vals for v in val] for vals in values]
        dict_.update({key: vals for key, vals in zip(keys, zip(*values))})
    kws_outer = tuple(map(_create_dicts, kws_outer))
    kws_pair = _create_dicts(*kws_pair)
    return *kws_outer, *kws_pair, kwargs


def split_specs(key, specs, **kwargs):
    """
    Split feedback and variable specification lists.

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


def feedback_breakdown(
    breakdown=None,
    component=None,
    feedbacks=True,
    adjusts=False,
    forcing=False,
    sensitivity=False,
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
    feedbacks, adjusts, forcing, sensitivity : bool, optional
        Whether to include various components.
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
    options = ', '.join(map(repr, FEEDBACK_BREAKDOWNS))
    init_names = lambda ncols: ((names := np.array([[None] * ncols] * 25)), names.flat)
    if not component and not feedbacks and not forcing and not sensitivity:
        raise ValueError('Invalid keyword argument combination.')
    if breakdown is None:
        components = component or 'net'
    elif FEEDBACK_BREAKDOWNS:
        components = FEEDBACK_BREAKDOWNS[breakdown]
    else:
        raise ValueError(f'Invalid breakdown {breakdown!r}. Options are: {options}')
    components = (components,) if isinstance(components, str) else tuple(components)

    # Generate plot layouts
    # NOTE: This is relevant for general_subplots() style figures when we wrap have
    # wrapped rows or columns of components but not as useful for e.g. summary_rows()
    lams = [ALIAS_FEEDBACKS[name] for name in components]
    erfs = [name.replace('_lam', '_erf') for name in lams]
    if len(lams) == 1 or len(lams) == 2:  # user input breakdowns
        gridskip = None
        names, iflat = init_names(maxcols)
        for i, name in enumerate(component):  # assign possibly correlation tuples!
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
    elif len(lams) == 5:
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
    # TODO: Consider re-implementing automatic application of color cycle
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
    ic(names, gridskip)
    return names, kwargs


def transport_breakdown(breakdown=None, component=None, transport=None, maxcols=None):
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
    options = ', '.join(map(repr, FEEDBACK_BREAKDOWNS))
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
    if transport not in TRANSPORT_BREAKDOWNS:
        raise ValueError(f'Invalid breakdown {transport!r}. Options are: {options}')
    if component is not None:
        names = (component,)
    else:
        names = TRANSPORT_BREAKDOWNS[transport]
        shading = 7  # open color shading level
        kwargs['cycle'] = [colors[name] + str(shading) for name in names]
        names = [name + transport for name in names]
    return names, kwargs
