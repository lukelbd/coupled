#!/usr/bin/env python3
"""
Helper functions for creating figures from some generalized templates.
"""
from pathlib import Path

import climopy as climo  # noqa: F401
import numpy as np
import xarray as xr
from climopy import ureg, vreg  # noqa: F401
from coupled import plotting, _warn_coupled
from icecream import ic  # noqa: F401

from .translate import create_specs, split_specs

__all__ = [
    'general_subplots',
    'summary_rows',
    'constraint_rows',
    'components_rows',
]

SAVE = Path(__file__).parent.parent

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


def _bar_size(refsize=None, project=None, institute=None, **kwargs):
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
    hatches : list
        The hatching instructions for significance indicators.
    levels : list
        The level boundaries for hatching instructions.
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


def _update_warming(dataset, source='~/scratch/cmip-processed'):
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
    # TODO: Remove this once transition to 'annual' files is complete.
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


def general_subplots(data, forward=True, save=True, **kwargs):
    """
    Plot multiple single-variable results per subplot (e.g. bars, boxes, lines).

    Parameters
    ----------
    data : xarray.Dataset
        The source dataset.
    forward : bool, optional
        Whether to apply name pair forward or backward.
    save : bool, optional
        Whether to save the result.
    rowsplit, colsplit : optional
        Passed to `split_specs`.
    **kwargs
        Passed to `create_specs`.
    """
    # NOTE: In constraint_rows() support e.g. name=('ts', None) combined with
    # breakdown='cld' or component=('swcld', 'lwcld') because the latter vector
    # is placed in outer specs while the former is placed in subspecs. However here
    # often need to vectorize breakdown inside subspecs (e.g. bar plots with many
    # feedback components) so approach is e.g. name='ts' and forward=True or False.
    defaults = {'save': SAVE} if save else {}
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
        *figspecs, subspecs1, subspecs2, kwargs = create_specs(**kwargs)
        rowspecs = figspecs[0] if len(figspecs) > 0 else [{}]
        colspecs = figspecs[1] if len(figspecs) > 1 else [{}]
    name = kwargs.pop('name', None)
    results = []
    for rowspecs, kwargs in split_specs('row', rowspecs, **kwargs):
        if globe and not compare and not spread and subspecs1 == subspecs2:
            kws = (*subspecs1, *subspecs2, kwargs)
            projects = set(kw['project'] for kw in kws if 'project' in kw)
            institutes = set(kw['institute'] for kw in kws if 'institute' in kw)
            project = projects.pop() if len(projects) == 1 else None
            institute = institutes.pop() if len(institutes) == 1 else None
            kw_size = {**kwargs, 'project': project, 'institute': institute}
            refsize, altsize = _bar_size(**kw_size)
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
        for cspecs, kwargs in split_specs('col', colspecs, **kwargs):
            result = plotting.create_plot(data, rspecs, cspecs, **kwargs)
            results.append(result)
    result = results[0] if len(results) == 1 else results
    return result


def constraint_rows(data, method=None, contours=True, hatching=True, save=True, **kwargs):  # noqa: E501
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
    save : bool, optional
        Whether to save the result.
    rowsplit, colsplit : optional
        Passed to `split_specs`.
    **kwargs
        Passed to `create_specs`.

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
    rowspecs, *colspecs, subspecs1, subspecs2, kwargs = create_specs(maxcols=1, **kwargs)  # noqa: E501
    if len(subspecs1) != 1 or len(subspecs2) != 1:
        raise ValueError(f'Too many constraints {subspecs1} and {subspecs2}. Check outer argument.')  # noqa: E501
    (subspec1,), (subspec2,) = subspecs1, subspecs2
    defaults = {'save': SAVE} if save else {}
    kwargs = {**defaults, **kwargs}
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
    for rowspecs, kwargs in split_specs('row', rowspecs, **kwargs):
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
        for cspecs, kwargs in split_specs('col', cspecs, **kwargs):
            result = plotting.create_plot(data, rspecs, cspecs, **kwargs)
            results.append(result)
    result = results[0] if len(results) == 1 else results
    return result


def components_rows(data, method=None, save=True, **kwargs):
    """
    Plot average of constraint components and their relationship per row (e.g. maps).

    Parameters
    ----------
    data : xarray.Dataset
        The source dataset.
    method : bool, optional
        The two-variable method to use.
    save : bool, optional
        Whether to save the result.
    rowsplit, colsplit : optional
        Passed to `split_specs`.
    **kwargs
        Passed to `create_specs`.
    """
    # TODO: Update this. It is out of date with constraint_rows.
    if 'breakdown' not in kwargs and 'component' not in kwargs and 'outer' not in kwargs:  # noqa: E501
        raise RuntimeError
    rowspecs, colspecs1, colspecs2, kwargs = create_specs(maxcols=1, **kwargs)
    if len(colspecs1) != 1 or len(colspecs2) != 1:
        raise ValueError(f'Too many constraints {colspecs1} and {colspecs2}. Check outer argument.')  # noqa: E501
    defaults = {'save': SAVE} if save else {}
    kwargs = {**defaults, **kwargs}
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
    for rowspecs, kwargs in split_specs('row', rowspecs, **kwargs):
        rspecs = rowspecs  # possibly none
        colspecs = [
            [{**colspecs1[0], **spec} for spec in rplots],
            [{**colspecs2[0], **spec} for spec in rplots],
            [({**colspecs1[0], **spec}, {**colspecs2[1], **spec}) for spec in cplots]
        ]
        for cspecs, kwargs in split_specs('col', colspecs, **kwargs):
            # ic(rspecs, cspecs)
            result = plotting.create_plot(data, rspecs, cspecs, **kwargs)
            results.append(result)
    result = results[0] if len(results) == 1 else results
    return result


def summary_rows(data, method=None, shading=True, contours=True, save=True, **kwargs):
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
    save : bool, optional
        Whether to save the result.
    rowsplit, colsplit : optional
        Passed to `split_specs`.
    **kwargs
        Passed to `create_specs`.
    """
    # NOTE: For simplicity pass scalar 'outer' and other vectors are used in columns.
    if not isinstance(method := method or 'avg', str):
        method1, method2 = method
    elif method in ('avg', 'med'):
        method1, method2 = method, 'std'
    elif method in ('std', 'var', 'pctile'):
        method1, method2 = method, 'avg'
    else:
        raise ValueError(f'Invalid summary_rows() method {method!r}.')
    if 'breakdown' not in kwargs and 'component' not in kwargs and 'outer' not in kwargs:  # noqa: E501
        raise RuntimeError
    rowspecs, colspecs, *_, kwargs = create_specs(maxcols=1, **kwargs)
    defaults = {'save': SAVE} if save else {}
    kwargs = {**defaults, **kwargs}
    kw_shading = {'method': method1} if shading is True else dict(shading or {})
    kw_shading.update({key: kwargs.pop(key) for key in KEYS_SHADING if key in kwargs})
    kw_contour = {'method': method2} if contours is True else dict(contours or {})
    kw_contour.update({key: kwargs.pop(key) for key in KEYS_CONTOUR if key in kwargs})
    results = []
    for rowspecs, kwargs in split_specs('row', rowspecs, **kwargs):
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
        for cspecs, kwargs in split_specs('col', cspecs, **kwargs):
            # ic(rspecs, cspecs)
            result = plotting.create_plot(data, rspecs, cspecs, **kwargs)
            results.append(result)
    result = results[0] if len(results) == 1 else results
    return result
