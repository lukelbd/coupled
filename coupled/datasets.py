#!/usr/bin/env python3
"""
Load and combine coupled model climate and feedback data.
"""
# TODO: Should create climate feedback files before refactoring 'facets' and 'version'
# coordinates since that is primary benefit, otherwise datasets are mostly empty when
# combining climate and feedback data on 'parameters'. First move coupled/climate.py
# functions to cmip_data/climate.py and save time series of core linearly-additive
# components (e.g. dse and lse instead of mse). Then generate circulation 'feedback'
# files by regressing against temperature for different time periods. Finally add
# derivations to get_result as with flux variables (e.g. mse = dse + lse).
# TODO: Should have 'facets' coordinate with project / institute / model / ensemble
# and 'parameters' coordinate with source / experiment / period / style / region.
# For circulation data, will have 'ratio' style for year 120-150 abrupt 4xCO2 changes
# normalized by temperature while 'monthly' and 'annual' are pre-industrial or abrupt
# 4xCO2 regressions against temperature (as with feedback calculations). Will rename
# 'source' on both to e.g. simply 'internal' or 'external' since 'eraint' source is not
# meaningful for climate data (or can add as distinct feedback-only coordinate) while
# 'period' and 'region' indicate integration or averaging periods and temperature
# normalization settings. All changes normalized by temperature will have 'lam' suffix
# while 'erf' suffix indicates rapid adjustments and unperturbed climatology will be
# stored without suffix or 'parameters' coordinate.
# TODO: Auto-construct 'ratio' style changes normalized by global temperature and use
# get_result() to build e.g. unnormalized 4xCO2 changes over years 120-150 using 'del'
# suffix, or absolute years 120-150 average using 'abs' suffix (equivalent to 'del'
# plus climatology). Would skip radiative flux data, since these are saved as special
# case with 'forcing' subtracted from the numerator (when subequently constructing
# budgets or breakdowns from radiative flux would simply get net change from 'del' plus
# 'erf', and note 'erf' will be present under 'ratio' style even though feedbacks.py
# simply sets this to the 'erf' from one of the regression estimates). The data for
# unnormalizing would be stored under 'ratio' style 'tstd' variable, since ratio-
# equivalent of scaling 'tpat' by global temperature standard deviation is to scale
# the ratio of local-to-global surface temperature change by the global change again.
import itertools
from pathlib import Path

import climopy as climo  # noqa: F401  # add accessor
import numpy as np
import pandas as pd
import xarray as xr
from icecream import ic  # noqa: F401

from cmip_data.climate import HYDRO_COMPONENTS
from cmip_data.climate import _add_energetics, _add_transport, _add_hydrology
from .climate import climate_datasets
from .feedbacks import _update_terms, feedback_datasets, feedback_jsons, feedback_texts
from .specs import _pop_kwargs

__all__ = ['open_scalar', 'open_dataset']

# Facet settings
FACETS_NAME = 'facets settings'
FACETS_LEVELS = (
    'project',
    'institute',  # auto-constructed institute index
    'model',
    'experiment',
    'ensemble',
)

# Version settings
VERSION_NAME = 'eeedback settings'
VERSION_LEVELS = (
    'source',
    'style',
    'start',  # initial year of regression or 'forced' climate average
    'stop',  # final year of regression or 'forced' climate average
    'region',
)

# Other options
FACETS_EXCLUDE = (
    'MCM-UA-1-0',
    'FIO-ESM-2-0',
    'IPSL-CM6A-LR-INCA',
)
FACETS_RENAME = {
    'piControl': 'picontrol',  # or 'control' but too general
    'control-1950': 'control1950',
    'abrupt4xCO2': 'abrupt4xco2',  # or 'response' but sounds like perturbation
    'abrupt-4xCO2': 'abrupt4xco2',
}


def _standardize_order(dataset):
    """
    Standardize insertion order of dataset for user convenience.

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset.

    Returns
    -------
    dataset : xarray.Dataset
        The ordered dataset.
    """
    # Create automatic variable lists
    eddies = ('', 'm', 's', 't')  # total, zonal-mean, stationary, transient
    statics = ('gse', 'hse', 'dse', 'lse', 'mse', 'ocean', 'total')
    locals = ('t', 'f', 'c', 'r')  # transport, flux, convergence, residual
    ends = ('', '_alt', '_exp')
    params = ('', 'lam', 'rho', 'kap', 'erf', 'ecs', 'tcr')
    parts = ('', 'pl', 'pl*', 'lr', 'lr*', 'hus', 'hur', 'atm', 'alb', 'cl', 'ncl', 'resid')  # noqa: E501
    moisture = list(name for names in HYDRO_COMPONENTS for name in names[:4])
    fluxes = list(itertools.product(('f', 'l', 's'), ('t', 's', 'a'), ('', 'cs', 'ce')))
    fluxes = list(f'r{wav}n{bnd}{sky}' for wav, bnd, sky in fluxes)  # capture fluxes
    transports = list(itertools.product(eddies, statics, locals, ends))
    transports = list(''.join(tup) for tup in transports)
    feedbacks = list(itertools.product(parts, fluxes, params))
    feedbacks = list('_'.join(tup).strip('_') for tup in feedbacks)
    # Update insertion order
    basics = ('ta', 'ts', 'tpat', 'tstd', 'tdev', 'tabs', 'ps', 'psl', 'pbot', 'ptop')
    circulation = ('zg', 'ua', 'va', 'uas', 'vas', 'tauu', 'tauv')
    humidity = ('hus', 'hur', 'huss', 'hurs', 'prw', 'cl', 'clt', 'cct')
    surface = ('albedo', 'hfls', 'hfss')
    names = []
    names.extend(basics)
    names.extend(circulation)
    names.extend(humidity)
    names.extend(moisture)
    names.extend(surface)
    names.extend(fluxes)
    names.extend(transports)
    names.extend(feedbacks)
    unknowns = [name for name in dataset.data_vars if name not in names]
    if unknowns:
        print('Warning: Unknown order for variables:', ', '.join(unknowns))
    results = {name: dataset[name] for name in (*names, *unknowns) if name in dataset}
    dataset = xr.Dataset(results)
    return dataset


def open_scalar(path=None, ceres=False):
    """
    Get the observational constraint estimate.

    Parameters
    ----------
    path : path-like, optional
        The source feedback path.
    ceres : bool, optional
        Whether to load global CERES or CMIP feedbacks.
    """
    source = 'CERES' if ceres else 'CMIP'
    base = Path('~/data/global-feedbacks').expanduser()
    file = f'feedbacks_{source}_global.nc'
    if isinstance(path, str) and '/' not in path:
        path = base / path
    elif path:
        path = Path(path).expanduser()
    if not path:
        path = base / file
    elif not path.suffix:
        path = path / file
    data = xr.open_dataset(path)
    names = [key for key, coord in data.coords.items() if coord.dims == ('facets',)]
    if names:  # facet levels
        data = data.set_index(facets=names)
    names = [key for key, coord in data.coords.items() if coord.dims == ('version',)]
    if names:  # version levels
        data = data.set_index(version=names)
    return data


def open_dataset(
    project=None,
    climate=True,
    feedbacks=True,
    feedbacks_json=False,
    feedbacks_text=False,
    standardize=True,
    **constraints,
):
    """
    Load climate and feedback data into a single dataset.

    Parameters
    ----------
    project : sequence, optional
        The project(s) to use.
    climate : bool or path-like, optional
        Whether to load processed climate data. Default path is ``~/data``.
    feedbacks : bool or path-like, optional
        Whether to load processed feedback files. Default path is ``~/data``.
    feedbacks_json, feedbacks_text : bool or path-like, optional
        Whether to load external feedback files. Default path is ``~/data/cmip-tables``.
    standardize : bool, optional
        Whether to standardize the resulting order.
    **kwargs
        Passed to relevant functions.
    **constraints
        Passed to constrain the results.
    """
    # Initial stuff
    # TODO: Support loading with .open_mfdataset() or else .open_dataset() followed by
    # .concat() that permits lazy loading of component variables. See utils 'load_file'
    # NOTE: Here 'source' refers to either the author of a cmip-tables file or the
    # creator of the kernels used to build custom feedbacks, and 'region' always refers
    # to the denominator. For external feedbacks, the region is always 'globe' and its
    # value is constant in longitude and latitude, while for internal feedbacks, the
    # values vary in longitude and latitude, and global feedbacks are generated by
    # taking the average (see notes -- also tested outdated feedback files directly).
    # So can compare e.g. internal and external feedbacks with ``region='globe'`` and
    # ``area='avg'`` -- this is a no-op for the spatially uniform external feedbacks.
    climate_funcs = (_add_energetics, _add_transport, _add_hydrology)
    kw_datasets = _pop_kwargs(constraints, 'nodrift', 'average')  # dataset functions
    kw_climate = _pop_kwargs(constraints, climate_datasets, *climate_funcs)
    kw_feedback = _pop_kwargs(constraints, feedback_datasets)
    kw_terms = _pop_kwargs(constraints, _update_terms)
    kw_jsons = _pop_kwargs(constraints, feedback_jsons)
    kw_texts = _pop_kwargs(constraints, feedback_texts)
    kw_climate = {**kw_datasets, **kw_climate}  # climate_datasets
    kw_feedback = {**kw_datasets, **kw_feedback, **kw_terms}  # feedback_datasets()
    kw_jsons = {**kw_jsons, **kw_terms}  # feedback_jsons()
    kw_texts = {**kw_texts, **kw_terms}  # feedback_texts()
    dirs_table = ('cmip-tables',)
    dirs_climate = ('cmip-climate',)
    dirs_feedback = ('cmip-feedbacks', 'ceres-feedbacks')
    bases = ('~/data', '~/scratch')
    datasets = {}
    projects = project.split(',') if isinstance(project, str) else ('cmip5', 'cmip6', 'ceres')  # noqa: E501

    # Open the datasets
    # WARNING: Using dataset.update() instead of xr.combine_by_coords() below can
    # result in silently replacing existing data with NaNs (verified with test). The
    # latter is required when adding new 'facets' and 'version' coordinate values.
    for project in map(str.upper, projects):
        print(f'Project: {project}')
        for b, function, dirs, kw in (
            (climate, climate_datasets, dirs_climate, kw_climate),
            (feedbacks, feedback_datasets, dirs_feedback, kw_feedback),
            (feedbacks_json, feedback_jsons, dirs_table, kw_jsons),
            (feedbacks_text, feedback_texts, dirs_table, kw_texts),
        ):
            if not b:
                continue
            if isinstance(b, (str, Path)):
                paths = (Path(b).expanduser(),)
            elif isinstance(b, (tuple, list)):
                paths = tuple(Path(_).expanduser() for _ in b)
            else:
                paths = tuple(Path(_).expanduser() / d for _ in bases for d in dirs)
            kwargs = {**constraints, 'project': project, 'standardize': False, **kw}
            parts = function(*paths, **kwargs)
            for facets, dataset in parts.items():
                if facets in datasets:  # e.g. combine 'version' coordinates
                    comb = (datasets[facets], dataset)
                    dataset = xr.combine_by_coords(comb, combine_attrs='override')
                datasets[facets] = dataset

    # Concatenate and standardize datasets
    # NOTE: Critical to use 'override' for combine_attrs in case models
    # use different naming conventions for identical variables.
    names = {name: da for ds in datasets.values() for name, da in ds.data_vars.items()}
    print('Adding missing variables.')
    if datasets:
        print('Model:', end=' ')
    for facets, dataset in tuple(datasets.items()):  # interpolated datasets
        print('_'.join(facets[1:4]), end=' ')
        for name in names.keys() - dataset.data_vars.keys():
            array = names[name]  # *sample* from another model or project
            array = xr.full_like(array, np.nan)  # preserve attributes as well
            if all('version' in keys for keys in (array.dims, dataset, dataset.sizes)):
                array = array.isel(version=0, drop=True)
                array = array.expand_dims(version=dataset.version.size)
                array = array.assign_coords(version=dataset.version)
            dataset[name] = array
    print()
    print('Concatenating datasets.')
    if not datasets:
        raise ValueError('No datasets found.')
    facets = xr.DataArray(
        pd.MultiIndex.from_tuples(datasets, names=FACETS_LEVELS),
        dims='facets',
        name='facets',
        attrs={'long_name': FACETS_NAME},
    )
    dataset = xr.concat(
        datasets.values(),
        dim=facets,
        coords='minimal',
        compat='override',
        combine_attrs='override',
    )
    print('Standardizing result.')
    if 'version' in dataset.dims:
        dataset = dataset.transpose('version', ...)
    if standardize:
        dataset = _standardize_order(dataset)
    dataset = dataset.climo.standardize_coords(prefix_levels=True)
    dataset = dataset.climo.add_cell_measures(surface=('ps' in dataset))
    if 'plev_bot' in dataset:  # created by add_cell_measures()
        dataset = dataset.drop_vars('plev_bot')
    return dataset
