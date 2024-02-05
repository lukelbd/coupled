#!/usr/bin/env python3
"""
Load coupled model climate data.
"""
import climopy as climo  # noqa: F401  # add accessor
import numpy as np
import xarray as xr
from icecream import ic  # noqa: F401
from climopy import const, ureg, vreg  # noqa: F401

from .specs import _pop_kwargs, FACETS_LEVELS, FACETS_RENAME, MODELS_EXCLUDE
from cmip_data.facets import Database, glob_files, _item_dates
from cmip_data.utils import assign_dates, load_file
from cmip_data.climate import _add_energetics, _add_hydrology, _add_transport  # noqa: E501


__all__ = ['climate_datasets']

# Unit short label constants
# NOTE: These entries inform the translation from standard unit strings to short
# names like 'energy flux' and 'energy transport' used in figure functions. In
# future should group all of these into cfvariables with standard units.
UNIT_LABELS = {
    'K': 'temperature',
    'hPa': 'pressure',
    'dam': 'surface height',
    'mm': 'liquid depth',
    'mm day^-1': 'accumulation',
    'm s^-1': 'wind speed',
    'Pa': 'wind stress',  # default tau units
    'g kg^-1': 'concentration',
    'W m^-2': 'flux',
    'PW': 'transport',
}

# Unit conversion constants
# NOTE: For now use the standard 1e3 kg/m3 water density (i.e. snow and ice terms
# represent melted equivalent depth) but could also use 1e2 kg/m3 snow density where
# relevant. See: https://www.sciencelearn.org.nz/resources/1391-snow-and-ice-density
UNIT_SCALINGS = {  # scaling prior to final unit transformation
    'prw': 1 / const.rhow,  # water vapor path not precip
    'pr': 1 / const.rhow,
    'prl': 1 / const.rhow,  # derived
    'pri': 1 / const.rhow,
    'ev': 1 / const.rhow,
    'evl': 1 / const.rhow,  # derived
    'evi': 1 / const.rhow,
    'clwvi': 1 / const.rhow,
    'cllvi': 1 / const.rhow,  # derived
    'clivi': 1 / const.rhow,
}
UNIT_VARIABLES = {
    'K': ('ta', 'ts'),
    'hPa': ('pbot', 'ptop', 'psl', 'ps'),
    'dam': ('zg',),
    'mm': ('prw', 'clwvi', 'cllvi', 'clivi'),
    'mm day^-1': ('pr', 'prl', 'pri', 'ev', 'evl', 'evi'),  # translated
    'm s^-1': ('ua', 'va', 'uas', 'vas'),
    'Pa': ('tauu', 'tauv'),
    'g kg^-1': ('hus', 'huss', 'clw', 'cll', 'cli'),
    'W m^-2': ('hfls', 'hfss'),
    'PW': ('intuadse', 'intvadse', 'intuaw', 'intvaw'),
}


def _update_units(dataset):
    """
    Convert dataset units into human-readable form, possibly
    multiplying by constants, and add a unit-based short name.

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset.
    """
    # NOTE: For converting snow precipitation we assume a simple 10:1 snow:liquid
    # ratio. See: https://www.sciencelearn.org.nz/resources/1391-snow-and-ice-density
    for unit, variables in UNIT_VARIABLES.items():
        for variable in variables:
            if variable not in dataset:
                continue
            scale = UNIT_SCALINGS.get(variable, 1.0)
            data = dataset[variable]
            if ureg.parse_units(unit) == data.climo.units:
                continue
            if quantify := not data.climo._is_quantity:
                with xr.set_options(keep_attrs=True):
                    data = scale * data.climo.quantify()
            data = data.climo.to_units(unit)
            if quantify:
                with xr.set_options(keep_attrs=True):
                    data = data.climo.dequantify()
            dataset[variable] = data
    for source in (dataset.data_vars, dataset.coords):
        for variable, data in source.items():
            long = data.attrs.get('long_name', None)
            unit = data.attrs.get('units', None)
            short = UNIT_LABELS.get(unit, long)  # default long name, e.g. longitude
            if variable == 'time':  # possibly missing long name
                short = 'time'
            if short is not None:
                data.attrs['short_name'] = short
    return dataset


def climate_datasets(
    *paths, years=None, anomaly=True, average=False,
    ignore=None, nodrift=False, standardize=True, **constraints
):
    """
    Return a dictionary of datasets containing processed files.

    Parameters
    ----------
    *paths : path-like
        The search paths.
    years : 2-tuple of int
        The climate year range.
    anomaly : bool, optional
        Whether to load response data in anomaly form.
    average : bool, optional
        Whether to average across the time dimension.
    ignore : str or sequence, optional
        The variables to optionally ignore.
    nodrift : bool, optional
        Whether to use drift corrections.
    standardize : bool, optional
        Whether to standardize the resulting order.
    **indexers
        Optional indexers.
    **constraints
        Passed to `Database`.

    Returns
    -------
    datasets : dict
        A dictionary of datasets.
    """
    # Initial stuff
    # TODO: Support regressions of each variable onto global temperature to go
    # along with ratio-style climate responses. See notes under VERSION_LEVELS.
    # NOTE: Non-flagship simulations can be added with flagship_translate=True
    # as with other processing functions. Ensembles are added in a MultiIndex
    from .datasets import _standardize_order
    kw_energetics = _pop_kwargs(constraints, _add_energetics)
    kw_transport = _pop_kwargs(constraints, _add_transport)
    kw_hydrology = _pop_kwargs(constraints, _add_hydrology)
    files, *_ = glob_files(*paths, project=constraints.get('project', None))
    constraints.setdefault('table', ['Amon', 'Emon'])
    database = Database(files, FACETS_LEVELS, flagship_translate=True, **constraints)
    database.filter(always_exclude={'variable': ['pfull']})  # skip dependencies
    nodrift = nodrift and '-nodrift' or ''
    datasets = {}
    print(f'Climate files: <dates>-climate{nodrift}')
    print(f'Number of climate file groups: {len(database)}.')

    # Open datasets for concatenation
    # WARNING: Critical to place time averaging after transport calculations so that
    # time-covariance of surface pressure and near-surface flux terms is factored
    # in (otherwise would need to include cell heights before averaging).
    if ignore is None:  # default value
        ignore = ('huss', 'hurs', 'uas', 'vas')
    if database:
        print('Model:', end=' ')
    for facets, data in database.items():
        # Load the data
        # NOTE: Critical to overwrite the time coordinates after loading or else xarray
        # coordinate matching will apply all-NaN values for climatolgoies with different
        # base years (e.g. due to control data availability or response calendar diffs).
        for sub, replace in FACETS_RENAME.items():
            facets = tuple(facet.replace(sub, replace) for facet in facets)
        if not data or facets[2] in MODELS_EXCLUDE:
            continue
        range_ = (120, 150) if facets[3] == 'abrupt4xco2' else (0, 150)
        range_ = years if years is not None else range_
        dates = f'{range_[0]:04d}-{range_[1]:04d}-climate{nodrift}'
        print(f'{facets[2]}_{facets[3]}_{range_[0]:04d}-{range_[1]:04d}', end=' ')
        dataset = xr.Dataset()
        for key, paths in data.items():
            variable = key[database.key.index('variable')]
            paths = [path for path in paths if _item_dates(path) == dates]
            if not paths or ignore and variable in ignore:
                continue
            if len(paths) > 1:
                print(f'Warning: Skipping ambiguous duplicate paths {list(map(str, paths))}.', end=' ')  # noqa: E501
                continue
            array = load_file(paths[0], variable, project=facets[0], validate=True)
            descrip = array.attrs.pop('title', variable)  # in case long_name missing
            descrip = array.attrs.pop('long_name', descrip)
            descrip = ' '.join(s if s == 'TOA' else s.lower() for s in descrip.split())
            array.attrs['long_name'] = descrip
            if 'time' in array.sizes:
                array = assign_dates(array, year=1800)  # exactly 12 months required
                if average:
                    days = array.time.dt.days_in_month.astype(np.float32)
                    array = array.weighted(days).mean('time', skipna=False, keep_attrs=True)  # noqa: E501
            dataset[variable] = array

        # Standardize the data
        # NOTE: Empirical testing revealed limiting integration to troposphere
        # often prevented strong transient heat transport showing up in overturning
        # cells due to aliasing of overemphasized stratospheric geopotential transport.
        if 'ps' not in dataset and 'plev' in dataset.coords:
            print('Warning: Surface pressure is unavailable.', end=' ')
        dataset = dataset.climo.add_cell_measures(surface=('ps' in dataset))
        dataset = _add_energetics(dataset, **kw_energetics)  # before transport  # noqa: E501
        dataset = _add_transport(dataset, **kw_transport)  # transport
        dataset = _add_hydrology(dataset, **kw_hydrology)  # after transport
        dataset = _update_units(dataset)  # after transport
        drop = ['cell_', '_bot', '_top']
        drop = [key for key in dataset.coords if any(o in key for o in drop)]
        dataset = dataset.drop_vars(drop).squeeze()
        if 'plev' in dataset:
            dataset = dataset.sel(plev=slice(None, 7000))
        if standardize:
            dataset = _standardize_order(dataset)
        datasets[facets] = dataset

    # Translate abrupt 4xCO2 datasets into anomaly form
    # TODO: Should still somehow support 'response' climopy and 'process.py' variable
    # suffix for specifying abrupt4xCO2 experiment instead of picontrol experiment.
    # Would require detecting anomalies are present and rebuilding original data.
    if datasets:
        print()
    if anomaly:
        print('Transforming abrupt 4xCO2 data into anomalies.')
        for facets, dataset in tuple(datasets.items()):
            control = (*facets[:3], 'picontrol', *facets[4:])
            if facets[3] != 'abrupt4xco2':
                continue
            if control not in datasets and datasets.pop(facets):  # one line
                continue
            climate = datasets[control]
            for name, data in dataset.data_vars.items():
                if name[:4] in ('ps', 'cell'):
                    continue
                if name not in climate:
                    dataset = dataset.drop_vars(name)
                else:  # WARNING: use .data to avoid subtracting coords
                    data.data -= climate[name].data
                    if name == 'ts':
                        data.attrs['long_name'] = 'surface warming'
                        data.attrs['short_name'] = 'warming'
                    else:
                        if 'long_name' in data.attrs:
                            data.attrs['long_name'] += ' response'
                        if 'short_name' in data.attrs:
                            data.attrs['short_name'] += ' response'
            datasets[facets] = dataset
    return datasets
