#!/usr/bin/env python3
"""
Utilities for loading coupled model output.
"""
import itertools
import json
import re
from pathlib import Path

import climopy as climo  # noqa: F401  # add accessor
import numpy as np
import pandas as pd
import xarray as xr
from climopy import const, ureg
from icecream import ic  # noqa: F401
from metpy import calc, units

from cmip_data.internals import FLAGSHIP_ENSEMBLES
from cmip_data.internals import _glob_files, _item_dates, _parse_constraints
from cmip_data import Database, open_file, time_averages

__all__ = [
    'open_all',
    'open_climate',
    'open_feedbacks',
    'open_json',
    'open_text',
]


# Feedback definitions
# NOTE: These are used to both translate tables from external sources into the more
# descriptive naming convention, and to translate inputs to plotting functions for
# convenience (default for each shorthand is to use combined longwave + shortwave toa).
FEEDBACK_DEFINITIONS = {
    'ecs': ('rfnt_ecs2x', 'K'),  # zelinka definition
    'tcr': ('tcr_ecs2x', 'K'),  # forster definition
    'f2x': ('rfnt_erf2x', 'W m^-2'),  # forster definition
    'f4x': ('rfnt_erf2x', 'W m^-2'),  # geoffroy definition
    'erf': ('rfnt_erf2x', 'W m^-2'),
    'erf2x': ('rfnt_erf2x', 'W m^-2'),  # zelinka definition
    'erf4x': ('rfnt_erf2x', 'W m^-2'),  # for consistency only
    'net': ('rfnt_lambda', 'W m^-2 K^-1'),
    'rho': ('tcr_lambda', 'W m^-2 K^-1'),  # forster definition
    'kap': ('ohc_lambda', 'W m^-2 K^-1'),  # forster definition
    'cs': ('rfntcs_lambda', 'W m^-2 K^-1'),
    'swcs': ('rsntcs_lambda', 'W m^-2 K^-1'),  # forster definition
    'lwcs': ('rlntcs_lambda', 'W m^-2 K^-1'),  # forster definition
    'cre': ('rfntce_lambda', 'W m^-2 K^-1'),  # forster definition
    'swcre': ('rsntce_lambda', 'W m^-2 K^-1'),
    'lwcre': ('rlntce_lambda', 'W m^-2 K^-1'),
    'pl': ('pl_rfnt_lambda', 'W m^-2 K^-1'),  # zelinka definition
    'pl*': ('pl*_rfnt_lambda', 'W m^-2 K^-1'),  # zelinka definition
    'lr': ('lr_rfnt_lambda', 'W m^-2 K^-1'),  # zelinka definition
    'lr*': ('lr*_rfnt_lambda', 'W m^-2 K^-1'),  # zelinka definition
    'wv': ('hus_rfnt_lambda', 'W m^-2 K^-1'),  # zelinka definition
    'rh': ('hur_rfnt_lambda', 'W m^-2 K^-1'),  # zelinka definition
    'alb': ('alb_rfnt_lambda', 'W m^-2 K^-1'),  # zelinka definition (full is 'albedo')
    'cld': ('cl_rfnt_lambda', 'W m^-2 K^-1'),  # zelinka definition
    'swcld': ('cl_rsnt_lambda', 'W m^-2 K^-1'),  # zelinka definition
    'lwcld': ('cl_rlnt_lambda', 'W m^-2 K^-1'),  # zelinka definition
    'resid': ('resid_rfnt_lambda', 'W m^-2 K^-1'),
    'err': ('resid_rfnt_lambda', 'W m^-2 K^-1'),  # forster definition
}

# Variable components
# NOTE: Consult Donohoe et al. for details on transport terms. Precipitation appears
# in the dry static energy formula because unlike surface evaporation, this process
# deposits heat inside the atmosphere, i.e. it remains in the energy budget after
# subtracting net surface and top-of-atmosphere loss terms.
# NOTE: Consult Duffy et al. for the snow correction... basically turbulent latent heat
# flux is always calculated under the assumption of water-to-vapor transition so cmip
# terms neglect the additional energy required for snow melting or sublimation. The
# adjustment accounts for atmospheric energy required to melt the snow. Alternatively
# can use separate sublimation and evaporation components (see _add_transport_terms).
COMPONENTS_RADIATION = {
    'rlnt': ('rlut',),  # out of the atmosphere
    'rsnt': ('rsut', 'rsdt'),  # out of the atmosphere (include constant rsdt)
    'rlntcs': ('rlutcs',),  # out of the atmosphere
    'rsntcs': ('rsutcs', 'rsdt'),  # out of the atmosphere (include constant rsdt)
    'rlns': ('rlds', 'rlus'),  # out of and into the atmosphere
    'rsns': ('rsds', 'rsus'),  # out of and into the atmosphere
    'rlnscs': ('rldscs', 'rlus'),  # out of and into the atmosphere
    'rsnscs': ('rsdscs', 'rsuscs'),  # out of and into the atmosphere
    'albedo': ('rsds', 'rsus'),  # full name to differentiate from 'alb' feedback
}
COMPONENTS_TRANSPORT = {
    'total': (('rlnt', 'rsnt'), ()),
    'ocean': ((), ('hfss', 'hfls', 'rlns', 'rsns')),  # flux into atmosphere
    'moist static': (('hfss', 'hfls', 'rlns', 'rsns', 'rlnt', 'rsnt', 'prsf'), ()),
    'latent static': (('hfls',), ('pr', 'prsn')),
    'dry static': (('hfss', 'rlns', 'rsns', 'rlnt', 'rsnt', 'pr', 'prsn'), ()),
}

# Variable unit conversions
# NOTE: For now use the standard 1e3 kg/m3 water density (i.e. snow and ice terms
# represent melted equivalent depth) but could also use 1e2 kg/m3 snow density where
# relevant. See: https://www.sciencelearn.org.nz/resources/1391-snow-and-ice-density
CONVERSIONS_TRANSPORT = {
    'pr': const.Lv,
    'prra': const.Lv,
    'prsn': const.Ls - const.Lv,  # remove the 'prsn * Lv' implied inside 'pr' term
    'evspsbl': const.Lv,
    'evsp': const.Lv,
    'sbl': const.Ls - const.Lv,  # remove the 'evspsbl * Lv' implied inside 'pr' term
}
CONVERSIONS_STANDARD = {
    'clw': (1, 'g kg^-1'),
    'cli': (1, 'g kg^-1'),
    'cll': (1, 'g kg^-1'),
    'hus': (1, 'g kg^-1'),
    'huss': (1, 'g kg^-1'),
    'psl': (1, 'hPa'),
    'ps': (1, 'hPa'),
    'zg': (1, 'dam'),
    'clivi': (1 / const.rhow, 'mm'),
    'clwvi': (1 / const.rhow, 'mm'),
    'cllvi': (1 / const.rhow, 'mm'),
    'prw': (1 / const.rhow, 'mm'),  # water vapor path
    'pr': (1 / const.rhow, 'mm day^-1'),
    'prra': (1 / const.rhow, 'mm day^-1'),
    'prsn': (1 / const.rhow, 'mm day^-1'),
    'evspsbl': (1 / const.rhow, 'mm day^-1'),
    'evsp': (1 / const.rhow, 'mm day^-1'),
    'sbl': (1 / const.rhow, 'mm day^-1'),
}


def _add_humidity_terms(dataset):
    """
    Add relative humidity to the dataset.

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset.
    """
    # NOTE: Generally forego downloading relative humidity variables... true that
    # operation is non-linear so relative humidity of climate is not climate of
    # relative humiditiy, but we already use this approach with feedback kernels.
    humids = {'hur': ('plev', 'ta', 'hus'), 'hurs': ('ps', 'ts', 'huss')}
    for name, keys in humids.items():
        if all(key in dataset for key in keys):
            datas = []
            for key, unit in zip(keys, ('Pa', 'K', '')):
                data = dataset.climo[key].climo.to_units(unit)
                data.data = data.data.magnitude * units.units(unit)
                datas.append(data)
            descrip = 'near-surface ' if name[-1] == 's' else ''
            data = calc.relative_humidity_from_specific_humidity(*datas)
            data = data.climo.dequantify()  # works with metpy registry
            data = data.climo.to_units('%').clip(0, 100)
            data.attrs['long_name'] = f'{descrip}relative humidity'
            dataset[name] = data
    return dataset


def _add_radiation_terms(dataset):
    """
    Add albedo and net fluxes from upwelling and downwelling components.

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset.
    """
    regex = re.compile(r'(upwelling|downwelling|outgoing|incident)')
    for name, keys in COMPONENTS_RADIATION.items():
        if all(key in dataset for key in keys):  # skip partial data
            if name == 'albedo':
                dataset[name] = 100 * dataset[keys[1]] / dataset[keys[0]]
                long_name = 'albedo'
                unit = '%'
            elif len(keys) == 2:
                dataset[name] = dataset[keys[1]] - dataset[keys[0]]
                long_name = dataset[keys[0]]
                unit = 'W m^-2'
            else:
                dataset[name] = -1 * dataset[keys[0]]
                long_name = dataset[keys[0]]
                unit = 'W m^-2'
            long_name = long_name.replace('radiation', 'flux')
            long_name = regex.sub('net', long_name)
            dataset[name].attrs.update({'units': unit, 'long_name': long_name})
    return dataset


def _add_transport_terms(dataset):
    """
    Add zonally-integrated meridional transport and pointwise
    transport convergence to the dataset.

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset.
    """
    # NOTE: Not necessary to weight by atmosphere thickness because we are
    # working in energy units, so the columnwise heat capacity factor affected
    # by spatially varying surface pressure is implicitly factored in.
    suffixes = ('', '_alt')
    transports = {}
    for suffix, descrip in itertools.product(suffixes, COMPONENTS_TRANSPORT):
        constants = {}
        plus, minus = map(list, COMPONENTS_TRANSPORT[descrip])
        for i, names in zip((1, -1), (plus, minus)):
            conversions = CONVERSIONS_TRANSPORT.copy()
            if 'hfls' in names:
                idx = names.index('hfls')
                names[idx:idx + 1] = ('evspsbl', 'sbl') if suffix else ('hfls', 'prsn')
                conversions['prsn'] = -const.Lf  # unused if suffix is true
            constants.update({n: i * conversions.get(n, 1) for n in names})
        if any(n not in dataset for names in (plus, minus) for n in names):
            continue
        resid = sum(c * dataset.climo.get(n, add_cell_measures=False) for n, c in constants.items())  # noqa: E501
        data = -1 * resid.climo.to_units('W m^-2')  # negative of residual
        data.attrs['long_name'] = f'{descrip} energy transport convergence'
        data.name = f'c{descrip[0]}sef' if 'static' in descrip else f'c{descrip}f'
        dataset[data.name] = transports[data.name] = data.climo.dequantify()
    for name, data in transports.items():
        long_name = data.long_name.replace(' convergence', '')
        data = data.climo.add_cell_measures().climo.average('lon')
        data = data.climo.integral('lon')  # total energy convergence
        data = data.climo.cumintegral('lat', reverse=True)  # northward transport
        data = data.climo.to_units('PW')
        data = data.drop_vars(data.coords.keys() - data.sizes.keys())
        data.attrs['long_name'] = long_name
        data.name = name[1:]
        dataset[name] = data
    return dataset


def _adjust_moisture_terms(dataset):
    """
    Add ice and liquid water terms.

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset.
    """
    if 'clw' in dataset:  # unlike related variables, this is only liquid component
        if 'cli' not in dataset:
            dataset = dataset.rename_vars(clw='cll')
        else:
            with xr.set_options(keep_attrs=True):
                dataset['clw'] = dataset['cli'] + dataset['clw']
    for (ice, both, liquid, ratio, descrip) in (
        ('cli', 'clw', 'cll', 'clp', 'mass fraction cloud %s'),
        ('clivi', 'clwvi', 'cllvi', 'clpvi', 'condensed %s water path'),
        ('prsn', 'pr', 'prrn', 'prp', '%s precipitation'),
        ('sbl', 'evspsbl', 'evsp', 'sblp', '%s evaporation'),
    ):
        if ice in dataset and both in dataset:  # note the clw variables include ice
            da = (100 * dataset[ice] / dataset[both]).clip(0, 100)
            da.attrs = {'units': '%', 'long_name': descrip % 'ice' + ' ratio'}
            dataset[ratio] = da
            da = dataset[both] - dataset[ice]
            da.attrs = {'units': dataset[both].attrs['units']}
            dataset[liquid] = da
        for name, string in zip((ice, both, liquid), ('ice', 'water', 'liquid')):
            if name in dataset and string not in descrip:
                dataset[name].attrs['long_name'] = descrip % string
    return dataset


def _adjust_standard_units(dataset):
    """
    Convert dataset units into a human-readable form.

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset.
    """
    # NOTE: For converting snow precipitation we assume a simple 10:1 snow:liquid
    # ratio. See: https://www.sciencelearn.org.nz/resources/1391-snow-and-ice-density
    for variable, (constant, unit) in CONVERSIONS_STANDARD.items():
        if variable not in dataset:
            continue
        with xr.set_options(keep_attrs=True):
            da = dataset[variable]
            if quantify := not da.climo._is_quantity:
                da = constant * da.climo.quantify()
            da = da.climo.to_units(unit)
            if quantify:
                da = da.climo.dequantify()
            dataset[variable] = da
    return dataset


def _adjust_feedback_attrs(dataset, boundary=None):
    """
    Adjust feedback term attributes for old files.

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset.
    """
    # NOTE: Here replace outdated feedback names with shorter names suitable
    # for row and column figure plot labels. Should remove eventually.
    boundary = boundary or 't'
    replacements = [
        ('', 'net', 'net ', ''),
        ('', 'radiative', 'radiative ', ''),  # effective radiative forcing
        ('', 'climate', 'climate ', ''),  # effective climate sensitivity
        ('', 'parameter', ' parameter', ''),  # radiative feedback parameter
        ('', 'traditional', 'traditional ', ''),  # traditional lapse rate
        ('*', 'preserving', 'relative humidity-preserving', 'adjusted'),
        ('cl_', 'adjusted', 'adjusted cloud forcing', 'cloud'),
        ('res', 'kernel', 'kernel residual', 'residual'),
        ('rsn', 'shortwave', r'(\S+\s\S+)', r'\1 shortwave'),
        ('rln', 'longwave', r'(\S+\s\S+)', r'\1 longwave'),
    ]
    if len(''.join(boundary)) == 1:
        for bound in ('TOA', 'surface', 'atmosphere'):
            replacements.append(('', bound, f'{bound} ', ''))
    for key, da in tuple(dataset.data_vars.items()):
        if 'units' not in da.attrs and key in ('plev_bot', 'plev_top'):
            da.attrs['units'] = 'Pa'
        for part, test, reg, sub in replacements:
            if test in sub:
                change = test not in da.long_name  # i.e. 'shortwave' stuff
            else:
                change = test in da.long_name
            if part in key and change:
                da.attrs['long_name'] = re.sub(reg, sub, da.long_name)
        if da.climo.units == ureg.Pa:  # default standard units
            da.attrs['standard_units'] = 'hPa'
    return dataset


def _adjust_feedback_terms(
    dataset,
    boundary=None,
    ecsparts=None,
    erfparts=None,
    wavparts=None,
):
    """
    Add net cloud effect and net atmospheric feedback terms, and possibly
    filter out unnecessary terms depending on input arguments.

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset.
    boundary : {'t', 's', 'a'}, optional
        The boundari(es) to load. Pass a tuple or string of a combination for multiple.
    erfparts : bool, optional
        Whether to include the kernel-derived effective forcing components.
    ecsparts : bool, optional
        Whether to include the wavelength-specific climate sensitivity estimates.
    wavparts : bool, optional
        Whether to include the non-cloud and non-raw-flux wavelength components.
    """
    # NOTE: This also adds 'cloud effect' parameters. While they include
    # masked components they are not dependent on kernels and possibly more
    # robust for using as constraints (some paper does this... not sure which).
    keys = {'t': [], 's': [], 'a': []}
    regex = re.compile(r'((?:\A|[^_]*_)r)([lsf])([udn])([ts])(_|cs_)')
    boundary = boundary or 't'
    for b in 'ts':
        keys[b][:] = (
            key for key in dataset.data_vars
            if (m := regex.search(key)) and m.group(4) == b
            and (ecsparts or 'ecs2x' not in key or m.group(2) == 'f')
            and (erfparts or m.group(1) in ('r', 'cl_r') or 'erf2x' not in key)
            and (wavparts or m.group(1) in ('r', 'cl_r') or m.group(2) == 'f')
        )
    for toa in keys['t']:
        if 'a' in boundary:
            sfc = regex.sub(r'\1\2\3s\5', toa)
            if sfc in keys['s']:
                key = regex.sub(r'\1\2\3a\5', toa)
                keys['a'].append(key)
                with xr.set_options(keep_attrs=True):
                    dataset[key] = dataset[toa] + dataset[sfc]
                if long_name := dataset[toa].attrs.get('long_name', None):
                    long_name = long_name.replace('TOA', 'atmospheric')
                    dataset[key].attrs['long_name'] = long_name
    for b, rads in keys.items():  # TODO: support cloudy climate sensitivity
        for rad in tuple(rads):
            m = regex.search(rad)
            if m.group(1) == 'r' and m.group(5) == '_' and 'ecs2x' not in rad:
                rcs = regex.sub(r'\1\2\3\4cs_', rad)
                rce = regex.sub(r'\1\2\3\4ce_', rad)
                if rcs not in rads:
                    continue
                rads.append(rce)
                with xr.set_options(keep_attrs=True):
                    dataset[rce] = dataset[rad] - dataset[rcs]
                if long_name := dataset[rcs].attrs.get('long_name', None):
                    long_name = long_name.replace('clear-sky', 'cloud')
                    dataset[rce].attrs['long_name'] = long_name
    keys = sorted(k for b, key in keys.items() for k in key if b in boundary)
    keys = sorted(k for k in dataset.data_vars if k not in keys)
    dataset = dataset.drop_vars(keys)
    return dataset


def open_climate(path='~/data', climate=None, nodrift=True, **constraints):
    """
    Return a dictionary of datasets containing climate files.

    Parameters
    ----------
    path : path-like
        The base path.
    climate : 2-tuple of int
        The climate range.
    nodrift : bool, optional
        Whether to use drift corrections.
    **indexers
        Optional indexers.
    **constraints
        Passed to `Database`.
    """
    # NOTE: Non-flagship simulations can be added with flagship_translate=True
    # as with other processing functions. Ensembles are added in a MultiIndex
    # NOTE: This loads all available variables by default, but can be
    # restricted to a few with e.g. variable=['ts', 'ta'].
    files, *_ = _glob_files(path, project=constraints.get('project', None))
    facets = ('project', 'model', 'ensemble')
    kwargs = {key: constraints.pop(key, 1) for key in ('annual', 'seasonal', 'monthly')}
    variables = [
        'ps', 'psl', 'ts', 'ta', 'huss', 'hus',
        'uas', 'ua', 'vas', 'va', 'tauu', 'tauv', 'zg',
        'prw', 'pr', 'prsn', 'hfss', 'hfls', 'evspsbl', 'sbl',
        'cl', 'clw', 'cli', 'clt', 'clwvi', 'clivi',
        *(key for keys in COMPONENTS_RADIATION.values() for key in keys),
    ]
    constraints.setdefault('variable', variables)
    constraints.setdefault('experiment', 'piControl')
    database = Database(files, facets, flagship_translate=True, **constraints)
    nodrift = nodrift and '-nodrift' or ''
    climate = climate or (0, 150)
    climate = f'{climate[0]:04d}-{climate[1]:04d}-climate{nodrift}'
    datasets = {}
    print(f'Climate files: {climate}')
    print(f'Number of climate files: {len(database)}.')
    print('Model:', end=' ')
    for group, data in database.items():
        # Load the data
        if not data:
            continue
        print(group[1], end=' (')
        dataset = xr.Dataset()
        for key, paths in data.items():
            variable = key[database.key.index('variable')]
            paths = [path for path in paths if _item_dates(path) == climate]
            if len(paths) != 1:
                continue
            print(variable, end=' ')
            da = open_file(paths[0], variable, project=database.project)
            default = da.attrs.pop('title', variable)  # in case long_name is missing
            long_name = da.attrs.pop('long_name', default)
            long_name = ' '.join(s if s == 'TOA' else s.lower() for s in long_name.split())  # noqa: E501
            da.attrs['long_name'] = long_name
            if da.climo.standard_units:
                da = da.climo.to_standard_units()
            dataset[variable] = da

        # Average, merge, and filter the data
        # TODO: Should place this stuff inside main function.
        print('other', end='')
        dataset = _add_humidity_terms(dataset)
        dataset = _add_radiation_terms(dataset)
        dataset = _add_transport_terms(dataset)
        dataset = _adjust_moisture_terms(dataset)
        dataset = _adjust_standard_units(dataset)
        print(')', end=' ')
        drop = (key for keys in COMPONENTS_RADIATION.values() for key in keys)
        dataset = dataset.drop_vars(drop, errors='ignore')  # remove directional
        dataset = time_averages(dataset, **kwargs)
        if 'plev' in dataset:
            dataset = dataset.sel(plev=slice(None, 7000))
        datasets[group] = dataset

    print()
    return datasets


def open_feedbacks(
    path='~/scratch2/data-outdated', kernels=None, boundary=None, nodrift=True,
    **constraints,
):
    """
    Return a dictionary of datasets containing feedback files.

    Parameters
    ----------
    path : path-like
        The base path.
    nodrift : bool, optional
        Whether to use drift corrections.
    kernels : str, optional
        The kernel source.
    **kwargs
        Passed to `_adjust_feedback_terms`.
    **constraints
        Passed to `Database`.
    """
    # Initial stuff
    # TODO: Adapt this for when regions are stored as a ('numerator', 'denominator')
    # MultiIndex. Should call .set_index(regions=('numerator', 'denominator')).
    keys = ('erfparts', 'ecsparts', 'wavparts')
    kwargs = {key: constraints.pop(key) for key in keys if key in constraints}
    files, *_ = _glob_files(path, project=constraints.get('project', None))
    facets = ('project', 'model', 'ensemble')
    constraints['variable'] = 'feedbacks'
    constraints.pop('experiment', None)
    database = Database(files, facets, flagship_translate=True, **constraints)
    nodrift = nodrift and '-nodrift' or ''
    kernels = kernels or 'eraint'
    datasets = {}
    print(f'Feedback files: {kernels}-<statistic>{nodrift}')
    print(f'Number of feedback files: {len(database)}.')
    print('Model:', end=' ')
    for group, data in database.items():
        # Get the files
        pairs = tuple(
            (key, file) for key, files in data.items() for file in files
            if kernels in file.name and bool(nodrift) == bool('nodrift' in file.name)
            and any(s in file.name for s in ('local-local', 'local-global', 'global-global'))  # noqa: E501
            # and all(s not in file.name for s in ('local-local', 'local-global', 'global-global'))  # noqa: E501
        )
        if not pairs:
            continue

        # Load the data
        # TODO: Adapt this for the new files with region and time dimensions
        print(group[1], end=' (')
        bnds, parts = {}, {}
        names = ('author', 'series', 'statistic', 'numerator', 'denominator')
        for i, (key, file) in enumerate(pairs):
            *_, regions, suffix = file.stem.split('_')
            statistic = 'ratio' if 'ratio' in suffix else 'regression'
            translate = {'local': 'point', 'global': 'globe'}  # translate this
            numerator, denominator = (translate[s] for s in regions.split('-'))
            series = 'response' if '4xCO2' in file.name else 'control'
            print(i + 1, end=' ' * int(i + 1 < len(pairs)))
            dataset = open_file(file, project=database.project)
            if 'plev_bot' in dataset:
                bnds['plev_bot'] = dataset['plev_bot']
            if 'plev_top' in dataset:
                bnds['plev_top'] = dataset['plev_top']
            dataset = _adjust_feedback_terms(dataset, boundary=boundary, **kwargs)
            dataset = _adjust_feedback_attrs(dataset, boundary=boundary)
            key = ('davis', series, statistic, numerator, denominator)
            parts[key] = dataset

        # Concatenate the data
        # NOTE: When concatenating along the MultiIndex xarray seems to automatically
        # broadcast global feedbacks across all longitudes and latitudes.
        print(')', end=' ')
        index = xr.DataArray(
            pd.MultiIndex.from_tuples(parts, names=names),
            dims='feedback',
            name='feedback',
            attrs={'long_name': 'feedback information'},
        )
        dataset = xr.concat(
            parts.values(),
            dim=index,
            compat='override',
            coords='minimal'
        )
        dataset.update(bnds)
        if group in datasets:
            datasets[group].update(dataset)
        else:
            datasets[group] = dataset

    print()
    return datasets


def open_json(path='~/data/cmip-tables', **constraints):
    """
    Return a dictionary of datasets containing json-provided feedback data.

    Parameters
    ----------
    path : path-like, optional
        The base path.
    **constraints
        Passed to `_parse_constraints`.
    """
    # NOTE: These are also available in tables but json is easier
    path = Path(path).expanduser()
    project, constraints = _parse_constraints(reverse=True, **constraints)
    datasets = {}
    for file in sorted(path.glob('cmip*.json')):
        print(f'External file: {file.name}')
        author = file.stem.split('_')[1]
        names = ('author', 'series', 'statistic', 'numerator', 'denominator')
        index = (author, 'response', 'regression', 'globe', 'globe')
        index = xr.DataArray(
            pd.MultiIndex.from_tuples([index], names=names),
            dims='feedback',
            name='feedback',
            attrs={'long_name': 'feedback information'},
        )
        with open(file, 'r') as f:
            source = json.load(f)
        for model, ensembles in source[project].items():
            if model not in constraints.get('model', (model,)):
                continue
            for ensemble, data in ensembles.items():
                experiment = 'abrupt4xCO2' if project == 'CMIP5' else 'abrupt-4xCO2'
                key_flagship = (project, experiment, model)
                ens_default = FLAGSHIP_ENSEMBLES[project, None, None]
                ens_flagship = FLAGSHIP_ENSEMBLES.get(key_flagship, ens_default)
                ensemble = 'flagship' if ensemble == ens_flagship else ensemble
                if ensemble not in constraints.get('ensemble', (ensemble,)):
                    continue
                group = (project, model, ensemble)
                dataset = xr.Dataset()
                for key, value in data.items():
                    name, units = FEEDBACK_DEFINITIONS[key.lower()]
                    if units == 'K':
                        long_name = 'climate sensitivity'
                    elif units == 'W m^-2':
                        long_name = 'radiative forcing'
                    else:
                        long_name = 'feedback parameter'
                    attrs = {'units': units, 'long_name': long_name}
                    dataset[name] = xr.DataArray(value, attrs=attrs)
                dataset = dataset.expand_dims(feedback=1)
                dataset = dataset.assign_coords(feedback=index)
                if group in datasets:
                    datasets[group].update(dataset)
                else:
                    datasets[group] = dataset
    return datasets


def open_text(path='~/data/cmip-tables', **constraints):
    """
    Return a dictionary of datasets containing text-provided feedback data.

    Parameters
    ----------
    path : path-like, optional
        The base path.
    **constraints
        Passed to `_parse_constraints`.
    """
    # TODO: Possibly add alternative cmip6 feedback estimates
    path = Path(path).expanduser()
    project, constraints = _parse_constraints(reverse=True, **constraints)
    datasets = {}
    for file in sorted(path.glob(f'{project.lower()}*.txt')):
        author = file.stem.split('_')[1]
        if author == 'zelinka':
            continue
        print(f'External file: {file.name}')
        names = ('author', 'series', 'statistic', 'numerator', 'denominator')
        index = (author, 'response', 'regression', 'globe', 'globe')
        index = xr.DataArray(
            pd.MultiIndex.from_tuples([index], names=names),
            dims='feedback',
            name='feedback',
            attrs={'long_name': 'feedback information'},
        )
        table = pd.read_table(
            file,
            header=1,
            skiprows=[2],
            index_col=0,
            delimiter=r'\s{2,}',
            engine='python',
        )
        table.index.name = 'model'
        dataset = table.to_xarray()
        dataset = dataset.expand_dims(feedback=1)
        dataset = dataset.assign_coords(feedback=index)
        for key, da in dataset.data_vars.items():
            name, units = FEEDBACK_DEFINITIONS[key.lower()]
            if units == 'K':
                long_name = 'climate sensitivity'
            elif units == 'W m^-2':
                long_name = 'radiative forcing'
            else:
                long_name = 'feedback parameter'
            for model in dataset.model.values:
                group = ('CMIP5', model, 'flagship')
                if model not in constraints.get('model', (model,)):
                    continue
                if 'flagship' not in constraints.get('ensemble', ('flagship',)):
                    continue
                if any(lab in model for lab in ('mean', 'deviation', 'uncertainty')):
                    continue
                data = da.sel(model=model, drop=True)
                data = data * 0.5 if '4x' in key else data * 1.0
                data.name = name
                data.attrs.update({'units': units, 'long_name': long_name})
                data = data.to_dataset()
                if group in datasets:  # combine new feedback coordinates
                    tup = (data, datasets[group])
                    data = xr.combine_by_coords(tup, combine_attrs='override')
                datasets[group] = data
    return datasets


def open_all(
    path='~/data', project=None, control=True, feedbacks=True, json=True, text=True,
    **constraints,
):
    """
    Load CMIP variables for each model. Return a dictionary of datasets.

    Parameters
    ----------
    path : str, optional
        The default path.
    project : sequence, optional
        The project(s) to use.
    control : bool, optional
        Whether to load control climate data.
    feedbacks : bool, optional
        Whether to load internal feedback files.
    json, text : bool, optional
        Whether to load external feedback files.
    **kwargs
        Passed to relevant functions.
    **constraints
        Passed to constrain the results.
    """
    # Open the files
    # NOTE: Current paradigm is to separately load data with and without drift
    # corrections and data from distinct kernel sources, but always load different
    # different feedback estimate time periods, spatial averaging, and methods (e.g.
    # ols1 vs. ols2, control vs. abrupt experiment) into the same dataset.
    keys_climate = ('annual', 'seasonal', 'monthly', 'climate', 'nodrift')
    keys_feedbacks = ('kernels', 'boundary')
    kw_climate = {k: constraints.pop(k) for k in keys_climate if k in constraints}
    kw_feedbacks = {k: constraints.pop(k) for k in keys_feedbacks if k in constraints}
    if 'nodrift' in kw_climate:
        kw_feedbacks['nodrift'] = kw_climate['nodrift']
    datasets = {}
    if isinstance(project, str):
        projects = project.split(',')
    else:
        projects = list(project or ('CMIP5', 'CMIP6'))
    for project in map(str.upper, projects):
        print(f'Project: {project}')
        for b, function, folder, kw in (
            (control, open_climate, None, kw_climate),
            (feedbacks, open_feedbacks, None, kw_feedbacks),
            (json, open_json, 'cmip-tables', {}),
            (text, open_text, 'cmip-tables', {}),
        ):
            if not b:
                continue
            input = Path(path) / (folder or '')
            parts = function(input, **kw, **constraints, project=project)
            for group, dataset in parts.items():
                if group in datasets:  # combine new feedback coordinates
                    tup = (dataset, datasets[group])
                    dataset = xr.combine_by_coords(tup, combine_attrs='override')
                datasets[group] = dataset

    # Concatenate datasets and add derived quantities
    # NOTE: MultiIndex support was added here: https://github.com/pydata/xarray/pull/702
    names = {name: da for ds in datasets.values() for name, da in ds.data_vars.items()}
    print('Adding missing variables.')
    print('Model:', end=' ')
    for group, dataset in tuple(datasets.items()):  # interpolated datasets
        print(group[1], end=' ')
        for name in names.keys() - dataset.data_vars.keys():
            da = names[name]  # *sample* from another model or project
            da = xr.full_like(da, np.nan)
            if 'feedback' in da.sizes and 'feedback' in dataset:
                da = da.isel(feedback=0, drop=True)
                da = da.expand_dims(feedback=len(dataset.feedback))
                da = da.assign_coords(feedback=dataset.feedback)
            da.attrs.clear()  # default to non-null array attributes
            dataset[name] = da
    print()
    print('Concatenating datasets.')
    index = xr.DataArray(
        pd.MultiIndex.from_tuples(datasets, names=('project', 'model', 'ensemble')),
        dims='source',
        name='source',
        attrs={'long_name': 'source information'},
    )
    dataset = xr.concat(
        datasets.values(),
        dim=index,
        coords='minimal',
        combine_attrs='drop_conflicts',  # drop e.g. history but keep e.g. long_name
    )
    if 'feedback' in dataset.sizes:
        dataset = dataset.transpose('feedback', ...)
    dataset = dataset.climo.standardize_coords()
    dataset = dataset.climo.add_cell_measures(verbose=False)
    return dataset
