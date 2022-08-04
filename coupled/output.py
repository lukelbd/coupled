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
from climopy import diff, const, ureg
from icecream import ic  # noqa: F401
from metpy import calc, units

from cmip_data.feedbacks import FEEDBACK_DESCRIPTIONS
from cmip_data.internals import ENSEMBLES_FLAGSHIP, _item_dates, _parse_constraints
from cmip_data.utils import MODELS_INSTITUTIONS  # noqa: F401
from cmip_data import Database, average_periods, glob_files, open_file

__all__ = [
    'open_bulk',
    'open_climate',
    'open_feedbacks',
    'open_feedbacks_json',
    'open_feedbacks_text',
]

# Facets for the MultiIndex coordinate named 'facets'.
# NOTE: Here for sake of brevity rename abrupt4xCO2 experiments to 'response' and
# piControl experiments to 'control', while others are the same. Note the value
# 'experiment' for feedback parameters denotes the time series they were derived.
FACETS_CONCAT = (
    'project',
    'model',
    'ensemble'
    'experiment',
)
FACETS_RENAME = {
    'piControl': 'control',
    'abrupt4xCO2': 'response',
    'abrupt-4xCO2': 'response',
}

# Feedback definitions
# NOTE: These are used to both translate tables from external sources into the more
# descriptive naming convention, and to translate inputs to plotting functions for
# convenience (default for each shorthand is to use combined longwave + shortwave toa).
# The preferred names should also come first becuase these are reverse-translated to
# help make shorter automatic filenames.
FEEDBACK_DEFINITIONS = {
    'ecs': ('rfnt_ecs', 'K'),  # zelinka definition
    'tcr': ('tcr_ecs', 'K'),  # forster definition
    'erf': ('rfnt_erf', 'W m^-2'),
    'f2x': ('rfnt_erf', 'W m^-2'),  # forster definition
    'f4x': ('rfnt_erf', 'W m^-2'),  # geoffroy definition
    'erf2x': ('rfnt_erf', 'W m^-2'),  # zelinka definition
    'erf4x': ('rfnt_erf', 'W m^-2'),  # for consistency only
    'net': ('rfnt_lam', 'W m^-2 K^-1'),
    'rho': ('tcr_lam', 'W m^-2 K^-1'),  # forster definition
    'kap': ('ohc_lam', 'W m^-2 K^-1'),  # forster definition
    'cs': ('rfntcs_lam', 'W m^-2 K^-1'),
    'swcs': ('rsntcs_lam', 'W m^-2 K^-1'),  # forster definition
    'lwcs': ('rlntcs_lam', 'W m^-2 K^-1'),  # forster definition
    'cre': ('rfntce_lam', 'W m^-2 K^-1'),  # forster definition
    'swcre': ('rsntce_lam', 'W m^-2 K^-1'),
    'lwcre': ('rlntce_lam', 'W m^-2 K^-1'),
    'pl': ('pl_rfnt_lam', 'W m^-2 K^-1'),  # zelinka definition
    'pl*': ('pl*_rfnt_lam', 'W m^-2 K^-1'),  # zelinka definition
    'lr': ('lr_rfnt_lam', 'W m^-2 K^-1'),  # zelinka definition
    'lr*': ('lr*_rfnt_lam', 'W m^-2 K^-1'),  # zelinka definition
    'wv': ('hus_rfnt_lam', 'W m^-2 K^-1'),  # zelinka definition
    'rh': ('hur_rfnt_lam', 'W m^-2 K^-1'),  # zelinka definition
    'alb': ('alb_rfnt_lam', 'W m^-2 K^-1'),  # zelinka definition (full is 'albedo')
    'cld': ('cl_rfnt_lam', 'W m^-2 K^-1'),  # zelinka definition
    'swcld': ('cl_rsnt_lam', 'W m^-2 K^-1'),  # zelinka definition
    'lwcld': ('cl_rlnt_lam', 'W m^-2 K^-1'),  # zelinka definition
    'resid': ('resid_rfnt_lam', 'W m^-2 K^-1'),
    'err': ('resid_rfnt_lam', 'W m^-2 K^-1'),  # forster definition
}

# Component variables for derivations and dataset loading
# NOTE: See Donohoe et al. (2020) for details on transport terms. Precipitation appears
# in the dry static energy formula because unlike surface evaporation, it deposits heat
# inside the atmosphere, i.e. it remains after subtracting surface and TOA loss terms.
REGEX_EXPONENTS = re.compile(  # ignore exponential scale factors
    r'([a-df-zA-DF-Z]+)([-+]?[0-9]+)'
)
COMPONENTS_EXPLICIT = {
    'dry static': (1, ('intuadse', 'intvadse')),
    'latent static': (const.Lv, ('intuaw', 'intvaw')),
}
COMPONENTS_TRANSPORT = {
    'dry static': (('hfss', 'rlns', 'rsns', 'rlnt', 'rsnt', 'pr', 'prsn'), ()),
    'latent static': (('hfls',), ('pr', 'prsn')),
    'moist static': (('hfss', 'hfls', 'rlns', 'rsns', 'rlnt', 'rsnt'), ()),
    'ocean': ((), ('hfss', 'hfls', 'rlns', 'rsns')),  # flux into atmosphere
    'total': (('rlnt', 'rsnt'), ()),
}
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

# Unit conversions for use in transport derivations and standardization for plots
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


def _print_info(*datas):
    """
    Print summary of the input data.

    Parameters
    ----------
    *data : xarray.DataArray
        The data array(s).
    """
    for data in datas:
        data = data.sel(lat=slice(0, None)).climo.dequantify()
        data = data.climo.add_scalar_coords().climo.add_cell_measures()
        min_, max_, mean = data.min(), data.max(), data.climo.average('area').mean()
        print(f'{data.name: <10s} range:', end=' ')
        print(f'min {min_.item():.1f} max {max_.item():.1f} mean {mean.item():.1f}')


def _update_climate_humidity(dataset):
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
                src = dataset.climo.coords if key == 'plev' else dataset.climo.vars
                data = src[key].climo.to_units(unit)  # climopy units
                data.data = data.data.magnitude * units.units(unit)  # metpy units
                datas.append(data)
            descrip = 'near-surface ' if name[-1] == 's' else ''
            data = calc.relative_humidity_from_specific_humidity(*datas)
            data = data.climo.dequantify()  # works with metpy registry
            data = data.climo.to_units('%').clip(0, 100)
            data.attrs['long_name'] = f'{descrip}relative humidity'
            dataset[name] = data
    return dataset


def _update_climate_moisture(dataset):
    """
    Add ice and liquid water terms.

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset.
    """
    # NOTE: Unlike related variables (including, confusingly, clwvi), 'clw' includes
    # only liquid component rather than combined liquid plus ice. Adjust it to match
    # convention from other variables and add other component terms.
    if 'clw' in dataset:
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
            da.attrs = {'units': dataset[both].units, 'long_name': descrip % 'liquid'}
            dataset[liquid] = da
        for name, string in zip((ice, both, liquid), ('ice', 'water', 'liquid')):
            if name in dataset and string not in descrip:
                dataset[name].attrs['long_name'] = descrip % string
    return dataset


def _update_climate_radiation(dataset):
    """
    Add albedo and net fluxes from upwelling and downwelling components.

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset.
    """
    # NOTE: Here also add 'albedo' term and generate long name by replacing directional
    # terms in existing long name. Remove the directional components when finished.
    regex = re.compile(r'(upwelling|downwelling|outgoing|incident)')
    for name, keys in COMPONENTS_RADIATION.items():
        if all(key in dataset for key in keys):  # skip partial data
            if name == 'albedo':
                dataset[name] = 100 * dataset[keys[1]] / dataset[keys[0]]
                long_name = 'surface albedo'
                unit = '%'
            elif len(keys) == 2:
                dataset[name] = dataset[keys[1]] - dataset[keys[0]]
                long_name = dataset[keys[1]].attrs['long_name']
                unit = 'W m^-2'
            else:
                dataset[name] = -1 * dataset[keys[0]]
                long_name = dataset[keys[0]].attrs['long_name']
                unit = 'W m^-2'
            long_name = long_name.replace('radiation', 'flux')
            long_name = regex.sub('net', long_name)
            dataset[name].attrs.update({'units': unit, 'long_name': long_name})

    dataset = dataset.drop_vars(
        key for keys in COMPONENTS_RADIATION.values()
        for key in keys if key in dataset
    )
    return dataset


def _update_climate_transport(dataset):
    """
    Add zonally-integrated meridional transport and pointwise
    transport convergence to the dataset.

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset.
    explicit : bool, optional
        Whether to load explicit data.
    """
    # NOTE: It is not necessary to weight by atmosphere thickness because we are
    # working in energy units, so the columnwise heat capacity factor associated
    # with spatially varying surface pressure is implicitly factored in.
    # NOTE: Duffy et al. (2018) and Mayer et al. (2020) suggest a snow correction of
    # energy budget is necessary, and Armour et al. (2019) suggests correcting for
    # "latent heat associated with falling snow", but this is relative to estimate of
    # hfls from idealized expression for *evaporation over ocean* based on temperature
    # and humidity differences between surface and boundary layer. Since model output
    # hfls = Lv * evsp + Ls * sbl exactly (compare below terms), where the sbl term is
    # equivalent to adding the latent heat of fusion required to melt snow before a
    # liquid-vapor transition, an additional correction is not needed here.
    alts = ('', '_alt')
    for (descrip, parts), alt in itertools.product(COMPONENTS_TRANSPORT.items(), alts):
        # Get component data
        constants = {}
        for i, keys in zip((1, -1), parts):
            keys = list(keys)
            source = CONVERSIONS_TRANSPORT.copy()
            if alt and 'hfls' in keys:
                idx = keys.index('hfls')
                keys[idx:idx + 1] = ('evspsbl', 'sbl')
            constants.update({key: i * source.get(key, 1) for key in keys})
        if alt and 'sbl' not in constants:  # skip redundant calculation
            continue

        # Get implicit convergence and transport terms
        name = f'c{descrip[0]}sef{alt}' if 'static' in descrip else f'c{descrip}f{alt}'
        prefix = alt and 'alternative '
        if all(key in dataset for key in constants):
            resid = sum(c * dataset[key].climo.quantify() for key, c in constants.items())  # noqa: E501
            data = -1 * resid.climo.to_units('W m^-2')  # negative of residual
            data.attrs['long_name'] = f'{prefix}{descrip} energy transport convergence'
            dataset[name] = data.climo.dequantify()
            data = data.climo.add_cell_measures().climo.integral('lon')
            data = data.climo.cumintegral('lat', reverse=True).climo.to_units('PW')
            data = data.drop_vars(data.coords.keys() - data.sizes.keys())
            data.attrs['long_name'] = f'{prefix}{descrip} energy transport'
            dataset[name[1:]] = data.climo.dequantify()
            # _print_info(dataset[name[1:]])
        if alt:  # only run explicit calcs after first run
            continue

        # Get explicit convergence and transport terms
        name = f'{name}_exp'
        pair = COMPONENTS_EXPLICIT.get(descrip, None)
        if pair and all(key in dataset for key in pair[1]):
            scale, (ukey, vkey) = pair
            utrans, vtrans = dataset[ukey], dataset[vkey]
            utrans *= ureg(REGEX_EXPONENTS.sub(r'\1^\2', utrans.attrs.pop('units')))
            vtrans *= ureg(REGEX_EXPONENTS.sub(r'\1^\2', vtrans.attrs.pop('units')))
            lon, lat = dataset.climo.coords['lon'], dataset.climo.coords['lat']
            x = (const.a * lon).climo.to_units('m')
            y = (const.a * lat).climo.to_units('m')
            uderiv = diff.deriv_even(x, utrans, keepedges=True) / np.cos(lat)
            vderiv = diff.deriv_even(y, np.cos(lat) * vtrans, keepedges=True) / np.cos(lat)  # noqa: E501
            data = -scale * (uderiv + vderiv)  # convergence i.e. negative divergence
            data = data.climo.to_units('W m^-2')
            data.attrs['long_name'] = f'explicit {descrip} energy transport convergence'
            dataset[name] = data.climo.dequantify()
            data = scale * vtrans.climo.add_cell_measures().climo.integral('lon')
            data = data.drop_vars(data.coords.keys() - data.sizes.keys())
            data = data.climo.to_units('PW')
            data.attrs['long_name'] = f'explicit {descrip} energy transport'
            dataset[name[1:]] = data.climo.dequantify()
            # _print_info(dataset[name[1:]])

        # Combine latent and dry explicit components
        for idx in (0, 1):
            name = name[idx:]  # convergence and transport
            if 'mse' not in name:
                continue
            dse, lse = name.replace('mse', 'dse'), name.replace('mse', 'lse')
            if dse not in dataset or lse not in dataset:
                continue
            data = dataset[dse] + dataset[lse]
            data.attrs['units'] = dataset[lse].units
            data.attrs['long_name'] = dataset[lse].long_name.replace('latent', 'moist')
            dataset[name] = data

    drop = set(key for _, parts in COMPONENTS_EXPLICIT.values() for key in parts)
    dataset = dataset.drop_vars(drop & dataset.data_vars.keys())
    return dataset


def _update_climate_units(dataset):
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
        data = dataset[variable]
        if quantify := not data.climo._is_quantity:
            with xr.set_options(keep_attrs=True):
                data = constant * data.climo.quantify()
        data = data.climo.to_units(unit)
        if quantify:
            with xr.set_options(keep_attrs=True):
                data = data.climo.dequantify()
        dataset[variable] = data
    return dataset


def _update_feedback_info(
    dataset, boundary=None, annual=True, seasonal=True, monthly=True
):
    """
    Adjust feedback term attributes before plotting.

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset.
    boundary : {'t', 's', 'a'}, optional
        The boundari(es) to load. If one is passed then the indiator is stripped.
    annual, seasonal, monthly : bool, optoinal
        Whether to load different periods of data.
    """
    # Flux metadata repairs
    # NOTE: This also drops pre-loaded climate sensitivity parameters, and only
    # keeps the boundary indicator if more than one was requested.
    options = set(boundary or 't')
    for boundary, wavelength, (component, descrip) in itertools.product(
        ('surface', 'TOA'),
        ('full', 'longwave', 'shortwave'),
        FEEDBACK_DESCRIPTIONS.items()
    ):
        rad = f'r{wavelength[0].lower()}n{boundary[0].lower()}'
        if wavelength == 'full':
            descrip = 'net' if descrip == '' else descrip
            if len(options) > 1:
                descrip = f'{boundary} {descrip}'
        else:
            descrip = wavelength if descrip == '' else f'{wavelength} {descrip}'
            if len(options) > 1:
                descrip = f'{boundary} {descrip}'
        for suffix, outdated, param in zip(
            ('lam', 'erf', 'ecs'),
            ('lambda', 'erf2x', 'ecs2x'),
            ('feedback parameter', 'effective forcing', '')
        ):
            if component in ('', 'cs'):
                prefix = f'{rad}{component}'
            else:
                prefix = f'{component}_{rad}'
            name = f'{prefix}_{suffix}'
            outdated = f'{prefix}_{outdated}'
            if outdated in dataset:
                dataset = dataset.rename({outdated: name})
            if name in dataset:
                if suffix == 'ecs':
                    dataset = dataset.drop_vars(name)
                else:
                    dataset[name].attrs['long_name'] = f'{descrip} {param}'

    # Other metadata repairs
    # NOTE: This mimics the behavior of average_periods in open_climate
    # by optionally skipping unneeded periods to save space.
    for name, outdated, boundary in (
        ('plev_bot', 'lower', 'surface'),
        ('plev_top', 'upper', 'tropopause')
    ):
        if outdated in dataset:
            dataset = dataset.rename({outdated: name})
        if name in dataset:
            data = dataset[name]
            data.attrs['standard_units'] = 'hPa'
            data.attrs.setdefault('units', 'Pa')
            data.attrs.setdefault('long_name', f'{boundary} pressure')
    if 'period' in dataset:  # this is for consistency with open_climate
        drop = []
        time = pd.date_range('2000-01-01', '2000-12-01', freq='MS')
        if not annual:
            drop.append('ann')
        if not seasonal:
            drop.extend(('djf', 'mam', 'jja', 'son'))
        if not monthly:
            drop.extend(time.strftime('%b').str.lower().values)
        if drop:
            dataset = dataset.drop_sel(period=drop, errors='ignore')
    return dataset


def _update_feedback_parts(
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
    # Add keys for components
    # WARNING: This must come after _update_feedback_parts
    # NOTE: This will add 'cloud radiative effect' feedback parameters. While they
    # include masked components they are not dependent on kernels and possibly more
    # robust for using as constraints (some paper does this... not sure which).
    keys = {'t': [], 's': [], 'a': []}
    regex = re.compile(r'((?:\A|[^_]*_)r)([lsf])([udn])([ts])(_|cs_)')
    boundary = boundary or 't'
    for b in 'ts':
        keys[b][:] = (
            key for key in dataset.data_vars
            if (m := regex.search(key)) and m.group(4) == b
            and (ecsparts or 'ecs' not in key or m.group(2) == 'f')
            and (erfparts or m.group(1) in ('r', 'cl_r') or 'erf' not in key)
            and (wavparts or m.group(1) in ('r', 'cl_r') or m.group(2) == 'f')
        )
    # Add climate sensitivity estimate
    # NOTE: This will include climate sensitivity estiamtes from non-global feedback
    # parameters, which should be inaccurate but will be useful for comparisons.
    if 'rfnt_lam' in dataset and 'rfnt_erf' in dataset:
        data = (
            -1 * dataset.rfnt_erf.climo.add_cell_measures().climo.average('area')
            / dataset.rfnt_lam.climo.add_cell_measures().climo.average('area')
        )
        data.attrs['units'] = 'K'
        data.attrs['long_name'] = 'effective climate sensitivity'
        dataset['rfnt_ecs'] = data

    # Add atmospheric component
    for toa in keys['t']:
        if 'a' in boundary:
            sfc = regex.sub(r'\1\2\3s\5', toa)
            if sfc in keys['s']:
                key = regex.sub(r'\1\2\3a\5', toa)
                keys['a'].append(key)
                with xr.set_options(keep_attrs=True):
                    dataset[key] = dataset[toa] + dataset[sfc]
                if len(boundary) > 1:
                    long_name = dataset[toa].attrs['long_name']
                    long_name = long_name.replace('TOA', 'atmospheric')
                    dataset[key].attrs['long_name'] = long_name

    # Add cloud effect terms
    for b, rads in keys.items():  # TODO: support cloudy climate sensitivity
        for rad in tuple(rads):
            m = regex.search(rad)
            if m.group(1) == 'r' and m.group(5) == '_' and 'ecs' not in rad:
                rcs = regex.sub(r'\1\2\3\4cs_', rad)
                rce = regex.sub(r'\1\2\3\4ce_', rad)
                if rcs not in rads:
                    continue
                rads.append(rce)
                with xr.set_options(keep_attrs=True):
                    dataset[rce] = dataset[rad] - dataset[rcs]
                if True:
                    long_name = dataset[rcs].attrs.get('long_name', None)
                    long_name = long_name.replace('clear-sky', 'cloud')
                    dataset[rce].attrs['long_name'] = long_name

    keep = [key for b in boundary for key in keys[b]]
    keep.extend(('plev_bot', 'plev_top', 'rfnt_ecs'))
    return dataset.drop_vars(dataset.data_vars.keys() - set(keep))


def open_climate(
    *paths, years=None, nodrift=False, **constraints
):
    """
    Return a dictionary of datasets containing processed files.

    Parameters
    ----------
    *paths : path-like
        The search paths.
    years : 2-tuple of int
        The year range.
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
    keys_times = ('annual', 'seasonal', 'monthly')
    kw_times = {key: constraints.pop(key, True) for key in keys_times}
    files, *_ = glob_files(*paths, project=constraints.get('project', None))
    constraints.setdefault('table', ['Amon', 'Emon'])
    constraints.setdefault('experiment', ['piControl', 'abrupt4xCO2'])
    database = Database(files, FACETS_CONCAT, flagship_translate=True, **constraints)
    database.filter(always_exclude={'variable': ['ps', 'pfull']})  # skip dependencies
    nodrift = nodrift and '-nodrift' or ''
    datasets = {}
    print(f'Processed files: <dates>-climate{nodrift}')
    print(f'Number of processed files: {len(database)}.')
    if database:
        print('Model:', end=' ')
    for group, data in database.items():
        # Initial stuff
        # NOTE: Critical to overwrite the time coordinates after loading or else xarray
        # coordinate matching will apply all-NaN values for climatolgoies with different
        # base years (e.g. due to control data availability or response calendar diffs).
        if not data:
            continue
        for sub, replace in FACETS_RENAME.items():
            group = tuple(s.replace(sub, replace) for s in group)
        if years is not None:
            range_ = years
        elif 'response' in group:
            range_ = (120, 150)
        else:
            range_ = (0, 150)
        dates = f'{range_[0]:04d}-{range_[1]:04d}-climate{nodrift}'
        print(f'{group[1]}_{group[3]}_{range_[0]:04d}-{range_[1]:04d}', end=' ')

        # Load the data
        # NOTE: Here open_file automatically populates the mapping MODELS_INSTITUTIONS
        att = {'axis': 'T', 'standard_name': 'time'}
        time = pd.date_range('2000-01-01', '2000-12-01', freq='MS')
        time = xr.DataArray(time, name='time', dims='time', attrs=att)
        dataset = xr.Dataset()
        for key, paths in data.items():
            variable = key[database.key.index('variable')]
            paths = [path for path in paths if _item_dates(path) == dates]
            if len(paths) != 1:
                print(f'Warning: Skipping ambiguous duplicate paths {list(map(str, paths))}.')  # noqa: E501
                continue
            array = open_file(paths[0], variable, project=database.project)
            if array.time.size != 12:
                print(f'Warning: Skipping path {paths[0]} with time length {array.time.size}.')  # noqa: E501
                continue
            months = array.time.dt.month
            if sorted(months.values) != sorted(range(1, 13)):
                print(f'Warning: Skipping path {paths[0]} with month values {months.values}.')  # noqa: E501
                continue
            array = array.assign_coords(time=time)
            descrip = array.attrs.pop('title', variable)  # in case long_name missing
            descrip = array.attrs.pop('long_name', descrip)
            descrip = ' '.join(s if s == 'TOA' else s.lower() for s in descrip.split())
            if array.climo.standard_units:
                array = array.climo.to_standard_units()
            array.attrs['long_name'] = descrip
            dataset[variable] = array

        # Standardize the data
        # NOTE: Critical to put average_periods here to avoid repeated overhead.
        dataset = _update_climate_humidity(dataset)
        dataset = _update_climate_moisture(dataset)
        dataset = _update_climate_radiation(dataset)
        dataset = _update_climate_transport(dataset)
        dataset = _update_climate_units(dataset)
        if 'plev' in dataset:  # remove unneeded stratosphere levels
            dataset = dataset.sel(plev=slice(None, 7000))
        dataset = average_periods(dataset, **kw_times)
        dataset = dataset.squeeze()
        datasets[group] = dataset

    if datasets:
        print()
    return datasets


def open_feedbacks(
    *paths, source=None, boundary=None, nodrift=False, **constraints,
):
    """
    Return a dictionary of datasets containing feedback files.

    Parameters
    ----------
    *path : path-like
        The search paths.
    nodrift : bool, optional
        Whether to use drift corrections.
    source : str or sequence, optional
        The kernel source(s) to optionally filter.
    **kwargs
        Passed to `_update_feedback_parts`.
    **constraints
        Passed to `Database`.
    """
    # NOTE:
    keys_terms = ('erfparts', 'ecsparts', 'wavparts')
    keys_times = ('annual', 'seasonal', 'monthly')
    kw_times = {key: constraints.pop(key, True) for key in keys_times}
    kw_terms = {key: constraints.pop(key) for key in keys_terms if key in constraints}
    files, *_ = glob_files(*paths, project=constraints.get('project', None))
    constraints['variable'] = 'feedbacks'
    database = Database(files, FACETS_CONCAT, flagship_translate=True, **constraints)
    sources = (source,) if isinstance(source, str) else tuple(source or ())
    nodrift = nodrift and '-nodrift' or ''
    datasets = {}
    print(f'Feedback files: <source>-<statistic>{nodrift}')
    print(f'Number of feedback files: {len(database)}.')
    if database:
        print('Model:', end=' ')
    for group, data in database.items():
        # Filter the files
        bnds, parts, versions = {}, {}, {}
        names = ('source', 'statistic', 'region')
        for sub, replace in FACETS_RENAME.items():
            group = tuple(s.replace(sub, replace) for s in group)
        files = tuple(
            file for files in data.values() for file in files
            if bool(nodrift) == bool('nodrift' in file.name)
        )
        if not files:
            continue

        # Load the data
        # NOTE: This accounts for files with dedicated regions indicated in the name,
        # files with numerator and denominator multi-index coordinates, and files with
        # just a denominator region coordinate. Note open_file builds the multi-index.
        print(f'{group[1]}_{group[3]}', end=' ')
        for file in files:
            *_, indicator, suffix = file.stem.split('_')
            source, statistic, *_ = suffix.split('-')
            if sources and source not in sources:
                continue
            dataset = open_file(file, project=database.project, validate=False)
            outdated = 'local' in indicator or 'global' in indicator
            if outdated := 'local' in indicator or 'global' in indicator:
                if indicator.split('-')[0] != 'local':
                    continue
            if dataset.coords.get('numerator') is not None:
                dataset = dataset.sel(numerator='point', drop=True)
                dataset = dataset.rename(denominator='region')
            dataset = _update_feedback_info(dataset, boundary=boundary, **kw_times)
            dataset = _update_feedback_parts(dataset, boundary=boundary, **kw_terms)
            if 'plev_bot' in dataset:
                bnds['plev_bot'] = dataset['plev_bot']
            if 'plev_top' in dataset:
                bnds['plev_top'] = dataset['plev_top']
            dataset = dataset.drop_vars(('plev_bot', 'plev_top'), errors='ignore')
            if outdated:
                region = 'point' if indicator.split('-')[1] == 'local' else 'globe'
                versions[source, statistic, region] = dataset
            else:
                for region in dataset.region.values:
                    sel = dataset.sel(region=region, drop=True)
                    versions[source, statistic, region] = sel

        # Concatenate the data
        # NOTE: Concatenation automatically broadcasts global feedbacks across lons and
        # lats. Also critical to use 'override' for combine_attrs in case conventions
        # changed between running feedback calculations on different models.
        for key, dataset in versions.items():
            dataset = _update_feedback_info(dataset, boundary=boundary, **kw_times)
            dataset = _update_feedback_parts(dataset, boundary=boundary, **kw_terms)
            if 'plev_bot' in dataset:
                bnds['plev_bot'] = dataset['plev_bot']
            if 'plev_top' in dataset:
                bnds['plev_top'] = dataset['plev_top']
            dataset = dataset.drop_vars(('plev_bot', 'plev_top'), errors='ignore')
            parts[key] = dataset
        index = xr.DataArray(
            pd.MultiIndex.from_tuples(parts, names=names),
            dims='feedback',
            name='feedback',
            attrs={'long_name': 'feedback version information'},
        )
        dataset = xr.concat(
            parts.values(),
            dim=index,
            coords='minimal',
            compat='override',
            combine_attrs='override',
        )
        dataset.update(bnds)
        dataset = dataset.squeeze()
        datasets[group] = dataset

    if datasets:
        print()
    return datasets


def open_feedbacks_json(path='~/data/cmip-tables', **constraints):
    """
    Return a dictionary of datasets containing json-provided feedback data.

    Parameters
    ----------
    path : path-like, optional
        The base path.
    **constraints
        Passed to `_parse_constraints`.
    """
    # NOTE: When combinining with 'open_bulk' the non-descriptive long names
    # here should be overwritten by long names from custom feedbacks.
    path = Path(path).expanduser()
    project, constraints = _parse_constraints(reverse=True, **constraints)
    datasets = {}
    for file in sorted(path.glob('cmip*.json')):
        print(f'External file: {file.name}')
        source = file.stem.split('_')[1]
        names = ('source', 'statistic', 'region')
        index = (source, 'regression', 'globe')
        index = xr.DataArray(
            pd.MultiIndex.from_tuples([index], names=names),
            dims='feedback',
            name='feedback',
            attrs={'long_name': 'feedback version'},
        )
        with open(file, 'r') as f:
            source = json.load(f)
        for model, ensembles in source[project].items():
            if model not in constraints.get('model', (model,)):
                continue
            for ensemble, data in ensembles.items():
                key_flagship = (project, 'abrupt-4xCO2', model)
                ens_default = ENSEMBLES_FLAGSHIP[project, None, None]
                ens_flagship = ENSEMBLES_FLAGSHIP.get(key_flagship, ens_default)
                ensemble = 'flagship' if ensemble == ens_flagship else ensemble
                if ensemble not in constraints.get('ensemble', (ensemble,)):
                    continue
                group = (project, model, ensemble, 'response')
                dataset = xr.Dataset()
                for key, value in data.items():
                    name, units = FEEDBACK_DEFINITIONS[key.lower()]
                    if units == 'K':
                        long_name = 'effective climate sensitivity'
                    elif units == 'W m^-2':
                        long_name = 'effective forcing'
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


def open_feedbacks_text(path='~/data/cmip-tables', **constraints):
    """
    Return a dictionary of datasets containing text-provided feedback data.

    Parameters
    ----------
    path : path-like, optional
        The base path.
    **constraints
        Passed to `_parse_constraints`.
    """
    # NOTE: The Zelinka, Geoffry, and Forster papers and sources only specify a
    # CO2 multiple of '2x' or '4x' in the forcing entry and just say 'ecs' for the
    # climate sensitivity. So detect the multiple by scanning all keys in the table.
    path = Path(path).expanduser()
    project, constraints = _parse_constraints(reverse=True, **constraints)
    datasets = {}
    for file in sorted(path.glob(f'{project.lower()}*.txt')):
        source = file.stem.split('_')[1]
        if source == 'zelinka':
            continue
        print(f'External file: {file.name}')
        names = ('source', 'statistic', 'region')
        index = (source, 'regression', 'globe')
        index = xr.DataArray(
            pd.MultiIndex.from_tuples([index], names=names),
            dims='feedback',
            name='feedback',
            attrs={'long_name': 'feedback version'},
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
        factor = 0.5 if any('4x' in key for key in dataset.data_vars) else 1.0
        for key, da in dataset.data_vars.items():
            name, units = FEEDBACK_DEFINITIONS[key.lower()]
            if units == 'K':
                scale = factor
                long_name = 'effective climate sensitivity'
            elif units == 'W m^-2':
                scale = factor
                long_name = 'effective forcing'
            else:
                scale = 1.0
                long_name = 'feedback parameter'
            for model in dataset.model.values:
                group = ('CMIP5', model, 'flagship', 'response')
                if model not in constraints.get('model', (model,)):
                    continue
                if 'flagship' not in constraints.get('ensemble', ('flagship',)):
                    continue
                if any(lab in model for lab in ('mean', 'deviation', 'uncertainty')):
                    continue
                data = scale * da.sel(model=model, drop=True)
                data.name = name
                data.attrs.update({'units': units, 'long_name': long_name})
                data = data.to_dataset()
                if group in datasets:  # combine new feedback coordinates
                    tup = (data, datasets[group])
                    data = xr.combine_by_coords(tup, combine_attrs='override')
                datasets[group] = data
    return datasets


def open_bulk(
    path='~/data',
    project=None,
    climate=True,
    feedbacks=True,
    feedbacks_json=True,
    feedbacks_text=True,
    **constraints,
):
    """
    Load climate and feedback data into a single dataset.

    Parameters
    ----------
    path : str, optional
        The default path.
    project : sequence, optional
        The project(s) to use.
    climate : bool, optional
        Whether to load processed climate data.
    feedbacks : bool, optional
        Whether to load processed feedback files.
    feedbacks_json, feedbacks_text : bool, optional
        Whether to load external feedback files.
    **kwargs
        Passed to relevant functions.
    **constraints
        Passed to constrain the results.
    """
    # Open the files
    # NOTE: Here 'source' refers to either the author of a cmip-tables file or the
    # creator of the kernels used to build custom feedbacks, and 'region' always refers
    # to the denominator. For external feedbacks, the region is always 'globe' and its
    # value is constant in longitude and latitude, while for internal feedbacks, the
    # values vary in longitude and latitude, and global feedbacks are generated by
    # taking the average (see notes -- also tested outdated feedback files directly).
    # So can compare e.g. internal and external feedbacks with ``region='globe'`` and
    # ``area='avg'`` -- this is a no-op for the spatially uniform external feedbacks.
    # WARNING: Using dataset.update() instead of xr.combine_by_coords() below can
    # result in silently replacing existing data with NaNs (verified with test). The
    # latter is required when adding new 'facets' and 'feedback' coordinate values.
    keys_both = ('nodrift', 'annual', 'seasonal', 'monthly')
    keys_climate = ('years',)
    keys_feedbacks = ('source', 'boundary')
    kw_both = {k: constraints.pop(k) for k in keys_both if k in constraints}
    kw_climate = {k: constraints.pop(k) for k in keys_climate if k in constraints}
    kw_feedbacks = {k: constraints.pop(k) for k in keys_feedbacks if k in constraints}
    datasets = {}
    projects = project.split(',') if isinstance(project, str) else ('cmip5', 'cmip6')
    for project in map(str.upper, projects):
        print(f'Project: {project}')
        path_climate = ''
        # path_climate = '../scratch2/outdated-processed-nonlatest'
        # path_feedbacks = '../scratch2/outdated-feedbacks-regional'
        path_feedbacks = '../scratch2/outdated-feedbacks-numerator'
        for b, function, folder, kw in (
            (climate, open_climate, path_climate, {**kw_climate, **kw_both}),
            (feedbacks, open_feedbacks, path_feedbacks, {**kw_feedbacks, **kw_both}),
            (feedbacks_json, open_feedbacks_json, 'cmip-tables', {}),
            (feedbacks_text, open_feedbacks_text, 'cmip-tables', {}),
        ):
            if not b:
                continue
            kw = {**kw, **constraints, 'project': project}
            parts = function(Path(path).expanduser() / folder, **kw)
            for group, data in parts.items():
                if group in datasets:
                    comb = (datasets[group], data)
                    data = xr.combine_by_coords(comb, combine_attrs='override')
                datasets[group] = data

    # Concatenate datasets and add derived quantities
    # NOTE: Critical to use 'override' for combine_attrs in case models
    # use different naming conventions for identical variables.
    names = {name: da for ds in datasets.values() for name, da in ds.data_vars.items()}
    print('Adding missing variables.')
    if datasets:
        print('Model:', end=' ')
    for group, dataset in tuple(datasets.items()):  # interpolated datasets
        print(f'{group[1]}_{group[3]}', end=' ')
        for name in names.keys() - dataset.data_vars.keys():
            da = names[name]  # *sample* from another model or project
            da = xr.full_like(da, np.nan)  # preserve attributes as well
            if 'feedback' in da.sizes and 'feedback' in dataset:
                da = da.isel(feedback=0, drop=True)
                da = da.expand_dims(feedback=len(dataset.feedback))
                da = da.assign_coords(feedback=dataset.feedback)
            dataset[name] = da
    print()
    print('Concatenating datasets.')
    index = xr.DataArray(
        pd.MultiIndex.from_tuples(datasets, names=FACETS_CONCAT),
        dims='facets',
        name='facets',
        attrs={'long_name': 'facet information'},
    )
    dataset = xr.concat(
        datasets.values(),
        dim=index,
        compat='override',
        coords='minimal',
        combine_attrs='override',
    )
    if 'feedback' in dataset.sizes:
        dataset = dataset.transpose('feedback', ...)
    dataset = dataset.climo.standardize_coords()
    dataset = dataset.climo.add_cell_measures(verbose=False)
    return dataset
