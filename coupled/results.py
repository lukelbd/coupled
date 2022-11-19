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
import metpy.calc as mcalc
import metpy.units as munits
from climopy import diff, const, ureg
from icecream import ic  # noqa: F401

from cmip_data.feedbacks import FEEDBACK_DESCRIPTIONS
from cmip_data.internals import ENSEMBLES_FLAGSHIP
from cmip_data.internals import Database, glob_files, _item_dates, _parse_constraints
from cmip_data.utils import average_periods, load_file

__all__ = [
    'open_dataset',
    'climate_datasets',
    'feedback_datasets',
    'feedback_datasets_json',
    'feedback_datasets_text',
]

# Levels for the MultiIndex feedback version coordinate
# NOTE: Currently years for 'ratio' type feedbacks always correspond to the abrupt4xCO2
# climate average; the pre-industrial climate average is always over the full 150 years.
VERSION_LEVELS = (
    'source',
    'statistic',
    'region',
    'start',  # iniital year of regression or 'forced' climate average
    'stop',  # final year of regression or 'forced' climate average
)

# Levels for the MultiIndex coordinate 'facets'.
# NOTE: Previously renamed piControl and abrupt-4xCO2 to 'control' and 'response'
# but this was confusing as 'response' sounds like a perturbation (also considered
# 'unperturbed' and 'perturbed'). Now simply use 'picontrol' and 'abrupt4xco2'.
FACETS_LEVELS = (
    'project',
    'model',
    'experiment',
    'ensemble',
)
FACETS_RENAME = {
    'piControl': 'picontrol',
    'control-1950': 'control1950',
    'abrupt4xCO2': 'abrupt4xco2',
    'abrupt-4xCO2': 'abrupt4xco2',
}

# Climate constants
# NOTE: For now use the standard 1e3 kg/m3 water density (i.e. snow and ice terms
# represent melted equivalent depth) but could also use 1e2 kg/m3 snow density where
# relevant. See: https://www.sciencelearn.org.nz/resources/1391-snow-and-ice-density
# NOTE: These entries inform the translation from standard unit strings to short
# names like 'energy flux' and 'energy transport' used in figure functions. In
# future should group all of these into cfvariables with standard units.
CLIMATE_SCALES = {  # scaling prior to final unit transformation
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
CLIMATE_SHORTS = {
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
CLIMATE_UNITS = {
    'ta': 'K',
    'ts': 'K',
    'hus': 'g kg^-1',
    'huss': 'g kg^-1',
    'hfls': 'W m^-2',
    'hfss': 'W m^-2',
    'prw': 'mm',  # water vapor path not precip
    'pr': 'mm day^-1',
    'prl': 'mm day^-1',  # derived
    'pri': 'mm day^-1',  # renamed
    'ev': 'mm day^-1',  # renamed
    'evl': 'mm day^-1',  # derived
    'evi': 'mm day^-1',  # renamed
    'clwvi': 'mm',
    'cllvi': 'mm',  # derived
    'clivi': 'mm',
    'clw': 'g kg^-1',
    'cll': 'g kg^-1',
    'cli': 'g kg^-1',
    'ua': 'm s^-1',
    'va': 'm s^-1',
    'uas': 'm s^-1',
    'vas': 'm s^-1',
    'tauu': 'Pa',
    'tauv': 'Pa',
    'pbot': 'hPa',
    'ptop': 'hPa',
    'psl': 'hPa',
    'ps': 'hPa',
    'zg': 'dam',
}

# Feedback constants
# NOTE: These are used to both translate tables from external sources into the more
# descriptive naming convention, and to translate inputs to plotting functions for
# convenience (default for each shorthand is to use combined longwave + shortwave toa).
FEEDBACK_TRANSLATIONS = {
    'ecs': ('rfnt_ecs', 'K'),  # zelinka definition
    'tcr': ('rfnt_tcr', 'K'),  # forster definition
    'pattern': ('ts_pattern', 'K / K'),
    'erf2x': ('rfnt_erf', 'W m^-2'),  # zelinka definition
    'erf4x': ('rfnt_erf', 'W m^-2'),
    'f2x': ('rfnt_erf', 'W m^-2'),  # forster definition
    'f4x': ('rfnt_erf', 'W m^-2'),  # geoffroy definition
    'erf': ('rfnt_erf', 'W m^-2'),  # preferred name last (for reverse translation)
    'net': ('rfnt_lam', 'W m^-2 K^-1'),
    'lw': ('rlnt_lam', 'W m^-2 K^-1'),
    'sw': ('rsnt_lam', 'W m^-2 K^-1'),
    'rho': ('rfnt_rho', 'W m^-2 K^-1'),  # forster definition
    'kap': ('rfnt_kap', 'W m^-2 K^-1'),  # forster definition
    'cre': ('rfntce_lam', 'W m^-2 K^-1'),  # forster definition
    'swcre': ('rsntce_lam', 'W m^-2 K^-1'),
    'lwcre': ('rlntce_lam', 'W m^-2 K^-1'),
    'cs': ('rfntcs_lam', 'W m^-2 K^-1'),
    'swcs': ('rsntcs_lam', 'W m^-2 K^-1'),  # forster definition
    'lwcs': ('rlntcs_lam', 'W m^-2 K^-1'),  # forster definition
    'cld': ('cl_rfnt_lam', 'W m^-2 K^-1'),  # zelinka definition
    'lwcld': ('cl_rlnt_lam', 'W m^-2 K^-1'),  # zelinka definition
    'swcld': ('cl_rsnt_lam', 'W m^-2 K^-1'),  # zelinka definition
    'alb': ('alb_rfnt_lam', 'W m^-2 K^-1'),  # zelinka definition (full is 'albedo')
    'atm': ('atm_rfnt_lam', 'W m^-2 K^-1'),
    'lwatm': ('atm_rlnt_lam', 'W m^-2 K^-1'),
    'swatm': ('atm_rsnt_lam', 'W m^-2 K^-1'),
    'pl': ('pl_rfnt_lam', 'W m^-2 K^-1'),  # zelinka definition
    'pl*': ('pl*_rfnt_lam', 'W m^-2 K^-1'),  # zelinka definition
    'lr': ('lr_rfnt_lam', 'W m^-2 K^-1'),  # zelinka definition
    'lr*': ('lr*_rfnt_lam', 'W m^-2 K^-1'),  # zelinka definition
    'wv': ('hus_rfnt_lam', 'W m^-2 K^-1'),  # zelinka definition
    'rh': ('hur_rfnt_lam', 'W m^-2 K^-1'),  # zelinka definition
    'err': ('resid_rfnt_lam', 'W m^-2 K^-1'),  # forster definition
    'resid': ('resid_rfnt_lam', 'W m^-2 K^-1'),  # preferred name last (for reverse translation)  # noqa: E501
}

# Energetics constants
# NOTE: Here the flux components table was copied from feedbacks.py. Critical
# to get net fluxes and execute renames before running transport derivations.
# NOTE: Rename native surface budget terms to look more like cloud water and
# ice terms. Then use 'prl'/'prp' and 'evl'/'evp' for ice components.
ENERGETICS_RENAMES = {
    'prsn': 'pri',
    'evspsbl': 'ev',
    'sbl': 'evi',
}
ENERGETICS_COMPONENTS = {
    'rlnt': ('rlut',),  # out of the atmosphere
    'rsnt': ('rsut', 'rsdt'),  # out of the atmosphere (include constant rsdt)
    'rlns': ('rlds', 'rlus'),  # out of and into the atmosphere
    'rsns': ('rsds', 'rsus'),  # out of and into the atmosphere
    'rlntcs': ('rlutcs',),  # out of the atmosphere
    'rsntcs': ('rsutcs', 'rsdt'),  # out of the atmosphere (include constant rsdt)
    'rlnscs': ('rldscs', 'rlus'),  # out of and into the atmosphere
    'rsnscs': ('rsdscs', 'rsuscs'),  # out of and into the atmosphere
    'albedo': ('rsds', 'rsus'),  # full name to differentiate from 'alb' feedback
}

# Moisture-related constants
# NOTE: Here the %s is filled in with water, liquid, or ice depending
# on the component of the particular variable category.
MOISTURE_DEPENDENCIES = {
    'hur': ('plev', 'ta', 'hus'),
    'hurs': ('ps', 'ts', 'huss'),
}
MOISTURE_COMPONENTS = [
    ('ev', 'evl', 'evi', 'evp', '%s evaporation'),
    ('pr', 'prl', 'pri', 'prp', '%s precipitation'),
    ('clw', 'cll', 'cli', 'clp', 'mass fraction cloud %s'),
    ('clwvi', 'cllvi', 'clivi', 'clpvi', 'condensed %s water path'),
]

# Transport constants
# NOTE: See Donohoe et al. (2020) for details on transport terms. Precipitation appears
# in the dry static energy formula because unlike surface evaporation, it deposits heat
# inside the atmosphere, i.e. it remains after subtracting surface and TOA loss terms.
# NOTE: Duffy et al. (2018) and Mayer et al. (2020) suggest a snow correction of
# energy budget is necessary, and Armour et al. (2019) suggests correcting for
# "latent heat associated with falling snow", but this is relative to estimate of
# hfls from idealized expression for *evaporation over ocean* based on temperature
# and humidity differences between surface and boundary layer. Since model output
# hfls = Lv * evsp + Ls * sbl exactly (compare below terms), where the sbl term is
# equivalent to adding the latent heat of fusion required to melt snow before a
# liquid-vapor transition, an additional correction is not needed here.
TRANSPORT_DESCRIPTIONS = {
    'gse': 'potential static',
    'hse': 'sensible static',
    'dse': 'dry static',
    'lse': 'latent static',
    'mse': 'moist static',
    'ocean': 'storage + ocean',
    'total': 'total',
}
TRANSPORT_SCALES = {  # scaling prior to implicit transport calculations
    'pr': const.Lv,
    'prl': const.Lv,
    'pri': const.Ls - const.Lv,  # remove the 'pri * Lv' implied inside 'pr' term
    'ev': const.Lv,
    'evl': const.Lv,
    'evi': const.Ls - const.Lv,  # remove the 'evi * Lv' implied inside 'pr' term
}
TRANSPORT_IMPLICIT = {
    'dse': (('hfss', 'rlns', 'rsns', 'rlnt', 'rsnt', 'pr', 'pri'), ()),
    'lse': (('hfls',), ('pr', 'pri')),
    'ocean': ((), ('hfss', 'hfls', 'rlns', 'rsns')),  # flux into atmosphere
    'total': (('rlnt', 'rsnt'), ()),
}
TRANSPORT_INDIVIDUAL = {
    'gse': ('zg', const.g, 'dam m s^-1'),
    'hse': ('ta', const.cp, 'K m s^-1'),
    'lse': ('hus', const.Lv, 'g kg^-1 m s^-1'),
}
TRANSPORT_INTEGRATED = {
    'dse': (1, 'intuadse', 'intvadse'),
    'mse': (1, 'intuamse', 'intvamse'),  # not currently provided in cmip
    'lse': (const.Lv, 'intuaw', 'intvaw'),
}


def _standardize_order(dataset):
    """
    Standardize insertion order of dataset for user convenience.

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset.
    """
    # Climate order
    names = ['ta', 'ts', 'zg', 'ps', 'psl', 'pbot', 'ptop']
    names += ['ua', 'va', 'uas', 'vas', 'tauu', 'tauv']
    names += ['hus', 'hur', 'huss', 'hurs', 'prw', 'cl', 'clt', 'cct']
    names += [name for names in MOISTURE_COMPONENTS for name in names[:4]]
    # Feedback order
    iter_ = itertools.product('fls', 'tsa', ('', 'cs', 'ce'))
    fluxes = ['hfls', 'hfss', 'albedo']
    fluxes += [f'r{wav}n{bnd}{sky}' for wav, bnd, sky in iter_]
    parameters = ['', 'lam', 'rho', 'kap', 'erf', 'ecs', 'tcr']
    components = ['', 'pl', 'pl*', 'lr', 'lr*', 'hus', 'hur', 'atm', 'alb', 'cl', 'resid']  # noqa: E501
    iter_ = itertools.product(components, fluxes, parameters)
    names.extend(f'{component}_{flux}_{parameter}'.strip('_') for component, flux, parameter in iter_)  # noqa: E501
    # Transport order
    prefixes = ('', 'm', 's', 't')  # total, zonal-mean, stationary, transient
    suffixes = ('t', 'f', 'c', 'r')  # transport, flux, convergence, residual
    parts = ('gse', 'hse', 'dse', 'lse', 'mse', 'ocean', 'total')
    ends = ('', '_alt', '_exp')
    iter_ = itertools.product(parts, prefixes, suffixes, ends)
    names.extend(f'{prefix}{part}{suffix}{end}' for part, prefix, suffix, end in iter_)
    # Update insertion order
    unknowns = [name for name in dataset.data_vars if name not in names]
    if unknowns:
        print('Warning: Unknown order for variables:', ', '.join(unknowns))
    results = {name: dataset[name] for name in (*names, *unknowns) if name in dataset}
    dataset = xr.Dataset(results)
    return dataset
    # dataset = dataset.drop_vars(results.keys())
    # dataset.update(results)
    # return dataset


def _transport_implicit(data, descrip=None, prefix=None, adjust=True):
    """
    Convert implicit flux residuals to convergence and transport terms.

    Parameters
    ----------
    data : xarray.DataArray
        The data array.
    descrip : str, optional
        The long name description.
    prefix : str, optional
        The optional description prefix.
    adjust : bool, optional
        Whether to adjust with global-average residual.

    Returns
    -------
    cdata : xarray.DataArray
        The convergence.
    rdata : xarray.DataArray
        The global-average residual.
    tdata : xarray.DataArray
        The meridional transport.
    """
    # Get convergence and residual
    # NOTE: This requires cell measures are already present for consistency with the
    # explicit transport function, which requires a surface pressure dependence.
    data = data.climo.quantify()
    descrip = f'{descrip} ' if descrip else ''
    prefix = f'{prefix} ' if prefix else ''
    cdata = -1 * data.climo.to_units('W m^-2')  # convergence equals negative residual
    cdata.attrs['long_name'] = f'{prefix}{descrip}energy convergence'
    rdata = cdata.climo.average('area').drop_vars(('lon', 'lat'))
    rdata.attrs['long_name'] = f'{prefix}{descrip}energy residual'
    # Get meridional transport
    # WARNING: Cumulative integration in forward or reverse direction will produce
    # estimates respectively offset-by-one, so compensate by taking average of both.
    tdata = cdata - rdata if adjust else cdata
    tdata = tdata.climo.integral('lon')
    tdata = 0.5 * (
        -1 * tdata.climo.cumintegral('lat', reverse=False)
        + tdata.climo.cumintegral('lat', reverse=True)
    )
    tdata = tdata.climo.to_units('PW')
    tdata = tdata.drop_vars(tdata.coords.keys() - tdata.sizes.keys())
    tdata.attrs['long_name'] = f'{prefix}{descrip}energy transport'
    tdata.attrs['standard_units'] = 'PW'  # prevent cfvariable auto-inference
    return cdata.climo.dequantify(), rdata.climo.dequantify(), tdata.climo.dequantify()


def _transport_explicit(udata, vdata, qdata, descrip=None, prefix=None):
    """
    Convert explicit advection to convergence and transport terms.

    Parameters
    ----------
    udata : xarray.DataArray
        The zonal wind.
    vdata : xarray.DataArray
        The meridional wind.
    qdata : xarray.DataArray
        The advected quantity.
    descrip : str, optional
        The long name description.
    prefix : str, optional
        The optional description prefix.

    Returns
    -------
    cdata : xarray.DataArray
        The stationary convergence.
    sdata : xarray.DataArray
        The stationary eddy transport.
    mdata : xarray.DataArray
        The zonal-mean transport.
    """
    # Get convergence
    # NOTE: This requires cell measures are already present rather than auto-adding
    # them, since we want to include surface pressure dependence supplied by dataset.
    descrip = descrip and f'{descrip} ' or ''
    qdata = qdata.climo.quantify()
    udata, vdata = udata.climo.quantify(), vdata.climo.quantify()
    lon, lat = qdata.climo.coords['lon'], qdata.climo.coords['lat']
    x, y = (const.a * lon).climo.to_units('m'), (const.a * lat).climo.to_units('m')
    udiff = diff.deriv_even(x, udata * qdata, cyclic=True) / np.cos(lat)
    vdiff = diff.deriv_even(y, np.cos(lat) * vdata * qdata, keepedges=True) / np.cos(lat)  # noqa: E501
    cdata = -1 * (udiff + vdiff)  # convergence i.e. negative divergence
    cdata = cdata.assign_coords(qdata.coords)  # re-apply missing cell measures
    if 'plev' in cdata.dims:
        cdata = cdata.climo.integral('plev')
    string = f'{prefix} ' if prefix else 'stationary '
    cdata = cdata.climo.to_units('W m^-2')
    cdata.attrs['long_name'] = f'{string}{descrip}energy convergence'
    cdata.attrs['standard_units'] = 'W m^-2'  # prevent cfvariable auto-inference
    # Get transport components
    # NOTE: This depends on the implicit weight cell_height getting converted to its
    # longitude-average value during the longitude-integral (see _integral_or_average).
    vmean, qmean = vdata.climo.average('lon'), qdata.climo.average('lon')
    sdata = (vdata - vmean) * (qdata - qmean)  # width and height measures removed
    sdata = sdata.assign_coords(qdata.coords)  # reassign measures lost to conflicts
    sdata = sdata.climo.integral('lon')
    if 'plev' in sdata.dims:
        sdata = sdata.climo.integral('plev')
    string = f'{prefix} ' if prefix else 'stationary '
    sdata = sdata.climo.to_units('PW')
    sdata.attrs['long_name'] = f'{string}{descrip}energy transport'
    sdata.attrs['standard_units'] = 'PW'  # prevent cfvariable auto-inference
    mdata = vmean * qmean
    mdata = 2 * np.pi * np.cos(mdata.climo.coords.lat) * const.a * mdata
    # mdata = mdata.climo.integral('lon')  # integrate scalar coordinate
    if 'plev' in mdata.dims:
        mdata = mdata.climo.integral('plev')
    string = f'{prefix} ' if prefix else 'zonal-mean '
    mdata = mdata.climo.to_units('PW')
    mdata.attrs['long_name'] = f'{string}{descrip}energy transport'
    mdata.attrs['standard_units'] = 'PW'  # prevent cfvariable auto-inference
    return cdata.climo.dequantify(), sdata.climo.dequantify(), mdata.climo.dequantify()


def _update_climate_units(dataset):
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
    for variable, unit in CLIMATE_UNITS.items():
        if variable not in dataset:
            continue
        scale = CLIMATE_SCALES.get(variable, 1.0)
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
    for src in (dataset.data_vars, dataset.coords):
        for variable, data in src.items():
            long = data.attrs.get('long_name', None)
            unit = data.attrs.get('units', None)
            short = CLIMATE_SHORTS.get(unit, long)  # default long name, e.g. longitude
            if variable == 'time':  # possibly missing long name
                short = 'time'
            if short is not None:
                data.attrs['short_name'] = short
    return dataset


def _update_climate_energetics(dataset, clear=False, drop_components=True):
    """
    Add albedo and net fluxes from upwelling and downwelling components and remove
    the original directional variables.

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset.
    clear : bool, default: False
        Whether to compute clear-sky components.
    drop_components : bool, optional
        Whether to drop the directional components
    """
    # NOTE: Here also add 'albedo' term and generate long name by replacing directional
    # terms in existing long name. Remove the directional components when finished.
    regex = re.compile(r'(upwelling|downwelling|outgoing|incident)')
    for key, name in ENERGETICS_RENAMES.items():
        if key in dataset:
            dataset = dataset.rename({key: name})
    keys_components = set()
    for name, keys in ENERGETICS_COMPONENTS.items():
        keys_components.update(keys)
        if any(key not in dataset for key in keys):  # skip partial data
            continue
        if not clear and name[-2:] == 'cs':
            continue
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
    if drop_components:
        dataset = dataset.drop_vars(keys_components & dataset.data_vars.keys())
    return dataset


def _update_climate_moisture(dataset, ratio=True, liquid=False):
    """
    Add relative humidity and ice and liquid water terms and standardize
    the insertion order for the resulting dataset variables.

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset.
    ratio : bool, default: True
        Whether to add ice-liquid ratio estimate.
    liquid : bool, default: False
        Whether to add liquid components.
    """
    # Add humidity terms
    # NOTE: Generally forego downloading relative humidity variables... true that
    # operation is non-linear so relative humidity of climate is not climate of
    # relative humiditiy, but we already use this approach with feedback kernels.
    for name, keys in MOISTURE_DEPENDENCIES.items():
        if any(key not in dataset for key in keys):
            continue
        datas = []
        for key, unit in zip(keys, ('Pa', 'K', '')):
            src = dataset.climo.coords if key == 'plev' else dataset.climo.vars
            data = src[key].climo.to_units(unit)  # climopy units
            data.data = data.data.magnitude * munits.units(unit)  # metpy units
            datas.append(data)
        descrip = 'near-surface ' if name[-1] == 's' else ''
        data = mcalc.relative_humidity_from_specific_humidity(*datas)
        data = data.climo.dequantify()  # works with metpy registry
        data = data.climo.to_units('%').clip(0, 100)
        data.attrs['long_name'] = f'{descrip}relative humidity'
        dataset[name] = data

    # Add cloud terms
    # NOTE: Unlike related variables (including, confusingly, clwvi), 'clw' includes
    # only liquid component rather than combined liquid plus ice. Adjust it to match
    # convention from other variables and add other component terms.
    if 'clw' in dataset:
        if 'cli' not in dataset:
            dataset = dataset.rename_vars(clw='cll')
        else:
            with xr.set_options(keep_attrs=True):
                dataset['clw'] = dataset['cli'] + dataset['clw']
    for both, liq, ice, rat, descrip in MOISTURE_COMPONENTS:
        if not liquid and liq in dataset:
            dataset = dataset.drop_vars(liq)
        if liquid and both in dataset and ice in dataset:
            da = dataset[both] - dataset[ice]
            da.attrs = {'units': dataset[both].units, 'long_name': descrip % 'liquid'}
            dataset[liq] = da
        if ratio and both in dataset and ice in dataset:
            da = (100 * dataset[ice] / dataset[both]).clip(0, 100)
            da.attrs = {'units': '%', 'long_name': descrip % 'ice' + ' ratio'}
            dataset[rat] = da
    for both, liq, ice, _, descrip in MOISTURE_COMPONENTS:
        for name, string in zip((ice, both, liq), ('ice', 'water', 'liquid')):
            if name not in dataset:
                continue
            data = dataset[name]
            if string not in data.long_name:
                data.attrs['long_name'] = descrip % string
    return dataset


def _update_climate_transport(
    dataset, fluxes=False, drop_implicit=True, drop_explicit=True
):
    """
    Add zonally-integrated meridional transport and pointwise transport convergence
    to the dataset with dry-moist and transient-stationary-mean breakdowns.

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset.
    fluxes : bool, default: False
        Whether to compute additional flux estimates.
    drop_implicit : bool, optional
        The boundaries to drop for implicit transport terms.
    drop_explicit : bool, optional
        Whether to drop explicit transport terms.
    """
    # Get total, ocean, dry, and latent transport
    # NOTE: The below regex prefixes exponents expressed by numbers adjacent to units
    # with the carat ^, but ignores the scientific notation 1.e6 in scaling factors,
    # so dry static energy convergence units can be parsed as quantities.
    # WARNING: This must come after _update_climate_energetics
    ends = ('', '_alt', '_exp')  # implicit, alternative, explicit
    keys_keep = {'pr', 'pri', 'ev', 'evi', 'rlnt', 'rsnt'}
    keys_implicit = set(ENERGETICS_COMPONENTS)  # components should be dropped
    keys_explicit = set()
    for (name, pair), end in itertools.product(TRANSPORT_IMPLICIT.items(), ends[:2]):
        # Implicit transport
        constants = {}  # store component data
        for c, keys in zip((1, -1), map(list, pair)):
            if end and 'hfls' in keys:
                keys[(idx := keys.index('hfls')):idx + 1] = ('ev', 'evi')
            keys_implicit.update(keys)
            constants.update({k: c * TRANSPORT_SCALES.get(k, 1) for k in keys})
        if end and 'evi' not in constants:  # skip redundant calculation
            continue
        descrip = TRANSPORT_DESCRIPTIONS[name]
        kwargs = {'descrip': descrip, 'prefix': 'alternative' if end else ''}
        if all(k in dataset for k in constants):
            data = sum(c * dataset.climo.vars[k] for k, c in constants.items())
            cdata, rdata, tdata = _transport_implicit(data, **kwargs)
            cname, rname, tname = f'{name}c{end}', f'{name}r{end}', f'{name}t{end}'
            dataset.update({cname: cdata, rname: rdata, tname: tdata})
        # Explicit transport
        if end and True:  # only run explicit calcs after first run
            continue
        scale, ukey, vkey = TRANSPORT_INTEGRATED.get(name, (None, None, None))
        keys_explicit.update((ukey, vkey))
        kwargs = {'descrip': descrip, 'prefix': 'explicit'}
        regex = re.compile(r'([a-df-zA-DF-Z]+)([-+]?[0-9]+)')
        if ukey and vkey and ukey in dataset and vkey in dataset:
            udata = dataset[ukey].copy(deep=False)
            vdata = dataset[vkey].copy(deep=False)
            udata *= scale * ureg(regex.sub(r'\1^\2', udata.attrs.pop('units')))
            vdata *= scale * ureg(regex.sub(r'\1^\2', vdata.attrs.pop('units')))
            qdata = ureg.dimensionless * xr.ones_like(udata)  # placeholder
            cdata, _, tdata = _transport_explicit(udata, vdata, qdata, **kwargs)
            cname, tname = f'{name}c_exp', f'{name}t_exp'
            dataset.update({cname: cdata, tname: tdata})
    # Optionally drop terms
    drop = keys_implicit & dataset.data_vars.keys() - keys_keep
    if drop_implicit:
        dataset = dataset.drop_vars(drop)
    drop = keys_explicit & dataset.data_vars.keys()
    if drop_explicit:
        dataset = dataset.drop_vars(drop)

    # Get basic transport terms
    # NOTE: Here convergence can be broken down into just two components: a
    # stationary term and a transient term.
    # NOTE: Here transient sensible transport is calculated from the residual of the dry
    # static energy minus both the sensible and geopotential stationary components. The
    # all-zero transient geopotential is stored for consistency if sensible is present.
    for name, (quant, scale, _) in TRANSPORT_INDIVIDUAL.items():
        # Get mean transport, stationary transport, and stationary convergence
        if 'ps' not in dataset or 'va' not in dataset or quant not in dataset:
            continue
        descrip = TRANSPORT_DESCRIPTIONS[name]
        qdata = scale * dataset.climo.vars[quant]
        udata, vdata = dataset.climo.vars['ua'], dataset.climo.vars['va']
        cdata, sdata, mdata = _transport_explicit(udata, vdata, qdata, descrip=descrip)
        cname, sname, mname = f's{name}c', f's{name}t', f'm{name}t'
        dataset.update({cname: cdata, sname: sdata, mname: mdata})
    iter_ = itertools.product(TRANSPORT_INDIVIDUAL, ('cs', 'tsm'), ends)
    for name, (suffix, *prefixes), end in iter_:
        # Get missing transient components and total sensible and geopotential terms
        ref = f'{prefixes[0]}{name}{suffix}'  # reference component
        total = 'lse' if name == 'lse' else 'dse'
        total = f'{total}{suffix}{end}'
        parts = [f'{prefix}{name}{suffix}' for prefix in prefixes]
        resids = ('lse',) if name == 'lse' else ('gse', 'hse')
        resids = [f'{prefix}{resid}{suffix}' for prefix in prefixes for resid in resids]
        if total not in dataset or any(resid not in dataset for resid in resids):
            continue
        data = xr.zeros_like(dataset[ref])
        if name != 'gse':
            with xr.set_options(keep_attrs=True):
                data += dataset[total] - sum(dataset[resid] for resid in resids)
        data.attrs['long_name'] = data.long_name.replace('stationary', 'transient')
        dataset[f't{name}{suffix}{end}'] = data
        if name != 'lse':  # total is already present
            with xr.set_options(keep_attrs=True):  # add non-transient components
                data = data + sum(dataset[part] for part in parts)
            data.attrs['long_name'] = data.long_name.replace('transient ', '')
            dataset[f'{name}{suffix}{end}'] = data

    # Get combined components and inferred average fluxes
    # NOTE: Here a residual between total and storage + ocean would also suffice
    # but this also gets total transient and stationary static energy terms.
    # NOTE: Flux values will have units K/s, m2/s2, and g/kg m/s and are more
    # relevant to local conditions on a given latitude band. Could get vertically
    # resolved values for stationary components, but impossible for residual transient
    # component, so decided to only store this 'vertical average' for consistency.
    replacements = {'dse': ('sensible', 'dry'), 'mse': ('dry', 'moist')}
    dependencies = {'dse': ('hse', 'gse'), 'mse': ('dse', 'lse')}
    prefixes = ('', 'm', 's', 't')  # total, zonal-mean, stationary, transient
    suffixes = ('t', 'c', 'r')  # convergence, residual, transport
    iter_ = itertools.product(('dse', 'mse'), prefixes, suffixes, ends)
    for name, prefix, suffix, end in iter_:
        # Get dry and moist static energy from component transport and convergence terms
        variable = f'{prefix}{name}{suffix}{end}'
        parts = [variable.replace(name, part) for part in dependencies[name]]
        if variable in dataset or any(part not in dataset for part in parts):
            continue
        with xr.set_options(keep_attrs=True):
            data = sum(dataset[part] for part in parts)
        data.attrs['long_name'] = data.long_name.replace(*replacements[name])
        dataset[variable] = data
    iter_ = itertools.product(('hse', 'gse', 'lse'), prefixes, ends)
    for name, prefix, end in iter_:
        # Get average flux terms from the integrated terms
        if not fluxes:
            continue
        variable = f'{prefix}{name}t{end}'
        if variable not in dataset:
            continue
        _, scale, unit = TRANSPORT_INDIVIDUAL[name]
        denom = 2 * np.pi * np.cos(dataset.climo.coords.lat) * const.a
        denom = denom * dataset.climo.vars.ps.climo.average('lon') / const.g
        data = dataset[variable].climo.quantify()
        data = (data / denom / scale).climo.to_units(unit)
        data.attrs['long_name'] = dataset[variable].long_name.replace('transport', 'flux')  # noqa: E501
        data = data.climo.dequantify()
        dataset[f'{prefix}{name}f{end}'] = data
    return dataset


def _update_feedback_attrs(
    dataset,
    boundary=None,
    annual=True,
    seasonal=True,
    monthly=True,
    quadruple=False
):
    """
    Adjust feedback term attributes before plotting.

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset.
    boundary : {'t', 's', 'a'}, optional
        The boundari(es) to load. If one is passed then the indicator is stripped.
    annual, seasonal, monthly : bool, optoinal
        Whether to load different periods of data.
    quadruple : bool, optional
        Whether to label climate sensitivity assuming doubled or quadrupled CO$_2$.
    """
    # Flux metadata repairs
    # NOTE: Scaling of forcing or climate sensitivity terms for doubled or quadrupled
    # CO2 is handled by individual loading functions. Here just add a label indication.
    # NOTE: This drops pre-loaded climate sensitivity parameters, and only
    # keeps the boundary indicator if more than one boundary was requested.
    options = set(boundary or 't')
    wavelengths = ('full', 'longwave', 'shortwave')
    boundaries = ('surface', 'TOA')
    iter_ = itertools.product(boundaries, wavelengths, FEEDBACK_DESCRIPTIONS.items())
    for boundary, wavelength, (component, descrip) in iter_:
        rad = f'r{wavelength[0].lower()}n{boundary[0].lower()}'
        num = 4 if quadruple else 2
        for suffix, outdated, short in (
            ('lam', 'lambda', 'feedback'),
            ('erf', 'erf2x', 'forcing'),
            ('ecs', 'ecs2x', 'climate sensitivity'),
        ):
            if wavelength == 'full':  # WARNING: do not overwrite itertools 'descrip'
                tail = descrip if descrip else 'effective' if suffix == 'ecs' else 'net'
            else:
                tail = f'{wavelength} {descrip}' if descrip else wavelength
            if component in ('', 'cs'):
                prefix = f'{rad}{component}'
            else:
                prefix = f'{component}_{rad}'
            name = f'{prefix}_{suffix}'
            outdated = f'{prefix}_{outdated}'
            if outdated in dataset:
                dataset = dataset.rename({outdated: name})
            if name not in dataset:
                continue
            data = dataset[name]
            spec = rf'{num}$\times$CO$_2$' if suffix in ('erf', 'ecs') else ''
            head = boundary if len(options) > 1 else ''
            long = f'{spec} {head} {tail} {short}'
            long = re.sub('  +', ' ', long).strip()
            data.attrs['long_name'] = long
            data.attrs['short_name'] = short
            if suffix == 'ecs' and 'lon' in dataset[name].dims:
                dataset = dataset.drop_vars(name)

    # Other metadata repairs
    # NOTE: This mimics the behavior of average_periods in climate_datasets
    # by optionally skipping unneeded periods to save space.
    components = [
        ('pbot', 'plev_bot', 'lower', 'surface'),
        ('ptop', 'plev_top', 'upper', 'tropopause')
    ]
    for name, original, outdated, boundary in components:
        if outdated in dataset:
            dataset = dataset.rename({outdated: original})
        if original in dataset:
            dataset = dataset.rename({original: name})
        if name in dataset:
            dataset[name].attrs['standard_units'] = 'hPa'
            dataset[name].attrs.setdefault('units', 'Pa')
            dataset[name].attrs.setdefault('long_name', f'{boundary} pressure')
    if 'period' in dataset:  # this is for consistency with climate_datasets
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


def _update_feedback_terms(
    dataset,
    boundary=None,
    erfextra=None,
    wavextra=None,
    quadruple=False,
):
    """
    Add net cloud effect and net atmospheric feedback terms, possibly filter out
    unnecessary terms, and standardize the insertion order for the dataset variables.

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset.
    boundary : {'t', 's', 'a'}, optional
        The boundari(es) to load. Pass a tuple or longer string for more than one.
    erfextra : bool, optional
        Whether to include the kernel-derived effective forcing components.
    wavextra : bool, optional
        Whether to include the humidity feedback wavelength components.
    quadruple : bool, optional
        Whether to label forcing and sensitivity assuming doubled or quadrupled CO$_2$.
    """
    # Helper function
    # TODO: Consider placing this function *after* _update_feedback_attrs rather
    # than before and having that function handle derived component labels.
    # NOTE: This is necessary so e.g. we can add atmospheric versions of combined
    # cloud effect and non-cloud feedbacks.
    regex = re.compile(r'(\A|[^_]*)(_?r)([lsf])([udn])([tsa])(|cs|ce)(?=_)')
    boundary = boundary or 't'
    def _iter_dataset(dataset, boundary=boundary, erfextra=erfextra, wavextra=wavextra):  # noqa: E301, E501
        erfparts = ('', 'cl', 'alb', 'atm', 'resid')
        wavparts = ('', 'cl')
        for key in dataset:
            if 'ecs' in key and 'lon' in dataset[key].dims:  # ignore outdated values
                continue
            if not (m := regex.search(key)):
                continue
            if boundary and m.group(5) not in boundary:
                continue
            if not erfextra and 'erf' in key and m.group(1) not in erfparts:
                continue
            if not wavextra and m.group(3) != 'f' and m.group(1) not in wavparts:
                continue
            yield key, m

    # Add atmospheric feedbacks
    # NOTE: This requires iterating over both surface and top-of-atmosphere boundary
    # variables. If user requested e.g. *only* 'a' feedbacks then skipped below.
    for key, m in _iter_dataset(dataset, boundary=None):
        if m.group(5) == 't':
            atm = regex.sub(r'\1\2\3\4a\6', key)
            sfc = regex.sub(r'\1\2\3\4s\6', key)
            if sfc in dataset and 'a' in boundary:
                with xr.set_options(keep_attrs=True):  # keep units and short_name
                    dataset[atm] = dataset[key] + dataset[sfc]
                long_name = dataset[key].attrs['long_name']
                long_name = long_name.replace('TOA', 'atmospheric')
                dataset[atm].attrs['long_name'] = long_name

    # Add non-cloud temperature + humidity feedbacks
    # NOTE: Want to include 'non-cloud' effective forcing similar to cloud effective
    # forcing due to fast adjustments, so skip the erfextra check.
    for key, m in _iter_dataset(dataset, erfextra=True, wavextra=False):
        if m.group(1) == 'pl':
            atm = regex.sub(r'atm\2\3\4\5\6', key)
            hus = regex.sub(r'hus\2\3\4\5\6', key)
            lr = regex.sub(r'lr\2\3\4\5\6', key)
            if lr in dataset and hus in dataset:
                with xr.set_options(keep_attrs=True):  # keep units and short_name
                    dataset[atm] = dataset[key] + dataset[hus] + dataset[lr]
                long_name = dataset[key].attrs.get('long_name', '')
                long_name = long_name.replace('Planck', 'temperature + humidity')
                dataset[atm].attrs['long_name'] = long_name

    # Add cloud effect feedbacks
    # NOTE: Here use 'ce' for 'cloud effect' to differentiate from the loaded term
    # 'forcing' used in e.g. effective forcing estimates 'cl_rfnt_erf'.
    for key, m in _iter_dataset(dataset, erfextra=erfextra, wavextra=wavextra):
        if m.group(1) == '' and m.group(6) == '':
            ce = regex.sub(r'\1\2\3\4\5ce', key)
            cs = regex.sub(r'\1\2\3\4\5cs', key)
            if cs in dataset:
                with xr.set_options(keep_attrs=True):  # keep units and short_name
                    dataset[ce] = dataset[key] - dataset[cs]
                long_name = dataset[cs].attrs.get('long_name', '')
                long_name = long_name.replace('clear-sky', 'cloud')
                dataset[ce].attrs['long_name'] = long_name

    # Add climate sensitivity estimates
    # NOTE: Previously computed climate sensitivity 'components' based on individual
    # effective forcings and feedbacks but interpretation is not useful. Now store
    # zero sensitivity components and only compute after the fact.
    if 't' in boundary and 'rfnt_lam' in dataset and 'rfnt_erf' in dataset:
        if 'rfnt_ecs' not in dataset or 'lon' in dataset['rfnt_ecs'].dims:
            num = 4 if quadruple else 2
            label = rf'{num}$\times$CO$_2$ effective climate sensitivity'
            numer = dataset.rfnt_erf.climo.add_cell_measures()
            denom = dataset.rfnt_lam.climo.add_cell_measures()
            data = -1 * numer.climo.average('area') / denom.climo.average('area')
            data.attrs['units'] = 'K'
            data.attrs['short_name'] = 'temperature'
            data.attrs['long_name'] = label
            dataset['rfnt_ecs'] = data
    keys = ['pbot', 'ptop', 'rfnt_ecs', *(key for key, _ in _iter_dataset(dataset))]
    dataset = dataset.drop_vars(dataset.data_vars.keys() - set(keys))
    return dataset


def climate_datasets(
    *paths, years=None, nodrift=False, standardize=True, **constraints
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
    # NOTE: Non-flagship simulations can be added with flagship_translate=True
    # as with other processing functions. Ensembles are added in a MultiIndex
    # NOTE: This loads all available variables by default, but can be
    # restricted to a few with e.g. variable=['ts', 'ta'].
    keys_periods = ('annual', 'seasonal', 'monthly')
    keys_energetics = ('clear', 'drop_components')
    keys_transport = ('fluxes', 'drop_explicit', 'drop_implicit')
    keys_water = ('liquid', 'ratio')
    kw_energetics = {key: constraints.pop(key) for key in keys_energetics if key in constraints}  # noqa: E501
    kw_transport = {key: constraints.pop(key) for key in keys_transport if key in constraints}  # noqa: E501
    kw_water = {key: constraints.pop(key) for key in keys_water if key in constraints}
    kw_periods = {key: constraints.pop(key) for key in keys_periods if key in constraints}  # noqa: E501
    files, *_ = glob_files(*paths, project=constraints.get('project', None))
    constraints.setdefault('table', ['Amon', 'Emon'])
    constraints.setdefault('experiment', ['piControl', 'abrupt4xCO2'])
    database = Database(files, FACETS_LEVELS, flagship_translate=True, **constraints)
    database.filter(always_exclude={'variable': ['pfull']})  # skip dependencies
    nodrift = nodrift and '-nodrift' or ''
    datasets = {}
    print(f'Climate files: <dates>-climate{nodrift}')
    print(f'Number of climate files: {len(database)}.')
    if database:
        print('Model:', end=' ')
    for group, data in database.items():
        # Initial stuff
        if not data:
            continue
        for sub, replace in FACETS_RENAME.items():
            group = tuple(s.replace(sub, replace) for s in group)
        if years is not None:
            range_ = years
        elif 'abrupt4xco2' in group or 'response' in group:  # TODO: deprecate
            range_ = (120, 150)
        elif 'picontrol' in group or 'control' in group:  # TODO: deprecate
            range_ = (0, 150)
        dates = f'{range_[0]:04d}-{range_[1]:04d}-climate{nodrift}'
        print(f'{group[1]}_{group[2]}_{range_[0]:04d}-{range_[1]:04d}', end=' ')
        att = {'axis': 'T', 'standard_name': 'time'}
        time = pd.date_range('2000-01-01', '2000-12-01', freq='MS')
        time = xr.DataArray(time, name='time', dims='time', attrs=att)

        # Load the data
        # NOTE: Critical to overwrite the time coordinates after loading or else xarray
        # coordinate matching will apply all-NaN values for climatolgoies with different
        # base years (e.g. due to control data availability or response calendar diffs).
        dataset = xr.Dataset()
        for key, paths in data.items():
            variable = key[database.key.index('variable')]
            paths = [path for path in paths if _item_dates(path) == dates]
            if not paths:
                continue
            if len(paths) > 1:
                print(f'Warning: Skipping ambiguous duplicate paths {list(map(str, paths))}.', end=' ')  # noqa: E501
                continue
            array = load_file(paths[0], variable, project=database.project)
            if array.time.size != 12:
                print(f'Warning: Skipping path {paths[0]} with time length {array.time.size}.', end=' ')  # noqa: E501
                continue
            months = array.time.dt.month
            if sorted(months.values) != sorted(range(1, 13)):
                print(f'Warning: Skipping path {paths[0]} with month values {months.values}.', end=' ')  # noqa: E501
                continue
            array = array.assign_coords(time=time)
            descrip = array.attrs.pop('title', variable)  # in case long_name missing
            descrip = array.attrs.pop('long_name', descrip)
            descrip = ' '.join(s if s == 'TOA' else s.lower() for s in descrip.split())
            array.attrs['long_name'] = descrip
            dataset[variable] = array

        # Standardize the data
        # NOTE: Empirical testing revealed limiting integration to troposphere
        # often prevented strong transient heat transport showing up in overturning
        # cells due to aliasing of overemphasized stratospheric geopotential transport.
        # WARNING: Critical to place average_periods after adjustments so that
        # time-covariance of surface pressure and near-surface flux terms is
        # effectively factored in (since average_periods only includes explicit
        # month-length weights and ignores implicit cell height weights).
        if 'ps' not in dataset and 'plev' in dataset.coords:
            print('Warning: Surface pressure is unavailable.', end=' ')
        dataset = dataset.climo.add_cell_measures(surface=('ps' in dataset))
        dataset = _update_climate_energetics(dataset, **kw_energetics)  # must come first  # noqa: E501
        dataset = _update_climate_transport(dataset, **kw_transport)
        dataset = _update_climate_moisture(dataset, **kw_water)
        dataset = _update_climate_units(dataset)  # must come after transport
        if 'time' in dataset:
            dataset = average_periods(dataset, **kw_periods)
        if standardize:
            dataset = _standardize_order(dataset)
        if 'plev' in dataset:
            dataset = dataset.sel(plev=slice(None, 7000))
        drop = ['cell_', '_bot', '_top']
        drop = [key for key in dataset.coords if any(o in key for o in drop)]
        dataset = dataset.drop_vars(drop)
        dataset = dataset.squeeze()
        datasets[group] = dataset

    if datasets:
        print()
    return datasets


def feedback_datasets(
    *paths,
    source=None,
    nodrift=False,
    standardize=True,
    quadruple=False,
    **constraints,
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
    standardize : bool, optional
        Whether to standardize the resulting order.
    quadruple : bool, optional
        Whether to adjust parameters to correspond to quadrupled CO$_2$.
    **kwargs
        Passed to `_update_feedback_terms`.
    **constraints
        Passed to `Database`.

    Returns
    -------
    datasets : dict
        A dictionary of datasets.
    """
    # NOTE: To reduce the number of variables this filters out
    # unneeded boundaries and effective forcings automatically.
    keys_periods = ('annual', 'seasonal', 'monthly')
    keys_shared = ('boundary',)
    keys_terms = ('erfextra', 'wavextra')
    kw_shared = {key: constraints.pop(key) for key in keys_shared if key in constraints}
    kw_terms = {key: constraints.pop(key) for key in keys_terms if key in constraints}
    kw_periods = {key: constraints.pop(key) for key in keys_periods if key in constraints}  # noqa: E501
    kw_terms['quadruple'] = kw_periods['quadruple'] = quadruple
    files, *_ = glob_files(*paths, project=constraints.get('project', None))
    constraints['variable'] = 'feedbacks'  # TODO: similar for climate_datasets
    database = Database(files, FACETS_LEVELS, flagship_translate=True, **constraints)
    sources = (source,) if isinstance(source, str) else tuple(source or ())
    nodrift = nodrift and '-nodrift' or ''
    factor = 2.0 if quadruple else 1.0  # TODO: possibly change conventions
    datasets = {}
    print(f'Feedback files: <source>-<statistic>{nodrift}')
    print(f'Number of feedback files: {len(database)}.')
    if database:
        print('Model:', end=' ')
    for group, data in database.items():
        # Filter the files
        bnds, parts, versions = {}, {}, {}
        for sub, replace in FACETS_RENAME.items():
            group = tuple(s.replace(sub, replace) for s in group)
        paths = tuple(
            path for paths in data.values() for path in paths
            if bool(nodrift) == bool('nodrift' in path.name)
        )
        if not paths:
            continue

        # Load the data
        # NOTE: This accounts for files with dedicated regions indicated in the name,
        # files with numerator and denominator multi-index coordinates, and files with
        # just a denominator region coordinate. Note load_file builds the multi-index.
        print(f'{group[1]}_{group[2]}', end=' ')
        for path in paths:
            *_, indicator, suffix = path.stem.split('_')
            start, stop, source, statistic, *_ = suffix.split('-')  # ignore -nodrift
            start, stop = map(int, (start, stop))
            if sources and source not in sources:
                continue
            dataset = load_file(path, project=database.project, validate=False)
            if outdated := 'local' in indicator or 'global' in indicator:
                if indicator.split('-')[0] != 'local':  # ignore local vs. global
                    continue
            for array in dataset.data_vars.values():
                if '_erf' in array.name or '_ecs' in array.name:
                    array *= factor  # NOTE: array.data *= fails for unloaded data
            if outdated:
                region = 'point' if indicator.split('-')[1] == 'local' else 'globe'
                versions[source, statistic, region, start, stop] = dataset
            else:
                for region in dataset.region.values:  # includes ts_pattern, pbot, ptop
                    sel = dataset.sel(region=region, drop=True)
                    versions[source, statistic, region, start, stop] = sel

        # Concatenate the data
        # NOTE: Concatenation automatically broadcasts global feedbacks across lons and
        # lats. Also critical to use 'override' for combine_attrs in case conventions
        # changed between running feedback calculations on different models.
        for key, dataset in versions.items():
            dataset = _update_feedback_attrs(dataset, **kw_shared, **kw_periods)
            dataset = _update_feedback_terms(dataset, **kw_shared, **kw_terms)
            if 'pbot' in dataset:  # always represents control pressure
                bnds['pbot'] = dataset['pbot']
            if 'ptop' in dataset:
                bnds['ptop'] = dataset['ptop']
            dataset = dataset.drop_vars({'pbot', 'ptop'} & dataset.keys())
            parts[key] = dataset
        index = xr.DataArray(
            pd.MultiIndex.from_tuples(parts, names=VERSION_LEVELS),
            dims='version',
            name='version',
            attrs={'long_name': 'feedback version'},
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
        if standardize:
            dataset = _standardize_order(dataset)
        datasets[group] = dataset

    if datasets:
        print()
    return datasets


def feedback_datasets_json(
    path='~/data/cmip-tables',
    boundary=None,
    standardize=True,
    quadruple=False,
    **constraints
):
    """
    Return a dictionary of datasets containing json-provided feedback data.

    Parameters
    ----------
    path : path-like, optional
        The base path.
    boundary : str, optional
        The boundary components.
    standardize : bool, optional
        Whether to standardize the resulting order.
    quadruple : bool, optional
        Whether to adjust parameters to correspond to quadrupled CO$_2$.
    **constraints
        Passed to `_parse_constraints`.

    Returns
    -------
    datasets : dict
        A dictionary of datasets.
    """
    # NOTE: When combinining with 'open_dataset' the non-descriptive long names
    # here should be overwritten by long names from custom feedbacks.
    path = Path(path).expanduser()
    boundary = boundary or 't'
    project, constraints = _parse_constraints(reverse=True, **constraints)
    factor = 2.0 if quadruple else 1.0
    datasets = {}
    if 't' not in boundary:  # only top-of-atmosphere feedbacks available
        return datasets
    for file in sorted(path.glob('cmip*.json')):
        source = file.stem.split('_')[1]
        print(f'External file: {file.name}')
        index = (source, 'slope', 'globe', 0, 150)
        index = xr.DataArray(
            pd.MultiIndex.from_tuples([index], names=VERSION_LEVELS),
            dims='version',
            name='version',
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
                group = (project, model, 'abrupt4xco2', ensemble)
                dataset = xr.Dataset()
                for key, value in data.items():
                    name, units = FEEDBACK_TRANSLATIONS[key.lower()]
                    attrs = {'units': units}  # long name assigned below
                    if '_erf' in name or '_ecs' in name:
                        value = factor * value
                    dataset[name] = xr.DataArray(value, attrs=attrs)
                dataset = dataset.expand_dims(version=1)
                dataset = dataset.assign_coords(version=index)
                if group in datasets:
                    datasets[group].update(dataset)
                else:
                    datasets[group] = dataset
    for group in tuple(datasets):
        dataset = datasets[group]
        dataset = _update_feedback_attrs(dataset, boundary='t')
        dataset = _update_feedback_terms(dataset, boundary='t')
        if standardize:
            dataset = _standardize_order(dataset)
        datasets[group] = dataset
    return datasets


def feedback_datasets_text(
    path='~/data/cmip-tables',
    boundary=None,
    standardize=True,
    quadruple=False,
    **constraints,
):
    """
    Return a dictionary of datasets containing text-provided feedback data.

    Parameters
    ----------
    path : path-like, optional
        The base path.
    boundary : str, optional
        The boundary components.
    standardize : bool, optional
        Whether to standardize the resulting order.
    quadruple : bool, optional
        Whether to adjust parameters to correspond to quadrupled CO$_2$.
    **constraints
        Passed to `_parse_constraints`.

    Returns
    -------
    datasets : dict
        A dictionary of datasets.
    """
    # NOTE: The Zelinka, Geoffry, and Forster papers and sources only specify a
    # CO2 multiple of '2x' or '4x' in the forcing entry and just say 'ecs' for the
    # climate sensitivity. So detect the multiple by scanning all keys in the table.
    path = Path(path).expanduser()
    boundary = boundary or 't'
    project, constraints = _parse_constraints(reverse=True, **constraints)
    datasets = {}
    if 't' not in boundary:  # only top-of-atmosphere feedbacks available
        return datasets
    for file in sorted(path.glob(f'{project.lower()}*.txt')):
        source = file.stem.split('_')[1]
        if source == 'zelinka':
            continue
        print(f'External file: {file.name}')
        index = (source, 'slope', 'globe', 0, 150)
        index = xr.DataArray(
            pd.MultiIndex.from_tuples([index], names=VERSION_LEVELS),
            dims='version',
            name='version',
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
        dataset = dataset.expand_dims(version=1)
        dataset = dataset.assign_coords(version=index)
        factor = 0.5 if any('4x' in key for key in dataset.data_vars) else 1.0
        factor = 2 * factor if quadruple else factor
        for key, array in dataset.data_vars.items():
            name, units = FEEDBACK_TRANSLATIONS[key.lower()]
            for model in dataset.model.values:
                group = ('CMIP5', model, 'abrupt4xco2', 'flagship')
                if model not in constraints.get('model', (model,)):
                    continue
                if 'flagship' not in constraints.get('ensemble', ('flagship',)):
                    continue
                if any(lab in model for lab in ('mean', 'deviation', 'uncertainty')):
                    continue
                data = array.sel(model=model, drop=True)
                if units in ('K', 'W m^-2'):  # avoid in-place operation
                    data = factor * data
                data.name = name
                data.attrs.update({'units': units})  # long name assigned below
                data = data.to_dataset()
                if group in datasets:  # combine new feedback coordinates
                    tup = (data, datasets[group])
                    data = xr.combine_by_coords(tup, combine_attrs='override')
                datasets[group] = data
    for group in tuple(datasets):
        dataset = datasets[group]
        dataset = _update_feedback_attrs(dataset, boundary='t')
        dataset = _update_feedback_terms(dataset, boundary='t')
        if standardize:
            dataset = _standardize_order(dataset)
        datasets[group] = dataset
    return datasets


def open_dataset(
    project=None,
    climate=True,
    feedbacks=True,
    feedbacks_json=None,
    feedbacks_text=None,
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
    # latter is required when adding new 'facets' and 'version' coordinate values.
    keys_both = ('nodrift', 'annual', 'seasonal', 'monthly')
    keys_joint = ('boundary',)
    keys_climate = ('years',)
    keys_feedbacks = ('source', 'erfextra', 'wavextra')
    kw_both = {k: constraints.pop(k) for k in keys_both if k in constraints}
    kw_joint = {k: constraints.pop(k) for k in keys_joint if k in constraints}
    kw_climate = {k: constraints.pop(k) for k in keys_climate if k in constraints}
    kw_feedbacks = {k: constraints.pop(k) for k in keys_feedbacks if k in constraints}
    base = Path('~/data').expanduser()
    datasets = {}
    projects = project.split(',') if isinstance(project, str) else ('cmip5', 'cmip6')
    if feedbacks_json is None:
        feedbacks_json = feedbacks
    if feedbacks_text is None:
        feedbacks_text = feedbacks
    for project in map(str.upper, projects):
        print(f'Project: {project}')
        for b, function, folder, kw in (
            (climate, climate_datasets, 'cmip-climate', {**kw_climate, **kw_both}),
            (feedbacks, feedback_datasets, 'cmip-feedbacks', {**kw_feedbacks, **kw_joint, **kw_both}),  # noqa: E501
            (feedbacks_json, feedback_datasets_json, 'cmip-tables', {**kw_joint}),  # noqa: E501
            (feedbacks_text, feedback_datasets_text, 'cmip-tables', {**kw_joint}),  # noqa: E501
        ):
            if not b:
                continue
            if isinstance(b, (tuple, list)):
                paths = tuple(Path(_).expanduser() for _ in b)
            elif isinstance(b, (str, Path)):
                paths = (Path(b).expanduser(),)
            else:
                paths = (sub if (sub := base / folder).is_dir() else base,)
            kwargs = {**constraints, 'project': project, **kw}
            parts = function(*paths, standardize=False, **kwargs)
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
        print(f'{group[1]}_{group[2]}', end=' ')
        for name in names.keys() - dataset.data_vars.keys():
            da = names[name]  # *sample* from another model or project
            da = xr.full_like(da, np.nan)  # preserve attributes as well
            if 'version' in da.dims and 'version' in dataset:
                da = da.isel(version=0, drop=True)
                da = da.expand_dims(version=len(dataset.version))
                da = da.assign_coords(version=dataset.version)
            dataset[name] = da
    print()
    print('Concatenating datasets.')
    index = xr.DataArray(
        pd.MultiIndex.from_tuples(datasets, names=FACETS_LEVELS),
        dims='facets',
        name='facets',
        attrs={'long_name': 'source facets'},
    )
    dataset = xr.concat(
        datasets.values(),
        dim=index,
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
    if 'plev_bot' in dataset:  # causes issues as coordinate variable and duplicates ps
        dataset = dataset.drop_vars('plev_bot')
    return dataset
