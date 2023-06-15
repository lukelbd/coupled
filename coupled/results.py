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

# Regular expressions
# NOTE: This omits the final possible suffixes e.g. '_lam' or '_ecs'. The
# specifiers relevant for use are in groups \1, \3, \5, and \6.
# NOTE: Here direction 'e' stands for effective forcing and 'r' stands for
# radiative response (i.e. net minus effective forcing). This is good idea.
REGEX_EXP = re.compile(r'([a-df-zA-DF-Z]+)([-+]?[0-9]+)')
REGEX_FLUX = re.compile(r'(\A|[^_]*)(_?r)([lsf])([udner])([tsa])(cs|ce|)')
REGEX_DIREC = re.compile(r'(upwelling|downwelling|outgoing|incident)')
REGEX_TRANSPORT = re.compile(r'(mse|total)')

# Keyword arguments
# NOTE: Shared between functions
KEYS_PERIODS = ('annual', 'seasonal', 'monthly')
KEYS_REGIONS = ('point', 'latitude', 'hemisphere')
KEYS_RANGES = ('early', 'late', 'historical')
KEYS_FEEDBACKS = ('parts_erf', 'parts_wav', 'parts_clear', 'parts_kernels', 'parts_planck', 'parts_relative', 'parts_absolute')  # noqa: E501
KEYS_ENERGY = ('drop_clear', 'drop_directions', 'skip_solar')
KEYS_TRANSPORT = ('parts_local', 'parts_eddies', 'parts_static', 'parts_fluxes')
KEYS_VERSION = ('implicit', 'alternative', 'explicit', 'drop_components')
KEYS_MOIST = ('parts_cloud', 'parts_precip', 'parts_phase')

# Models to skip
# NOTE: Went through trouble of processing these models but cannot compute cloud
# feedbacks... would be confusing to include them in net feedback analyses but
# exclude them from cloud feedback analyses. Skip until they provide more data.
MODELS_SKIP = (
    'MCM-UA-1-0',
    'FIO-ESM-2-0',
)

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
FEEDBACK_SETTINGS = {
    'ecs': ('rfnt_ecs', 'K'),  # zelinka definition
    'tcr': ('rfnt_tcr', 'K'),  # forster definition
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
    'cs': ('rfntcs_lam', 'W m^-2 K^-1'),
    'swcs': ('rsntcs_lam', 'W m^-2 K^-1'),  # forster definition
    'lwcs': ('rlntcs_lam', 'W m^-2 K^-1'),  # forster definition
    'ce': ('rfntce_lam', 'W m^-2 K^-1'),
    'swce': ('rsntce_lam', 'W m^-2 K^-1'),
    'lwce': ('rlntce_lam', 'W m^-2 K^-1'),
    'cre': ('rfntce_lam', 'W m^-2 K^-1'),  # forster definition
    'swcre': ('rsntce_lam', 'W m^-2 K^-1'),
    'lwcre': ('rlntce_lam', 'W m^-2 K^-1'),
    'cld': ('cl_rfnt_lam', 'W m^-2 K^-1'),  # zelinka definition
    'lwcld': ('cl_rlnt_lam', 'W m^-2 K^-1'),  # zelinka definition
    'swcld': ('cl_rsnt_lam', 'W m^-2 K^-1'),  # zelinka definition
    'ncl': ('ncl_rfnt_lam', 'W m^-2 K^-1'),
    'lwncl': ('ncl_rlnt_lam', 'W m^-2 K^-1'),  # not currently used
    'swncl': ('ncl_rsnt_lam', 'W m^-2 K^-1'),  # not currently used
    'alb': ('alb_rfnt_lam', 'W m^-2 K^-1'),  # zelinka definition (full is 'albedo')
    'atm': ('atm_rfnt_lam', 'W m^-2 K^-1'),
    'pl': ('pl_rfnt_lam', 'W m^-2 K^-1'),  # zelinka definition
    'pl*': ('pl*_rfnt_lam', 'W m^-2 K^-1'),  # zelinka definition
    'lr': ('lr_rfnt_lam', 'W m^-2 K^-1'),  # zelinka definition
    'lr*': ('lr*_rfnt_lam', 'W m^-2 K^-1'),  # zelinka definition
    'wv': ('hus_rfnt_lam', 'W m^-2 K^-1'),  # zelinka definition
    'rh': ('hur_rfnt_lam', 'W m^-2 K^-1'),  # zelinka definition
    'err': ('resid_rfnt_lam', 'W m^-2 K^-1'),  # forster definition
    'resid': ('resid_rfnt_lam', 'W m^-2 K^-1'),  # preferred name last (for reverse translation)  # noqa: E501
}

# Add additional feedback aliases
# NOTE: This is used for plotting stuff and things
_feedback_aliases = {alias: name for alias, (name, _) in FEEDBACK_SETTINGS.items()}
ALIAS_FEEDBACKS = {
    **{f't{alias}': name for alias, name in _feedback_aliases.items()},
    **{f's{alias}': name.replace('t_', 's_') for alias, name in _feedback_aliases.items()},  # noqa: E501
    **{f'a{alias}': name.replace('t_', 'a_') for alias, name in _feedback_aliases.items()},  # noqa: E501
    **_feedback_aliases,
}
FEEDBACK_ALIASES = {value: key for key, value in ALIAS_FEEDBACKS.items()}

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
    'albedo': ('rsds', 'rsus'),  # evaluates to rsus / rsds (out over in)
    'rlnt': ('rlut',),  # evaluates to -rlut (minus out)
    'rsnt': ('rsut', 'rsdt'),  # evaluates to rsdt - rsut (in minus out)
    'rlns': ('rlds', 'rlus'),  # evaluates to rlus - rlds (in minus out)
    'rsns': ('rsds', 'rsus'),  # evaluates to rsus - rsds (in minus out)
    'rlntcs': ('rlutcs',),  # evaluates to -rlutcs (minus out)
    'rsntcs': ('rsutcs', 'rsdt'),  # evaluates to rsdt - rsutcs (in minus out)
    'rlnscs': ('rldscs', 'rlus'),  # evaluates to rlus - rldscs (in minus out)
    'rsnscs': ('rsdscs', 'rsuscs'),  # evaluates to rsuscs - rsdscs (in minus out)
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
    'ocean': 'storage + ocean',
}
TRANSPORT_SCALES = {  # scaling prior to implicit transport calculations
    'pr': const.Lv,
    'prl': const.Lv,
    'pri': const.Ls - const.Lv,  # remove the 'pri * Lv' implied inside 'pr' term
    'ev': const.Lv,
    'evl': const.Lv,
    'evi': const.Ls - const.Lv,  # remove the 'evi * Lv' implied inside 'pr' term
}
TRANSPORT_EXPLICIT = {
    'dse': (1, 'intuadse', 'intvadse'),
    'lse': (const.Lv, 'intuaw', 'intvaw'),
}
TRANSPORT_IMPLICIT = {
    'dse': (('hfss', 'rlns', 'rsns', 'rlnt', 'rsnt', 'pr', 'pri'), ()),
    'lse': (('hfls',), ('pr', 'pri')),
    'ocean': ((), ('hfss', 'hfls', 'rlns', 'rsns')),  # flux into atmosphere
}
TRANSPORT_INDIVIDUAL = {
    'gse': ('zg', const.g, 'dam m s^-1'),
    'hse': ('ta', const.cp, 'K m s^-1'),
    'lse': ('hus', const.Lv, 'g kg^-1 m s^-1'),
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
    moisture = list(name for names in MOISTURE_COMPONENTS for name in names[:4])
    fluxes = list(itertools.product(('f', 'l', 's'), ('t', 's', 'a'), ('', 'cs', 'ce')))
    fluxes = list(f'r{wav}n{bnd}{sky}' for wav, bnd, sky in fluxes)  # capture fluxes
    transports = list(itertools.product(eddies, statics, locals, ends))
    transports = list(''.join(tup) for tup in transports)
    feedbacks = list(itertools.product(parts, fluxes, params))
    feedbacks = list('_'.join(tup).strip('_') for tup in feedbacks)
    # Update insertion order
    basics = ('ta', 'ts', 'tstd', 'tpat', 'tabs', 'ps', 'psl', 'pbot', 'ptop')
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


def _update_climate_energetics(
    dataset, drop_clear=False, drop_directions=True, correct_solar=True,
):
    """
    Add albedo and net fluxes from upwelling and downwelling components and remove
    the original directional variables.

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset.
    drop_clear : bool, default: False
        Whether to drop the clear-sky components.
    drop_directions : bool, optional
        Whether to drop the directional components
    correct_solar : bool, optional
        Whether to correct zonal variations in insolation by replacing with averages.
    """
    # NOTE: Here also add 'albedo' term and generate long name by replacing directional
    # terms in existing long name. Remove the directional components when finished.
    for key, name in ENERGETICS_RENAMES.items():
        if key in dataset:
            dataset = dataset.rename({key: name})
    keys_directions = set()
    if correct_solar and 'rsdt' in dataset:
        data = dataset.rsdt.mean(dim='lon', keepdims=True)
        dataset.rsdt[:] = data  # automatically broadcasts
    for name, keys in ENERGETICS_COMPONENTS.items():
        keys_directions.update(keys)
        if any(key not in dataset for key in keys):  # skip partial data
            continue
        if drop_clear and name[-2:] == 'cs':
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
        long_name = REGEX_DIREC.sub('net', long_name)
        dataset[name].attrs.update({'units': unit, 'long_name': long_name})
    if drop_directions:
        dataset = dataset.drop_vars(keys_directions & dataset.data_vars.keys())
    return dataset


def _update_climate_hydrology(
    dataset, parts_cloud=True, parts_precip=False, parts_phase=False,
):
    """
    Add relative humidity and ice and liquid water terms and standardize
    the insertion order for the resulting dataset variables.

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset.
    parts_cloud : bool, optional
        Whether to keep cloud columnwise and layerwise ice components.
    parts_precip : bool, optional
        Whether to keep evaporation and precipitation ice components.
    parts_phase : bool, default: False
        Whether to keep separate liquid and ice parts in addition to ice ratio.
    """
    # Add humidity terms
    # NOTE: Generally forego downloading relative humidity variables... true that
    # operation is non-linear so relative humidity of climate is not climate of
    # relative humiditiy, but we already use this approach with feedback kernels.
    for name, keys in MOISTURE_DEPENDENCIES.items():
        descrip = 'near-surface ' if name[-1] == 's' else ''
        long_name = f'{descrip}relative humidity'
        if all(key in dataset for key in keys):
            datas = []
            for key, unit in zip(keys, ('Pa', 'K', '')):
                src = dataset.climo.coords if key == 'plev' else dataset.climo.vars
                data = src[key].climo.to_units(unit)  # climopy units
                data.data = data.data.magnitude * munits.units(unit)  # metpy units
                datas.append(data)
            data = mcalc.relative_humidity_from_specific_humidity(*datas)
            data = data.climo.dequantify()  # works with metpy registry
            data = data.climo.to_units('%').clip(0, 100)
            data.attrs['long_name'] = long_name
            dataset[name] = data

    # Add cloud terms
    # NOTE: Unlike related variables (including, confusingly, clwvi), 'clw' includes
    # only liquid component rather than combined liquid plus ice. Adjust it to match
    # convention from other variables and add other component terms.
    if 'clw' in dataset:
        dataset = dataset.rename_vars(clw='cll')
        if 'cli' in dataset:
            with xr.set_options(keep_attrs=True):
                dataset['clw'] = dataset['cli'] + dataset['cll']
    for name, lname, iname, rname, descrip in MOISTURE_COMPONENTS:
        skip_parts = not parts_cloud and name in ('clw', 'clwvi')
        skip_parts = skip_parts or not parts_precip and name in ('ev', 'pr')
        if not skip_parts and name in dataset and iname in dataset:
            da = (100 * dataset[iname] / dataset[name]).clip(0, 100)
            da.attrs = {'units': '%', 'long_name': descrip % 'ice' + ' ratio'}
            dataset[rname] = da
        if not skip_parts and parts_phase and name in dataset and iname in dataset:
            da = dataset[name] - dataset[iname]
            da.attrs = {'units': dataset[name].units, 'long_name': descrip % 'liquid'}
            dataset[lname] = da
        if skip_parts:  # remove everything except total
            drop = dataset.data_vars.keys() & {lname, iname, rname}
            dataset = dataset.drop_vars(drop)
        if not parts_phase:  # remove everything except ratio and total
            drop = dataset.data_vars.keys() & {lname, iname}
            dataset = dataset.drop_vars(drop)
    for name, lname, iname, _, descrip in MOISTURE_COMPONENTS:
        names = (name, lname, iname)
        strings = ('water', 'liquid', 'ice')
        for name, string in zip(names, strings):
            if name not in dataset:
                continue
            data = dataset[name]
            if string not in data.long_name:
                data.attrs['long_name'] = descrip % string
    return dataset


def _update_climate_transport(
    dataset,
    implicit=True, alternative=False, explicit=False, drop_implicit=False, drop_explicit=True,  # noqa: E501
    parts_local=True, parts_static=False, parts_eddies=False, parts_fluxes=False,
):
    """
    Add local transport convergence and zonal-mean meridional transport including
    sensible-geopotential-latent and mean-stationary-transient breakdowns.

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset.
    implicit : bool, optional
        Whether to include implicit transport estimates.
    alternative : bool, optional
        Whether to include alternative transport estimates.
    explicit : bool, optional
        Whether to include explicit transport estimates.
    drop_implicit : bool, optional
        Whether to drop energetic components used for implicit transport.
    drop_explicit : bool, optional
        Whether to drop flux components used for explicit transport.
    parts_local : bool, optional
        Whether to include local transport components.
    parts_eddies : bool, optional
        Whether to include zonal-mean stationary and transient components.
    parts_static : bool, optional
        Whether to include separate sensible and geopotential components.
    parts_fluxes : bool, default: False
        Whether to compute additional flux estimates.
    """
    # Get implicit ocean, dry, and latent transport
    # WARNING: This must come after _update_climate_energetics introduces 'net'
    # components. Also previously removed contributors to implicit calculations
    # but not anymore... might consider adding back.
    idxs, ends = (0,), ('',)
    if alternative:
        idxs, ends = (0, 1), ('', '_alt')
    keys_implicit = set()  # components should be dropped
    for (name, pair), idx in itertools.product(TRANSPORT_IMPLICIT.items(), idxs):
        end, constants = ends[idx], {}
        for c, keys in zip((1, -1), map(list, pair)):
            if end and 'hfls' in keys:
                keys[(idx := keys.index('hfls')):idx + 1] = ('ev', 'evi')
            keys_implicit.update(keys)
            constants.update({k: c * TRANSPORT_SCALES.get(k, 1) for k in keys})
        if implicit and (not end or 'evi' in constants):  # avoid redundancy
            descrip = TRANSPORT_DESCRIPTIONS[name]
            kwargs = {'descrip': descrip, 'prefix': 'alternative' if end else ''}
            if all(key in dataset for key in constants):
                data = sum(c * dataset.climo.vars[k] for k, c in constants.items())
                cdata, rdata, tdata = _transport_implicit(data, **kwargs)
                dataset[f'{name}t{end}'] = tdata  # zonal-mean transport
                if parts_local:  # include local convergence
                    dataset[f'{name}c{end}'] = cdata
                if parts_eddies:  # include residual estimate
                    dataset[f'{name}r{end}'] = rdata
    drop = keys_implicit & dataset.data_vars.keys()
    if drop_implicit:
        dataset = dataset.drop_vars(drop)

    # Get explicit dry, latent, and moist transport.
    # NOTE: The below regex prefixes exponents expressed by numbers adjacent to units
    # with the carat ^, but ignores the scientific notation 1.e6 in scaling factors,
    # so dry static energy convergence units can be parsed as quantities.
    keys_explicit = set()
    if explicit:  # later processing
        ends += ('_exp',)
    for name, (scale, ukey, vkey) in TRANSPORT_EXPLICIT.items():
        keys_explicit.update((ukey, vkey))
        descrip = TRANSPORT_DESCRIPTIONS[name]
        kwargs = {'descrip': descrip, 'prefix': 'explicit'}
        if explicit and ukey in dataset and vkey in dataset:
            udata = dataset[ukey].copy(deep=False)
            vdata = dataset[vkey].copy(deep=False)
            udata *= scale * ureg(REGEX_EXP.sub(r'\1^\2', udata.attrs.pop('units')))
            vdata *= scale * ureg(REGEX_EXP.sub(r'\1^\2', vdata.attrs.pop('units')))
            qdata = ureg.dimensionless * xr.ones_like(udata)  # placeholder
            cdata, _, tdata = _transport_explicit(udata, vdata, qdata, **kwargs)
            dataset[f'{name}t_exp'] = tdata
            if parts_local:  # include local convergence
                dataset[f'{name}c_exp'] = cdata
    drop = keys_explicit & dataset.data_vars.keys()
    if drop_explicit:
        dataset = dataset.drop_vars(drop)

    # Get mean transport, stationary transport, and stationary convergence
    # TODO: Stop storing these. Instead implement in loading func or as climopy
    # derivations. Currently get simple summation, product, and difference terms
    # on-the-fly (e.g. mse and total transport) but need to support more complex stuff.
    iter_ = tuple(TRANSPORT_INDIVIDUAL.items())
    for name, (quant, scale, _) in iter_:
        descrip = TRANSPORT_DESCRIPTIONS[name]
        if (parts_eddies or parts_static) and (  # TODO: then remove after adding?
            'ps' in dataset and 'va' in dataset and quant in dataset
        ):
            qdata = scale * dataset.climo.vars[quant]
            udata, vdata = dataset.climo.vars['ua'], dataset.climo.vars['va']
            cdata, sdata, mdata = _transport_explicit(udata, vdata, qdata, descrip=descrip)  # noqa: E501
            dataset[f's{name}c'] = cdata  # stationary compoennt of convergence
            dataset[f's{name}t'] = sdata  # stationary component of meridional transport
            dataset[f'm{name}t'] = mdata  # mean component of meridional transport

    # Get missing transient components and total sensible and geopotential terms
    # NOTE: Transient sensible transport is calculated from the residual of the dry
    # static energy minus both the sensible and geopotential stationary components. The
    # all-zero transient geopotential is stored for consistency if sensible is present.
    iter_ = itertools.product(TRANSPORT_INDIVIDUAL, ('cs', 'tsm'), ends)
    for name, (suffix, *prefixes), end in iter_:
        ref = f'{prefixes[0]}{name}{suffix}'  # reference component
        total = 'lse' if name == 'lse' else 'dse'
        total = f'{total}{suffix}{end}'
        parts = [f'{prefix}{name}{suffix}' for prefix in prefixes]
        others = ('lse',) if name == 'lse' else ('gse', 'hse')
        others = [f'{prefix}{other}{suffix}' for prefix in prefixes for other in others]
        if (parts_eddies or parts_static) and (
            total in dataset and all(other in dataset for other in others)
        ):
            data = xr.zeros_like(dataset[ref])
            if name != 'gse':  # total transient is zero (aliased into sensible)
                with xr.set_options(keep_attrs=True):
                    data += dataset[total] - sum(dataset[other] for other in others)
            data.attrs['long_name'] = data.long_name.replace('stationary', 'transient')
            dataset[f't{name}{suffix}{end}'] = data
            if name != 'lse':  # total sensible and total geopotential
                with xr.set_options(keep_attrs=True):
                    data = data + sum(dataset[part] for part in parts)
                data.attrs['long_name'] = data.long_name.replace('transient ', '')
                dataset[f'{name}{suffix}{end}'] = data

    # Get average flux terms from the integrated terms
    # NOTE: Flux values will have units K/s, m2/s2, and g/kg m/s and are more relevant
    # to local conditions on a given latitude band. Could get vertically resolved values
    # for stationary components, but impossible for residual transient component, so
    # decided to only store this 'vertical average' for consistency.
    prefixes = ('', 'm', 's', 't')  # transport components
    names = ('hse', 'gse', 'lse')  # flux components
    iter_ = itertools.product(prefixes, names, ends)
    if parts_fluxes:
        for prefix, name, end in iter_:
            _, scale, unit = TRANSPORT_INDIVIDUAL[name]
            variable = f'{prefix}{name}t{end}'
            if variable not in dataset:
                denom = 2 * np.pi * np.cos(dataset.climo.coords.lat) * const.a
                denom = denom * dataset.climo.vars.ps.climo.average('lon') / const.g
                data = dataset[variable].climo.quantify()
                data = (data / denom / scale).climo.to_units(unit)
                data.attrs['long_name'] = dataset[variable].long_name.replace('transport', 'flux')  # noqa: E501
                dataset[f'{prefix}{name}f{end}'] = data.climo.dequantify()

    # Get dry static energy components from sensible and geopotential components
    # NOTE: Here a residual between total and storage + ocean would also suffice
    # but this also gets total transient and stationary static energy terms. Also
    # have support for adding geopotential plus
    prefixes = ('', 'm', 's', 't')  # total, zonal-mean, stationary, transient
    suffixes = ('t', 'c', 'r')  # transport, convergence, residual
    iter_ = itertools.product(prefixes, suffixes, ends)
    if not parts_static:  # only wanted combined dse eddies not components
        for prefix, suffix, end in iter_:
            variable = f'{prefix}{name}{suffix}{end}'
            parts = [variable.replace(name, part) for part in ('hse', 'gse')]
            if variable not in dataset and all(part in dataset for part in parts):
                with xr.set_options(keep_attrs=True):
                    data = sum(dataset[part] for part in parts)
                data.attrs['long_name'] = data.long_name.replace('sensible', 'dry')
                dataset[variable] = data
            drop = [part for part in parts if part in dataset]
            dataset = dataset.drop_vars(drop)
    return dataset


def _update_feedback_attrs(
    dataset, quadruple=True, boundary=None, annual=True, seasonal=True, monthly=True,
):
    """
    Adjust feedback term attributes before plotting.

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset.
    quadruple : bool, optional
        Whether to label climate sensitivity assuming doubled or quadrupled CO$_2$.
    boundary : {'t', 's', 'a'}, optional
        The boundari(es) to load. If one is passed then the indicator is stripped.
    annual, seasonal, monthly : bool, optoinal
        Whether to load different periods of data.
    """
    # Flux metadata repairs
    # NOTE: Scaling of forcing or climate sensitivity terms for doubled or quadrupled
    # CO2 is handled by individual loading functions. Here just add a label indication.
    # NOTE: This drops pre-loaded climate sensitivity parameters, and only
    # keeps the boundary indicator if more than one boundary was requested.
    options = set(boundary or 't')
    boundaries = ('surface', 'TOA')
    wavelengths = ('full', 'longwave', 'shortwave')
    iter_ = itertools.product(boundaries, wavelengths, FEEDBACK_DESCRIPTIONS.items())
    for boundary, wavelength, (component, descrip) in iter_:
        rad = f'r{wavelength[0].lower()}n{boundary[0].lower()}'
        scale = '' if quadruple else r'2$\times$CO$_2$'
        for suffix, outdated, short in (
            ('lam', 'lambda', 'feedback'),
            ('erf', 'erf2x', 'forcing'),
            ('ecs', 'ecs2x', 'climate sensitivity'),
        ):
            if wavelength != 'full':  # WARNING: do not overwrite itertools 'descrip'
                tail = f'{wavelength} {descrip}' if descrip else wavelength
            elif suffix != 'lam':
                tail = descrip if descrip else 'effective'
            else:
                tail = descrip if descrip else 'net'
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
            spec = scale if suffix in ('erf', 'ecs') else ''
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
    renames = {
        'pbot': ('plev_bot', 'lower'),
        'ptop': ('plev_top', 'upper'),
        'tstd': ('ts_projection',),  # not realy but why not
        'tpat': ('ts_pattern',),
    }
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
    for name, options in renames.items():
        for option in options:
            if option in dataset:
                dataset = dataset.rename({option: name})
        if name in dataset:
            data = dataset[name]
            boundary = 'surface' if name == 'pbot' else 'tropopause'
            if name == 'tstd':
                data.attrs['short_name'] = 'warming'
                data.attrs['long_name'] = 'regional warming'
                data.attrs.setdefault('units', 'K')
            if name == 'tpat':
                data.attrs['short_name'] = 'relative warming'
                data.attrs['long_name'] = 'relative warming'
                data.attrs['standard_units'] = 'K / K'  # otherwise not used in labels
                data.attrs.setdefault('units', 'K / K')
            if name in ('pbot', 'ptop'):
                data.attrs['short_name'] = 'pressure'
                data.attrs['long_name'] = f'{boundary} pressure'
                data.attrs['standard_units'] = 'hPa'  # differs from units
                data.attrs.setdefault('units', 'Pa')
    return dataset


def _update_feedback_terms(
    dataset, quadruple=True, boundary=None, parts_clear=True, parts_kernels=True,
    parts_planck=None, parts_relative=None, parts_absolute=None, parts_erf=False, parts_wav=False,  # noqa: E501
):
    """
    Add net cloud effect and net atmospheric feedback terms, possibly filter out
    unnecessary terms, and standardize the insertion order for the dataset variables.

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset.
    quadruple : bool, optional
        Whether to label forcing and sensitivity assuming doubled or quadrupled CO$_2$.
    boundary : {'t', 's', 'a'}, optional
        The boundari(es) to load. Pass a tuple or longer string for more than one.
    parts_clear : bool, optional
        Whether to include clear-sky components alongside all-sky.
    parts_kernels : bool, optional
        Whether to include any kernel-derived variables in the output.
    parts_planck : bool, optional
        Whether to include relative and absolute Planck feedback components.
    parts_relative : bool, optional
        Whether to include relative humidity-style atmospheric components.
    parts_absolute : bool, optional
        Whether to include absolute humidity-style atmospheric components.
    parts_wav : bool, optional
        Whether to include non-cloud related feedback wavelength components.
    parts_erf : bool, optional
        Whether to include non-net effective radiative forcing components.
    """
    # Add climate sensitivity estimate
    # NOTE: Previously computed climate sensitivity 'components' based on individual
    # effective forcings and feedbacks but interpretation is not useful. Now store
    # zero sensitivity components and only compute after the fact.
    check = lambda keys: all(key in dataset for key in keys)
    scale = '' if quadruple else r'2$\times$CO$_2$ '  # avoid 'abrupt 4xCO2 4xCO2'
    if 'rfnt_ecs' in dataset and 'lon' not in dataset['rfnt_ecs'].dims:
        numer = denom = None
    elif check(numer := ('rfnt_erf',)) and check(denom := ('rfnt_lam',)):
        pass  # full wavelength
    elif check(numer := ('rlnt_erf', 'rsnt_erf')) and check(denom := ('rlnt_lam', 'rsnt_lam')):  # noqa: E501
        pass  # wavelength components
    else:
        numer = denom = None
    if numer and denom and 'rfnt_ecs' not in dataset:
        with xr.set_options(keep_attrs=True):
            numer = sum(dataset[key] for key in numer)
        with xr.set_options(keep_attrs=True):
            denom = sum(dataset[key] for key in denom)
        numer = numer.climo.add_cell_measures().climo.average('area')
        denom = denom.climo.add_cell_measures().climo.average('area')
        data = -1 * numer / denom
        data.attrs['units'] = 'K'
        data.attrs['short_name'] = 'climate sensitivity'
        data.attrs['long_name'] = f'{scale}effective climate sensitivity'
        dataset['rfnt_ecs'] = data

    # Update and potentially drop variables
    # NOTE: Previously full wavelength was included in saved files but newest code only
    # keeps longwave and shortwave to save storage space. Must test on new files.
    # NOTE: Below renames 'net' feedback to 'longwave' feedback in case of parts_wav
    # is `False` so that external sources without partitioning can be used. This is
    # not needed for component feedbacks; Piers Forster includes 'cre' but we can drop
    # since also provides 'lwcs' and 'swcs' from which get_data() can derive 'cre', and
    # Mark Zelinka includes 'lwcld' and 'swcld' components alongside net 'cld'.
    if parts_relative is None:
        parts_relative = parts_kernels
    if parts_absolute is None:
        parts_absolute = False
    parts_relative = parts_kernels if parts_relative is None else parts_relative
    parts_absolute = False if parts_absolute is None else parts_absolute
    keep_wav = ('', 'cl')  # keep net all-sky, net clear-sky, cloud kernel wavelengths
    keys = {'pbot', 'ptop', 'tstd', 'tpat', 'tabs', 'rfnt_ecs'}
    ignore = ()  # figure out parts to ignore
    ignore += () if parts_planck or parts_relative else ('pl*',)
    ignore += () if parts_planck or parts_absolute else ('pl',)
    ignore += () if parts_relative else ('lr*', 'hur')
    ignore += () if parts_absolute else ('lr', 'hus')
    ignore += () if parts_kernels else ('cl', 'alb', 'resid')
    boundary = boundary or 't'
    variables = list(dataset.data_vars)
    for key in variables:  # possibly augmented during iteration
        if 'ecs' in key and 'lon' in dataset[key].dims:  # ignore outdated values
            continue
        if not (m := REGEX_FLUX.search(key)):
            continue
        # Add or rename full-wavelength components
        long = dataset[key].attrs.get('long_name', '')
        long = long.replace('longwave', '').replace('shortwave', '').replace('  ', ' ')
        if m.group(1) in ('pl', 'lr', 'alb'):  # rename indicator to 'f' for 'full'
            full = REGEX_FLUX.sub(r'\1\2f\4\5\6', key)
            if key in dataset and full not in dataset:
                dataset = dataset.rename({key: full})
                dataset[full].attrs['long_name'] = long
                variables.append(full)  # augment for later iteration
        if m.group(3) == 'l':  # get 'full' from 'l' and 's' components
            full = REGEX_FLUX.sub(r'\1\2f\4\5\6', key)
            short = REGEX_FLUX.sub(r'\1\2s\4\5\6', key)
            if short in dataset and full not in dataset:
                with xr.set_options(keep_attrs=True):  # keep units and short_name
                    dataset[full] = dataset[key] + dataset[short]
                dataset[full].attrs['long_name'] = long
                variables.append(full)  # augment for later iteration
        if 'ecs' not in key and m.group(3) == 'f' and m.group(1) == m.group(6) == '':
            long = REGEX_FLUX.sub(r'\1\2l\4\5\6', key)
            short = REGEX_FLUX.sub(r'\1\2s\4\5\6', key)
            if not parts_wav and long not in dataset and short not in dataset:
                dataset = dataset.rename({key: long})  # pretend longwave is 'full'
                dataset[short] = xr.zeros_like(dataset[long])
                variables.extend((long, short))  # augment for later iteration
                continue
        # Skip various keys
        if boundary is not None and m.group(5) not in boundary:
            continue
        if not parts_clear and m.group(6) != '':
            continue
        if ignore is not None and m.group(1) in ignore:
            continue
        if not parts_wav and m.group(3) == 'f' and m.group(1) in keep_wav:
            continue  # no need to store full wavelength as well
        if not parts_wav and m.group(3) != 'f' and m.group(1) not in keep_wav:
            continue
        if not parts_erf and 'erf' in key and m.group(1) != '':
            continue
        keys.add(key)
    drop = dataset.data_vars.keys() - keys
    drop.update(key for key in dataset if 'cell' in key)
    dataset = dataset.drop_vars(drop)
    return dataset


def climate_datasets(
    *paths,
    years=None, anomaly=True, ignore=None, nodrift=False, standardize=True,
    **constraints
):
    """
    Return a dictionary of datasets containing processed files.

    Parameters
    ----------
    *paths : path-like
        The search paths.
    years : 2-tuple of int
        The year range.
    anomaly : bool, optional
        Whether to load 4xCO2 data in anomaly form.
    nodrift : bool, optional
        Whether to use drift corrections.
    ignore : bool, optional
        The variables to optionally ignore.
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
    kw_energetics = {key: constraints.pop(key) for key in KEYS_ENERGY if key in constraints}  # noqa: E501
    kw_transport = {key: constraints.pop(key) for key in KEYS_TRANSPORT if key in constraints}  # noqa: E501
    kw_version = {key: constraints.pop(key) for key in KEYS_VERSION if key in constraints}  # noqa: E501
    kw_moist = {key: constraints.pop(key) for key in KEYS_MOIST if key in constraints}
    kw_periods = {key: constraints.pop(key) for key in KEYS_PERIODS if key in constraints}  # noqa: E501
    files, *_ = glob_files(*paths, project=constraints.get('project', None))
    constraints.setdefault('table', ['Amon', 'Emon'])
    constraints.setdefault('experiment', ['piControl', 'abrupt4xCO2'])
    database = Database(files, FACETS_LEVELS, flagship_translate=True, **constraints)
    database.filter(always_exclude={'variable': ['pfull']})  # skip dependencies
    nodrift = nodrift and '-nodrift' or ''
    datasets = {}
    print(f'Climate files: <dates>-climate{nodrift}')
    print(f'Number of climate file groups: {len(database)}.')
    if ignore is None:  # default value
        ignore = ('huss', 'hurs', 'uas', 'vas')
    if database:
        print('Model:', end=' ')
    for (project, model, experiment, ensemble), data in database.items():
        # Initial stuff
        if not data:
            continue
        if model in MODELS_SKIP:
            continue
        for sub, replace in FACETS_RENAME.items():
            experiment = experiment.replace(sub, replace)
        if years is not None:
            range_ = years
        elif experiment == 'abrupt4xco2':
            range_ = (120, 150)
        elif experiment == 'picontrol':
            range_ = (0, 150)
        dates = f'{range_[0]:04d}-{range_[1]:04d}-climate{nodrift}'
        print(f'{model}_{experiment}_{range_[0]:04d}-{range_[1]:04d}', end=' ')
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
            if ignore and variable in ignore:
                continue
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
        dataset = _update_climate_energetics(dataset, **kw_energetics)  # before all
        dataset = _update_climate_transport(dataset, **kw_transport, **kw_version)
        dataset = _update_climate_hydrology(dataset, **kw_moist)  # after transport
        dataset = _update_climate_units(dataset)  # after transport
        if 'time' in dataset:  # TODO: remove this and just keep months
            dataset = average_periods(dataset, **kw_periods)
        if 'plev' in dataset:
            dataset = dataset.sel(plev=slice(None, 7000))
        drop = ['cell_', '_bot', '_top']
        drop = [key for key in dataset.coords if any(o in key for o in drop)]
        dataset = dataset.drop_vars(drop).squeeze()
        if standardize:
            dataset = _standardize_order(dataset)
        datasets[project, model, experiment, ensemble] = dataset

    # Translate abrupt 4xCO2 datasets into anomaly form
    # TODO: Still somehow support 'response' suffix for abrupt4xCO2
    # experiment instead of picontrol experiment after concatenating.
    if datasets:
        print()
    if anomaly:
        print('Transforming abrupt 4xCO2 data into anomalies.')
        for group, dataset in tuple(datasets.items()):
            (project, model, experiment, ensemble) = group
            if experiment != 'abrupt4xco2':
                continue
            group0 = (project, model, 'picontrol', ensemble)
            if group0 not in datasets:
                del datasets[group]
                continue
            dataset0 = datasets[group0]
            for name, data in dataset.data_vars.items():
                if name[:4] in ('ps', 'cell'):
                    continue
                if name not in dataset0:
                    dataset = dataset.drop_vars(name)
                else:  # WARNING: use .data to avoid subtracting coords
                    data.data -= dataset0[name].data
                    if name == 'ts':
                        data.attrs['long_name'] = 'surface warming'
                        data.attrs['short_name'] = 'warming'
                    else:
                        if 'long_name' in data.attrs:
                            data.attrs['long_name'] += ' response'
                        if 'short_name' in data.attrs:
                            data.attrs['short_name'] += ' response'
            datasets[group] = dataset
    return datasets


def feedback_datasets(
    *paths,
    source=None, nodrift=False, quadruple=True, boundary=None, standardize=True,
    point=True, latitude=True, hemisphere=True, early=True, late=True, historical=True,
    **constraints,
):
    """
    Return a dictionary of datasets containing feedback files.

    Parameters
    ----------
    *path : path-like
        The search paths.
    source : str or sequence, optional
        The kernel source(s) to optionally filter.
    nodrift : bool, optional
        Whether to use drift corrections.
    quadruple : bool, optional
        Whether to adjust parameters to correspond to quadrupled CO$_2$.
    boundary : bool or str, optional
        The boundaries to include.
    standardize : bool, optional
        Whether to standardize the resulting order.
    point, latitude, hemisphere : bool, optional
        Whether to include or drop extra regional feedbacks.
    early, late, historical : bool, optional
        Whether to include or drop extra period feedbacks.
    **kwargs
        Passed to `_update_feedback_terms`.
    **constraints
        Passed to `Database`.

    Returns
    -------
    datasets : dict
        A dictionary of datasets.
    """
    # WARNING: Recent xarray versions produce bugs when running xr.concat or
    # xr.update with multi-index. See: https://github.com/pydata/xarray/issues/7695
    # NOTE: To reduce the number of variables this filters out
    # unneeded boundaries and effective forcings automatically.
    kw_both = {'quadruple': quadruple, 'boundary': boundary}
    kw_terms = {key: constraints.pop(key) for key in KEYS_FEEDBACKS if key in constraints}  # noqa: E501
    kw_periods = {key: constraints.pop(key) for key in KEYS_PERIODS if key in constraints}  # noqa: E501
    files, *_ = glob_files(*paths, project=constraints.get('project', None))
    constraints['variable'] = 'feedbacks'  # TODO: similar for climate_datasets
    database = Database(files, FACETS_LEVELS, flagship_translate=True, **constraints)
    sources = (source,) if isinstance(source, str) else tuple(source or ())
    nodrift = nodrift and '-nodrift' or ''
    scale = 2.0 if quadruple else 1.0  # TODO: possibly change convention
    datasets = {}
    print(f'Feedback files: <source>-<statistic>{nodrift}')
    print(f'Number of feedback file groups: {len(database)}.')
    if database:
        print('Model:', end=' ')
    for group, data in database.items():
        # Load the data
        # NOTE: This accounts for files with dedicated regions indicated in the name,
        # files with numerator and denominator multi-index coordinates, and files with
        # just a denominator region coordinate. Note load_file builds the multi-index.
        for sub, replace in FACETS_RENAME.items():
            group = tuple(s.replace(sub, replace) for s in group)
        paths = [
            path for paths in data.values() for path in paths
            if bool(nodrift) == bool('nodrift' in path.name)
        ]
        if not paths:
            continue
        if group[1] in MODELS_SKIP:
            continue
        versions = {}
        print(f'{group[1]}_{group[2]}', end=' ')
        for path in paths:
            # Load file
            *_, indicator, suffix = path.stem.split('_')
            start, stop, source, statistic, *_ = suffix.split('-')  # ignore -nodrift
            start, stop = map(int, (start, stop))
            if sources and source not in sources:
                continue
            if not early and (start, stop) == (0, 20):
                continue
            if not late and (start, stop) == (20, 150):
                continue
            if not historical and stop - start == 50:
                continue
            if outdated := 'local' in indicator or 'global' in indicator:
                if indicator.split('-')[0] != 'local':  # ignore global vs. global
                    continue
            dataset = load_file(path, project=database.project, validate=False)
            for array in dataset.data_vars.values():
                if '_erf' in array.name or '_ecs' in array.name:
                    array *= scale  # NOTE: array.data *= fails for unloaded data
            # Filter and add to dictionary
            if outdated:
                region = 'point' if indicator.split('-')[1] == 'local' else 'globe'
                if not point and region == 'point':
                    continue
                versions[source, statistic, region, start, stop] = dataset
            else:
                for region in dataset.region.values:  # includes pbot and ptop
                    sel = dataset.sel(region=region, drop=True)
                    if not point and region == 'point':
                        continue
                    if not latitude and region == 'latitude':
                        continue
                    if not hemisphere and region == 'hemisphere':
                        continue
                    versions[source, statistic, region, start, stop] = sel
            del dataset

        # Concatenate the data
        # NOTE: Concatenation automatically broadcasts global feedbacks across lons and
        # lats. Also critical to use 'override' for combine_attrs in case conventions
        # changed between running feedback calculations on different models.
        concat, noncat = {}, {}
        for key, dataset in versions.items():
            names = ('pbot', 'ptop')
            dataset = _update_feedback_attrs(dataset, **kw_both, **kw_periods)
            dataset = _update_feedback_terms(dataset, **kw_both, **kw_terms)
            for name in names:
                if name in dataset:
                    data = dataset[name]
                    if 'plev' in data.dims:  # error in _fluxes_from_anomalies
                        data = data.isel(plev=0, drop=True)
                    noncat[name] = data.expand_dims('version')
            drop = set(names) & dataset.keys()
            dataset = dataset.drop_vars(drop)
            concat[key] = dataset
        index = xr.DataArray(
            pd.MultiIndex.from_tuples(concat, names=VERSION_LEVELS),
            dims='version',
            name='version',
            attrs={'long_name': 'feedback version'},
        )
        dataset = xr.concat(
            concat.values(),
            dim=index,
            coords='minimal',
            compat='override',
            combine_attrs='override',
        )
        # TODO: Work around xarray error.
        # for key, array in noncat.items():
        #     dataset[key] = array
        dataset = dataset.squeeze()
        if standardize:
            dataset = _standardize_order(dataset)
        datasets[group] = dataset

    if datasets:
        print()
    return datasets


def feedback_datasets_json(
    *paths,
    quadruple=True, boundary=None, standardize=True, nonflag=False, **constraints,
):
    """
    Return a dictionary of datasets containing json-provided feedback data.

    Parameters
    ----------
    *paths : path-like, optional
        The base path(s).
    quadruple : bool, optional
        Whether to adjust parameters to correspond to quadrupled CO$_2$.
    boundary : str, optional
        The boundary components.
    standardize : bool, optional
        Whether to standardize the resulting order.
    nonflag : bool, optional
        Whether to include non-flagship feedback estimates.
    **kwargs
        Used to filter and adjust the data. See `feedback_datasets`.
    **constraints
        Passed to `_parse_constraints`.

    Returns
    -------
    datasets : dict
        A dictionary of datasets.
    """
    # NOTE: When combinining with 'open_dataset' the non-descriptive long names
    # here should be overwritten by long names from custom feedbacks.
    kw_both = {'quadruple': quadruple, 'boundary': boundary}
    kw_terms = {key: constraints.pop(key) for key in KEYS_FEEDBACKS if key in constraints}  # noqa: E501
    paths = paths or ('~/data/cmip-tables',)
    paths = tuple(Path(path).expanduser() for path in paths)
    boundary = boundary or 't'
    project, constraints = _parse_constraints(reverse=True, **constraints)
    scale = 2.0 if quadruple else 1.0
    datasets = {}
    if 't' not in boundary:  # only top-of-atmosphere feedbacks available
        return datasets
    for file in sorted(file for path in paths for file in path.glob('cmip*.json')):
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
            if model == 'IPSL-CM6ALR-INCA':  # no separate control version
                continue
            if model not in constraints.get('model', (model,)):
                continue
            for ensemble, data in ensembles.items():
                key_flagship = (project, 'abrupt-4xCO2', model)
                ens_default = ENSEMBLES_FLAGSHIP[project, None, None]
                ens_flagship = ENSEMBLES_FLAGSHIP.get(key_flagship, ens_default)
                ensemble = 'flagship' if ensemble == ens_flagship else ensemble
                if not nonflag and ensemble != 'flagship':
                    continue
                if ensemble not in constraints.get('ensemble', (ensemble,)):
                    continue
                group = (project, model, 'abrupt4xco2', ensemble)
                dataset = xr.Dataset()
                for key, value in data.items():
                    name, units = FEEDBACK_SETTINGS[key.lower()]
                    attrs = {'units': units}  # long name assigned below
                    if '_erf' in name or '_ecs' in name:
                        value = scale * value
                    dataset[name] = xr.DataArray(value, attrs=attrs)
                dataset = dataset.expand_dims(version=1)
                dataset = dataset.assign_coords(version=index)
                if group in datasets:
                    datasets[group].update(dataset)
                else:
                    datasets[group] = dataset
    for group in tuple(datasets):
        dataset = datasets[group]
        dataset = _update_feedback_attrs(dataset, **kw_both)
        dataset = _update_feedback_terms(dataset, **kw_both, **kw_terms)
        if standardize:
            dataset = _standardize_order(dataset)
        datasets[group] = dataset
    return datasets


def feedback_datasets_text(
    *paths,
    quadruple=True, boundary=None, standardize=True, transient=False, **constraints,
):
    """
    Return a dictionary of datasets containing text-provided feedback data.

    Parameters
    ----------
    *paths : path-like, optional
        The base path(s).
    quadruple : bool, optional
        Whether to adjust parameters to correspond to quadrupled CO$_2$.
    boundary : str, optional
        The boundary components.
    standardize : bool, optional
        Whether to standardize the resulting order.
    transient : bool, optional
        Whether to include transient components.
    **kwargs
        Used to filter and adjust the data. See `feedback_datasets`.
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
    kw_both = {'quadruple': quadruple, 'boundary': boundary}
    kw_terms = {key: constraints.pop(key) for key in KEYS_FEEDBACKS if key in constraints}  # noqa: E501
    paths = paths or ('~/data/cmip-tables',)
    paths = tuple(Path(path).expanduser() for path in paths)
    boundary = boundary or 't'
    project, constraints = _parse_constraints(reverse=True, **constraints)
    datasets = {}
    if 't' not in boundary:  # only top-of-atmosphere feedbacks available
        return datasets
    for file in sorted(file for path in paths for file in path.glob(f'{project.lower()}*.txt')):  # noqa: E501
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
        scale = 0.5 if any('4x' in key for key in dataset.data_vars) else 1.0
        scale = 2 * scale if quadruple else scale
        for key, array in dataset.data_vars.items():
            name, units = FEEDBACK_SETTINGS[key.lower()]
            transients = ('rfnt_tcr', 'rfnt_rho', 'rfnt_kap')
            if not transient and name in transients:
                continue
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
                    data = scale * data
                data.name = name
                data.attrs.update({'units': units})  # long name assigned below
                data = data.to_dataset()
                if group in datasets:  # combine new feedback coordinates
                    tup = (data, datasets[group])
                    data = xr.combine_by_coords(tup, combine_attrs='override')
                datasets[group] = data
    for group in tuple(datasets):
        dataset = datasets[group]
        dataset = _update_feedback_attrs(dataset, **kw_both)
        dataset = _update_feedback_terms(dataset, **kw_both, **kw_terms)
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
    keys_both = ('nodrift', *KEYS_PERIODS)
    keys_climate = ('years', 'anomaly', 'ignore', *KEYS_ENERGY, *KEYS_MOIST, *KEYS_TRANSPORT, *KEYS_VERSION)  # noqa: E501
    keys_internal = ('source', *KEYS_REGIONS, *KEYS_RANGES)
    keys_feedback = ('quadruple', 'boundary', *KEYS_FEEDBACKS)
    kw_json = {'nonflag': constraints.pop('nonflag', False)}
    kw_text = {'transient': constraints.pop('transient', False)}
    kw_both = {k: constraints.pop(k) for k in keys_both if k in constraints}
    kw_climate = {k: constraints.pop(k) for k in keys_climate if k in constraints}
    kw_internal = {k: constraints.pop(k) for k in keys_internal if k in constraints}
    kw_feedback = {k: constraints.pop(k) for k in keys_feedback if k in constraints}
    kw_internal = {**kw_internal, **kw_feedback}  # slightly simpler
    dirs_table = ('cmip-tables',)
    dirs_climate = ('cmip-climate',)
    dirs_feedback = ('cmip-feedbacks', 'historical-feedbacks')
    bases = ('~/data', '~/scratch2', '~/scratch5')
    datasets = {}
    projects = project.split(',') if isinstance(project, str) else ('cmip5', 'cmip6')
    if feedbacks_json is None:
        feedbacks_json = feedbacks
    if feedbacks_text is None:
        feedbacks_text = feedbacks
    for project in map(str.upper, projects):
        print(f'Project: {project}')
        for b, function, dirs, kw in (
            (climate, climate_datasets, dirs_climate, {**kw_climate, **kw_both}),
            (feedbacks, feedback_datasets, dirs_feedback, {**kw_internal, **kw_both}),
            (feedbacks_json, feedback_datasets_json, dirs_table, {**kw_json, **kw_feedback}),  # noqa: E501
            (feedbacks_text, feedback_datasets_text, dirs_table, {**kw_text, **kw_feedback}),  # noqa: E501
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
            for group, data in parts.items():
                if group in datasets:
                    comb = (datasets[group], data)
                    data = xr.combine_by_coords(comb, combine_attrs='override')
                datasets[group] = data

    # Concatenate and standardize datasets
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
            if all('version' in dict_ for dict_ in (da.dims, dataset, dataset.sizes)):
                da = da.isel(version=0, drop=True)
                da = da.expand_dims(version=dataset.sizes['version'])
                da = da.assign_coords(version=dataset.version)
            dataset[name] = da
    print()
    print('Concatenating datasets.')
    if not datasets:
        raise ValueError('No datasets found.')
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
