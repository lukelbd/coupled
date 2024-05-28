#!/usr/bin/env python3
"""
Load coupled model feedback data.
"""
import itertools
import warnings
import json
import os
import re
from pathlib import Path

import climopy as climo  # noqa: F401  # add accessor
import numpy as np
import pandas as pd
import xarray as xr
from climopy import var, ureg, vreg  # noqa: F401
from icecream import ic  # noqa: F401

from cmip_data.facets import ENSEMBLES_FLAGSHIP, MODELS_INSTITUTES
from cmip_data.facets import Database, glob_files, _item_member, _parse_constraints
from cmip_data.feedbacks import FEEDBACK_DESCRIPTIONS
from cmip_data.utils import assign_dates, load_file
from .process import get_parts
from .specs import TRANSLATE_PATHS
from .specs import _pop_kwargs

__all__ = [
    'feedback_datasets',
    'feedback_jsons',
    'feedback_texts',
    'process_scalar',
]

# Regular expression
# NOTE: The specifiers relevant for use are in groups \1, \3, \5, and \6. Also add
# to the net/upwelling/downwelling indicator 'e' for 'effective forcing' and 'r'
# for 'radiative response' (i.e. net minus effective forcing).
REGEX_FLUX = re.compile(
    r'(\A|[^_]*)(_?r)([lsf])([udner])([tsa])(cs|ce|)'
)

# Period label translations
# NOTE: Observed feedback utility concatenates along 'version' coordinate with keys and
# labels constructed from input keyword arguments, but have no concept of 'early' or
# 'late' response for observed data, so can supply additional translations manually.
LABELS_YEARS = {
    ('years', value): ('period', label)
    for (key, value), label in TRANSLATE_PATHS.items() if key == 'startstop'
}

# Default feedback settings
# NOTE: This calculates both full 150-year and early and late perturbed feedback
# estimates. Also skip 'month' 'anomaly' and 'detrend' options irrelevant for abrupt
# experiments and 'x' and 'y' detrend options for control runs since stationary.
PARAMS_ABRUPT = {
    'years': ((0, 150), (0, 20), (20, 150)),
    'month': (None,),
    'anomaly': (False,),
    'detrend': ('',),
}
PARAMS_CONTROL = {
    'years': (None, 20, 50),
    'month': ('jan', 'jul'),
    'anomaly': (True, False),
    'detrend': ('', 'xy'),
}

# Feedback aliases
# NOTE: These are used to both translate tables from external sources into the more
# descriptive naming convention, and to translate inputs to plotting functions for
# convenience (default for each shorthand is to use combined longwave + shortwave toa).
VARIABLE_DEFINITIONS = {
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
ALIAS_VARIABLES = {
    **{f'a{alias}': name.replace('t_', 'a_') for alias, (name, _) in VARIABLE_DEFINITIONS.items()},  # noqa: E501
    **{f's{alias}': name.replace('t_', 's_') for alias, (name, _) in VARIABLE_DEFINITIONS.items()},  # noqa: E501
    **{f't{alias}': name for alias, (name, _) in VARIABLE_DEFINITIONS.items()},
    **{alias: name for alias, (name, _) in VARIABLE_DEFINITIONS.items()},
}
VARIABLE_ALIASES = {value: key for key, value in ALIAS_VARIABLES.items()}


def _update_attrs(dataset, boundary=None):
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
    """
    # Flux metadata repairs
    # NOTE: This drops pre-loaded climate sensitivity parameters, and only
    # keeps the boundary indicator if more than one boundary was requested.
    options = set(boundary or 't')
    boundaries = ('surface', 'TOA')
    wavelengths = ('full', 'longwave', 'shortwave')
    iter_ = itertools.product(boundaries, wavelengths, FEEDBACK_DESCRIPTIONS.items())
    for boundary, wavelength, (component, descrip) in iter_:
        for suffix, outdated, short, units in (
            ('lam', 'lambda', 'feedback', 'W m^-2 K^-1'),
            ('erf', 'erf2x', 'forcing', 'W m^-2'),
            ('ecs', 'ecs2x', 'climate sensitivity', 'K'),
        ):
            if wavelength != 'full':  # WARNING: do not overwrite itertools 'descrip'
                tail = f'{wavelength} {descrip}' if descrip else wavelength
            elif suffix != 'lam':
                tail = descrip if descrip else 'effective'
            else:
                tail = descrip if descrip else 'net'
            flux = f'r{wavelength[0].lower()}n{boundary[0].lower()}'
            if component in ('', 'cs', 'ce'):
                prefix = f'{flux}{component}'
            else:
                prefix = f'{component}_{flux}'
            name = f'{prefix}_{suffix}'
            outdated = f'{prefix}_{outdated}'
            if outdated in dataset:
                dataset = dataset.rename({outdated: name})
            if name not in dataset:
                continue
            data = dataset[name]
            head = boundary if len(options) > 1 else ''
            long = f'{head} {tail} {short}'
            long = re.sub('  +', ' ', long).strip()
            data.attrs['long_name'] = long
            data.attrs['short_name'] = short
            data.attrs.setdefault('units', units)
            if suffix == 'ecs' and 'lon' in dataset[name].dims:
                dataset = dataset.drop_vars(name)

    # Other metadata repairs
    # NOTE: This optionally skips unneeded periods to save space. However now simply
    # load monthly data and take annual averages when needed. Should remove.
    aliases = {
        'pbot': ('plev_bot', 'lower'),
        'ptop': ('plev_top', 'upper'),
        'tstd': ('ts_projection',),  # not really but why not
        'tpat': ('ts_pattern',),
    }
    for name, aliases in aliases.items():
        for alias in aliases:
            if alias in dataset:
                dataset = dataset.rename({alias: name})
        if name in dataset:
            data = dataset[name]
            boundary = 'surface' if name == 'pbot' else 'tropopause'
            if name == 'tstd':
                data.attrs['short_name'] = 'temperature change'
                data.attrs['long_name'] = 'temperature change'
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


def _update_terms(
    dataset, boundary=None, parts_clear=True, parts_kernels=True, parts_planck=None,
    parts_relative=None, parts_absolute=None, parts_erf=False, parts_wav=False,
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
    # Initial stuff
    # NOTE: Previously full wavelength was included in saved files but newest code
    # only keeps longwave and shortwave to save storage space. Try to bypass 'full'.
    if parts_relative is None:
        parts_relative = parts_kernels
    if parts_absolute is None:
        parts_absolute = False
    parts_relative = parts_kernels if parts_relative is None else parts_relative
    parts_absolute = False if parts_absolute is None else parts_absolute
    parts_keep = ('', 'cl')  # keep net all-sky, clear-sky, cloud kernel wavelengths
    parts_ignore = () if parts_planck or parts_relative else ('pl*',)
    parts_ignore += () if parts_planck or parts_absolute else ('pl',)
    parts_ignore += () if parts_relative else ('lr*', 'hur')
    parts_ignore += () if parts_absolute else ('lr', 'hus')
    parts_ignore += () if parts_kernels else ('cl', 'alb', 'resid')

    # Iterate over variables
    # NOTE: Previously this included more but has mostly been moved to get_result().
    # Idea is to calculate on-the-fly whenever possible unless the component is almost
    # never needed on its own (e.g. shortwave component of 'hur', 'pl*', or 'lr*').
    keys_keep = {'pbot', 'ptop', 'tstd', 'tpat', 'tabs'}
    boundary = boundary or 't'  # string or tuple
    variables = list(dataset.data_vars)  # augmented during iteration
    for key in variables:
        # Rename longwave to full for special case of external data
        # NOTE: This is not needed for component feedbacks. Forster includels 'cre'
        # but we can drop since also provides 'lwcs' and 'swcs' from which get_result()
        # can derive 'cre' and Zelinka includes 'lwcld' and 'swcld' alongside 'cld'.
        if not (m := REGEX_FLUX.search(key)):
            continue
        if 'ecs' in key and ('lon' in dataset[key].dims or 'lat' in dataset[key].dims):
            continue  # ignore outdated regional breakdowns
        if 'ecs' in key and (m.group(1) != '' or m.group(6) != '' or m.group(3) != 'f'):
            continue  # ignore unneeded breakdowns
        if m.group(3) == 'f' and m.group(1) == m.group(6) == '':
            long = REGEX_FLUX.sub(r'\1\2l\4\5\6', key)
            short = REGEX_FLUX.sub(r'\1\2s\4\5\6', key)
            if not parts_wav and long not in dataset and short not in dataset:
                dataset = dataset.rename({key: long})  # pretend longwave is 'full'
                dataset[short] = xr.zeros_like(dataset[long])
                variables.extend((long, short))  # augment for later iteration
                continue

        # Add or rename shortwave and longwave components
        # WARNING: Critical to rename 'alb' for consistency.
        # NOTE: Possibly faster to just keep water vapor components instead of
        # adding them at the start? Idea is to prevent future plotting slowdowns.
        if m.group(3) != 'f' and m.group(1) in {'pl', 'lr', 'alb'} - {*parts_ignore}:
            full = REGEX_FLUX.sub(r'\1\2f\4\5\6', key)
            if key in dataset and full not in dataset:
                dataset = dataset.rename({key: full})
                variables.append(full)  # augment for later iteration
        if not parts_wav and m.group(3) == 'l' and m.group(1) not in (*parts_keep, *parts_ignore):  # noqa: E501
            full = REGEX_FLUX.sub(r'\1\2f\4\5\6', key)
            short = REGEX_FLUX.sub(r'\1\2s\4\5\6', key)
            if short in dataset and full not in dataset:
                with xr.set_options(keep_attrs=True):  # keep units and short_name
                    dataset[full] = dataset[key] + dataset[short]
                long_name = dataset[full].attrs.get('long_name', '')
                long_name = re.sub(r'(longwave|shortwave)\s*', long_name, '')
                dataset[full].attrs['long_name'] = long_name
                variables.append(full)  # augment for later iteration

        # Bypass variables based on user input
        # NOTE: Effective radiative forcing components can be used to estimate
        # forcing adjustments (find citation)? However skip by default.
        if boundary is not None and 'a' not in boundary and m.group(5) not in boundary:
            continue
        if not parts_clear and m.group(6) != '':
            continue
        if parts_ignore is not None and m.group(1) in parts_ignore:
            continue
        if not parts_wav and m.group(3) == 'f' and m.group(1) in parts_keep:
            continue  # no need to load full wavelength as well
        if not parts_wav and m.group(3) != 'f' and m.group(1) not in parts_keep:
            continue  # ignore full wavelength parts
        if not parts_erf and 'erf' in key and (m.group(1) != '' or m.group(6) != ''):
            continue
        keys_keep.add(key)

    # Add climate sensitivity estimate
    # NOTE: Previously computed climate sensitivity 'components' based on individual
    # effective forcings and feedbacks but interpretation is not useful. Now store
    # zero sensitivity components and only compute after the fact.
    denoms = [('rfnt_lam',), ('rlnt_lam', 'rsnt_lam'), ()]
    numers = [('rfnt_erf',), ('rlnt_erf', 'rsnt_erf'), ()]
    for denom, numer in zip(denoms, numers):
        if all(name in dataset for name in (*numer, *denom)):  # noqa: E501
            break
    short_name = 'climate sensitivity'
    long_name = 'effective climate sensitivity'
    attrs = {'units': 'K', 'short_name': short_name, 'long_name': long_name}
    if numer and denom and 'rfnt_ecs' not in dataset:
        with xr.set_options(keep_attrs=True):
            numer = sum(dataset[key] for key in numer)
            denom = sum(dataset[key] for key in denom)
        if 'lon' in numer.sizes and 'lat' in numer.sizes:
            parts = ('width', 'depth')
            numer = numer.climo.add_cell_measures(parts).climo.average('area')
            denom = denom.climo.add_cell_measures(parts).climo.average('area')
        if 'time' in numer.sizes and 'time' in denom.sizes:  # average over months
            wgts = numer.time.dt.days_in_month / numer.time.dt.days_in_month.sum()
            numer = (numer * wgts).sum('time', skipna=False)
            denom = (denom * wgts).sum('time', skipna=False)
        dataset['rfnt_ecs'] = -1 * numer / denom
        dataset['rfnt_ecs'].attrs.update(attrs)  # 2xCO2 sensitivity from 2xCO2 forcing

    # Return filtered dataset
    drop = dataset.data_vars.keys() - keys_keep - {'rfnt_ecs'}
    drop.update(key for key in dataset if 'cell' in key)
    dataset = dataset.drop_vars(drop)
    return dataset


def feedback_jsons(
    *paths, boundary=None, nonflag=False, standardize=True, **constraints,
):
    """
    Return a dictionary of datasets containing json-provided feedback data.

    Parameters
    ----------
    *paths : path-like, optional
        The base path(s).
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
    # NOTE: Use non-descriptive long names here but when combining with custom feedbacks
    # in open_dataset() should be overwritten by more descriptive long names.
    from .datasets import FACETS_EXCLUDE, VERSION_LEVELS
    kw_terms = _pop_kwargs(constraints, _update_terms)
    paths = paths or ('~/data/cmip-tables',)
    paths = tuple(Path(path).expanduser() for path in paths)
    boundary = boundary or 't'
    project, constraints = _parse_constraints(decode=True, **constraints)
    datasets = {}
    if 't' not in boundary:  # only top-of-atmosphere feedbacks available
        return datasets
    for file in sorted(file for path in paths for file in path.glob('cmip*.json')):
        source = file.stem.split('_')[1]
        print(f'External file: {file.name}')
        version = (source, 'annual', 0, 150, 'globe')
        version = xr.DataArray(
            pd.MultiIndex.from_tuples([version], names=VERSION_LEVELS),
            dims='version',
            name='version',
            attrs={'long_name': 'feedback version'},
        )
        with open(file, 'r') as f:
            source = json.load(f)
        options = source.get(project, {})
        for model, ensembles in options.items():
            institute = MODELS_INSTITUTES.get(model, 'UNKNOWN')
            if model in FACETS_EXCLUDE:  # no separate control version
                continue
            if model not in constraints.get('model', (model,)):
                continue
            if institute not in constraints.get('institute', (institute,)):
                continue
            for idx, ensemble in enumerate(sorted(ensembles, key=_item_member)):
                index = (project, 'abrupt-4xCO2', model)
                flagship = ENSEMBLES_FLAGSHIP[project, None, None]
                flagship = ENSEMBLES_FLAGSHIP.get(index, flagship)
                standard = 'flagship' if ensemble == flagship else f'ensemble{idx:02d}'
                if not nonflag and standard != 'flagship':
                    continue
                if standard not in constraints.get('ensemble', (ensemble, standard)):
                    continue
                facets = (project, institute, model, 'abrupt4xco2', standard)
                arrays = {}
                for key, value in ensembles[ensemble].items():
                    name, units = VARIABLE_DEFINITIONS[key.lower()]
                    attrs = {'units': units}  # long name assigned below
                    arrays[name] = xr.DataArray(value, attrs=attrs)
                dataset = xr.Dataset(arrays)
                dataset = dataset.expand_dims(version=1)
                dataset = dataset.assign_coords(version=version)
                if facets in datasets:
                    datasets[facets].update(dataset)
                else:
                    datasets[facets] = dataset
    from .datasets import _standardize_order
    for facets in tuple(datasets):
        dataset = datasets[facets]
        dataset = _update_attrs(dataset, boundary=boundary)
        dataset = _update_terms(dataset, boundary=boundary, **kw_terms)
        if standardize:
            dataset = _standardize_order(dataset)
        datasets[facets] = dataset
    return datasets


def feedback_texts(
    *paths, boundary=None, transient=False, standardize=True, **constraints,
):
    """
    Return a dictionary of datasets containing text-provided feedback data.

    Parameters
    ----------
    *paths : path-like, optional
        The base path(s).
    boundary : str, optional
        The boundary components.
    transient : bool, optional
        Whether to include transient components.
    standardize : bool, optional
        Whether to standardize the resulting order.
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
    from .datasets import FACETS_EXCLUDE, VERSION_LEVELS
    kw_terms = _pop_kwargs(constraints, _update_terms)
    paths = paths or ('~/data/cmip-tables',)
    paths = tuple(Path(path).expanduser() for path in paths)
    boundary = boundary or 't'
    project, constraints = _parse_constraints(decode=True, **constraints)
    datasets = {}
    if 't' not in boundary:  # only top-of-atmosphere feedbacks available
        return datasets
    for file in sorted(file for path in paths for file in path.glob(f'{project.lower()}*.txt')):  # noqa: E501
        source = file.stem.split('_')[1]
        if source == 'zelinka':
            continue
        print(f'External file: {file.name}')
        version = (source, 'annual', 0, 150, 'globe')
        version = xr.DataArray(
            pd.MultiIndex.from_tuples([version], names=VERSION_LEVELS),
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
        dataset = dataset.assign_coords(version=version)
        scale = 0.5 if any('4x' in key for key in dataset.data_vars) else 1.0
        for key, array in dataset.data_vars.items():
            name, units = VARIABLE_DEFINITIONS[key.lower()]
            transients = ('rfnt_tcr', 'rfnt_rho', 'rfnt_kap')
            if not transient and name in transients:
                continue
            for model in dataset.model.values:
                institute = MODELS_INSTITUTES.get(model, 'UNKNOWN')
                if any(s in model for s in ('mean', 'deviation', 'uncertainty')):
                    continue
                if model in FACETS_EXCLUDE:
                    continue
                if model not in constraints.get('model', (model,)):
                    continue
                if institute not in constraints.get('institute', (institute,)):
                    continue
                if 'flagship' not in constraints.get('ensemble', ('flagship',)):
                    continue
                facets = ('CMIP5', institute, model, 'abrupt4xco2', 'flagship')
                select = array.sel(model=model, drop=True)
                if scale != 1 and units in ('K', 'W m^-2'):  # avoid in-place operation
                    select = scale * select
                select.name = name
                select.attrs.update({'units': units})  # long name assigned below
                select = select.to_dataset()
                if facets in datasets:  # combine new version coordinates
                    args = (select, datasets[facets])
                    select = xr.combine_by_coords(args, combine_attrs='override')
                datasets[facets] = select
    from .datasets import _standardize_order
    for facets in tuple(datasets):
        dataset = datasets[facets]
        dataset = _update_attrs(dataset, boundary=boundary)
        dataset = _update_terms(dataset, boundary=boundary, **kw_terms)
        if standardize:
            dataset = _standardize_order(dataset)
        datasets[facets] = dataset
    return datasets


def feedback_datasets(
    *paths,
    boundary=None, source=None, style=None,
    early=True, late=True, delay=False, fifty=False,
    point=True, latitude=True, hemisphere=False,
    annual=True, seasonal=False, monthly=False,
    average=False, nodrift=False, standardize=True,
    **constraints,
):
    """
    Return a dictionary of datasets containing feedback files.

    Parameters
    ----------
    *path : path-like
        The search paths.
    boundary : bool or str, optional
        The boundaries to include.
    source, style : str or sequence, optional
        The kernel source(s) and feedback style(s) to optionally filter.
    point, latitude, hemisphere : bool, optional
        Whether to include or drop extra regional feedbacks.
    early, late, delay, fifty : bool, optional
        Whether to include or drop extra range feedbacks.
    annual, seasonal, monthly : bool, optional
        Whether to include or drop extra period feedbacks.
    average : bool, optional
        Whether to average feedbacks with 'time' dimension.
    nodrift : bool, optional
        Whether to use drift corrections.
    standardize : bool, optional
        Whether to standardize the resulting order.
    **kwargs
        Passed to `_update_terms`.
    **constraints
        Passed to `Database`.

    Returns
    -------
    datasets : dict
        A dictionary of datasets.
    """
    # Initial stuff
    # TODO: Support subtracting global anomaly within get_result() by adding suffix
    # to the variable string or allowing user override of 'relative' key? Tricky in
    # context of e.g. regressions of anomalies against something not anomalous.
    from .datasets import FACETS_EXCLUDE, FACETS_LEVELS, FACETS_RENAME
    from .datasets import VERSION_LEVELS, VERSION_NAME
    kw_terms = _pop_kwargs(constraints, _update_terms)
    sample = pd.date_range('2000-01-01', '2000-12-01', freq='MS')
    regions = [(point, 'point'), (latitude, 'latitude'), (hemisphere, 'hemisphere')]
    regions = [region for b, region in regions if not b]  # whether to drop
    periods = [] if annual else ['ann']  # dropped selections
    if not seasonal:
        periods.extend(('djf', 'mam', 'jja', 'son'))
    if not monthly:
        periods.extend(sample.strftime('%b').str.lower().values)

    # Open datasets for concatenation
    # WARNING: Recent xarray versions produce bugs when running xr.concat or
    # xr.update with multi-index. See: https://github.com/pydata/xarray/issues/7695
    # NOTE: Current xr.concat() version loads arrays into memory. Might figure out how
    # to make all load_file() and below utilities compatible with an open_mfdataset()
    # implementation. Would require switching to workflow where below annual averaging
    # is removed, and we repeat 'monthly' style feedbacks along 'annual' style time
    # dimension on concatenation, but not sure if open_mfdataset supports this? Could
    # also switch to dask arrays. See: https://github.com/pydata/xarray/issues/4628
    files, *_ = glob_files(*paths, project=constraints.get('project', None))
    constraints['variable'] = 'feedbacks'  # TODO: similar 'climate_Amon' dataset files
    database = Database(files, FACETS_LEVELS, flagship_translate=True, **constraints)
    sources = (source,) if isinstance(source, str) else tuple(source or ())
    styles = (style,) if isinstance(style, str) else tuple(style or ())
    nodrift = nodrift and '-nodrift' or ''
    datasets = {}
    print(f'Feedback files: <source>-<style>{nodrift}')
    print(f'Number of feedback file groups: {len(database)}.')
    if database:
        print('Model:', end=' ')
    for facets, data in database.items():
        # Load the data
        # TODO: Figure out if lots of filled NaN sectors increase dataset size? Could
        # try to prevent intersection of 'picontrol' with non-default startstop.
        # NOTE: This accounts for files with dedicated regions indicated in the name,
        # files with numerator and denominator multi-index coordinates, and files with
        # just a denominator region coordinate. Note load_file builds the multi-index.
        for sub, replace in FACETS_RENAME.items():
            facets = tuple(facet.replace(sub, replace) for facet in facets)
        if facets[2] in FACETS_EXCLUDE:
            continue
        paths = [
            path for paths in data.values() for path in paths
            if bool(nodrift) == bool('nodrift' in path.name)
        ]
        if not paths:
            continue
        print(f'{facets[2]}_{facets[3]}', end=' ')
        versions = {}
        for path in paths:
            *_, other, suffix = path.stem.split('_')
            start, stop, source, style, *_ = suffix.split('-')  # _ = nodrift, climate
            start, stop = map(int, (start, stop))
            version = (source, style, start, stop)
            if sources and source not in sources:
                continue
            if styles and style not in styles:
                continue
            if not early and (start, stop) == (0, 20):
                continue
            if not late and (start, stop) == (20, 150):
                continue
            if not delay and start in range(1, 20):
                continue
            if not fifty and stop - start == 50:
                continue
            if outdated := 'local' in other or 'global' in other:
                if other.split('-')[0] != 'local':  # ignore global vs. global
                    continue
            dataset = load_file(path, project=database.project, validate=False)
            if outdated:
                region = 'point' if other.split('-')[1] == 'local' else 'globe'
                dataset = dataset.expand_dims('region').assign_coords(region=[region])
                if facets in datasets:  # combine new feedback coordinates
                    args = (datasets[version], dataset)
                    dataset = xr.combine_by_coords(args, combine_attrs='override')
            versions[version] = dataset
            del dataset

        # Standardize and concatenate data
        # NOTE: To reduce the number of variables this filters out unrequested regions,
        # periods, and feedback variables. See _update_terms for details.
        # NOTE: Integration bounds 'pbot' and 'ptop' are currently based on control
        # climate data so 'version' coordinate redundant. Simplify below.
        concat, noncat = {}, {}
        for version, dataset in versions.items():
            if regions:
                dataset = dataset.drop_sel(region=regions, errors='ignore')
            if periods and 'period' in dataset.sizes:
                dataset = dataset.drop_sel(period=periods, errors='ignore')
            arrays = {name: dataset[name] for name in ('pbot', 'ptop') if name in dataset}  # noqa: E501
            dataset = dataset.drop_vars(arrays)
            dataset = _update_attrs(dataset, boundary=boundary)
            dataset = _update_terms(dataset, boundary=boundary, **kw_terms)
            for name, data in arrays.items():  # address _fluxes_from_anomalies bug
                if 'plev' in data.sizes:
                    data = data.isel(plev=0, drop=True)
                if name not in noncat:
                    noncat[name] = data
            if 'time' in dataset.sizes:
                dataset = assign_dates(dataset, year=1800)
                if average:  # use assigned dates so results will be consistent
                    days = dataset.time.dt.days_in_month.astype(np.float32)
                    dataset = dataset.weighted(days).mean('time', skipna=False, keep_attrs=True)  # noqa: E501
            concat[version] = dataset
        from .datasets import _standardize_order
        dataset = xr.concat(
            concat.values(),
            dim='concat',
            coords='minimal',
            compat='override',
            combine_attrs='override',
        )
        dataset = dataset.stack(version=['concat', 'region'])
        dataset = dataset.transpose('version', ...)
        version = tuple(concat)  # original version values
        version = [(*version[num], region) for num, region in dataset.version.values]
        version = xr.DataArray(
            pd.MultiIndex.from_tuples(version, names=VERSION_LEVELS),
            dims='version',
            name='version',
            attrs={'long_name': VERSION_NAME},
        )
        dataset = dataset.assign_coords(version=version)
        dataset = dataset.squeeze()
        for name, array in noncat.items():
            dataset[name] = array
        if standardize:
            dataset = _standardize_order(dataset)
        datasets[facets] = dataset

    if datasets:
        print()
    return datasets


def process_scalar(
    *paths, output=None, names=None, project=None,
    standardize=True, kernels=None, restrict=False, **kwargs,
):
    """
    Process global average unperturbed and perturbed feedback estimates.

    Parameters
    ----------
    *paths : str or pathlib.Path
        The search paths.
    output : str or pathlib.Path, optional
        The output directory or name.
    names : str or sequence, optional
        The variables to select. Default is net, cloud, and clear-sky components.
    project : str, optional
        The project to search.
    kernels : bool, optional
        Whether to get kernel-adjusted cloud instead of raw cloud by default.
    restrict : bool, optional
        Whether to restrict the output for faster computation.
    testing : bool, optional
        Whether to calculate single versions for a single component.
    standardize : bool, optional
        Whether to standardize the resulting order.
    **kwargs
        Passed to `_parse_kwargs`.

    Returns
    -------
    result : xarray.Dataset
        The resulting terms.
    """
    # Initial stuff
    # NOTE: This is used to get mean and internal variability estimates for use with
    # tables and eventual emergent constraints. Includes different estimates.
    from observed.feedbacks import _parse_kwargs, process_scalar
    testing = bool(kwargs.get('testing', None))
    kwargs.setdefault('annual', (None,) if restrict else (False, True))
    kwargs.setdefault('detrend', ('xy', '')[:restrict or None])
    kwargs.setdefault('month', ('jan', 'jul')[:restrict or None])
    kwargs.setdefault('anomaly', (False, True)[:restrict or None])
    params, _, constraints = _parse_kwargs('source', testing=testing, **kwargs)  # noqa: E501 skip source
    constraints['variable'] = 'fluxes'
    correct = constraints.pop('correct', None)
    translate = {('years', None): ('period', 'full'), **LABELS_YEARS}
    kwargs = {'output': False, 'correct': correct, 'translate': translate}
    heads = ('', 'sw', 'lw')  # default wavelengths
    tails = ('net', 'cld', 'ncl') if kernels else ('net', 'cre', 'cs')
    default = tuple(  # default variables
        head or tail if tail == 'net' else head + tail for tail in tails
        for head in (('',) if restrict and tail not in ('cld', 'cre') else heads)
    )
    args = (names,) if isinstance(names, str) else names
    aliases = tuple(VARIABLE_ALIASES.get(name, name) for name in args or default)
    variables = tuple(ALIAS_VARIABLES.get(name, name) for name in args or default)
    label = '-'.join(aliases) if args else 'cld' if kernels else 'cre'
    paths = paths or ('~/data/cmip-fluxes', '~/scratch/cmip-fluxes')
    suffix = ['0000', '0150', 'eraint', 'series']
    projects = project.split(',') if isinstance(project, str) else ('cmip5', 'cmip6')
    def _find_dependencies(data, args, names=None):  # noqa: E301
        if isinstance(args, str) or type(args) not in (list, tuple):
            args = (args,)
        for arg in args:
            if isinstance(arg, str):
                if m := REGEX_FLUX.match(arg):
                    arg = m.group()  # remove e.g. 'lam' suffix
                try:
                    arg = get_parts(data, arg)
                except KeyError:
                    continue
            for part in arg.parts:
                if isinstance(part, xr.DataArray):
                    names = names or {}
                    names.setdefault(arg.__class__.__name__, set()).add(part.name)
                else:  # namedtuple with parent and dependencies
                    part.__class__.__name__ = arg.__class__.__name__
                    names = _find_dependencies(data, part, names)
        return names

    # Calculate feedback parameters
    # NOTE: Here calculate both 150-year control and 20-year and 50-year
    # internal variability estimates. Also skip 'month' 'anomaly' and 'detrend'
    # options irrelevant for abrupt experiments and skip 'x' and 'y' detrend
    # options for control runs for simplicity / since climate is stationary.
    results = {}
    print('Names:', *aliases)
    from .datasets import FACETS_LEVELS, FACETS_NAME, FACETS_RENAME
    for project in map(str.upper, projects):
        constraints['project'] = project
        files, *_ = glob_files(*paths, project=project)
        database = Database(files, FACETS_LEVELS, flagship_translate=True, **constraints)  # noqa: E501
        if database:
            print('Model:', end=' ')
        for facets, data in database.items():
            # Initial stuff
            paths = [
                path for paths in data.values() for path in paths
                if path.stem.split('_')[-1].split('-') == suffix
            ]
            if not paths:
                continue
            if len(paths) > 1:
                warnings.warn('Ambiguous', '_'.join(facets), 'paths:', ', '.join(map(str, paths)))  # noqa: E501
            for sub, replace in FACETS_RENAME.items():
                facets = tuple(facet.replace(sub, replace) for facet in facets)
            if facets[3] == 'picontrol':  # use default 'annual' 'correct'
                years = (None, 20)  # default value
                others = dict()
                annual = False  # default value
            elif facets[3] == 'abrupt4xco2':  # use default 'annual' 'correct'
                years = ((0, 150), (0, 20), (20, 150))
                others = dict(month=(None,), anomaly=(False,), detrend=('',))
                annual = True  # default value
            else:
                continue
            # Calculate feedback parameters
            inames, iparams = variables[:1] if testing else variables, params.copy()
            iparams.update(years=years, **others)
            iparams = {key: vals[:1] if testing else vals for key, vals in iparams.items()}  # noqa: E501
            iparams['annual'] = tuple(annual if ann is None else annual for ann in iparams['annual'])  # noqa: E501
            series = load_file(paths[0], lazy=True, project=project)  # speed-up
            start = series.time.dt.strftime('%b').values[0].lower()
            fluxes = _find_dependencies(series, inames)  # 'name': [*dependencies]
            retain = {'ts', *(key for keys in fluxes.values() for key in keys)}
            series = series.drop_vars(series.keys() - retain)
            series = series.climo.add_cell_measures()
            series = xr.Dataset({name: data.climo.average('area') for name, data in series.items()})  # noqa: E501
            source = paths[0].stem.split('-')[-2]  # e.g. '0000-0150-eraint-series.nc'
            print(f'{facets[2]}_{facets[3]}_{start} ({len(fluxes)})', end=' ')
            kw_process = {'name': tuple(fluxes), 'source': source, **iparams, **kwargs}
            result = process_scalar(series, **kw_process)
            levels = ('experiment', 'ensemble', *result.indexes['version'].names)
            version = [(*facets[-2:], *index) for index in result.version.values]
            version = xr.DataArray(
                pd.MultiIndex.from_tuples(version, names=levels),
                dims='version',
                name='version',
                attrs={'long_name': 'feedback version'},
            )
            facets = facets[:3]  # project institute model
            result = result.assign_coords(version=version)
            if facets in results:  # combine new version coordinates
                args = (result, results[facets])
                result = xr.concat(args, dim='version')
            results[facets] = result

    # Concatenate along facets and save result
    # WARNING: Here .reset_index() also moves coordinates to left-most position
    # so pass allstart of the
    # NOTE: Here xarray cannot save multi-index so have to reset index
    # then stack back into multi-index with .stack(version=[...]).
    from .datasets import _standardize_order
    names = {name: da for ds in results.values() for name, da in ds.data_vars.items()}
    print('Adding missing variables.')
    if results:
        print('Model:', end=' ')
    for facets, result in tuple(results.items()):  # interpolated datasets
        print('_'.join(facets[1:4]), end=' ')
        for name in names.keys() - result.data_vars.keys():
            array = names[name]  # *sample* from another model or project
            array = xr.full_like(array, np.nan)  # preserve attributes as well
            if all('version' in keys for keys in (array.dims, result, result.sizes)):
                array = array.isel(version=0, drop=True)
                array = array.expand_dims(version=result.version.size)
                array = array.assign_coords(version=result.version)
            result[name] = array
    print()
    print('Concatenating datasets.')
    if not results:
        raise RuntimeError('No datasets found.')
    facets = xr.DataArray(
        pd.MultiIndex.from_tuples(results, names=FACETS_LEVELS[:3]),
        dims='facets',
        name='facets',
        attrs={'long_name': FACETS_NAME},
    )
    dataset = xr.concat(
        results.values(),
        dim=facets,
        coords='minimal',
        compat='override',
        combine_attrs='override',
    )
    dataset = dataset.transpose('facets', 'version', 'statistic', ...)
    dataset = _update_attrs(dataset)
    if standardize:
        dataset = _standardize_order(dataset)
    proj = projects[0] if len(projects) == 1 else 'cmip'
    base = Path('~/data/global-feedbacks').expanduser()
    file = 'tmp.nc' if testing else f'feedbacks_{proj.upper()}_global-{label}.nc'
    if isinstance(output, str) and '/' not in output:
        output = base / output
    elif output:
        output = Path(output).expanduser()
    if not output:
        output = base / file
    elif not output.suffix:
        output = output / file
    if not output.parent.is_dir():
        os.mkdir(output.parent)
    if output.is_file():
        os.remove(output)
    print(f'Saving file: {output.name}')
    dataset.attrs['facets_levels'] = tuple(dataset.indexes['facets'].names)
    dataset.attrs['version_levels'] = tuple(dataset.indexes['version'].names)
    dataset.reset_index(('facets', 'version')).to_netcdf(output)
    return dataset
