#!/usr/bin/env python
"""
Rename files so that 'region' comes after 'start' and 'stop'.
"""
import sys
import warnings
from pathlib import Path

from icecream import ic  # noqa: F401

# Global constants
PART_RENAMES = {'start': 'early50', 'end': 'late50'}
REGION_ABBREVS = ['globe', 'hemi', 'lat', 'pt', 'aglobe', 'ahemi', 'alat', 'apt']
STARTSTOP_ABBREVS = ['full', 'start', 'end', 'early', 'late', 'early1', 'early2']

# Iterate over figure folders
for path in sys.argv[1:]:
    path = Path(path).expanduser()
    if path.is_file():
        paths = (path,)
    elif path.is_dir():
        paths = tuple(path.glob('*.pdf'))
        print(f'Folder: {path.name}')
    else:
        raise ValueError(f'Invalid path {str(path)!r}.')
    for previous in paths:
        # Parse the path regions
        original = [parts.split('-') for parts in previous.stem.split('_')]
        ridxs = [
            i for i, parts in enumerate(original)
            if any(name in REGION_ABBREVS for name in parts)
        ]
        if len(ridxs) > 1:
            warnings.warn(f'File {previous.name!r} has ambiguous regions.')
        if len(ridxs) != 1:
            continue  # no rearrangement necessary

        # Parse and rename the periods
        groups = [
            [PART_RENAMES.get(part, part) for part in parts]
            for parts in original
        ]
        sidxs = [
            i for i, parts in enumerate(groups)
            if any(name in STARTSTOP_ABBREVS for name in parts)
        ]
        if len(sidxs) > 1:
            warnings.warn(f'File {previous.name!r} has ambiguous periods.')
        if len(sidxs) != 1 and groups == original:
            continue  # no rearrangement necessary

        # Update the path
        ridx = ridxs[0] if len(ridxs) == 1 else 1  # always true
        sidx = sidxs[0] if len(sidxs) == 1 else 0
        difference = ridx - sidx
        if abs(difference) != 1:
            warnings.warn(  # should equal 1, fix if equals -1
                f'File {previous.name!r} has unexpected region and period '
                f'positions {ridx} and {sidx}. Should be only one apart.'
            )
        if difference == -1 and len(sidxs) == 1:
            groups = [*groups[:ridx], groups[sidx], groups[ridx], *groups[sidx + 1:]]
        if groups == original:
            continue
        updated = '_'.join('-'.join(parts) for parts in groups)
        updated = previous.parent / (updated + previous.suffix)
        print(f'Previous: {previous.name}')
        print(f' Updated: {updated.name}')
        previous.rename(updated)
