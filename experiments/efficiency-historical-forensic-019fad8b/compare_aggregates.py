#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def compare(expected, observed, path='$', errors=None):
    errors = [] if errors is None else errors
    if isinstance(expected, bool) or isinstance(observed, bool):
        if expected != observed:
            errors.append(f'{path}: expected {expected!r}, got {observed!r}')
        return errors
    if isinstance(expected, (int, float)) and isinstance(observed, (int, float)):
        if not math.isclose(expected, observed, rel_tol=1e-12, abs_tol=1e-15):
            errors.append(f'{path}: expected {expected!r}, got {observed!r}')
        return errors
    if isinstance(expected, dict) and isinstance(observed, dict):
        expected_keys = set(expected)
        observed_keys = set(observed)
        if expected_keys != observed_keys:
            errors.append(
                f'{path}: key mismatch, missing={sorted(expected_keys - observed_keys)!r}, '
                f'extra={sorted(observed_keys - expected_keys)!r}'
            )
        for key in sorted(expected_keys & observed_keys):
            compare(expected[key], observed[key], f'{path}.{key}', errors)
        return errors
    if isinstance(expected, list) and isinstance(observed, list):
        if len(expected) != len(observed):
            errors.append(f'{path}: list length mismatch {len(expected)} != {len(observed)}')
        for index, (left, right) in enumerate(zip(expected, observed)):
            compare(left, right, f'{path}[{index}]', errors)
        return errors
    if type(expected) is not type(observed) or expected != observed:
        errors.append(f'{path}: expected {expected!r}, got {observed!r}')
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--expected', type=Path, required=True)
    parser.add_argument('--observed', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    expected = json.loads(args.expected.read_text(encoding='utf-8'))
    observed = json.loads(args.observed.read_text(encoding='utf-8'))
    errors = compare(expected, observed)
    output = {
        'schema_version': 1,
        'valid': not errors,
        'errors': errors,
        'relative_tolerance': 1e-12,
        'absolute_tolerance': 1e-15,
        'expected_sha256': sha256(args.expected),
        'observed_sha256': sha256(args.observed),
    }
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
