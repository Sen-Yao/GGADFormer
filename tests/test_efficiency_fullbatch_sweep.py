import importlib.util
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / 'experiments' / 'efficiency-fullbatch-k-scaling-019fad8b'


def load_runner():
    spec = importlib.util.spec_from_file_location('fullbatch_run_shard', EXPERIMENT / 'run_shard.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def arg_value(args, name):
    prefix = f'--{name}='
    values = [arg[len(prefix):] for arg in args if arg.startswith(prefix)]
    assert len(values) == 1
    return values[0]


def test_sweep_and_registry_have_exactly_the_same_18_trials():
    runner = load_runner()
    trials = runner.load_trials(EXPERIMENT / 'registry.json')
    sweep = yaml.safe_load((EXPERIMENT / 'fullbatch-sweep.yaml').read_text(encoding='utf-8'))
    sweep_ids = sweep['parameters']['trial_id']['values']
    assert len(trials) == len(sweep_ids) == 18
    assert {trial['id'] for trial in trials} == set(sweep_ids)
    assert len(sweep_ids) == len(set(sweep_ids))


def test_registry_enforces_full_batch_and_common_supervision():
    runner = load_runner()
    registry = json.loads((EXPERIMENT / 'registry.json').read_text(encoding='utf-8'))
    assert registry['full_batch_required'] is True
    assert registry['train_ratio'] == 0.05
    assert registry['data_split_seed'] == 42
    for trial in runner.load_trials(EXPERIMENT / 'registry.json'):
        args = trial['args']
        if trial['method'] == 'VecGAD':
            assert int(arg_value(args, 'batch_size')) == trial['num_nodes']
            assert float(arg_value(args, 'train_rate')) == 0.05
            assert int(arg_value(args, 'data_split_seed')) == 42
        elif trial['method'] == 'GGAD':
            assert float(arg_value(args, 'train_rate')) == 0.05
            assert int(arg_value(args, 'data_split_seed')) == 42
        else:
            assert int(arg_value(args, 'batch-size')) == 0
            assert float(arg_value(args, 'train-ratio')) == 0.05
            assert int(arg_value(args, 'data-split-seed')) == 42
