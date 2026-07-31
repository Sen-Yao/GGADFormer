import importlib.util
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / 'experiments' / 'efficiency-historical-forensic-019fad8b'


def load_wrapper():
    spec = importlib.util.spec_from_file_location(
        'historical_forensic_wrapper', EXPERIMENT / 'run_wandb_trial.py'
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_registry_and_sweep_define_the_same_24_trials():
    wrapper = load_wrapper()
    trials = wrapper.load_trials(EXPERIMENT / 'registry.json')
    sweep = yaml.safe_load(
        (EXPERIMENT / 'historical-forensic-sweep.yaml').read_text(encoding='utf-8')
    )
    sweep_ids = sweep['parameters']['trial_id']['values']
    assert len(trials) == len(sweep_ids) == 24
    assert [trial['id'] for trial in trials] == sweep_ids
    assert len(sweep_ids) == len(set(sweep_ids))


def test_source_and_batch_contracts_are_frozen():
    wrapper = load_wrapper()
    trials = wrapper.load_trials(EXPERIMENT / 'registry.json')
    expected_sources = {
        'e071ae6646451d94fc8e8c9e88305eb76c393089': (4, True, True),
        '5bf8205b0d4c54d583b13c547ae62122ffdf2f6a': (0, False, False),
    }
    for trial in trials:
        loader = trial['loader']
        assert (
            loader['num_workers'],
            loader['persistent_workers'],
            loader['pin_memory'],
        ) == expected_sources[trial['source_commit']]
        if trial['batch_mode'] == 'native':
            assert trial['batch_size'] < trial['num_nodes']
        else:
            assert trial['batch_size'] == trial['num_nodes']


def test_historical_wandb_configuration_provenance_is_preserved():
    registry = json.loads((EXPERIMENT / 'registry.json').read_text(encoding='utf-8'))
    amazon = registry['datasets']['Amazon']
    tfinance = registry['datasets']['t_finance']
    assert amazon['wandb_source'] == 'HCCS/GGADFormer/8ylmsq7q'
    assert amazon['native_batch_size'] == 1024
    assert amazon['config']['pp_k'] == 5
    assert amazon['config']['progregate_alpha'] == 0.3
    assert tfinance['wandb_source'] == 'HCCS/GGADFormer/iqxjqsdl'
    assert tfinance['wandb_source_commit'].startswith('e071ae66')
    assert tfinance['native_batch_size'] == 8192
    assert tfinance['config']['pp_k'] == 7
    assert tfinance['config']['progregate_alpha'] == 0.3


def test_manifest_keeps_forensic_internal_and_unlaunched():
    manifest = json.loads((EXPERIMENT / 'RUN_MANIFEST.json').read_text(encoding='utf-8'))
    assert manifest['state'] == 'prepared'
    assert manifest['scope']['evidence_class'] == 'internal_diagnostic'
    assert manifest['scope']['expected_trials'] == 24
    assert manifest['wandb']['sweep_id'] is None
