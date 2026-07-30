# VecGAD direction x magnitude strict ablation rerun

This protocol runs the AAAI 2027 strict perturbation-control matrix on HCCS-90:
Amazon, Elliptic, and Tolokers; five variants; seeds 0-4; 75 independent W&B
runs. Metrics are fixed-epoch `AUC.last` and `AP.last` from the final evaluation
step, without best-epoch or best-seed selection.

The five variants are:

- `none`: learned projected perturbation direction and native node-specific magnitude.
- `random_dir`: independent random unit direction and native projected magnitude.
- `random_mag`: learned direction and exact cyclic permutation of current-generation projected magnitudes.
- `random_both`: independent random unit direction and the same exact cyclic magnitude permutation helper as `random_mag`.
- `constant_mag`: learned direction and current-generation mean projected magnitude.

Random controls resample every forward. Their RNGs are separate from global
PyTorch/NumPy/Python RNG state and from each other. If not supplied explicitly,
each run records:

- `ablation_direction_seed = seed * 1000003 + 1729`
- `ablation_magnitude_seed = seed * 1000003 + 7919`

The unavailable Overleaf paths named in the task were not present in this
worktree, so authority configuration was decided from `TODOLIST.md`,
`reproduction.sh`, project W&B run/sweep evidence, and the existing HCCS-90
T-Finance protocol template.

## Final Status

Completed on HCCS-90 with three native W&B sweeps:

- Amazon: `HCCS/GGADFormer/qoxpewpp`
- Elliptic: `HCCS/GGADFormer/q7j5snr0`
- Tolokers: `HCCS/GGADFormer/njo17tej`

Independent W&B replay validated 75/75 expected trials, all `finished`, all
final history steps present, all resolved configs and ablation RNG seeds matching
the manifest, and all 12 agent exit codes equal to 0. Results are in
`results.json`; remote log hashes are in `remote-log-sha256.txt`.

| Dataset | Variant | AUC.last mean | AUC.last std | AP.last mean | AP.last std |
|---|---|---:|---:|---:|---:|
| Amazon | none | 0.9398 | 0.0163 | 0.8097 | 0.0112 |
| Amazon | random_dir | 0.5739 | 0.2592 | 0.1632 | 0.1527 |
| Amazon | random_mag | 0.9128 | 0.0107 | 0.7820 | 0.0124 |
| Amazon | random_both | 0.5323 | 0.2733 | 0.1511 | 0.1323 |
| Amazon | constant_mag | 0.9474 | 0.0041 | 0.8028 | 0.0148 |
| elliptic | none | 0.7681 | 0.0294 | 0.2949 | 0.0798 |
| elliptic | random_dir | 0.3636 | 0.0399 | 0.0705 | 0.0040 |
| elliptic | random_mag | 0.4308 | 0.0702 | 0.0798 | 0.0111 |
| elliptic | random_both | 0.3561 | 0.0235 | 0.0696 | 0.0022 |
| elliptic | constant_mag | 0.5618 | 0.0442 | 0.1034 | 0.0125 |
| tolokers | none | 0.6394 | 0.0612 | 0.2963 | 0.0382 |
| tolokers | random_dir | 0.5968 | 0.0985 | 0.2754 | 0.0546 |
| tolokers | random_mag | 0.4668 | 0.0816 | 0.2155 | 0.0351 |
| tolokers | random_both | 0.5967 | 0.0990 | 0.2751 | 0.0548 |
| tolokers | constant_mag | 0.5029 | 0.0939 | 0.2326 | 0.0409 |

Interpretation boundary: learned projected direction shows a clear benefit over
random direction on Amazon and Elliptic, and a smaller/noisier benefit on
Tolokers. Preserving native node-magnitude association helps on Amazon,
Elliptic, and Tolokers when comparing Full with exact-permuted magnitude.
Constant magnitude is not uniformly worse than Full: it is slightly higher on
Amazon AUC but lower on Amazon AP and clearly lower on Elliptic/Tolokers. These
results support a bounded claim that node-dependent learned projected
perturbation direction can add value beyond magnitude alone, but they do not
establish true anomaly direction, anomaly semantics, or a scalar-RDV reduction
interpretation.
