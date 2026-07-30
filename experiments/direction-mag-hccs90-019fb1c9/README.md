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

