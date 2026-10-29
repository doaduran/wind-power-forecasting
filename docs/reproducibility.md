# Reproducibility and artifact provenance

## What this release verifies

The two submission CSVs match the hashes recorded in the submitted notebooks. Each contains 9,408 unique IDs in the same order. Means lie in [0, 1], uncertainty standard deviations are positive, and all values are finite.

`python scripts/replay_artifacts.py` rebuilds the final mean ensemble from `m1..m9` and `d_v5_orth.npy`, and exports the saved uncertainty mean and sigma arrays. On the packaging environment, the mean replay differed from the reference by at most 7.87e-13 before eight-decimal serialization; the uncertainty replay differed by zero. Both serialized CSVs matched the original SHA-256 hashes exactly.

This is a numerical consistency check of supplied artifacts. It does not establish how the component predictions were trained, validate the hidden leaderboard labels, or reproduce the entire research process.

| Reference submission | SHA-256 |
| --- | --- |
| `submission_mean_final_v5.csv` | `1e4060c84d292eecd81b957c1cf9c1af61321866d9c9bac3b5e534268f12470f` |
| `submission_v14_final_decomposed_guided.csv` | `1015761f1edefcb0f3614816d717bcc3a41a876fd12fe00debdc89dc0fe1e36b` |

## Important artifact contracts

- `component_models.npz` contains nine prediction vectors, not trained model weights.
- `specialist_test_predictions.npz` contains cached specialist predictions. The compatibility functions in `export_aligned.py` return those arrays; they do not perform neural-network or forest inference.
- `final_v14_components.npz` contains the final uncertainty `mu_final` and `sigma_final` vectors. Exporting them does not rerun imputation, probabilistic training or variance calibration.
- The saved arrays do not include row IDs. Replay assigns the IDs from the checksum-verified reference CSVs and assumes original competition row order. Training-data alignment cannot be independently checked without the source data.
- `p5_expected.npy`, `p6_expected.npy` and `p8_expected.npy` can check consistency against the cached specialist arrays. They are not independent accuracy tests.
- The two `outer_evaluation.json` files under `dual_kriging_tensor_ensemble` and `final_v5_subspace_ensemble` are byte-identical in the supplied package. They must not be counted as separate validation experiments.

## Historical notebooks

The notebooks retain historical outputs for reading. Those outputs were not regenerated during portfolio packaging. A note was added to each report, student-identifying notebook filenames and metadata were replaced with author attribution, stale score metadata was removed, and `WORK_ROOT = ROOT` was added to the mean notebook setup to resolve an undefined variable.

Baseline feature and neural-network implementations remain in the reports. Some advanced training implementations, fitted preprocessing, seeds' trained weights, raw out-of-fold predictions and full experiment logs are not provided. Therefore, the notebooks are research reports with partially executable sections, rather than a verified Run All training pipeline.

Private/public subset explanations in the historical discussion are the author's interpretations. Hidden target distributions and causal explanations cannot be established from the included predictions alone. Public leaderboard feedback was used in the final mean ensemble constants.

## Data and access

Raw competition data is excluded. Authorized participants can obtain it through the competition pages linked in the README. The recorded forecasts and diagnostic images are included as supplied project artifacts. The replay requires only the repository files and NumPy/pandas; notebook exploration has additional dependencies.

## Packaging checks

- Opened all NumPy artifacts with `allow_pickle=False` and checked array shapes and finite values.
- Parsed the notebooks and checked code-cell syntax.
- Checked submission checksums, IDs and numeric bounds.
- Ran the final prediction replay and matched both CSV hashes.
- Checked for embedded access credentials and local absolute paths before publication.
- Did not rerun full training, recompute leaderboard scores, or reconstruct the original seasonal evaluation.
