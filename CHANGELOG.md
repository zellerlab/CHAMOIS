# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [Unreleased]
[Unreleased]: https://github.com/zellerlab/CHAMOIS/compare/v0.2.1...HEAD


## [v0.2.1] - 2026-01-24
[v0.2.1]: https://github.com/zellerlab/CHAMOIS/compare/v0.2.0...v0.2.1

### Changed
- Bump `pyhmmer` requirement to `v0.12.0`.
- Use SPDX license qualifiers in `pyproject.toml`.


## [v0.2.0] - 2025-12-07
[v0.2.0]: https://github.com/zellerlab/CHAMOIS/compare/v0.1.3...v0.2.0

### Added
- Training option to ignore classes and features in less than *N* groups.
- `cvi` command run independent cross-validation as done in the paper.
- Support for training and evaluating random forest models.
- Support for computing sample weights for observation groups in `fit`.
- Report information content of the prediction in `predict` output.

### Changed
- Relax `anndata` dependency to allow more recent releases.
- Update MIBiG 3.1 dataset to filter out erroneous partial clusters.
- Retrain CHAMOIS with filtered MIBIG 3.1 dataset and improved feature selection.
- Make `chamois.compositions.build_variables` drop empty columns.
- Rename `screen` command to `compare`.
- Incorrect use of chemical groups in cross-validation commands.
- Update Pfam to 38.0.

### Fixed
- Issue with computation of probabilistic Jaccard distance in `validate` command.
- Incorrect error message on non-existing class in `explain `command.
- Incorrect merging of multiple feature tables in CLI.
- Extraction of minimum class and features by groups for majority positive classes.
- Incorrect error message in `predict` command.
- Unsupported sample weighting in older `scikit-learn` releases.


## [v0.1.3] - 2025-03-30
[v0.1.3]: https://github.com/zellerlab/CHAMOIS/compare/v0.1.2...v0.1.3

### Fixed
- Build issue on missing `rich` package.

### Added
- Jaccard index to reported metrics in `chamois cv`, `chamois train` and `chamois validate` commands.


## [v0.1.2] - 2025-03-26
[v0.1.2]: https://github.com/zellerlab/CHAMOIS/compare/v0.1.1...v0.1.2

### Fixed
- Make installation procedure not require extra packages in most cases.
- Remove `universal` from `bdist_wheel` command in `setup.cfg`.
- Seeds not being passed from the CLI to the actual `ChemicalOntologyPredictor` instances.

### Documentation
- Add missing Javascript files for PyPI icons.
- Update installation instructions for Docker and Singularity.


## [v0.1.1] - 2025-03-21
[v0.1.1]: https://github.com/zellerlab/CHAMOIS/compare/v0.1.0...v0.1.1

### Fixed
- Automated deployment and setup of filtered Pfam HMMs from GitHub.


## [v0.1.0] - 2025-03-21
[v0.1.0]: https://github.com/zellerlab/CHAMOIS/compare/bfe081f...v0.1.0

Initial release.
