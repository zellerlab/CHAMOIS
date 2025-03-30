# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [Unreleased]
[Unreleased]: https://github.com/zellerlab/CHAMOIS/compare/v0.1.3...HEAD


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
