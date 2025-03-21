# 🐐 CHAMOIS [![Stars](https://img.shields.io/github/stars/zellerlab/CHAMOIS.svg?style=social&maxAge=3600&label=Star)](https://github.com/zellerlab/CHAMOIS/stargazers)

*Chemical Hierarchy Approximation for secondary Metabolism clusters Obtained In Silico.*

[![Actions](https://img.shields.io/github/actions/workflow/status/zellerlab/CHAMOIS/test.yml?branch=main&logo=github&style=flat-square&maxAge=300)](https://github.com/zellerlab/CHAMOIS/actions)
[![PyPI](https://img.shields.io/pypi/v/chamois-tool.svg?logo=pypi&style=flat-square&maxAge=3600)](https://pypi.org/project/chamois-tool)
[![Wheel](https://img.shields.io/pypi/wheel/chamois-tool.svg?style=flat-square&maxAge=3600)](https://pypi.org/project/chamois-tool/#files)
[![Python Versions](https://img.shields.io/pypi/pyversions/chamois-tool.svg?logo=python&style=flat-square&maxAge=3600)](https://pypi.org/project/chamois-tool/#files)
[![License](https://img.shields.io/badge/license-GPLv3-blue.svg?style=flat-square&maxAge=2678400)](https://choosealicense.com/licenses/gpl-3.0/)
[![Source](https://img.shields.io/badge/source-GitHub-303030.svg?maxAge=2678400&style=flat-square)](https://github.com/zellerlab/CHAMOIS/)
[![Mirror](https://img.shields.io/badge/mirror-EMBL-009f4d?style=flat-square&maxAge=2678400)](https://git.embl.de/larralde/CHAMOIS)
[![Issues](https://img.shields.io/github/issues/zellerlab/CHAMOIS.svg?style=flat-square&maxAge=600)](https://github.com/zellerlab/CHAMOIS/issues)
[![Docs](https://img.shields.io/readthedocs/chamois/latest?style=flat-square&maxAge=600)](https://chamois.readthedocs.io)
[![Changelog](https://img.shields.io/badge/keep%20a-changelog-8A0707.svg?maxAge=2678400&style=flat-square)](https://github.com/zellerlab/CHAMOIS/blob/main/CHANGELOG.md)
[![Preprint](https://img.shields.io/badge/preprint-bioRxiv-darkblue?style=flat-square&maxAge=2678400)](https://www.biorxiv.org/content/10.1101/2025.03.13.642868)

## 🗺️  ️Overview

CHAMOIS is a fast method for predicting chemical features of natural products 
produced by Biosynthetic Gene Clusters (BGCs) using only their genomic 
sequence. It can be used to get chemical features from BGCs predicted in 
silico with tools such as [GECCO](https://gecco.embl.de) or 
[antiSMASH](https://antismash.secondarymetabolites.org).

## 💡 Usage

This section shows only the basic commands for installing and running CHAMOIS. 
The [online documentation](https://chamois.readthedocs.io) 
contains a more detailed 
[installation guide](https://chamois.readthedocs.io/en/latest/guide/install.html),
[examples](https://chamois.readthedocs.io/en/latest/examples/index.html), 
an [API reference](https://chamois.readthedocs.io/en/latest/api/index.html), 
and a [CLI reference](https://chamois.readthedocs.io/en/latest/cli/index.html)

### 🔧 Installing CHAMOIS

CHAMOIS is implemented in [Python](https://www.python.org/), and supports 
[all versions](https://endoflife.date/python) from Python 3.7 onwards. 
It requires additional libraries that can be installed directly from
[PyPI](https://pypi.org), the Python Package Index.

Clone the repository and install it from the local folder. This will take 
a little bit of time, since it will download the Pfam HMMs used for annotation
and install dependencies:

```console
$ pip install chamois-tool
```

*Note that CHAMOIS uses [HMMER3](http://hmmer.org/), which can only run
on PowerPC, x86-64 and Aarch64 machines running a POSIX operating system.
Therefore, CHAMOIS **will work on Linux and OSX, but not on Windows.***

### 🧬 Running CHAMOIS

Once CHAMOIS is installed, you can run it from the terminal by providing
it with one or more GenBank file the genomic records of the BGCs to analyze,
and an output path where to write the results in HDF5 format. For instance to
predict the classes for [BGC0000703](https://mibig.secondarymetabolites.org/repository/BGC0000703.4/index.html#r1c1), 
a kanamycin-producing BGC from MIBiG:

```console
$ chamois predict -i tests/data/BGC0000703.4.gbk -o tests/data/BGC0000703.4.hdf5
```

### 🔎 Viewing results

The output file can be loaded with the `anndata` package, and corresponds
to a probability matrix where rows are the input BGCs, and columns are the
ChemOnt classes.

To get a summary for each predicted BGC, use the `render` command:

```console
$ chamois render -i tests/data/BGC0000703.4.hdf5
```

Predictions for each BGC will be shown as a tree with their computed 
probabilities:

```
CHEMONTID:0000002 (Organoheterocyclic compounds): 0.996
├── CHEMONTID:0002012 (Oxanes): 0.996│
└── CHEMONTID:0004140 (Oxacyclic compounds): 0.976
CHEMONTID:0004150 (Hydrocarbon derivatives): 0.999
CHEMONTID:0004557 (Organopnictogen compounds): 0.948
CHEMONTID:0004603 (Organic oxygen compounds): 1.000
└── CHEMONTID:0000323 (Organooxygen compounds): 1.000
    ├── CHEMONTID:0000011 (Carbohydrates and carbohydrate conjugates): 0.996
    │   ├── CHEMONTID:0001540 (Monosaccharides): 0.996
    │   ├── CHEMONTID:0002105 (Glycosyl compounds): 0.977
    │   │   └── CHEMONTID:0002207 (O-glycosyl compounds): 0.977
    │   └── CHEMONTID:0003305 (Aminosaccharides): 0.995
    │       └── CHEMONTID:0000282 (Aminoglycosides): 0.995
    │           └── CHEMONTID:0001675 (Aminocyclitol glycosides): 0.995
    │               └── CHEMONTID:0003575 (2-deoxystreptamine aminoglycosides): 0.961
    ├── CHEMONTID:0000129 (Alcohols and polyols): 1.000
    │   ├── CHEMONTID:0000286 (Primary alcohols): 0.891
    │   ├── CHEMONTID:0001292 (Cyclic alcohols and derivatives): 0.998
    │   │   └── CHEMONTID:0002509 (Cyclitols and derivatives): 0.996
    │   │       └── CHEMONTID:0002510 (Aminocyclitols and derivatives): 0.987
    │   ├── CHEMONTID:0001661 (Secondary alcohols): 0.999
    │   │   └── CHEMONTID:0002647 (Cyclohexanols): 0.995
    │   └── CHEMONTID:0002286 (Polyols): 0.972
    └── CHEMONTID:0000254 (Ethers): 0.959
        └── CHEMONTID:0001656 (Acetals): 0.959
CHEMONTID:0004707 (Organic nitrogen compounds): 0.999
└── CHEMONTID:0000278 (Organonitrogen compounds): 0.999
    ├── CHEMONTID:0002449 (Amines): 0.999
    │   ├── CHEMONTID:0002450 (Primary amines): 0.989
    │   │   └── CHEMONTID:0000469 (Monoalkylamines): 0.989
    │   └── CHEMONTID:0002460 (Alkanolamines): 0.999
    │       └── CHEMONTID:0001897 (1,2-aminoalcohols): 0.992
    └── CHEMONTID:0002674 (Cyclohexylamines): 0.987
```

## 🔖 Reference

CHAMOIS can be cited using the following preprint:

> **Machine learning inference of natural product chemistry across biosynthetic gene cluster types**.
> Martin Larralde, Georg Zeller.
> bioRxiv 2025.03.13.642868; [doi:10.1101/2025.03.13.642868](https://doi.org/10.1101/2025.03.13.642868)


## 💭 Feedback

### ⚠️ Issue Tracker

Found a bug ? Have an enhancement request ? Head over to the [GitHub issue
tracker](https://github.com/zellerlab/CHAMOIS/issues) if you need to report
or ask something. If you are filing in on a bug, please include as much
information as you can about the issue, and try to recreate the same bug
in a simple, easily reproducible situation.

### 🏗️ Contributing

Contributions are more than welcome! See [`CONTRIBUTING.md`](https://github.com/zellerlab/CHAMOIS/blob/master/CONTRIBUTING.md)
for more details.

## ⚖️ License

This software is provided under the [GNU General Public License v3.0 *or later*](https://choosealicense.com/licenses/gpl-3.0/). 
CHAMOIS is developped by the [Zeller Lab](https://zellerlab.org)
at the [European Molecular Biology Laboratory](https://www.embl.de/) in Heidelberg 
and the [Leiden University Medical Center](https://lumc.nl/en/) in Leiden.
