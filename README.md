# 🐐 CHAMOIS [![Stars](https://img.shields.io/github/stars/zellerlab/CHAMOIS.svg?style=social&maxAge=3600&label=Star)](https://github.com/zellerlab/CHAMOIS/stargazers)

*Chemical Hierarchy Approximation for secondary Metabolism clusters Obtained In Silico.*

[![License](https://img.shields.io/badge/license-GPLv3-blue.svg?style=flat-square&maxAge=2678400)](https://choosealicense.com/licenses/gpl-3.0/)
[![Source](https://img.shields.io/badge/source-GitHub-303030.svg?maxAge=2678400&style=flat-square)](https://github.com/zellerlab/CHAMOIS/)
[![Mirror](https://img.shields.io/badge/mirror-EMBL-009f4d?style=flat-square&maxAge=2678400)](https://git.embl.de/larralde/CHAMOIS)
[![Changelog](https://img.shields.io/badge/keep%20a-changelog-8A0707.svg?maxAge=2678400&style=flat-square)](https://github.com/zellerlab/CHAMOIS/blob/master/CHANGELOG.md)
[![Issues](https://img.shields.io/github/issues/zellerlab/CHAMOIS.svg?style=flat-square&maxAge=600)](https://github.com/zellerlab/CHAMOIS/issues)

## 🗺️  ️Overview

CHAMOIS is a fast method for predicting chemical features of natural products 
produced by Biosynthetic Gene Clusters (BGCs) using only their genomic 
sequence. It can be used to get chemical features from BGCs predicted in 
silico with tools such as [GECCO](https://gecco.embl.de) or 
[antiSMASH](https://antismash.secondarymetabolites.org).

## 🔧 Installing CHAMOIS

CHAMOIS is implemented in [Python](https://www.python.org/), and supports 
[all versions](https://endoflife.date/python) from Python 3.7 onwards. 
It requires additional libraries that can be installed directly from
[PyPI](https://pypi.org), the Python Package Index.

Clone the repository and install it from the local folder. This will take 
a little bit of time, since it will download the Pfam HMMs used for annotation
and install dependencies:

```console
$ pip install git+https://github.com/zellerlab/CHAMOIS
```

*Note that CHAMOIS uses [HMMER3](http://hmmer.org/), which can only run
on PowerPC, x86-64 and Aarch64 machines running a POSIX operating system.
Therefore, CHAMOIS **will work on Linux and OSX, but not on Windows.***

## 🧬 Running CHAMOIS

Once CHAMOIS is installed, you can run it from the terminal by providing
it with one or more GenBank file the genomic records of the BGCs to analyze,
and an output path where to write the results in HDF5 format:

```console
chamois predict -i records.gbk -o probas.hdf5
```

## 🔎 Results

The output file can be loaded with the `anndata` package, and corresponds
to a probability matrix where rows are the input BGCs, and columns are the
ChemOnt classes.

To get a summary for each predicted BGC, use the `render` command:

```console
chamois render -i probas.hdf5
```

Predictions for each BGC will be shown as a tree with their computed 
probabilities:

```console
╭─────────────────────────── CP123780.1_cluster1 ──────────────────────────────╮
│ CHEMONTID:0000002 (Organoheterocyclic compounds): 0.823                      │
│ ├── CHEMONTID:0000050 (Lactones): 0.638                                      │
│ └── CHEMONTID:0004140 (Oxacyclic compounds): 0.823                           │
│ CHEMONTID:0000012 (Lipids and lipid-like molecules): 0.587                   │
│ └── CHEMONTID:0003909 (Fatty Acyls): 0.587                                   │
│     └── CHEMONTID:0000262 (Fatty acids and conjugates): 0.587                │
│         └── CHEMONTID:0000339 (Unsaturated fatty acids): 0.587               │
│ CHEMONTID:0000264 (Organic acids and derivatives): 0.940                     │
│ └── CHEMONTID:0000265 (Carboxylic acids and derivatives): 0.833              │
│     ├── CHEMONTID:0001093 (Carboxylic acid derivatives): 0.679               │
│     ├── CHEMONTID:0001137 (Monocarboxylic acids and derivatives): 0.618      │
│     └── CHEMONTID:0001205 (Carboxylic acids): 0.517                          │
│ CHEMONTID:0004150 (Hydrocarbon derivatives): 0.997                           │
│ CHEMONTID:0004603 (Organic oxygen compounds): 0.997                          │
│ ├── CHEMONTID:0000323 (Organooxygen compounds): 0.994                        │
│ │   ├── CHEMONTID:0000129 (Alcohols and polyols): 0.893                      │
│ │   │   └── CHEMONTID:0001661 (Secondary alcohols): 0.893                    │
│ │   ├── CHEMONTID:0000254 (Ethers): 0.538                                    │
│ │   └── CHEMONTID:0001831 (Carbonyl compounds): 0.852                        │
│ └── CHEMONTID:0003940 (Organic oxides): 0.979                                │
╰──────────────────────────────────────────────────────────────────────────────╯
```


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
