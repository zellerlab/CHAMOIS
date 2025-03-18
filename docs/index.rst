.. CHAMOIS documentation master file, created by
   sphinx-quickstart on Tue Mar 18 21:40:56 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

CHAMOIS |Stars|
===============

.. |Stars| image:: https://img.shields.io/github/stars/zellerlab/CHAMOIS.svg?style=social&maxAge=3600&label=Star
   :target: https://github.com/zellerlab/CHAMOIS/stargazers
   :class: dark-light

*Chemical Hierarchy Approximation for secondary Metabolite clusters Obtained In Silico.*

|License| |Source| |Mirror| |Changelog| |Issues| |Preprint|

.. |License| image:: https://img.shields.io/badge/license-GPLv3-blue.svg?style=flat-square&maxAge=2678400
   :target: https://choosealicense.com/licenses/gpl-3.0/

.. |Source| image:: https://img.shields.io/badge/source-GitHub-303030.svg?maxAge=2678400&style=flat-square
   :target: https://github.com/zellerlab/CHAMOIS/

.. |Mirror| image:: https://img.shields.io/badge/mirror-EMBL-009f4d?style=flat-square&maxAge=2678400
   :target: https://git.embl.de/larralde/CHAMOIS

.. |Changelog| image:: https://img.shields.io/badge/keep%20a-changelog-8A0707.svg?maxAge=2678400&style=flat-square
   :target: https://github.com/zellerlab/CHAMOIS/blob/master/CHANGELOG.md

.. |Issues| image:: https://img.shields.io/github/issues/zellerlab/CHAMOIS.svg?style=flat-square&maxAge=600
   :target: https://github.com/zellerlab/CHAMOIS/issues

.. |Preprint| image:: https://img.shields.io/badge/preprint-bioRxiv-darkblue?style=flat-square&maxAge=2678400
   :target: https://www.biorxiv.org/content/10.1101/2025.03.13.642868


Overview
--------

CHAMOIS is a fast method for predicting chemical features of natural products 
produced by Biosynthetic Gene Clusters (BGCs) using only their genomic 
sequence. It can be used to get chemical features from BGCs predicted in 
silico with tools such as `GECCO <https://gecco.embl.de>`_ or 
`antiSMASH <https://antismash.secondarymetabolites.org>`_.


Setup
-----

Run ``pip install git+https://github.com/zellerlab/CHAMOIS`` in a shell to 
download the development version from GitHub, or have a look at the
:doc:`Installation page <guide/install>` to find other ways to install CHAMOIS.


Citation
--------

CHAMOIS is scientific software, with a
`preprint <https://www.biorxiv.org/content/10.1101/2025.03.13.642868>`_
in `BioRxiv <https://www.biorxiv.org>`_. Check the
:doc:`Publications page <guide/publications>` to see how to cite CHAMOIS.



Library
-------

.. toctree::
   :maxdepth: 2

   User Guide <guide/index>
   Examples <examples/index>
   API Reference <api/index>


Feedback
--------

Contact
^^^^^^^

If you have any question about CHAMOIS, if you run into any issue, or if you
would like to make a feature request, please create an
`issue in the GitHub repository <https://github.com/zellerlab/CHAMOIS/issues/new>`_.
You can also directly contact `Martin Larralde via email <mailto:martin.larralde@embl.de>`_.

Contributing
^^^^^^^^^^^^

If you want to contribute to CHAMOIS, please have a look at the
contribution guide first, and feel free to open a pull
request on the `GitHub repository <https://github.com/zellerlab/CHAMOIS>`_.



License
-------

This library is provided under the `GNU General Public License 3.0 or later <https://choosealicense.com/licenses/gpl-3.0/>`_.
See the :doc:`Copyright Notice <guide/copyright>` section for more information.

*This project was developed by* `Martin Larralde <https://github.com/althonos>`_ 
*during his PhD project at the* `European Molecular Biology Laboratory <https://www.embl.de/>`_
*and the* `Leiden University Medical Center <https://lumc.nl/en/>`_
*in the* `Zeller team <https://zellerlab.org>`_.

