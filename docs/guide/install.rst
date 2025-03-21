Installation
============

.. caution::

    Windows is not supported by HMMER, which is used by CHAMOIS to perform
    domain annotation. CHAMOIS therefore cannot be installed on Windows
    machines. Consider using a Python install inside the
    `Windows Subsystem for Linux <https://learn.microsoft.com/en-us/windows/wsl/install>`_
    if you need CHAMOIS on a Windows computer.

Local Setup
-----------

PyPi
^^^^

CHAMOIS is hosted on GitHub, but the easiest way to install it is to download
the latest release from its `PyPi repository <https://pypi.python.org/pypi/chamois-tool>`_.
It will install all dependencies, then install CHAMOIS and its required data:

.. code:: console

    $ pip install --user chamois-tool


.. Conda
.. ^^^^^

.. CHAMOIS is also available as a `recipe <https://anaconda.org/bioconda/chamois>`_
.. in the `bioconda <https://bioconda.github.io/>`_ channel. To install, simply
.. use the ``conda`` installer:

.. .. code:: console

..      $ conda install -c bioconda chamois


.. Arch User Repository
.. ^^^^^^^^^^^^^^^^^^^^

.. A package recipe for Arch Linux can be found in the Arch User Repository
.. under the name `python-chamois <https://aur.archlinux.org/packages/python-chamois>`_.
.. It will always match the latest release from PyPI.

.. Steps to install on ArchLinux depend on your `AUR helper <https://wiki.archlinux.org/title/AUR_helpers>`_
.. (``yaourt``, ``aura``, ``yay``, etc.). For ``aura``, you'll need to run:

.. .. code:: console

..     $ aura -A python-chamois


.. BioArchLinux
.. ^^^^^^^^^^^^

.. The `BioArchLinux <https://bioarchlinux.org>`_ project provides pre-compiled packages
.. based on the AUR recipe. Add the BioArchLinux package repository to ``/etc/pacman.conf``:

.. .. code:: ini

.. ..     [bioarchlinux]
..     Server = https://repo.bioarchlinux.org/$arch

.. Then install the latest version of the package and its dependencies with ``pacman``:

.. .. code:: console

..     $ pacman -Sy
..     $ pacman -S python-chamois


GitHub + ``pip``
^^^^^^^^^^^^^^^^

If, for any reason, you prefer to download the library from GitHub, you can clone
and install the repository with ``pip`` by running (with the admin rights):

.. code:: console

    $ pip install -U git+https://github.com/zellerlab/CHAMOIS

.. caution::

    Keep in mind this will install always try to install the latest commit,
    which may not even build, so consider using a versioned release instead.


GitHub + ``build``
^^^^^^^^^^^^^^^^^^

If you do not want to use ``pip``, you can still clone the repository and
use ``build`` and ``installer`` manually:

.. code:: console

    $ git clone https://github.com/zellerlab/CHAMOIS
    $ cd CHAMOIS
    $ python -m build .
    # python -m installer dist/*.whl

.. Danger::

    Installing packages without ``pip`` is strongly discouraged, as they can
    only be uninstalled manually, and may damage your system.


Containers
----------

Docker
^^^^^^

CHAMOIS is also distributed in a Docker container for reproducibility. An image
is built for every release. To get the latest image, run:

.. code:: console

    $ docker pull ghcr.io/zellerlab/chamois

Then, to run the image and analyze files in the local directory, make sure
to mount the currend working directory to the `/io` volume, enable terminal
emulation with `-t` to get a nice output, and run the rest of the command
line interface normally:

.. code:: console

    $ docker run -v $(pwd):/io -t ghcr.io/zellerlab/chamois predict -i tests/data/BGC0000703.4.gbk -o tests/data/BGC0000703.4.hdf5


Singularity / Apptainer
^^^^^^^^^^^^^^^^^^^^^^^

A recipe for Singularity / Apptainer containers is available in the project
repository. Clone the repository and then build the image with:

.. code:: console

    $ git clone https://github.com/zellerlab/CHAMOIS
    $ cd CHAMOIS
    $ singularity build --fakeroot chamois.sif pkg/singularity/chamois.def

Then run the image and analyze the files in the local directory:

.. code:: console

    $ singularity run chamois.sif predict -i tests/data/BGC0000703.4.gbk -o tests/data/BGC0000703.4.hdf5
