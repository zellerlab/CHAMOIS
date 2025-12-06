CLI Reference
=============

.. currentmodule:: chamois

This section documents the command line interface (CLI) of CHAMOIS.

.. note::

    When installing CHAMOIS with ``pip``, an executable named ``chamois`` will
    be created in ``/usr/bin`` or ``$HOME/.local/bin``. If the install path is
    not in your ``$PATH``, you can also invoke the command line as a
    Python module with:

    .. code:: console

        $ python -m chamois.cli ...


Inference
---------

These sub-commands allow using the inference mechanism of CHAMOIS.
``chamois predict`` is the basic entry-point to the CHAMOIS prediction
method, computing ChemOnt class predictions from one or more BGC in
GenBank format. ``chamois render`` can be used to render class predictions
stored in HDF5 as a tree in the terminal.

.. toctree::
    :maxdepth: 1
    :caption: Inference

    predict
    render


Training
--------

These sub-commands can be used for training and evaluating CHAMOIS.
``chamois annotate`` can be used to annotate the features of a set of
BGCs in a GenBank file, and create a ``features.hdf5`` file suitable
for training. ``chamois train`` can to train CHAMOIS from a dataset.
``chamois validate`` can check the performance of a trained model on
a given dataset. ``chamois cv`` can run (stratified grouped) cross-validation
to evaluate generalization.

.. note::

    Some of these commands have additional dependencies, such as
    ``chamois train`` which requires `scikit-learn <https://scikit-learn.org>`_
    to train the logistic regression classifiers. To install the
    required dependencies, make sure to install the ``train`` extra
    (e.g. ``pip install "chamois-tool[train]"``).

.. toctree::
    :maxdepth: 1
    :caption: Training

    annotate
    train
    validate
    cv
    cvi


Compound Search
---------------

These sub-commands can be used to explore CHAMOIS predictions.
``chamois gompare`` can be used to search which BGC of a dataset is most likely
to produce a query metabolite. ``chamois search`` can be used to search which
metabolite of a compound catalog (such as `NPAtlas <https://npatlas.org>`_)
is most similar to the predictions.


.. toctree::
    :maxdepth: 1
    :caption: Compound Search

    compare
    search


Model Interpretation
--------------------

These sub-commands help interpreting the CHAMOIS model.

.. toctree::
    :maxdepth: 1
    :caption: Model Interpretation

    explain

