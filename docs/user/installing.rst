.. _installing:

Installation
============

Installing Chem-I-Calc
----------------------

Chem-I-Calc is written in pure Python and can be easily installed :ref:`pip-install` (recommended for casual users) or :ref:`git-install` (recommended for advanced users).


.. _pip-install:

Using pip
+++++++++

The easiest way to install Chem-I-Calc is with `pip <https://pip.pypa.io>`_:

.. code-block:: bash

    pip install Chem-I-Calc


.. _git-install:

From Source
+++++++++++

The source code for Chem-I-Calc can be downloaded from GitHub and installed by running

.. code-block:: bash

    git clone https://github.com/nathansandford/Chem-I-Calc.git
    cd Chem-I-Calc
    python setup.py develop


**Testing**

You can test your installation by running `pytest <http://doc.pytest.org/>`_ in the installation directory.

.. code-block:: bash

    pip install pytest  # if pytest not already installed
    pytest -v

Note however, that the unit tests are still under active development. Many tests are presently skipped and several are expected to fail.


Downloading Data
----------------
Chem-I-Calc requires grids of high-resolution spectra to calculate partial derivatives of a star's spectrum  for a wide variety of instrumental configurations. Unfortunately, the file containing our pre-generated grids of spectra, **reference_spectra_300000.h5**, is too large to host on Github or PyPi (~11 Gb) and must be downloaded seperately. You have three options to acquire this file.


.. _runtime-download:

At Runtime
++++++++++

If you run any portion of the code that requires **reference_spectra_300000.h5** and the file cannot be found in ``<path_to_installation>/Chem-I-Calc/chemicalc/data/``, the file will automatically be downloaded to the appropriate directory in chunks of 32768 bytes. For example, the following code will initiate the download.

.. code-block:: python

    import chemicalc.reference_spectra as ref
    RGB = ref.ReferenceSpectra(reference="RGB_m1.5")


.. _download-all-files:

With chemicalc.file_mgmt.download_all_files()
+++++++++++++++++++++++++++++++++++++++++++++

You can also manually call the python function that begins the download process. This can also be used to re-download **reference_spectra_300000.h5** in the case that it has been updated. This will also re-download **reference_labels.h5**, though it should already be kept up to date through standard version control methods (i.e., via GitHub and PyPi).

.. code-block:: python

    from chemicalc.file_mgmt import download_all_files
    download_all_files(overwrite=True)


.. _manual-download:

Manually
++++++++

The file is hosted on Google Drive, so it is also possible to manually download **reference_spectra_300000.h5** using the following link:

- `reference_spectra_300000.h5 <https://drive.google.com/open?id=1I9GzorHm0KfqJ-wvZMVGbQDeyMwEu3n2>`_
- `reference_labels.h5 <https://drive.google.com/open?id=1-qCCjDXp2eNzRGCfIqI_2JZrzi22rFor>`_

| You will need to place this file in the appropriate directory:
| ``<path_to_installation>/Chem-I-Calc/chemicalc/data/``

If you are unsure what the full path to the directory is, you can check with the following code:

.. code-block:: python

    from chemicalc.file_mgmt import data_dir
    print(data_dir)
