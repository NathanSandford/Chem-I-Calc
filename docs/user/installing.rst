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

    cd <path_to_installation>
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


Optional Dependencies
---------------------

While some spectrographs have online exposure time calculators (ETCs) that can be queeried by Chem-I-Calc,
others have ETCs in the form of GitHub code repositories. To simplify the installation of Chem-I-Calc,
we do not include these repositories as dependencies. However, to ease the integration of those ETC's with Chem-I-Calc,
we have written several convenience functions into chemicalc.s2n. To use these functions, you will need to install the
relevant repositories following the instructions below.

.. warning:: Many of these repositories are undergoing constant revision so we recommend making sure that you have
the most recent version installed before making important calculations. If a ETC code-base changes sufficiently that it
breaks the Chem-I-Calc interface with them, please raise an issue on the
`Chem-I-Calc GitHub <https://github.com/NathanSandford/Chem-I-Calc>`_.

FOBOS ETC (enyo)
++++++++++++++++

To use chemicalc.s2n.calculate_fobos_snr() the fobos-enyo package must be installed as follows:

.. code-block:: bash

    cd <path_to_installation>
    git clone https://github.com/Keck-FOBOS/enyo
    cd enyo
    python setup.py develop

To update:

.. code-block:: bash

    cd <path_to_installation>/enyo
    git pull
    python setup.py develop

PFS ETC
+++++++

No convenience functions have been writted for the PFS ETC, but it was used for Sandford et al. (in prep).
To install:

.. code-block:: bash

    cd <path_to_installation>
    git clone https://github.com/Subaru-PFS/spt_ExposureTimeCalculator
    cd enyo
    python setup.py develop

To update:

.. code-block:: bash

    cd <path_to_installation>/spt_ExposureTimeCalculator
    git pull
    python setup.py develop

(Blue)MUSE ETC
++++++++++++++

The function chemicalc.s2n.calculate_muse_snr() is adapted from the calculation presented
`here <https://git-cral.univ-lyon1.fr/johan.richard/BlueMUSE-ETC/-/blob/master/BlueMUSE-ETC.py>`_ by Johan Richard.
While the function is self-contained in Chem-I-Calc, it does require several small external files,
which can be downloaded from the BlueMUSE-ETC with chemicalc.file_mgmt.download_bluemuse_files() as follows:

.. code-block:: python

    from chemicalc.file_mgmt import download_bluemuse_files
    download_bluemuse_files()