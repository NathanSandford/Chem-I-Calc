.. _refstars:

Reference Stars
===============

The :py:class:`ReferenceSpectra <chemicalc.reference_spectra.ReferenceSpectra>` class represents the spectra of the star
being observed. Here is an overview of the class, as well as some of its important attributes and methods:

Overview
--------

.. autoclass:: chemicalc.reference_spectra.ReferenceSpectra
    :no-members:

Attributes
----------

Spectra
+++++++
``ReferenceSpectra.spectra``
is a Python dictionary, each entry of which consists of a grid of spectra generated around the stellar labels of the reference star
Each spectral grid is a :math:`n_{\text{spectra}} \times n_{\text{pixel}}` ``numpy.ndarray``.
The order of spectra in the pre-computed spectral grids is described :ref:`below <pre-computed_refstars_org>`.

Upon initialization, the dictionary consists of only one entry with the key "init" that corresponds to the initial grid
of high-resolution (:math:`R \sim 300000`) normalized spectra.
It is read in from ``reference_spectra_300000.h5`` by default or from ``ref_spec_file`` if provided.

With each call to :py:meth:`ReferenceSpectra.convolve() <chemicalc.reference_spectra.ReferenceSpectra.convolve>`,
the dictionary is populated with a new grid of spectra that have been convolved and subsampled to match
the resolution and wavelength array of the passed :py:class:`InstConfig <chemicalc.instruments.InstConfif>` object.
The key for the new spectral grid is the name of the instrument configuration, ``InstConfig.name``.

Wavelength
++++++++++
``ReferenceSpectra.wavelength`` is a
Python dictionary containing the wavelength array (as a size `\times n_{\text{pixel}}` ``numpy.ndarray``)
corresponding to spectral grids.

Upon initialization, the dictionary consists of only one entry with the key "init" that corresponds to the wavelength
array for the initial high-resolution spectral grid.
It is read in from ``reference_spectra_300000.h5`` by default or from ``ref_spec_file`` if provided.

With each call to :py:meth:`ReferenceSpectra.convolve() <chemicalc.reference_spectra.ReferenceSpectra.convolve>`,
the dictionary is populated with a new entry corresponding to the wavelength array of the passed
:py:class:`InstConfig <chemicalc.instruments.InstConfif>` object.
The key for the new wavelength array is the name of the instrument configuration, ``InstConfig.name``.

Labels
++++++
``ReferenceSpectra.labels`` is a
``pandas.DataFrame`` of shape :math:`n_{\text{labels}} \times n_{\text{spectra}}`,
containing the stellar labels [#f1]_ corresponding to each spectrum in the grid.
The DataFrame's row names (indices) consist of the stellar labels included.
The DataFrame's column names consist of (arbitrary) ID's for the individual spectra in the grid.
It is read in from ``reference_labels.h5`` by default or from ``ref_label_file`` if provided.

Gradients
+++++++++
``ReferenceSpectra.gradients`` is initially an empty Python dictionary.

With each call to :py:meth:`ReferenceSpectra.calc_gradient() <chemicalc.reference_spectra.ReferenceSpectra.calc_gradient>`,
partial derivatives of the spectrum with respect to each stellar label is calculated for one of the sets of spectral grids.
They are added as a new entry in the form of an :math:`n_{\text{label}} \times n_{\text{pixel}}` ``pandas.DataFrame``.
The key for the new spectral gradient DataFrame is the name of the instrument configuration, ``InstConfig.name``.
The DataFrame's row names (indices) consist of the stellar labels included.
The DataFrame's column names consist of the wavelength of each pixel.

Methods
-------

Convolve Spectral Grids
+++++++++++++++++++++++

.. automethod:: chemicalc.reference_spectra.ReferenceSpectra.convolve

Calculate Spectral Gradients
++++++++++++++++++++++++++++
.. automethod:: chemicalc.reference_spectra.ReferenceSpectra.calc_gradient

Zero Out Gradients
++++++++++++++++++
.. automethod:: chemicalc.reference_spectra.ReferenceSpectra.zero_gradients

Mask Out Gradients
++++++++++++++++++
.. automethod:: chemicalc.reference_spectra.ReferenceSpectra.mask_wavelength

Get Names of Spectral Grids
+++++++++++++++++++++++++++
.. automethod:: chemicalc.reference_spectra.ReferenceSpectra.get_names

Duplicate Spectral Grid
+++++++++++++++++++++++
.. automethod:: chemicalc.reference_spectra.ReferenceSpectra.duplicate

Rest ReferenceSpectra Object
++++++++++++++++++++++++++++
.. automethod:: chemicalc.reference_spectra.ReferenceSpectra.reset

----

.. _pre-computed_refstars:

Pre-Computed Reference Stars
----------------------------

The spectral grids for the following Reference Stars have been pre-computed using :code:`atlas12` 1D-LTE model atmospheres
and the :code:`synthe` radiative transfer code.

.. note:: See Section 3.2 of Sandford et al. (In Press) for a detailed description of the spectral generation

+---------------+----------------------+-----------------+------------------+-------------------------+-----------------+-------+--------------------------+
| Name          | :math:`M_{V}` (Vega) | :math:`T_{eff}` | :math:`\\log(g)` | :math:`v_{turb}` (km/s) | :math:`\log(Z)` | [X/H] | :math:`\alpha` included? |
+===============+======================+=================+==================+=========================+=================+=======+==========================+
| RGB_m0.5      | -0.5                 | 4200            | 1.5              | 2.0                     | -0.5            | Solar | No                       |
+---------------+----------------------+-----------------+------------------+-------------------------+-----------------+-------+--------------------------+
| RGB_m1.0      | -0.5                 | 4530            | 1.7              | 1.9                     | -1.0            | Solar | Yes                      |
+---------------+----------------------+-----------------+------------------+-------------------------+-----------------+-------+--------------------------+
| RGB_m1.5      | -0.5                 | 4750            | 1.8              | 1.9                     | -1.5            | Solar | Yes                      |
+---------------+----------------------+-----------------+------------------+-------------------------+-----------------+-------+--------------------------+
| RGB_m2.0      | -0.5                 | 4920            | 1.9              | 1.9                     | -2.0            | Solar | Yes                      |
+---------------+----------------------+-----------------+------------------+-------------------------+-----------------+-------+--------------------------+
| RGB_m2.5      | -0.5                 | 5050            | 1.9              | 1.9                     | -2.5            | Solar | Yes                      |
+---------------+----------------------+-----------------+------------------+-------------------------+-----------------+-------+--------------------------+
| MSTO_m1.5     | +3.5                 | 6650            | 4.1              | 1.2                     | -1.5            | Solar | Yes                      |
+---------------+----------------------+-----------------+------------------+-------------------------+-----------------+-------+--------------------------+
| TRGB_m1.5     | -2.5                 | 4070            | 0.5              | 2.3                     | -1.5            | Solar | Yes                      |
+---------------+----------------------+-----------------+------------------+-------------------------+-----------------+-------+--------------------------+
| KGiant        | ---                  | 4800            | 2.5              | 1.7                     |  0.0            | Solar | No                       |
+---------------+----------------------+-----------------+------------------+-------------------------+-----------------+-------+--------------------------+
| Ph_k0i_m0.0   | ---                  | 4500            | 1.0              | 2.2                     |  0.0            | Solar | Yes                      |
+---------------+----------------------+-----------------+------------------+-------------------------+-----------------+-------+--------------------------+
| Ph_k0i_m1.0   | ---                  | 4500            | 1.0              | 2.2                     | -1.0            | Solar | Yes                      |
+---------------+----------------------+-----------------+------------------+-------------------------+-----------------+-------+--------------------------+
| Ph_k5iii_m0.0 | ---                  | 4800            | 1.5              | 2.0                     |  0.0            | Solar | Yes                      |
+---------------+----------------------+-----------------+------------------+-------------------------+-----------------+-------+--------------------------+
| Ph_k5iii_m1.0 | ---                  | 4800            | 1.5              | 2.0                     | -1.0            | Solar | Yes                      |
+---------------+----------------------+-----------------+------------------+-------------------------+-----------------+-------+--------------------------+

.. _pre-computed_refstars_org:

Pre-Computed Spectral Grid Organization
+++++++++++++++++++++++++++++++++++++++
* The first spectrum in each grid is generated with the stellar labels in the above table.
* The next 2 spectra are generated with :math:`T_{eff}` offset by :math:`\pm 50` K.
* The next 2 spectra are generated with :math:`\\log(g)` offset by :math:`\pm 0.1` dex.
* The next 2 spectra are generated with :math:`v_{turb}` offset by :math:`\pm 0.1` km/s.
* The next 2 spectra are generated with [:math:`\alpha`/H] offset by :math:`\pm 0.05` dex (if "Yes" in the ":math:`\alpha` included?" column above).
* The next 97x2 spectra are generated with [X/H] offset by :math:`\pm 0.05` dex, where X refers to elements with atomic numbers between 3 (Li) and 99 (Es).

In total, these spectral grids will consist of 203 (201) spectra if the offsets to [:math:`\alpha`/H] are (not) included.

----

Custom Reference Stars
----------------------
Some Chem-I-Calc users may wish to use their own spectral grids to calculate the CRLBs for additional reference stars or with
alternative spectral models (e.g., with 3D, non-LTE atmospheres).
This can be done by calling :py:class:`ReferenceSpectra <chemicalc.reference_spectra.ReferenceSpectra>` with the keyword
arguments: :code:`ref_spec_file` and :code:`ref_label_file`. At present, these files must be in the same format as the default
Chem-I-Calc files (``reference_spectra_300000.h5`` and ``reference_labels.h5``).

.. warning:: Use of custom spectral grids has  not been thoroughly tested. We welcome feedback and code contributions to improve this functionality (see :ref:`Contributing to Chem-I-Calc <contributing>.`)

----

.. rubric:: Footnotes
.. [#f1] "Stellar Labels" broadly encompasses both atmospheric parameters (e.g., effective temperature, surface gravity, and microturbulent velocity) and elemental abundances.