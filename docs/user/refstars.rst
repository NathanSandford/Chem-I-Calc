.. _refstars:

Reference Stars
===============

Upon initialization, :py:class:`ReferenceSpectra <chemicalc.reference_spectra.ReferenceSpectra>` class loads in a grid of
high-resolution (:math:`R \sim 300000`) *ab initio* normalized stellar spectra generated around the reference star's stellar labels [#f1]_.

This grid of spectra can be subsequently convolved and subsampled to match the resolution and wavelength array of an
instrument represented by an :py:class:`InstConfig <chemicalc.instruments.InstConfif>` object
(see :ref:`Defining Instrument Setups <instruments>`).

For each convolution, the new lower resolution spectral grid can be used to calculate the partial derivatives of the
reference star's spectrum with respect to each of its stellar labels.

In turn, these gradients (along with the S/N of the observation) are used to forecast the possible abundance precision
these instruments may achieve.


.. _pre-computed_refstars:

Pre-Computed Reference Stars
----------------------------

The spectral grids for the following Reference Stars have been pre-computed using :code:`atlas12` 1D-LTE model atmospheres
and the :code:`synthe` radiative transfer code.

.. note:: See Section 3.2 of Sandford et al. (2020) for a detailed description of the spectral generation

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

* The first spectrum in each grid is generated with the stellar labels in the above table.
* The next 2 spectra are generated with :math:`T_{eff}` offset by :math:`\pm 50` K.
* The next 2 spectra are generated with :math:`\\log(g)` offset by :math:`\pm 0.1` dex.
* The next 2 spectra are generated with :math:`v_{turb}` offset by :math:`\pm 0.1` km/s.
* The next 2 spectra are generated with [:math:`\alpha`/H] offset by :math:`\pm 0.05` dex (if "Yes" in the ":math:`\alpha` included?" column above).
* The next 97x2 spectra are generated with [X/H] offset by :math:`\pm 0.05` dex, where X refers to elements with atomic numbers between 3 (Li) and 99 (Es).

In total, these spectral grids will consist of 203 (201) spectra if the offsets to [:math:`\alpha`/H] are (not) included.
This can be confirmed by looking at the
:any:`labels <chemicalc.reference_spectra.ReferenceSpectra.labels>` attribute of your
:py:class:`ReferenceSpectra <chemicalc.reference_spectra.ReferenceSpectra>`.


.. rubric:: Footnotes
.. [#f1] "Stellar Labels" broadly encompasses both atmospheric parameters (e.g., effective temperature, surface gravity, and microturbulent velocity) and elemental abundances.