.. _api:

API Reference
=================


Reference Spectra
-----------------------------------

.. automodule:: chemicalc.reference_spectra
   :members:
   :undoc-members:
   :show-inheritance:

Instruments
----------------------------

.. automodule:: chemicalc.instruments
   :members:
   :undoc-members:
   :show-inheritance:

Signal/Noise Calculations
----------------------------

.. automodule:: chemicalc.s2n
   :members:
   :undoc-members:
   :show-inheritance:

Cramer-Rao Calculations
----------------------------

Functions for calculating and handling Cramer-Rao Lower Bounds.

.. automodule:: chemicalc.crlb
   :members:
   :undoc-members:
   :show-inheritance:

Plotting
---------------------

Pre-designed plotting functions. They may not satisfy everyone's use case,
but should at least provide a place to start.

.. automodule:: chemicalc.plot
   :members:
   :undoc-members:
   :show-inheritance:

External File Management
---------------------------

Functions for downloading necessary external data files.
In the future, this will also provide functions to check the format of custom user-provided data files.

.. automodule:: chemicalc.file_mgmt
   :members:
   :undoc-members:
   :show-inheritance:

Utility Functions
----------------------

Most of these functions are wrapped in chemicalc.reference_spectra.ReferenceSpectra and
chemicalc.instruments.InstConfig and should only be used for hacking/testing purposes.
Exceptions are kpc_to_mu and mu_to_kpc which provide conversions between distances in kpc and distance moduli.

.. automodule:: chemicalc.utils
   :members:
   :undoc-members:
   :show-inheritance:
