================================================
Chem-I-Calc: The Chemical Information Calculator
================================================

**Chem-I-Calc** is a python package for evaluating the chemical information content of resolved star spectroscopy.
It takes advantage of the Fisher information matrix and the Cramér-Rao inequality to quickly calculate the Cramér-Rao lower bounds (CRLBs),
which give the best theoretically achievable precision from a set of observations.

In this documentation, we hope to provide an overview of the underlying methodology and intended application of Chem-I-Calc.
However, a more detailed and complete discussion of the scientific motivation, statistical foundations,
and practical limitations of our methods can be found in `Sandford et al. (In Press) <https://arxiv.org/abs/2006.08640>`_.

.. toctree::
   :maxdepth: 1
   :caption: Getting Started:

   Scientific/Statistical Background <intro/background>
   Installing Chem-I-Calc <intro/installing>
   A Basic Example <intro/basic_example>
   Reporting Issues and Contributing <intro/contributing>

.. toctree::
   :maxdepth: 1
   :caption: User Guide:

   Reference Stars <user/refstars>
   Instrument Configurations <user/instruments>
   Setting the Signal/Noise <user/snr>
   Calculating CRLBs <user/crlb>
   Plotting CRLBs with Chem-I-Calc <user/plotting>
   Tutorials <user/tutorials>
   Sandford et al. (In Press) Notebooks <user/paper>
   API Reference <api/chemicalc>

Authors
-------

- Nathan Sandford (UC Berkeley, `nathan_sandford@berkeley.edu <nathan_sandford@berkeley.edu>`_)

Collaborators
+++++++++++++

- Dan Weisz
- Yuan-Sen Ting


License & Attribution
---------------------

Copyright 2019-2020 Nathan Sandford and contributors.

Chemi-I-Calc is being developed by `Nathan Sandford <http://w.astro.berkeley.edu/~nathan_sandford/>`_ in a
`public GitHub repository <https://github.com/NathanSandford/Chem-I-Calc>`_.
The source code is made available under the terms of the MIT license.

If you make use of this code, please cite `Sandford et al. (In Press) <https://arxiv.org/abs/2006.08640>`_
