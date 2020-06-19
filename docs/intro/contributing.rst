.. _contributing:

Contributing to Chem-I-Calc
===========================

| Like what Chem-I-Calc is doing but unsatisfied with some portion of it?
| Good news! There are many ways you can get involved!

Types of Contributions
----------------------

.. _report_bugs:

Reporting Bugs
++++++++++++++

If you have found a bug in Chem-I-Calc or find a feature that does not work as expected,
please create a new issue on the `Chem-I-Calc Issue Tracker <https://github.com/NathanSandford/Chem-I-Calc/issues>`_.
To help us identify and address the issue, please include a minimal,
reproducible example and a full Python stack trace (i.e., the error message).

.. _fix_bugs:

Fixing Bugs
++++++++++++++

Whether its associated with an existing issue on the
`Chem-I-Calc Issue Tracker <https://github.com/NathanSandford/Chem-I-Calc/issues>`_
or a bug you found yourself, we welcome your bugfixes!

.. _implement_features:

Implementing New Features / Improving Performance
+++++++++++++++++++++++++++++++++++++++++++++++++

If there is a feature that you think would make Chem-I-Calc more useful for the astronomical community,
please let us know (via the `Chem-I-Calc Issue Tracker <https://github.com/NathanSandford/Chem-I-Calc/issues>`_)!
Granted, we are people-power limited and welcome any and all efforts to implement these new features.

Suggestions for (or better yet, implementation of) improvments in code performance are also appreciated.

.. _write_documentation:

Writing Documentation
+++++++++++++++++++++

Chem-I-Calc's documentation is far from complete (and likely full of errata).
Contributions to documentation here on our readthedocs page, in the Chem-I-Calc docstrings,
or in the form of Jupyter Notebook tutorials are all welcome.
Even just correcting typos or pointing out places where the documentation is missing or unclear
(via the `Chem-I-Calc Issue Tracker <https://github.com/NathanSandford/Chem-I-Calc/issues>`_) is incredibly helpful.

.. _write_tests:

Writing Tests
+++++++++++++

To help us address bugs and implement new features without breaking existing features,
we would like to have a suite of tests to run on the Chem-I-Calc package whenever changes are made.
At present, these tests are woefully incomplete. All contributions to the test suite are very much appreciated.

Making a Contribution w/ GitHub
-------------------------------

Here is a brief primer on how to contribute to Chem-I-Calc via a GitHub pull request.

0. Make a GitHub account if you have not already.

1. Fork the `Chem-I-Calc repository on GitHub <https://github.com/NathanSandford/Chem-I-Calc>`_.

2. Install Chem-I-Calc and its dependencies following the :ref:`from GitHub <git-install>` instructions, except replacing

.. code-block:: bash

    git clone https://github.com/nathansandford/Chem-I-Calc.git

with

.. code-block:: bash

    git clone https://github.com/<YOUR-GITHUB-USERNAME>/Chem-I-Calc.git

3. Create a branch for local development stemming from the develop branch:

.. code-block:: bash
    cd Chem-I-Calc
    git checkout develop
    git pull
    git checkout -b name-of-bugfix-or-feature

4. Make your contribution! If you are changing or adding to the functionality of the code, please make the relevant changes or additions to the docstrings, the readthedocs documentation, and the test suite.

5. Run tests in your cloned repository and make sure nothing breaks as a result of your changes.

.. code-block:: bash

    pip install pytest  # if pytest not already installed
    pytest -v

6. Commit your changes and push your branch to GitHub.

.. code-block:: bash

    git add <CHANGED-FILES>
    git commit -m "Description of changes"
    git push origin name-of-bugfix-or-feature

7. Submit a pull request through the `Chem-I-Calc repository on GitHub <https://github.com/NathanSandford/Chem-I-Calc>`_.

If all looks good, your pull request will be accepted.
Otherwise, if changes are requested, repeat steps 4-6 until the outstanding issues have been addressed
at which point your pull request will be accepted. Thanks for your contribution!

----

.. rubric:: Acknowledgements

This page was adapted from the `page outlining contributions to specutils <https://specutils.readthedocs.io/en/stable/contributing.html>`_.
