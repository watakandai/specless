.. package_name documentation master file, created by
   sphinx-quickstart on Tue Aug  6 16:58:44 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to specless's documentation!
========================================

**specless** (SPECification LEarning and Strategy Synthesis) is a Python library for learningg specification from demonstrations and synthesizing strategies that satisfy the given specification.
It aims to offer a *simple* and *intuitive* API.
To checkout the code, please visit our repo on `GitHub <https://github.com/watakandai/specless/>`_.

.. note::
   This project is under active development.

Installation
----------------
from `PyPI`

.. code-block:: console

   $ pip install specless

from source

.. code-block:: console

   $ pip install git@github.com:watakandai/specless.git

or clone from github and install the library

.. code-block:: console

   $ git clone https://github.com/watakandai/specless.git
   $ cd specless
   $ pip install .


Development
----------------
If you want to contribute, set up your development environment as follows:

- Install `Poetry <https://python-poetry.org>_`

- Clone the repository:

.. code-block:: console

   $ git clone https://github.com/watakandai/specless.git && cd specless

- Install the dependencies:

.. code-block:: console

   $ poetry shell && poetry install

Tests
----------------

To run tests: `tox`

To run only the code tests: `tox`


Docs
----------------

Locally, run `make html` inside the `docs` directory.

Once you are ready, make a pull request and the documentations are built automatically with GitHub Actions.
See `.github/generate-documentation.yml`.


License
----------------

Apache 2.0 License

Copyright 2023- KandaiWatanabe



.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst
   :recursive:

   specless


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
