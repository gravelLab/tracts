.. tracts documentation master file, created by
   sphinx-quickstart on Wed Nov 19 12:38:28 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

tracts
======

A Python library for demographic inference from ancestry tracts.

.. grid:: 1 1 2 3
   :gutter: 3

   .. grid-item::

      .. card:: Using tracts
         :link: user_guide/index
         :link-type: doc
         :class-card: sd-h-100
         		
         Main guidelines to use ``tracts``.

   .. grid-item::

      .. card:: Examples
         :link: examples/index
         :link-type: doc
         :class-card: sd-h-100
         
         End-to-end examples with figures and code.
         
   .. grid-item::

      .. card:: Phase-type models
         :link: tutorials/index
         :link-type: doc
         :class-card: sd-h-100
         
         A tutorial to compute Phase-Type distributions
         from a migration matrix.

   .. grid-item::

      .. card:: API documentation
         :link: api/index
         :link-type: doc
         :class-card: sd-h-100
         
         Detailed reference for modules, functions, and classes.


----

``tracts`` models migration histories in admixed populations using ancestry tracts.
It supports multiple source populations, time-dependent migration, sex-biased admixture and inference for both autosomes and the X chromosome.

Installation
------------

.. code-block:: bash

   git clone https://github.com/gravelLab/tracts.git
   cd tracts
   pip install .

You can now import tracts as a python package.

.. note::


   ``tracts`` is currently not distributed on PyPI or Conda.

----

The papers
----------

A detailed introduction to the models and methods presented in ``tracts 2.0`` is available in `this paper (in preparation) </path/to/paper>`_. ``tracts 1.0`` is based on `this paper <https://pubmed.ncbi.nlm.nih.gov/22491189/>`_.

----

.. admonition:: Updates in ``tracts 2.0``
   :class: note

   **Software updates**

   - ``tracts`` is now a Python package rather than a single ``.py`` file.
   - ``tracts`` no longer requires writing a custom driver script.
   - Demographic models are now defined via YAML.

   **Methodological updates**

   - Phase-Type theory for tract distributions.
   - X chromosome support.
   - Sex-biased migration.
   - Improved admixture models.


Contact
-------
For any inquires, please `file an issue <https://github.com/gravelLab/tracts/issues>`_ or `contact us <mailto:javier.gonzalez-delgado@ensai.fr>`_.


.. toctree::
   :hidden:
   :maxdepth: 2

   user_guide/index
   examples/index
   tutorials/index
   api/index
   
