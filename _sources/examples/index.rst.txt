Examples
========

This section shows how to use ``tracts`` on a real dataset. It analyzes a single population from the 1000 Genomes Project, the African Americans from Southwest US (ASW), using trio-phased local ancestry calls. The examples walk through the main steps of an analysis: the input data, the driver files, the demographic models, running the inference and reading the outputs. Below, we provide a description of the data, the preprocessing steps, links to the examples, and the demographic models considered.

.. admonition:: Reproducing the 1000 Genomes analyses of the paper
   :class: tip

   The data used in these examples are not the same as the data analyzed in the ``tracts`` paper. Here, the ASW local ancestry calls are trio-phased, whereas the paper uses the population-phased local ancestry calls from the same 1000 Genomes release. The full analysis of the six admixed populations of the 1000 Genomes Project (ACB, ASW, CLM, MXL, PEL and PUR) presented in the paper is available in the folder `example/3pops_sexbiased <https://github.com/gravelLab/tracts/tree/devel/example/3pops_sexbiased>`_ of the GitHub repository. Its README explains how to download the data, run the inference and produce the tables and figures.

Data
----

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Dataset description
      :link: data-description
      :link-type: ref

      Description of the local ancestry datasets used in this section (1000 Genomes Project).

   .. grid-item-card:: Data processing
      :link: data-processing
      :link-type: ref

      Preprocessing and formatting steps applied to the data.

Example analyses
----------------

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: ASW population
      :link: ../auto_examples/ASW/index
      :link-type: doc

      Example analyses for the ASW population.

Demographic models
------------------

In this section, we consider three demographic models of varying complexity. The corresponding YAML files are specified below.

One pulse model
^^^^^^^^^^^^^^^

The following model is written in the file ``ppp.yaml``.

.. code-block:: yaml

   model_name: One pulse
   description: Represents a population X founded t generations ago by EUR, NAT, and AFR.
   time_units: generations
   demes:
      - name: EUR
      - name: AFR
      - name: NAT
      - name: X
   ancestors: [EUR, NAT, AFR]
   proportions: [REUR, RNAT, RAFR]
   start_time: t

Two pulses model
^^^^^^^^^^^^^^^^

The following model is written in the file ``ppp_pxx.yaml``.

.. code-block:: yaml

   model_name: Two pulses
   description: A population X founded t1 generations ago by EUR, AFR, and NAT, then subsequent EUR migration.
   time_units: generations
   demes:
      - name: EUR
      - name: AFR
      - name: NAT
      - name: X
   ancestors: [EUR, NAT, AFR]
   proportions: [REUR, RNAT, RAFR]
   start_time: t1

   pulses:
      - sources: [EUR]
        dest: X
        proportions: [REUR2]
        time: t2


Three pulses model
^^^^^^^^^^^^^^^^^^

The following model is written in the file ``ppp_xxp_pxx.yaml``.

.. code-block:: yaml

   model_name: Three pulses
   description: A demes model with flexible parameters. Represents a population X founded t1 generations ago by EUR and NAT, then subsequent AFR migration,
   then subsequent EUR migration.
   time_units: generations
   demes:
      - name: EUR
      - name: NAT
      - name: targetpop
   ancestors: [EUR, NAT]
   proportions: [REUR, RNAT]
   start_time: t1

   pulses:
      - sources: [AFR]
        dest: targetpop
        proportions: [RAFR]
        time: t2

      - sources: [EUR]
        dest: targetpop
        proportions: [REUR2]
        time: t3


.. toctree::
   :hidden:

   data_description
   data_processing
   ../auto_examples/ASW/index





