Examples
========

This section contains example analyses of local ancestry inference performed on admixed populations from the 1000 Genomes Project. Below, we provide a detailed description of data, preprocessing steps, links to the examples, and the considered demographic models.

Data
----

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Dataset description
      :link: data-description
      :link-type: ref

      Description of the local ancestry datasets used in the examples (1000 Genomes Project).

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

   .. grid-item-card:: MXL population
      :link: ../auto_examples/MXL/index
      :link-type: doc

      Example analyses for the MXL population.
      

Demographic models
------------------

In this section, we consider four demographic models of varying complexity. The corresponding YAML files are specified below.

One pulse model
^^^^^^^^^^^^^^^

The following model is written in the file ``ppp.yaml``.

.. code-block:: yaml

   model_name: One_Pulse
   description: Represents a population X founded a generations ago by EUR, NAT, and AFR.
   time_units: generations
   demes:
     - name: EUR
     - name: AFR
     - name: NAT
     - name: X
   ancestors: [EUR, NAT, AFR]
   proportions: [REUR,RNAT,1-REUR-RNAT]
   start_time: t

Two pulses model
^^^^^^^^^^^^^^^^

The following model is written in the file ``ppp_pxx.yaml``.

.. code-block:: yaml

   model_name: Two pulses
   description: A population X founded a generations ago by EUR, AFR, and NAT, then subsequent EUR migration.
   time_units: generations
   demes:
     - name: EUR
     - name: AFR
     - name: NAT
     - name: X
   ancestors: [EUR, NAT, AFR]
   proportions: [REUR,RNAT,1-REUR-RNAT]
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
   description: A demes model with flexible parameters. Represents a population X founded a generations ago by EUR and NAT, then subsequent AFT migration, 
   then subsequent EUR migration.
   time_units: generations
   demes:
     - name: EUR
     - name: NAT
     - name: X
   ancestors: [EUR, NAT]
   proportions: [REUR,1-REUR]
   start_time: t1

   pulses:
     - sources: [AFR]
       dest: X
       proportions: [RAFR]
       time: t2
     - sources: [EUR]
       dest: X
       proportions: [REUR2]
       time: t3
   
 
Continuous pulse model
^^^^^^^^^^^^^^^^^^^^^^

The following model is written in the file ``ccc.yaml``.

.. code-block:: yaml

   model_name: One_Pulse
   description: Represents a population X founded with a continuous event.
   time_units: generations
   demes:
     - name: EUR
     - name: NAT
     - name: AFR
     - name: X
   ancestors: [EUR, NAT, AFR]
   proportions: [REUR, RNAT,RAFR ]
   start_time: t1
   end_time: t2  
   
   
.. toctree::
   :hidden:

   data_description
   data_processing
   ../auto_examples/ASW/index
   ../auto_examples/MXL/index
   




