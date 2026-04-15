Phase-type models
=================

In this section, we present a tutorial for computing and illustrating Phase-Type distributions under the different admixture models implemented in ``tracts``. In ``tracts 2.0``, several new models are introduced with increasing levels of complexity. These models arise as successive simplifications of each other, starting from the so-called `pedigree-wide` and `ancestral-pedigree` models presented in the paper, but not implemented due to their high computational complexity.

At each stage of simplification, the state space is reduced by grouping states into super-states, and transitions between these super-states are assumed to be Markovian. In ``tracts 2.0``, we implement the last three stages of this simplification process: the `Dioecious-Fine` (DF) model, the `Dioecious-Coarse` (DC) model, and the `Monoecious` (M) model, the latter of which was already available in ``tracts 1.0``. Note that, unlike the Monoecious model, the Dioecious models introduced in this version allow for modelling admixture on the X chromosome. For further details, we refer the reader to the ``tracts 2.0`` paper.

.. figure:: admixture_models.png
   :width: 600px
   :align: center
  
  
  
  
Once the model is chosen, the corresponding transition probabilities are computed. Then, by setting the states corresponding to the population of interest as absorbing, we obtain a Phase-Type distribution that characterizes tract lengths. In this tutorial, we provide code illustrating how to use the functions in ``tracts 2.0`` to build an admixture model and compute the corresponding Phase-Type densities or histograms. 



  
.. figure:: phasetype.png
   :width: 600px
   :align: center
  
  
  
  
  
.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Phase-type tutorial
      :link: phase_type_models
      :link-type: doc
      :class-card: sd-shadow-sm

      Go to the notebook
      
     
.. toctree::
   :hidden:

   phase_type_models
