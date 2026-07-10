User guide
==========

``tracts`` is a Python library for inferring migration histories from the distribution of ancestry tracts in admixed individuals. It supports time-dependent gene flow from multiple populations,
including sex-biased migration and recombination, enabling modeling for autosomes and the X chromosome.

A typical workflow consists of four steps:

.. _user-guide:

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: 1. Define a demographic model
      :link: demographic-models	
      :link-type: ref

      Specify migration events in YAML.

   .. grid-item-card:: 2. Prepare data
      :link: input-data
      :link-type: ref

      Provide ancestry tracts.

   .. grid-item-card:: 3. Configure driver
      :link: driver-file
      :link-type: ref

      Set admixture model and optimization parameters.

   .. grid-item-card:: 4. Run ``tracts``
      :link: run-tracts
      :link-type: ref

      Run tracts and get the output.


.. _demographic-models:

1. Define a demographic model
-----------------------------


.. admonition:: Key idea
   :class: tip

   Instead of working with migration matrices, ``tracts`` uses **parametric demographic models** to reduce complexity.
   Demographic models describe migration using a small number of parameters. 


The demographic model has to be specified in a YAML file. It may contain:

A founding pulse
^^^^^^^^^^^^^^^^

.. code-block:: yaml

   demes:
     - name: EUR
     - name: AFR
     - name: X
       ancestors: [EUR, AFR]
       proportions: [R, 1-R]
       start_time: tx

The previous block specifies a founding pulse at generation ``tx`` with a proportion ``R`` of ``EUR`` individuals and a proportion ``1-R`` of ``AFR`` individuals. The code ``X`` denotes the admixed population receving migration.


Multiple migration pulses
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   pulses:
     - sources: [EUR]
       dest: X
       proportions: [P]
       time: t2
     - sources: [EUR]
       dest: X
       proportions: [P]
       time: t3

The previous block specifies one pulse from ``EUR`` population at generation ``t2`` in the past, replacing a proportion ``P`` of the admixed individuals in ``X``. Another pulse from the same population and with the same proportion is considered at generation ``t3``.

A continuous migration
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   migrations:
     - source: EUR
       dest: X
       rate: K
       start_time: t1
       end_time: t2

Continuous migration between generations ``t1`` and ``t2`` can be specified as below.

.. admonition:: Sex-bias specification
   :class: tip

   If allosomes are present in the sample, each migration proportion will be automatically associated with
   a corresponding sex-bias parameter, measuring the deviation from an unbiased migration setting. For each
   population :math:`i` and each generation :math:`t`, we consider the proportion :math:`R_{ti}^{f}` (resp. :math:`R_{ti}^{m}`)
   of female (resp. male) migrants at generation :math:`t` from source population :math:`i`. The average migration
   rate will then be :math:`R_{ti} = (R_{ti}^{f} + R_{ti}^{m})/2`, which corresponds to the balanced setting where
   :math:`R_{ti}^{f} = R_{ti}^{m}`. The departure from such scenario is measured by the sex-bias parameter, defined as:

   .. math::

      R_{ti}^{\text{sex-bias}} = \frac{R_{ti}^{f} - R_{ti}^{m}}{2\min(R_{ti},\, 1 - R_{ti})}.



   Consequently, the sex-specific rates are recovered from the mean rate and the sex-bias parameter as:

   .. math::

      R_{ti}^{f} &= R_{ti} + R_{ti}^{\text{sex-bias}} \cdot \min(R_{ti},\, 1 - R_{ti}),\\
      R_{ti}^{m} &= R_{ti} - R_{ti}^{\text{sex-bias}} \cdot \min(R_{ti},\, 1 - R_{ti}).



   Consequently, :math:`R_{ti}^{\text{sex-bias}} = 1` corresponds to exclusively female migration,
   :math:`R_{ti}^{\text{sex-bias}} = -1` to exclusively male migration, and :math:`R_{ti}^{\text{sex-bias}} = 0` to 
   unbiased migration.

   For all generations except the founder generation, the initial value of :math:`R_{ti}^{\text{sex-bias}}`
   must be specified by the user when configuring the driver file.
   In the founder generation, all individuals are migrants, leading to a dependency among the migration rates:

   .. math::

      \sum_i R_{Ti} = \sum_i R_{Ti}^{f} = \sum_i R_{Ti}^{m} = 1.

   This entails a relationship among sex bias parameters:

   .. math::

      \sum_{i} R_{Ti}^{\text{sex-bias}} \cdot \min(R_{Ti},\, 1 - R_{Ti}) = 0.

   If there are :math:`P` source populations, the user only specifies :math:`P-1` rates and sex biases,
   and the remaining parameters are inferred from these dependencies.


.. _input-data:

2. Prepare data
---------------

``tracts`` takes as input a **folder** containing a set of BED-style files describing the local ancestry of genomic segments. Each file includes two additional columns specifying the genetic (cM) positions of the segments.

.. code-block:: text

   chrom		begin		end		assignment	cmBegin	     cmEnd
   chr13		0	        18110261	UNKNOWN	        0.0	     0.19
   chr13		18110261	28539742	YRI		0.19	     22.193
   chr13		28539742	28540421	UNKNOWN	        22.193       22.193
   chr13		28540421	91255067	CEU		22.193	     84.7013

.. important::

   Each individual must have **two files (for each haploid genome copy)**, distinguished with user-specified labels e.g. ``indiv1_A.bed`` and ``indiv1_B.bed``.
   
   Consequently, ``tracts`` must be provided with a folder containing `2n` ``.bed`` files, for a sample of `n` individuals.

.. _driver-file:


3. Configure driver
-------------------

Finally, the user must provide a driver YAML file controlling the inference. It is composed of several groups of parameters:

Sample configuration
^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   samples:
     directory: ./G10/
     filename_format: "{name}_{label}.bed"
     individual_names: ["NA19700", "NA19701"]
     male_names : ["NA19649","NA19652"]
     labels: [A, B]
     chromosomes: 1-22
     allosomes: [X]

- ``directory``: the path to the folder where the ``.bed`` :ref:`data files <input-data>` are located.
- ``filename_format``: The file name format for the ``.bed`` :ref:`data files <input-data>`. It must indicate where the individual name ``{name}`` and the haploid copy label ``{label}`` are located in the file name.
- ``individual_names``: A list containing the names of the individuals in the sample.
- ``male_names``: A list containing the names of the male individuals in the sample. Needed for inference on the X chromosome.
- ``labels``: The labels chosen to distinguish the haploid copies.
- ``chromosomes``: The autosomes present in data, if any.
- ``allosomes``: The allosomes present in data, if any.

Models
^^^^^^

.. code-block:: yaml

   models:
     model_filename: ../models/ppp.yaml
     ad_model_autosomes: DC
     ad_model_allosomes: H-DC
     rho_f: 1
     rho_m: 1
     TP: 2

- ``model_filename``: The path to the :ref:`YAML file <demographic-models>` specifying the demographic model.
- ``ad_model_autosomes``: The admixture model used to perform inference on autosomes. Must be either ``M`` (Monoecious), ``DC`` (Dioecious-Coarse), ``DF`` (Dioecious-Fine), ``H-DC`` (The hybrid-pedigree refinement of the Dioecious-Coarse model) or ``H-DF`` (The hybrid-pedigree refinement of the Dioecious-Fine model).
- ``ad_model_allosomes``: The admixture model used to perform inference on allosomes. Must be either ``DC`` (Dioecious-Coarse), ``DF`` (Dioecious-Fine), ``H-DC`` (The hybrid-pedigree refinement of the Dioecious-Coarse model) or ``H-DF`` (The hybrid-pedigree refinement of the Dioecious-Fine model).
- ``rho_f``: The female-specific recombination rate. Default is 1.
- ``rho_m``: The male-specific recombination rate. Default is 1.
- ``TP``: The number of pedigree generations under the hybrid-pedigree refinements of the Dioecious models. Default is 2. Ignored if not applicable.

Starting parameters
^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   start_params:
     R: 0.1:0.2
     t: 10:11
     R_sex_bias: 0
     t2: 5.5

- ``start_params``: Initial values for the parameters defined in the :ref:`demographic model <demographic-models>`. The user can set a single value or an interval ``min:max``, from which an initial value is randomly selected.

Optimization
^^^^^^^^^^^^

.. code-block:: yaml

   optim:
     seed: 100
     repetitions: 3
     maximum_iterations: 1000
     exclude_tracts_below_cm: 2
     npts: 50
     unknown_labels_for_smoothing: ["UNK", "centromere", "miscall"]
     fix_parameters_from_ancestry_proportions: ['R', 'R_sex_bias']
     two_steps_optimization: True
     use_autosomes_for_sex_bias: False
     N_cores: 5

- ``seed``: The random seed.
- ``repetitions``: Number of independent optimization runs performed from different initial values, randomly chosen within the bounds set by the user. Since the optimizer may converge to different local optima, the algorithm repeats the optimization ``repetitions`` times and automatically retains the run with the highest likelihood.
- ``maximum_iterations``: The maximum number of iterations during likelihood optimization.
- ``exclude_tracts_below_cm``: The minimum tract length (in cM) required for a tract to be included in the optimization.
- ``npts``: The number of bins controlling the resolution of the tract length histogram.
- ``unknown_labels_for_smoothing``: Segments with these labels will be smoothed over, that is, will be filled with neighbouring ancestries up to their midpoints.
- ``fix_parameters_from_ancestry_proportions``: These parameters are analytically computed from the ancestry proportions, and the optimization is restricted to the remaining parameters.
- ``two_steps_optimization``: Whether to perform a two-step optimization, where non-sex-bias parameters are optimized first using autosomes, and then sex-bias parameters are optimized with the non-sex-bias parameters fixed. Defaults to ``True``.
- ``use_autosomes_for_sex_bias``: Whether to use both autosomal and allosomal data to optimize sex-bias parameters. Defaults to ``False``, which means that only allosomes are used to optimize sex-bias parameters.
- ``N_cores``: The number of CPU cores to use for parallel processing, when the hybrid-pedigree refinements of the DF or DC models are used to model autosomal or allosomal admixture. Ignored if the hybrid-pedigree refinements are not used. Default is 1.

.. admonition:: Using ``fix_parameters_from_ancestry_proportions``
   :class: tip

   This option fixes a specified subset of parameters to values computed from the observed ancestry proportions in the sample. These parameters are then excluded from the optimization, reducing the dimension of the parameter space and improving convergence speed. However, it also constrains the optimization problem, which may make it more difficult for the optimizer to reach a good optimum; in practice, this often results in a lower likelihood compared to leaving all parameters free. When using this option, we recommended to set ``repetitions > 1``.


Output
^^^^^^

.. code-block:: yaml

   output:
     output_directory: ./output_files/
     output_filename_format: "filename_{label}"
     log_filename: 'my_example.log'
     verbose_log: 1
     verbose_screen: 1
     log_scale: True
     plot_migration_matrices: True

- ``output_directory``: Path to the directory where output files are stored. The directory is created automatically if it does not exist.
- ``output_filename_format``: The file name format for the output files.
- ``log_filename``: The name of the log file where execution details are recorded. If not specified, a default filename (``tracts.log``) is used.
- ``verbose_log``: Controls the level of detail reported in the log file during execution. If greater than zero, logs optimization status every ``verbose`` steps.
- ``verbose_screen``: Controls the level of detail printed on screen during execution. If greater than zero, prints optimization status every ``verbose`` steps.
- ``log_scale``: Whether the tract length distributions are depicted in log-scaled counts. Default is ``True``.
- ``plot_migration_matrices``: Whether to include plots of the inferred migration matrices in the output. Default is ``True``.

.. _run-tracts:


4. Run ``tracts``
-----------------

Once the :ref:`driver file<driver-file>` is ready, the inference can be run using the ``run_tracts`` function:

.. code-block:: python

   from tracts.driver import run_tracts

   run_tracts(driver_filename = "driverfile.yaml", script_dir ="/path/to/folder/")

- ``driver_filename``: The name of the :ref:`driver file <driver-file>`.
- ``script_dir``: The path to the folder where ``driver_filename`` is located.

The software displays the initial parameters to be optimized, along with the ancestry proportions estimated from the sample. It then performs a two-stage optimization:

- First, the parameters unrelated to sex bias are optimized using only autosomal tracts. In this stage, the ``ad_model_autosomes`` admixture model specified in the :ref:`driver file<driver-file>` is considered.
- Next, these non–sex-bias parameters are fixed, and only the sex-bias parameters are optimized using (autosomal and) allosomal tracts. In this stage, the ``ad_model_allosomes`` admixture model specified in the :ref:`driver file<driver-file>` is considered.


Outputs
^^^^^^^

``tracts`` saves results in the ``output_directory`` specified in the :ref:`driver file<driver-file>`. For autosomes, allosomes in males and allosomes in females, these include:

- ``_tract_length_autosome_bins``: the bins used in the discretization for the autosomal distribution.
- ``_tract_length_allosome_bins``: if allosomes are present in the sample, the bins used in the discretization for the allosomal distributions.
- ``_autosome_sample_tract_distribution``: the observed counts in each bin for the autosomal distribution.
- ``_female_allosome_sample_tract_distribution``: if allosomes are present in the sample, the observed counts in each bin for the allosomal distribution in females.
- ``_male_allosome_sample_tract_distribution``: if allosomes are present in the sample, the observed counts in each bin for the allosomal distribution in males.
- ``_autosome_predicted_tract_distribution``: the predicted counts in each bin for the autosomal distribution, according to the predicted model.
- ``_female_allosome_predicted_tract_distribution``: if allosomes are present in the sample, the predicted counts in each bin for the allosomal distribution in females, according to the predicted model.
- ``_male_allosome_predicted_tract_distribution``: if allosomes are present in the sample, the predicted counts in each bin for the allosomal distribution in males, according to the predicted model.
- ``_female_migration_matrix``: the inferred female-specific migration matrix, with the most recent generation at the top, and one column per migrant population. Entry `(i,j)` in the matrix represents the proportion of individuals in the admixed population who originate from the source population `j` at generation `i` in the past.
- ``_male_migration_matrix``: the inferred male-specific migration matrix, with the most recent generation at the top, and one column per migrant population. Entry `(i,j)` in the matrix represents the proportion of individuals in the admixed population who originate from the source population `j` at generation `i` in the past.
- ``_optimal_parameters.txt``: the optimal parameters for the considered :ref:`demographic model<demographic-models>`, together with the inferred likelihood.
- ``_ancestry_per_individual``: a tab-separated file listing the ancestry proportions per individual for each source population.
- ``_ancestry_proportions.txt``: a table of the observed and predicted mean ancestry proportions for autosomes and, if present, allosomes.
- ``_admixture_plot.pdf``: a stacked bar chart of ancestry proportions per individual, sorted by the most common ancestry (ADMIXTURE-style).
- ``_migration_matrices.pdf``, ``_migration_matrices.png``: plots of the inferred mean migration matrix and sex-bias values per generation. Only produced if ``plot_migration_matrices: True`` in the driver file.
- ``_autosomes_all_populations.pdf``, ``_autosomes_all_populations.png``: a plot comparing the sample and the predicted tract length distribution for all source populations, for autosomes.
- ``_female_allosomes_all_populations.pdf``, ``_female_allosomes_all_populations.png``: if allosomes are present in the sample, a plot comparing the sample and the predicted tract length distribution for all source populations, for allosomes in females.
- ``_male_allosomes_all_populations.pdf``, ``_male_allosomes_all_populations.png``: if allosomes are present in the sample, a plot comparing the sample and the predicted tract length distribution for all source populations, for allosomes in males.



FAQ
---

.. dropdown:: The inferred distribution of tract lengths decreases as a function of tract length, but increases at the very last bin. This was not seen in the original paper.

   The last bin corresponds to chromosomes with no ancestry switches, i.e., tracts that span the entire chromosome.

.. dropdown:: When I have a single pulse of admixture, I would expect an exponential distribution of tract length, which is not observed. Why is that?

   Since ancestry tracts cannot extend beyond chromosomes, ``tracts`` transforms the Phase-Type distribution to consider the observed tracts `on a finite interval`. This yields a departure from an exponential distribution in the one-pulse case. See the paper for details on this transformation.

.. dropdown:: Individuals in my population vary considerably in their ancestry proportion. Is that a problem?

   This is not a problem as long as the population is close to random mating. If admixture is recent, random mating is not inconsistent with ancestry variance. If admixture is ancient, however, variation in ancestry proportion may indicate population structure, and the random mating assumption may fail.


