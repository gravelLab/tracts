"""
MXL inference - continuous pulse model
======================================

This example implements inference for the MXL population under a continuous pulse model of admixture, using the tracts package.
Inference is performed using autosomal and X chromosome data, allowing for the specification of sex-biased admixture. 

To implement this example, we use the following driver file:

.. code-block:: yaml

   samples:
     directory: ./MXL_TrioPhased/
     individual_names: [
      "NA19648","NA19649","NA19651","NA19652","NA19654","NA19655","NA19657","NA19658","NA19661","NA19663",
      "NA19664","NA19669","NA19670","NA19676","NA19678","NA19679","NA19681","NA19682","NA19684","NA19716",
      "NA19717","NA19719","NA19720","NA19722","NA19723","NA19725","NA19726","NA19728","NA19729","NA19731",
      "NA19732","NA19734","NA19735","NA19740","NA19741","NA19746","NA19747","NA19749","NA19750","NA19752",
      "NA19755","NA19756","NA19758","NA19759","NA19761","NA19762","NA19764","NA19770","NA19771","NA19773",
      "NA19774","NA19776","NA19777","NA19779","NA19780","NA19782","NA19783","NA19785","NA19786","NA19788",
      "NA19789","NA19792","NA19794","NA19795"] 
     male_names : [
      "NA19649","NA19652","NA19655","NA19658","NA19661","NA19664","NA19670","NA19676","NA19679","NA19682",
      "NA19717","NA19720","NA19723","NA19726","NA19729","NA19732","NA19735","NA19741","NA19747","NA19750",
      "NA19756","NA19759","NA19762","NA19771","NA19774","NA19777","NA19780","NA19783","NA19786","NA19789",
      "NA19792","NA19795"] #see Readme_dataprocessing.md for how this was generated
     filename_format: "{name}_{label}_final.bed"
     labels: [A, B] #If this field is omitted, 'A' and 'B' will be used by default
     chromosomes: 1-22 #The chromosomes to use for analysis. Can be specified as a list or a range
     allosomes: [X]

   models:
     model_filename: ../models/ccp.yaml
     ad_model_autosomes: M
     ad_model_allosomes: DC

   start_params:
     t1: 11.70:14.30
     REUR: 0.07:0.09
     RAFR: 0.04:0.05
     RNAT: 0.08:0.09
     t2: 2.29:2.79
     REUR_sex_bias: -0.99:-0.90 # more males
     RNAT_sex_bias: 0.44:0.54 # more females
     RAFR_sex_bias: 0.90:0.99

   bounds:
     t1: 10:15

   optim:
     repetitions: 3
     seed: 100
     maximum_iterations: 1000
     unknown_labels_for_smoothing: ["UNK", "centromere","miscall"] # segments with these labels will be smoother over, that is, will be filled with neighbouring ancestries up to their midpoints.
     exclude_tracts_below_cm: 2
     npts : 50
     n_reoptimizations: 5
     rerun_optimization_on_boundaries: False

   output:
     output_directory: "./output_ccp/"
     output_filename_format: "MXL_output_{label}"
     log_filename: 'MXL_continuous_pulse.log'
     verbose_log: 1
     verbose_screen: 30
     log_scale: True

Complete results from this analysis are saved in the output directory specified in the driver file. Below, we display the optimal parameters estimated from this analysis,
as well as the plots illustrating the inferred tract length distributions, compared to the observed histograms, for every source population and chromosome type (autosomes and X chromosome).

Optimal parameters
------------------

.. csv-table:: Estimated optimal parameters
   :file: output_ccp/MXL_output_optimal_parameters.txt
   :header-rows: 1
   :delim: tab
   
Optimal migration matrices
--------------------------

.. image:: output_ccp/MXL_output_migration_matrices.png
   :width: 500px

Tract length histograms
-----------------------

Autosomal admixture
^^^^^^^^^^^^^^^^^^^

.. image:: output_ccp/MXL_output_autosomes_all_populations.png
   :width: 700px
   :alt: Autosomal tract length distribution

X chromosome admixture
^^^^^^^^^^^^^^^^^^^^^^

.. image:: output_ccp/MXL_output_allosomes_all_populations.png
   :width: 700px
   :alt: X chromosome tract length distribution

"""
# sphinx_gallery_start_ignore
sphinx_gallery_thumbnail_path = 'auto_examples/MXL/output_ccp/MXL_output_autosomes_all_populations.png'
# sphinx_gallery_end_ignore

import sys
from pathlib import Path
from tracts.driver import run_tracts

# Read files automatically for online documentation
sys.path.append('.')
script_dir = Path.cwd()
driver_filename = script_dir / "MXL_continuous.yaml"

run_tracts(
    driver_filename=str(driver_filename),
    script_dir=str(script_dir),
)

# sphinx_gallery_start_ignore
from tracts.doc_utils import prepare_example_outputs_for_docs
prepare_example_outputs_for_docs("output_ccp")
# sphinx_gallery_end_ignore
