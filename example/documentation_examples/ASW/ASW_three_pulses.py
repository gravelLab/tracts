"""
ASW inference - Three pulses model
==================================

This example implements inference for the ASW population under a three pulses model of admixture, using the tracts package.
Inference is performed using autosomal and X chromosome data, allowing for the specification of sex-biased admixture. 

To implement this example, we use the following driver file:

.. code-block:: yaml

   samples:
     directory: ../ASW_TrioPhased
     individual_names: [
      "NA19625","NA19700","NA19701","NA19703","NA19704","NA19707","NA19711","NA19712","NA19713","NA19818","NA19819",
      "NA19834","NA19835","NA19900","NA19901","NA19904","NA19908","NA19909","NA19913","NA19914","NA19916","NA19917",
      "NA19920","NA19921","NA19922","NA19923","NA19982","NA19984","NA20126","NA20127","NA20274","NA20276","NA20278",
      "NA20281","NA20282","NA20287","NA20289","NA20291","NA20294","NA20296","NA20298","NA20299","NA20314","NA20317",
      "NA20318","NA20320","NA20321","NA20332","NA20334","NA20339","NA20340","NA20342","NA20346","NA20348","NA20351",
      "NA20355","NA20356","NA20357","NA20359","NA20362","NA20412"] 
   male_names : [
      "NA19700","NA19703","NA19711","NA19818","NA19834","NA19900","NA19904","NA19908","NA19916","NA19920",
      "NA19922","NA19982","NA19984","NA20126","NA20278","NA20281","NA20291","NA20298","NA20318","NA20340",
      "NA20342","NA20346","NA20348","NA20351","NA20356","NA20362"] #see Readme_dataprocessing.md for how this was generated
   filename_format: "{name}_{label}_final.bed"
   labels: [A, B] #If this field is omitted, 'A' and 'B' will be used by default
   chromosomes: 1-22 #The chromosomes to use for analysis. Can be specified as a list or a range
   allosomes: [X]

   models:
    model_filename: ../models/ppx_xxp_pxx.yaml
    ad_model_autosomes: M
    ad_model_allosomes: DC

   start_params: 
     t1: 9:12
     REUR: 0.8
     RAFR: 0.9
     REUR2: 0.2
     t2: 5:8
     t3: 1:4
     REUR_sex_bias: -0.2:0.2
     REUR2_sex_bias: -0.2:0.2
     RAFR_sex_bias: -0.2:0.2

   optim:
     repetitions: 5
     seed: 100
     maximum_iterations: 1000
     npts: 50
     exclude_tracts_below_cm: 2
     unknown_labels_for_smoothing: ["UNK", "centromere","miscall"] # segments with these labels will be smoother over, that is, will be filled with neighbouring ancestries up to their midpoints.
     n_reoptimizations: 5
     rerun_optimization_on_boundaries: True
   
   output:
    output_directory: ./output_three_pulses/
    output_filename_format: "ASW_test_output_{label}"
    log_filename: 'ASW_three_pulses.log'
    verbose_log: 1
    verbose_screen: 30
    log_scale: True
   
Complete results from this analysis are saved in the output directory specified in the driver file. Below, we display the optimal parameters estimated from this analysis,
as well as the plots illustrating the inferred tract length distributions, compared to the observed histograms, for every source population and chromosome type (autosomes and X chromosome).

Optimal parameters
------------------

.. csv-table:: Estimated optimal parameters
   :file: output_three_pulses/ASW_test_output_optimal_parameters.txt
   :header-rows: 1
   :delim: tab
   
Optimal migration matrices
--------------------------

.. image:: output_three_pulses/ASW_test_output_migration_matrices.png
   :width: 500px

Tract length histograms
-----------------------

Autosomal admixture
^^^^^^^^^^^^^^^^^^^

.. image:: output_three_pulses/ASW_test_output_autosomes_all_populations.png
   :width: 700px
   :alt: African ancestry tract histogram

X chromosome admixture in females
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: output_three_pulses/ASW_test_output_female_allosomes_all_populations.png
   :width: 700px
   :alt: European ancestry tract histogram

X chromosome admixture in males
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: output_three_pulses/ASW_test_output_male_allosomes_all_populations.png
   :width: 700px
   :alt: Native American ancestry tract histogram


"""

# sphinx_gallery_start_ignore
sphinx_gallery_thumbnail_path = 'auto_examples/ASW/output_three_pulses/ASW_test_output_autosomes_all_populations.png'
# sphinx_gallery_end_ignore

import sys
from pathlib import Path
from tracts.driver import run_tracts

# Read files automatically for online documentation
sys.path.append('.')
script_dir = Path.cwd()
driver_filename = script_dir / "ASW_three_pulses.yaml"

run_tracts(
    driver_filename=str(driver_filename),
    script_dir=str(script_dir),
)

# sphinx_gallery_start_ignore
from tracts.doc_utils import prepare_example_outputs_for_docs
prepare_example_outputs_for_docs("output_three_pulses")
# sphinx_gallery_end_ignore

