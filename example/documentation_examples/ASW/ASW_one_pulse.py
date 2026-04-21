"""
ASW inference - One pulse model
===============================

This example implements inference for the ASW population under a one pulse model of admixture, using the tracts package.
Inference is performed using autosomal and X chromosome data, allowing for the specification of sex-biased admixture. 

To implement this example, we use the following driver file:

.. code-block:: yaml
   
   samples:
    directory: ./TrioPhased/
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
   
   output_filename_format: "ASW_test_output_{label}"
   log_filename: 'ASW_one_pulse.log'
   output_directory: ./output_one_pulse/
   verbose_log: 1
   verbose_screen: 30
   log_scale : True  
   
   start_params: 
    t: 5:8

   repetitions: 3
   maximum_iterations: 1000
   seed: 100
   unknown_labels_for_smoothing: ["UNK", "centromere","miscall"] # segments with these labels will be smoother over, that is, will be filled with neighbouring ancestries up to their midpoints.  
   exclude_tracts_below_cm: 2
   npts : 50
   #fix_parameters_from_ancestry_proportions: ['REUR', 'RNAT', 'REUR_sex_bias', 'RNAT_sex_bias']
   output_directory: ./output_one_pulse/
   ad_model_autosomes : M
   ad_model_allosomes: DC



Complete results from this analysis are saved in the output directory specified in the driver file. Below, we display the optimal parameters estimated from this analysis,
as well as the plots illustrating the inferred tract length distributions, compared to the observed histograms, for every source population and chromosome type (autosomes and X chromosome).

Optimal parameters
------------------

.. csv-table:: Estimated optimal parameters
   :file: output_one_pulse/ASW_test_output_optimal_parameters.txt
   :header-rows: 1
   :delim: tab

Tract length histograms
-----------------------

Autosomal admixture
^^^^^^^^^^^^^^^^^^^

.. image:: output_one_pulse/ASW_test_output_autosomes_all_populations.png
   :width: 700px
   :alt: African ancestry tract histogram

X chromosome admixture in females
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: output_one_pulse/ASW_test_output_female_allosomes_all_populations.png
   :width: 700px
   :alt: European ancestry tract histogram

X chromosome admixture in males
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: output_one_pulse/ASW_test_output_male_allosomes_all_populations.png
   :width: 700px
   :alt: Native American ancestry tract histogram

"""

import sys
from pathlib import Path

sys.path.append('.')

from tracts.driver import run_tracts

script_path = Path(sys.argv[0]).resolve()

driver_filename = "ASW_one_pulse.yaml"

run_tracts(driver_filename = driver_filename, script_dir = script_path)



# Don't run the code below: for documentation purposes only.
from tracts.doc_utils import prepare_example_outputs_for_docs
prepare_example_outputs_for_docs("output_one_pulse")

