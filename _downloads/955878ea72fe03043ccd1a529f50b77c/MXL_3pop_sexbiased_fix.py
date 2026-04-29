"""
MXL inference - continuous pulse model
======================================

This example implements inference for the MXL population under a continuous pulse model of admixture, using the tracts package.
Inference is performed using autosomal and X chromosome data, allowing for the specification of sex-biased admixture. 

To implement this example, we use the following driver file:

.. code-block:: yaml

   samples:
     directory: ./TrioPhased/
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
     
   output_filename_format: "MXL_test_output_{label}"
   log_filename: 'ASW_continuous_pulse.log'
   output_directory: ./output_continuous_pulse/
   verbose_log: 1
   verbose_screen: 30
   log_scale : True
     
   model_filename: ../models/ccc.yaml
   start_params: 
     t1: 13.5
     REUR: 0.2
     RAFR: 0.02
     RNAT: 0.2
     t2: 6.8
  
     REUR_sex_bias: -0.99 # more males
     RNAT_sex_bias: 0.99 # more females
     RAFR_sex_bias: -0.1
   repetitions: 3
   seed: 100
   maximum_iterations: 1000
   unknown_labels_for_smoothing: ["UNK", "centromere","miscall"] # segments with these labels will be smoother over, that is, will be filled with neighbouring ancestries up to their midpoints.  
   exclude_tracts_below_cm: 2
   npts : 50
   #fix_parameters_from_ancestry_proportions: ['REUR', 'RAFR','REUR_sex_bias', 'RAFR_sex_bias']

   ad_model_autosomes : M
   ad_model_allosomes : DC


Complete results from this analysis are saved in the output directory specified in the driver file. Below, we display the optimal parameters estimated from this analysis,
as well as the plots illustrating the inferred tract length distributions, compared to the observed histograms, for every source population and chromosome type (autosomes and X chromosome).

Optimal parameters
------------------

.. csv-table:: Estimated optimal parameters
   :file: output_continuous_pulse/MXL_test_output_optimal_parameters.txt
   :header-rows: 1
   :delim: tab

Tract length histograms
-----------------------

Autosomal admixture
^^^^^^^^^^^^^^^^^^^

.. image:: output_continuous_pulse/MXL_test_output_autosomes_all_populations.png
   :width: 700px
   :alt: African ancestry tract histogram

X chromosome admixture in females
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: output_continuous_pulse/MXL_test_output_female_allosomes_all_populations.png
   :width: 700px
   :alt: European ancestry tract histogram

X chromosome admixture in males
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: output_continuous_pulse/MXL_test_output_male_allosomes_all_populations.png
   :width: 700px
   :alt: Native American ancestry tract histogram

"""

import sys
from pathlib import Path

sys.path.append('.')

from tracts.driver import run_tracts

script_path = Path(sys.argv[0]).resolve()
   
driver_filename = "MXL_continuous.yaml"

run_tracts(driver_filename = driver_filename, script_dir = script_path.parent)


# Don't run the code below: for documentation purposes only.
from tracts.doc_utils import prepare_example_outputs_for_docs
prepare_example_outputs_for_docs("output_continuous_pulse")
