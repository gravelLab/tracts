.. _data-processing:

Data Processing
===============

First steps
-----------

A list of individuals present in the local ancestry calls downloaded from 1KGP was first compiled. From the directory where the ASW archive was extracted, the trio-phased BED files are unpacked and the individual identifiers are listed (one per line):

.. code-block:: bash

   tar -xzf ./ASW/TrioPhased/bed_files/ASW_bed.tar.gz
   ls ./ASW/TrioPhased/*_A_final.bed | xargs -n 1 basename | cut -d _ -f 1 \
   > ./ASW/TrioPhased/individuals.txt

Sex information for individuals was obtained from the file:

::

   integrated_call_samples_v3.20130502.ALL.panel

available at:

https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/.

We then identified individuals present in the local ancestry data but missing from this panel.


Individuals Not Included in the Panel
-------------------------------------

The following command identifies individuals present in the ancestry calls but absent from the panel:

.. code-block:: bash

   awk '
   NR==FNR { seen[$1]=1; next }
   !($1 in seen) { print $1 }
   ' integrated_call_samples_v3.20130502.ALL.panel ./ASW/TrioPhased/individuals.txt

Output:

::

   NA19985
   NA20322
   NA20336
   NA20341
   NA20344


Identification of Related Individuals
-------------------------------------

These individuals correspond to related samples, as listed in:

::

   20140625_related_individuals.txt

available at:

https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/.

The following command confirms this:

.. code-block:: bash

   awk '
   NR==FNR { seen[$1]=1; next }
   ($1 in seen) { print $1 }
   ' 20140625_related_individuals.txt ./ASW/TrioPhased/individuals.txt

Output:

::

   NA19985
   NA20322
   NA20336
   NA20341
   NA20344


Filtering Related Individuals
-----------------------------

These individuals are present in the local ancestry calls but are not independent. They should therefore be removed.

To exclude related individuals:

.. code-block:: bash

   awk '
   NR==FNR { seen[$1]=1; next }
   !($1 in seen) { print $1 }
   ' 20140625_related_individuals.txt ./ASW/TrioPhased/individuals.txt

To save the filtered list:

.. code-block:: bash

   awk '
   NR==FNR { seen[$1]=1; next }
   !($1 in seen) { print $1 }
   ' 20140625_related_individuals.txt ./ASW/TrioPhased/individuals.txt \
   > ./ASW/TrioPhased/individuals_unrelated.txt


Output Formatting
-----------------

To format the list of unrelated individuals as a CSV string (e.g., for inclusion in a driver file):

.. code-block:: bash

   awk '
   NR==FNR { seen[$1]=1; next }
   !($1 in seen) { printf "\"%s\"," , $1 }
   ' 20140625_related_individuals.txt ./ASW/TrioPhased/individuals.txt


Male Individuals
^^^^^^^^^^^^^^^^

To extract the male individuals among the unrelated ones, in the same format, using the sex (``gender`` column) given in the panel file:

.. code-block:: bash

   awk '
   NR==FNR { if ($4 == "male") male[$1]=1; next }
   ($1 in male) { printf "\"%s\"," , $1 }
   ' integrated_call_samples_v3.20130502.ALL.panel ./ASW/TrioPhased/individuals_unrelated.txt

These two lists correspond to the fields ``individual_names`` and ``male_names`` of the driver files used in the ASW examples (61 unrelated individuals, of which 26 are males).
