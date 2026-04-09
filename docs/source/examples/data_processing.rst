.. _data-processing:

Data Processing
===============

First steps
-----------

A list of individuals present in the local ancestry calls downloaded from 1KGP was first compiled.

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

To format the list of individuals as a CSV string (e.g., for inclusion in a driver file):

.. code-block:: bash

   awk '
   NR==FNR { seen[$1]=1; next }
   ($1 in seen) { printf "\"%s\"," , $1 }
   ' 20140625_related_individuals.txt ./ASW/TrioPhased/individuals.txt


Male Individuals
^^^^^^^^^^^^^^^^

To extract male individuals in the same format:

.. code-block:: bash

   awk '
   NR==FNR { seen[$1]=1; next }
   ($1 in seen) { printf "\"%s\"," , $1 }
   ' ./males_ASW.panel ./ASW/TrioPhased/individuals_unrelated.txt
