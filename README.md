Tracts
======

Tracts is a set of classes and definitions used to model migration histories
based on ancestry tracts in admixed individuals. Time-dependent gene-flow from
multiple populations can be modeled.

Examples
========

Examples contains sample hapmap data and scripts to analyze them, including two
different gene flow models.  It also contains a 3-population model for 1000
genomes puerto Rican data

Installation
============

Installation support for tracts is rudimentary. Apologies. 

Copy all files and folders locally.  

"tracts.py" is a python module. The main branch should support python 2.7 and 3. This is a relatively new update, so 
please let me know if you encounter issues running or installing with either version of python 2.7. Moving ahead, however, I expect that most
development will be on python 3. 

To load the package, you currently have to tell python where to look for the package by doing something like

import sys
tractspath = path_to_tracts_directory  # the path to tracts if not in your default pythonpath
sys.path.append(tractspath)
import tracts. 

I have added tracts_conda_env.yml,  an anaconda environment file with the relevant dependencies. If you are using anaconda, you can load this environment using 

conda env create -f tracts_conda_env.yml
conda activate py2_tracts


Input
=====

Tracts input is a set bed-style file describing the local ancestry of segments
along the genome.  The file has 2 extra columns for the cM positions of the
segments. There are two input files per individuals (for each haploid genome
copy).

    chrom		begin		end			assignment	cmBegin	cmEnd
    chr13		0			18110261	UNKNOWN	0.0			0.19
    chr13		18110261	28539742	YRI			0.19		22.193
    chr13		28539742	28540421	UNKNOWN	22.193		22.193
    chr13		28540421	91255067	CEU		22.193		84.7013

Driver File
===========

Tracts is used by passing a driver yaml file to the method tracts.run_tracts().
An example driver file is:

    samples:
      directory: .\G10\
      individual_names: [
        "NA19700", "NA19701", "NA19704", "NA19703", "NA19819", "NA19818",
        "NA19835", "NA19834", "NA19901", "NA19900", "NA19909", "NA19908",
        "NA19917", "NA19916", "NA19713", "NA19982", "NA20127", "NA20126",
        "NA20357", "NA20356"
      ]
      filename_format: "{name}_{label}.bed"
      labels: [A, B] #If this field is omitted, 'A' and 'B' will be used by default
      chromosomes: 1-22 #The chromosomes to use for analysis. Can be specified as a list or a range
    model_filename: pp.yaml
    start_params: 
    - [0.173632,  0.0683211] #Run tracts with these start params
    - values: [0.173632,  6.83211] #Run tracts with these start params twice
    perturbation: [0.6, 2] #And multiply them by a random number in this range at each repetition
    repetitions: 2
    exclude_tracts_below_cM: 2
    time_scaling_factor: 100

The samples field indicates information about the location of the sample data.
The individual_names tells tracts which individuals use for analysis.
The filename_format allows tracts to find the filenames corresponding to each individual.
The model_filename corresponds to a yaml file of a demographic model, as detailed below.
Tracts also allows for the time parameter to be scaled, as some optimizers run better when
all parameters are on the same scale.

Output
======

The 3-population exemple files produce 5 output files, e.g.

     boot0_-252.11_bins	boot0_-252.11_liks	boot0_-252.11_ord	boot0_-252.11_pred
     boot0_-252.11_dat	boot0_-252.11_mig	boot0_-252.11_pars


boot0 means that this is bootstrap iteration 0, which in the convention used
here means the fit with the real data (in the two-population example, there is
no bootstrap, so the output is named "out" and "out2" instead) -252.11 is the
likelihood of the best-fit model

* _bins: the bins used in the discretization
* _dat: the observed counts in each bins
* _pred: the predicted counts in each bin, according to the model
* _mig: the inferred migration matrix, with the most recent generation at the
    top, and one column per migrant population. Entry i,j in the matrix represent 
    the proportion of individuals in the admixed population who originate
    from the source population j at generation i in the past. 
* _pars: the optimal parameters. I.e., if these models are passed to the 
    admixture model, it will return the inferred migration matrix.
* _liks: the likelihoods in the model parameter space in the output format of
    scipy.optimizes' "brute" function: the first number is the best likelihood,
    the top matrices define the grid of parameters usedin the search, and the
    last matrix defines the likelihood at all grid points. see
    http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brute.html

Setting up a demographic model
==============================

The Demes specification is a project with the goal of standardizing 
demographic models in computational biology. Populations, pulses of Tracts uses a modified version
of the Demes format to specify demographic models with flexible parameters,
which can then be optimized over. The details of the model are specified as a yaml file.

    model_name: One_Pulse
    description:
        A demes model with flexible parameters. Represents a population X founded R generations ago by EUR and AFR.
    time_units: generations
    demes:
      - name: EUR
      - name: AFR
      - name: X
      ancestors: [EUR, AFR]
      proportions: [R, 1-R] #This line represents the proportions from each ancestor when the population is founded. Should add up to 1.
      start_time: tx #This line represents the founding time of the population.
      
In Demes, the proportions and start_time would be given as numbers. 
In tracts, they are given as parameter names, which are then tuned
by the optimizer.

For now, the model only accepts one population having ancestors and pulses.
This is taken to be the sample population (in the example, X).
The other populations in the model should have labels corresponding to the
sources of admixture in the data.

The population is founded when two populations meet; at the first generation,
we consider all individuals in the population as “migrants”, so the sum of
migration frequencies at the first generation must be one.

Importantly, the optimizers in tracts assume that all parameters are
continuous, but the underlying markov model uses discrete generations.
The ParametrizedDemography class allows for pulses of migration at non-integer times
by splitting the pulse across the previous and subsequent generation.
The pulse is split such that the effective time of arrival corresponds to the time parameter,
while the expected number of migrants corresponds to the rate parameter.
This ensures that the output matrix is continuous in the parameters,
which prevents the optimizer from getting stuck.

Contact
=======

See the example files for example usage. If something isn't clear, please let
me know by filing an "new issue", or emailing me.

FAQ
===

> The distribution of tract lengths decreases as a function of tract length,
> but increases at the very last bin. This was not seen in the original paper.
> What is going on?

In tracts, the last bin represents the number of chromosomes with no ancestry
switches. It does not correspond to a specific length value, and for this
reason was not plotted in the tracts paper.


> When I have a single pulse of admixture, I would expect an exponential
> distribution of tract length, but the distribution of tract lengths shows
> steps in the expected length. Why is that?

"Tracts" takes into account the finite length of chromosomes. Since ancestry
tracts cannot extend beyond chromosomes, we expect this departure from an
exponential distribution

> I have migrants from the last generation. "tracts" tells me that migrants in
> the last two generations are not allowed. Why is that?

Haploid genomes from the last two generations have no ancestry switches and
should be easy to identify in well-phased data--they should be removed from the
sample before running tracts. If this is impossible (e.g., because of
inaccurate phasing across chromosomes), tracts will likely attempt to assign
last-generation migrants to two generations ago. This should be observable by
an excess of very long tracts in the data compared to the model.

> Individuals in my population vary considerably in their ancestry proportion.
> Is that a problem?

It is not a problem as long as the population was close to random mating. If
admixture is recent, random mating is not inconsistent with ancestry variance.
If admixture is ancient, however, variation in ancestry proportion may indicate
population structure, and the random mating assumption may fail.

> I ran the optimization steps many times, and found different optimal
> likelihoods. Why is that?

Optimizing functions in many dimensions is hard, and sometimes optimizers get
stuck in local maxima. If you haven tried already, you can attempt to fix the
ancestry proportions a priori (see the `_fix` examples in the documentation).
In most cases, the optimization will converge to the global maximum a
substantial proportion of the time: running the optimization a few times from
random starting positions and comparing the best values may help control for
this.

If you fail to revisit the same minimum after running say, 10 optimizations,
then something else might be going on. If the model is not continuous as a
function of a parameter, it could make the optimization much harder. Defining a
continuous model would help, or you could try the brute-force optimization
method if the number of parameters is small.
