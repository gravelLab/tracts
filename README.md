Tracts
======

Tracts is a set of classes and definitions used to model migration histories
based on ancestry tracts in admixed individuals. Time-dependent gene-flow from
multiple populations can be modeled.

Changes in Tracts 2
========

- Tracts is now a python package rather than a single .py file. Follow the installation instructions below.
- Tracts now uses the matrix exponential form of the Phase-Type Distribution to calculate the tractlength distribution. This should not have resulted to changes to the interface. If it has, please report it in the issues.
- Tracts no longer requires writing your own driver script. Instead, details about the simulation are read from a YAML file (examples below).
- Demographic models also do not have to be handcoded anymore. They are now specified by a Demes-like YAML file (examples below).
- Minor Patches: Fixed an issue with fixing multiple parameters from ancestry.

Examples
========

Examples contains sample hapmap data and scripts to analyze them, including two
different gene flow models.  It also contains a 3-population model for 1000
genomes puerto Rican data

Installation
============

To install:
1. Clone this repository
2. In your local copy, open a terminal.
3. Run pip install .

You can now import tracts as a python package.

Tracts is currently not distributed on PyPi or Conda. 


Setting up a demographic model
==============================

Tracts attempts to predict a population's migration history from the distribution of ancestry tracts.
The space of all migration matrices is very large: if we have `p` migrant populations over g generations, there can be `n*g` different migration rates.
In tracts, demographic models are used to describe the migration matrix as a reduced number of migration events with flexible parameters.
For example, a model can contain only a founding pulse of migration from two ancestral populations. 
In that case, the only parameters are the time of the pulse, and the migration rate from each population.
The yaml file for such a model would look like:

    demes:
      - name: EUR
      - name: AFR
      - name: X
      ancestors: [EUR, AFR]
      proportions: [R, 1-R]
      start_time: tx

Here, tracts deduces the sample population to be `X`. The parameter for the time of founding is named `tx`.
Since migration at the founding pulse must add to 1, there is only one other parameter for model: `R`.
The founding proportion from EUR is equal to `R`, and the founding proportion from AFR is `1-R`.

To add more migration pulses to the model, add a `pulses` field to the YAML file:

    pulses:
      - sources: [EUR]
        dest: X
        proportions: [P]
        time: t2

This represents a single pulse of migration from `EUR` to `X`. It occurs at time `t2` with proportion `P`.

The full model would then look like:

    demes:
      - name: EUR
      - name: AFR
      - name: X
      ancestors: [EUR, AFR]
      proportions: [R, 1-R]
      start_time: tx
    pulses:
      - sources: [EUR]
        dest: X
        proportions: [P]
        time: t2

The `pulses` field can also contain more than one pulse:

    pulses:
      - sources: [EUR]
        dest: X
        proportions: [P]
        time: t2
      - sources: [EUR]
        dest: X
        proportions: [P]
        time: t3

Here, the proportion of both pulse migrations is the same, but they occur at different times. Tracts allows for the linking of parameters in this way.
This model would have 5 parameters: `R`, `tx`, `P`, `t2`, `t3`. If the pulses had different rates, the model would have 6 parameters instead.

Similar to pulses, continuous migrations can be specified in the `migrations` field:

    migrations:
      - source: EUR
        dest: X
        rate: K
        start_time: t1
        end_time: t2

Driver File
===========

Tracts is used by passing a driver yaml file to the method tracts.run_tracts().
The first part of the driver file tells tracts how to load the sample data:

    samples:
      directory: .\G10\
      filename_format: "{name}_{label}.bed"
      individual_names: [
        "NA19700", "NA19701", "NA19704", "NA19703", "NA19819", "NA19818",
        "NA19835", "NA19834", "NA19901", "NA19900", "NA19909", "NA19908",
        "NA19917", "NA19916", "NA19713", "NA19982", "NA20127", "NA20126",
        "NA20357", "NA20356"
      ]
      labels: [A, B]
      chromosomes: 1-22

In this example, the samples are located in the 'G10' directory.
The individual 'NA19700' has sample data in the files 'NA19700_A.bed' and 'NA19700_B.bed'. <br>
The 'chromosomes' field tells tracts to use data from chromosomes 1 to 22. You can also specify a single chromosome or a list of chromosomes.

The details of the model are specified as a different YAML file. The model_filename field is used to tell tracts where to find this model YAML.

    model_filename: pp.yaml

Tracts optimizes the parameters of the model to best match the distribution of ancestry tracts. 
Starting values for the parameters can be specified as numbers or ranges.
Multiple repetitions can be run on the same data, and a seed can be used for repeatability.

    start_params:
      R: 0.1-0.2
      tx: 10-11
      P:  0.03-0.05
      t2: 5.5
    repetitions: 2
    seed: 100

Tracts also allows for the time parameter to be scaled, as some optimizers run better when
all parameters are on the same scale:

    time_scaling_factor: 100

Likewise, tracts below a certain length (in centimorgans) can be excluded from the analysis.

    exclude_tracts_below_cM: 10

Ancestry Fixing
===============



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
