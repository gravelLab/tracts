## About $\texttt{tracts}$

$\texttt{tracts}$ is a python library used to model migration histories
based on ancestry tracts in admixed individuals. Time-dependent gene-flow from
multiple populations can be modeled. In $\texttt{tracts 2.0}$, sex-biased migration
and recombination rates are supported, enabling modeling and inference for 
both autosomes and the X chromosome.

---

### Installing $\texttt{tracts}$

To install: 

1. Clone this repository
2. In your local copy, open a terminal
3. Run `pip install .`

You can now import $\texttt{tracts}$ as a python package.

$\texttt{tracts}$ is currently not distributed on PyPi or Conda. 

### Documentation

The documentation of $\texttt{tracts 2.0}$ can be found [here](https://gravellab.github.io/tracts/).

### The papers
A detailed introduction to the models and methods presented in $\texttt{tracts 2.0}$ is available in [this paper (_in preparation_)](). $\texttt{tracts 1.0}$ is based on [this paper](https://pubmed.ncbi.nlm.nih.gov/22491189/). 

### Contact

For any inquires, please file an [issue](https://github.com/gravelLab/tracts/issues) or [contact us](mailto:javier.gonzalez-delgado@ensai.fr).

---

### Updates in $\texttt{tracts 2.0}$

#### Software updates

- $\texttt{tracts}$ is now a python package rather than a single .py file.
- $\texttt{tracts}$ no longer requires writing the user's own driver script. Instead, the inference parameters are read from a YAML file (see examples below).
- Demographic models do not have to be hand-coded anymore. They are now specified by a Demes-like YAML file (see examples below).

#### Methdological updates

- The tract length distribution is computed using Phase-Type theory. 
- Admixture on the X chromosome is allowed.
- Sex-biased migration and recombination rates are allowed.
- Various admixture models are implemented with increasing levels of approximation.
- Migrations in the two last generations are now supported.

For details, see the $\texttt{tracts 2.0}$ [paper (_in preparation_)]().

---

## Running $\texttt{tracts}$

### 1. Setting up a demographic model

The goal of $\texttt{tracts}$ is to predict a population's migration history from the distribution of ancestry tracts.
The space of all migration matrices is very large: if we have $p$ migrant populations over $T$ generations, there can be $pT$ different migration rates. In $\texttt{tracts}$, demographic models are used to describe the migration matrix as a reduced number of migration events with flexible parameters.

For example, a model can contain only a founding **migration pulse** from two ancestral populations. 
In that case, the only parameters are the time of the pulse, and the migration rate from each population.
The **yaml file** for such a **demographic model** would look like:

    demes:
      - name: EUR
      - name: AFR
      - name: X
      ancestors: [EUR, AFR]
      proportions: [R, 1-R]
      start_time: tx

Here, $\texttt{tracts}$ deduces the sample population to be `X`. The parameter describing the time of founding is named `tx`.
Since migration at the founding pulse must sum up to 1, the only remaining parameter is `R`, corresponding to the founding proportion from EUR. The founding proportion from AFR is therefore `1-R`.

To add more migration pulses to the model, the user can add a `pulses` field to the YAML file:

    pulses:
      - sources: [EUR]
        dest: X
        proportions: [P]
        time: t2

This represents a **single migration pulse** from `EUR` to `X`, taking place at time `t2` with proportion `P`.

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

The `pulses` field can also contain **more than one pulse**:

    pulses:
      - sources: [EUR]
        dest: X
        proportions: [P]
        time: t2
      - sources: [EUR]
        dest: X
        proportions: [P]
        time: t3

Here, the proportion of both migration pulses is the same, but they occur at different times. This model would have 5 parameters: `R`, `tx`, `P`, `t2`, `t3`. If the pulses had different rates, the model would have 6 parameters instead.

Similar to pulses, **continuous migrations** can be specified in the `migrations` field:

    migrations:
      - source: EUR
        dest: X
        rate: K
        start_time: t1
        end_time: t2

### 2. Preparing the input data

$\texttt{tracts}$ input is a set bed-style file describing the local ancestry of segments
along the genome.  The file has 2 extra columns for the cM positions of the
segments. There are two input files per individuals (for each haploid genome
copy).

    chrom		begin		end			assignment	cmBegin	cmEnd
    chr13		0			18110261	UNKNOWN	0.0			0.19
    chr13		18110261	28539742	YRI			0.19		22.193
    chr13		28539742	28540421	UNKNOWN	22.193		22.193
    chr13		28540421	91255067	CEU		22.193		84.7013

### 3. Setting up the driver file

$\texttt{tracts}$ is run by passing a **driver yaml file** to `tracts.run_tracts()`.
The first part of the driver file tells $\texttt{tracts}$ how to load the sample data:

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
      allosomes: [X]`

In this example, the samples are located in the 'G10' directory. The individual 'NA19700' has sample data in the files 'NA19700_A.bed' and 'NA19700_B.bed'.

The `chromosomes` field tells $\texttt{tracts}$ to use data from chromosomes 1 to 22. You can also specify a single chromosome or a list of chromosomes. Adding the field `allosomes: [X]` makes $\texttt{tracts}$ include data from the X chromosome.

Then, the yaml file specifying the demographic model (see section above) is indicated:

    model_filename: pp.yaml

$\texttt{tracts}$ optimizes the parameters of the model to best match the distribution of ancestry tracts. 
Starting values for the parameters can be specified as numbers or ranges. Multiple repetitions can be run on the same data, and a seed can be used for replicability.

    start_params:
      R: 0.1-0.2
      tx: 10-11
      P:  0.03-0.05
      P_sex_bias: 0
      t2: 5.5
    repetitions: 2
    seed: 100

$\texttt{tracts}$ also allows for the time parameter to be scaled, as some optimizers run better when
all parameters are on the same scale:

    time_scaling_factor: 100

Finally, tracts below a certain length (in centimorgans) can be excluded from the analysis.

    exclude_tracts_below_cM: 10
    
### 4. Running tracts

Once the driver file is ready, the user can simply run:

    run_tracts(driver_filename  = driverfile.yaml, script_dir = '/path/to/folder/containing/driverfile.yaml/')
  
where `driverfile.yaml` is the driver yaml file previously prepared, and `script_dir` is the path
to the directory where the file is located.

### The output 

$\texttt{tracts}$ saves the results in the `output_directory` specified in the driver file. For autosomes, allosomes in males and allosomes in females, these include:

* _bins: the bins used in the discretization.
* _dat_sample_tract_distribution: the observed counts in each bin.
* _predicted_tract_distribution: the predicted counts in each bin, according to the model.
* _migration_matrix: the inferred migration matrix, with the most recent generation at the
    top, and one column per migrant population. Entry i,j in the matrix represent 
    the proportion of individuals in the admixed population who originate
    from the source population j at generation i in the past. 
* _optimal_parameters: the optimal parameters. I.e., if these models are passed to the 
    admixture model, it will return the inferred migration matrix.
* A plot comparing the sample and the predicted tract length distribution, for each source population.

### Examples

In the [example](https://github.com/gravelLab/tracts/tree/master/example) folder, we provide data and multiple yaml files to run $\texttt{tracts}$ as an example.

To produce tract length density functions or histograms from a pair of sex-specific migrations matrices, using any of the models presented in the [tracts 2.0 paper](), an easy-to-use jupyter notebook illustrating the functions of the `PhaseType` class is available [here](https://github.com/gravelLab/tracts/blob/devel/example/toy_example.ipynb).

---

### FAQ

> The distribution of tract lengths decreases as a function of tract length,
> but increases at the very last bin. This was not seen in the original paper.

The last bin represents the number of chromosomes with no ancestry
switches.

> When I have a single pulse of admixture, I would expect an exponential
> distribution of tract length, but the distribution of tract lengths shows
> steps in the expected length. Why is that?

$\texttt{tracts}$ takes into account the finite length of chromosomes. Since ancestry
tracts cannot extend beyond chromosomes, we expect a departure from an
exponential distribution.

> Individuals in my population vary considerably in their ancestry proportion.
> Is that a problem?

It is not a problem as long as the population was close to random mating. If
admixture is recent, random mating is not inconsistent with ancestry variance.
If admixture is ancient, however, variation in ancestry proportion may indicate
population structure, and the random mating assumption may fail.

> I ran the optimization steps many times, and found different optimal
> likelihoods.

Optimizing functions in many dimensions is hard, and sometimes optimizers get
stuck in local maxima. If you haven't tried already, you can attempt to fix the
ancestry proportions a priori (see the `_fix` examples in the documentation).
In most cases, the optimization will converge to the global maximum a
substantial proportion of the time: running the optimization a few times from
random starting positions and comparing the best values may help control for
this.

If you fail to revisit the same minimum after running say, 10 optimizations,
then something else might be going on. If the model is not continuous as a
function of a parameter, it could make the optimization much harder. Defining a
continuous model would help, or trying a brute-force optimization
method if the number of parameters is small.
