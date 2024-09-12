# TODO: Verify that the functions don't change the data
from tracts import Population

directory = "./G10/"


# number of short tract bins not used in inference.
cutoff = 2

# number of repetitions for each model (to ensure convergence of optimization)
rep_pp = 2
rep_pp_px = 2

# only trio individuals
names = [
    "NA19700", "NA19701", "NA19704", "NA19703", "NA19819", "NA19818",
    "NA19835", "NA19834", "NA19901", "NA19900", "NA19909", "NA19908",
    "NA19917", "NA19916", "NA19713", "NA19982", "NA20127", "NA20126",
    "NA20357", "NA20356"
]

chroms = ['%d' % (i,) for i in range(1, 23)]

# load the population
pop = Population(
    names=names, fname=(directory, "", ".bed"), selectchrom=chroms)
(bins, data) = pop.get_global_tractlengths(npts=50)


# choose order of populations and sort data accordingly
labels = ['EUR', 'AFR']
data = [data[poplab] for poplab in labels]
