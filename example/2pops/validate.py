#!/usr/bin/env python

from __future__ import print_function

import sys, os, json

sys.path.append('../..')
sys.path.append('..')

import tracts, pp, pp_px

import FancyPlot as fp
import numpy as np

from glob import glob
from collections import defaultdict

eprint = lambda *args, **kwargs: print(*args, file=sys.stderr, **kwargs)

# The haplotype specifiers, which are used in the path specifier for
# identifying files to load.
haplotype_spec = ('A', 'B')

# The path specifier, which is a pattern into which we interpolate the
# individual identifier and the haplotype specifier.
pathspec = 'ind_%d_%s.bedx'

# Identifiers given to the ancestries, as they appear in the .bed files.
ancestry_labels = ('EUR', 'AFR')

# A cutoff for bins with small counts of tracts.
cutoff = 2

names = [
    "NA19700", "NA19701", "NA19704", "NA19703", "NA19819", "NA19818",
    "NA19835", "NA19834", "NA19901", "NA19900", "NA19909", "NA19908",
    "NA19917", "NA19916", "NA19713", "NA19982", "NA20127", "NA20126",
    "NA20357", "NA20356"
]

def collect_paths(data_dir, pathspec):
    """ Builds a list of tuples of paths corresponding to a list of paths
        needed to build an individual in tracts.
        Specifically, this method will, for increasing values of i starting at
        1, file pairs of existing files whose name match the pathspec (a format
        string) formatted by supplying the number i and the haplotype
        specifier, in that order.
        This is fairly naive and works only in the present case. A more robust
        solution should be implemented later.
    """
    mkpath = lambda n, s: os.path.join(data_dir, pathspec % (n, s))

    paths = []

    i = 1
    while True:
        if os.path.exists(mkpath(i, haplotype_spec[0])):
            paths.append(tuple(mkpath(i, s) for s in haplotype_spec))
        else:
            i -= 1
            break
        i += 1

    eprint('found', i, 'individuals')

    return paths

def G10_paths(data_dir):
    paths = []
    for name in names:
        paths.append(
                tuple(
                    os.path.join(data_dir, name + suf)
                    for suf
                    in ['_A.bed', '_B.bed']))
    return paths

def load_population(path_pairs):
    """ Given a list of pairs of paths, each pair identifying the two
        haplotypes for an individual, build a tracts population.
    """
    eprint('loading population')
    return tracts.population([tracts.indiv.from_files(t) for t in path_pairs])

def dual_analysis(labels, pop, migration_fun, migration_outofbounds_fun,
        startparams, migration_fun_name):
    """ Analyse a population using a given migration function, with both the
        composite and classical models.
    """

    ## 1a. the composite demographic model.

    # Get the information about the tracts out of the population object, and
    # split the population into 2 groups, in order to perform an analysis as a
    # composite demographic model.
    bins, group_data = pop.get_global_tractlengths(npts=50, split_count=2)

    # The actual groups that we're splitting the population into.
    groups = pop.split_by_props(2)

    # Compute the counts of individuals in each group and the average ancestry
    # proportions across individuals. The counts of individuals in each groups
    # should differ by no more than one.
    ninds, group_ancestry_averages = zip(*map(
        lambda p: (
            len(p.indivs),
            p.get_mean_ancestry_proportions(ancestry_labels)),
        groups))

    # Rearrange the data for each group into a list ordered in the same way as
    # the population labels.
    group_data = [
            [g[ancestry_label] for ancestry_label in ancestry_labels]
            for g in group_data]

    ## Optimize the model parameters using COBYLA
    #composite_model_parameters = tracts.optimize_cob_multifracs(
    #        startparams, bins, pop.Ls, group_data, ninds, migration_fun,
    #        group_ancestry_averages, outofbounds_fun=migration_outofbounds_fun,
    #        cutoff=cutoff, epsilon=1e-12)

    # Optimize the model parameters using brute
    composite_model_parameters, _ = tracts.optimize_brute_multifracs(
            bins, pop.Ls, group_data, ninds, migration_fun,
            group_ancestry_averages, startparams,
            outofbounds_fun=migration_outofbounds_fun,
            cutoff=cutoff)

    # Construct the composite demographic model
    composite_model = tracts.composite_demographic_model(
            migration_fun, composite_model_parameters,
            group_ancestry_averages)


    ## 1b. the fracs2 model.

    nind = len(pop.indivs)

    _, data = pop.get_global_tractlengths(npts=50, split_count=1)
    data = [data[ancestry_label] for ancestry_label in ancestry_labels]

    ancestry_averages = pop.get_mean_ancestry_proportions(ancestry_labels)

    fracs2_model_parameters, _ = tracts.optimize_brute_fracs2(
            bins, pop.Ls, data, nind,
            migration_fun, ancestry_averages, startparams,
            outofbounds_fun=migration_outofbounds_fun, cutoff=cutoff)

    fracs2_model = tracts.demographic_model(
            migration_fun(fracs2_model_parameters, ancestry_averages))

    return {
            'classical': {
                'params': fracs2_model_parameters,
                'model': fracs2_model,
                'averages': ancestry_averages,
                'theories': dict(
                    (
                        name,
                        fp.Theory(
                            bins,
                            nind * np.array(
                                fracs2_model.expectperbin(
                                    pop.Ls, i, bins)),
                            migration_fun_name + ' classical',
                            name)
                    )
                    for i, name in enumerate(labels)),
            },
            'composite': {
                'params': composite_model_parameters,
                'model': composite_model,
                'ninds': ninds,
                'averages': group_ancestry_averages,
                'groups': group_data,
                'theories': dict(
                    (
                        name,
                        fp.Theory(
                            bins,
                            composite_model.expectperbin(
                                pop.Ls, i, bins, ninds),
                            migration_fun_name + ' composite',
                            name)
                    )
                    for i, name in enumerate(labels)),
            },
            'misc': {
                'startparams': startparams,
                'bins': bins,
                'data': data,
                'fun': migration_fun,
            },
    }

def combine(d):
    """ Takes a dictionary ModelName -> ModelData and joins it into something
        with less redundancy and consisting only of serializable data.
        The name of the model should be such that we can load the corresponding
        function back from an eponymous file.
        For example, if the name is "pp", then we should find a file called
        "pp.py" with a function "pp_fix" in it that we can call to compute a
        migration matrix according to that model.
    """
    q = {}
    for name, data in d.iteritems():
        if 'misc' not in q:
            del data['misc']['fun']
            del data['misc']['startparams']
            data['misc']['bins'] = list(data['misc']['bins'])
            q['misc'] = data['misc']
            q['misc']['bins'] = list(q['misc']['bins'])
        else:
            del data['misc']
        del data['classical']['model'] # nor demographic_model
        del data['composite']['model'] # nor composite_demographic_model
        del data['classical']['theories'] # nor FancyPlot Thoery
        del data['composite']['theories']
        data['classical']['params'] = list(data['classical']['params'])
        data['composite']['params'] = list(data['composite']['params'])
        q[name] = data
    return q

def main(data_dir, output_dir, pathspec):
    if pathspec == 'G10' or not pathspec:
        pop = load_population(G10_paths(data_dir))
    else:
        # load the population
        pop = load_population(collect_paths(data_dir, pathspec))

    nind = len(pop.indivs)


    # do the analyses

    pp_data = dual_analysis(ancestry_labels, pop, pp.pp_fix,
            pp.outofbounds_pp_fix, (slice(0.02, 0.06, 0.005),), 'pp')

    pp_px_data = dual_analysis(ancestry_labels, pop, pp_px.pp_px_fix,
            pp_px.outofbounds_pp_px_fix,
            (slice(0.02, 0.06, 0.01),
                slice(0.005, 0.04, 0.01),
                slice(0.01, 0.04, 0.01)),
            'pp_px')

    bins = pp_data['misc']['bins']
    ninds = pp_data['composite']['ninds']


    # Plotting

    eprint("plotting pp migration function... ", end='')

    pp_plot = fp.FancyPlot([
        fp.Population(bins, pp_data['misc']['data'][i], name)
            .add_theory(pp_data['classical']['theories'][name])
            .add_theory(pp_data['composite']['theories'][name])
        for i, name in enumerate(ancestry_labels)])

    pp_plot.bottom_limit = 0.45

    pp_plot.make_figure().savefig(output_dir + '_pp_validate.pdf')

    eprint("done")

    eprint("plotting pp_px migration function... ", end='')

    pp_px_plot = fp.FancyPlot([
        fp.Population(bins, pp_data['misc']['data'][i], name)
            .add_theory(pp_px_data['classical']['theories'][name])
            .add_theory(pp_px_data['composite']['theories'][name])
        for i, name in enumerate(ancestry_labels)])

    pp_px_plot.bottom_limit = 0.45

    pp_px_plot.make_figure().savefig(output_dir + '_pp_px_validate.pdf')

    eprint("done")

    eprint("plotting pp vs pp_px for classical model... ", end='')

    pp_vs_pp_px_plot_classical = fp.FancyPlot([
        fp.Population(bins, pp_data['misc']['data'][i], name)
            .add_theory(pp_data['classical']['theories'][name])
            .add_theory(pp_px_data['classical']['theories'][name])
        for i, name in enumerate(ancestry_labels)])

    pp_vs_pp_px_plot_classical.bottom_limit = 0.45

    pp_vs_pp_px_plot_classical.make_figure().savefig(
            output_dir + '_pp_vs_pp_px_classical.pdf')

    eprint("done")

    eprint("plotting pp vs pp_px for composite model... ", end='')

    pp_vs_pp_px_plot_composite = fp.FancyPlot([
        fp.Population(bins, pp_data['misc']['data'][i], name)
            .add_theory(pp_data['composite']['theories'][name])
            .add_theory(pp_px_data['composite']['theories'][name])
        for i, name in enumerate(ancestry_labels)])

    pp_vs_pp_px_plot_composite.bottom_limit = 0.45

    pp_vs_pp_px_plot_composite.make_figure().savefig(
            output_dir + '_pp_vs_pp_px_composite.pdf')

    eprint("done")

    eprint("writing migration matrices... ", end='')

    def write_migs(d, prefix):
        """ Using the given path prefix, this method generates a migration
            matrix file for the classical model and one migration matrix
            file for each component demographic model of a composite model.
            The argument "d" should be a dictionary produced by the
            "dual_analysis" function.
            Returns the list of files written.
        """
        names = [prefix+ '_classical_mig']

        with open(prefix + '_classical_mig', 'wt') as f:
            for line in d['classical']['model'].mig:
                f.write('\t'.join(map(str, line)) + '\n')

        for i, mig in enumerate(d['composite']['model'].migs()):
            path = prefix + '_composite_mig_%d' % (i,)
            names.append(path)
            with open(path, 'wt') as f:
                for line in mig:
                    f.write('\t'.join(map(str, line)) + '\n')

        return names

    wrote = write_migs(pp_data, output_dir + '_pp')
    wrote.extend(write_migs(pp_px_data, output_dir + '_pp_px'))

    eprint("wrote", ', '.join(wrote))

    # Compute the model likelihoods and save them

    pp_data['classical']['lik'] = pp_data['classical']['model'].loglik(
            bins, pop.Ls, pp_data['misc']['data'], nind, cutoff=cutoff)

    pp_data['composite']['lik'] = pp_data['composite']['model'].loglik(
            bins, pop.Ls, pp_data['composite']['groups'],
            pp_data['composite']['ninds'], cutoff=cutoff)

    pp_px_data['classical']['lik'] = pp_px_data['classical']['model'].loglik(
            bins, pop.Ls, pp_data['misc']['data'], nind, cutoff=cutoff)

    pp_px_data['composite']['lik'] = pp_px_data['composite']['model'].loglik(
            bins, pop.Ls, pp_data['composite']['groups'],
            pp_data['composite']['ninds'], cutoff=cutoff)

    def write_liks(f):
        print("pp classical", pp_data['classical']['lik'],
                sep='\t', file=f)
        print("pp composite", pp_data['composite']['lik'],
                sep='\t', file=f)
        print("pp_px classical", pp_px_data['classical']['lik'],
                sep='\t', file=f)
        print("pp_px composite", pp_px_data['composite']['lik'],
                sep='\t', file=f)

    print("likelihoods:")
    write_liks(sys.stdout)
    with open(output_dir + '_lik', 'wt') as f:
        write_liks(f)

    with open(output_dir + '_bins', 'wt') as f:
        f.write('\t'.join(map(str, pp_data['misc']['bins'])) + '\n')

    with open(output_dir + '_dat', 'wt') as f:
        for i in xrange(len(ancestry_labels)):
            f.write('\t'.join(map(str, pp_data['misc']['data'][i])) + '\n')

    # Combine the pp and pp_px data into one JSON-serializable dict and save
    # it.
    q = combine({
        'pp': pp_data,
        'pp_px': pp_px_data,
    })

    with open(output_dir + 'result.json', 'wt') as f:
        json.dump(q, f)

    print('Done. Have a nice day.')

if __name__ == '__main__':
    # "Parse" command line arguments.

    try:
        data_dir = sys.argv[1]
    except IndexError:
        data_dir = 'G10/'

    try:
        output_dir = sys.argv[2]
    except IndexError:
        output_dir = './all_fix'

    try:
        pathspec = sys.argv[3]
    except:
        pathspec = ''

    main(data_dir, output_dir, pathspec)
