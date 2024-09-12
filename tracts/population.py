from tracts.indiv import Indiv
from tracts.util import eprint
from tracts.chrom import Chrom

import numpy as np
import tkinter as tk

from matplotlib import pylab
from tkinter import filedialog
import bisect
from collections import defaultdict


def collect_pop(flatdat):
    """ Organize a list of tracts into a dictionary keyed on ancestry
        labels.
    """
    dic = defaultdict(list)
    for t in flatdat:
        dic[t.label].append(t)
    return dic


def preprocess_color_dict(colordict, dat):
    for pop, color in colordict.items():
        for pos in dat[1]:
            try:
                pos[0][pop]
            except KeyError:
                pos[0][pop] = 0
                pos[1][pop] = 0


def _split_indivs(indivs, count, sort_ancestry=None):
    """ Internal function used to split a list of individuals into equally
        sized groups after sorting the individuals according to the proportion
        of the ancestry identified by "sort_ancestry". When no "sort_ancestry"
        is provided, the ancestry of the first tract of the first chromosome
        copy of the first chromosome of the first individual in the list is
        used.
    """
    if sort_ancestry is None:
        sort_ancestry = indivs[0].chroms[0].copies[0].tracts[0].label

    s_indivs = sorted(indivs, key=lambda i: i.ancestryProps([sort_ancestry])[0])

    n = len(indivs)
    group_frac = 1.0 / count

    groups = [s_indivs[int(n * i * group_frac):int(n * (i + 1) * group_frac)] for i in range(count)]

    return groups


class Population:
    def __init__(self, list_indivs=None, names=None, fname=None,
                 labs=("_A", "_B"), selectchrom=None, ignore_length_consistency=False, filenames_by_individual=None):
        """ Construct a population of diploid individuals. A population is
            essentially a simple list of indiv objects.

            There are two ways to build populations, either from a dataset
            stored in files or from a list of individuals. The facilities for
            loading populations from files present in this constructor are
            deprecated. It is advised to instead load a list of individuals,
            using indiv.from_file, and to then pass that list to this
            constructor.

        The population can be initialized by providing it with a list of
            "individual" objects, or a file format fname and a list of names.
            If reading from a file, fname should be a tuple with the start
            middle and end of the file names., where an individual file is
            specified by start--Indiv--Middle--_A--End. Otherwise, provide list
            of individuals. Distinguishing labels for maternal and paternal
            chromosomes are given in lab.
        """
        self.currentplot = None
        self.win = None
        self.chro_canvas = None
        self.colordict = None
        self._flats = None
        self.canv = None
        if list_indivs is not None:
            self.indivs = list_indivs
            self.nind = len(list_indivs)
            # should probably check that all individuals have same length!
            self.Ls = self.indivs[0].Ls
            assert all(i.Ls == self.indivs[0].Ls for i in self.indivs), "individuals have genomes of different lengths"
            self.maxLen = max(self.Ls)
        elif filenames_by_individual is not None:
            self.indivs = []
            for name, files in filenames_by_individual.items():
                try:
                    self.indivs.append(
                        Indiv.from_files(
                            files,
                            name=name,
                            selectchrom=selectchrom))
                except Exception as e:
                    raise IndexError(f'Files for individiual {name} ({files}) could not be found.') from e

            self.nind = len(self.indivs)
            # Check that all individuals have the same length.
            self.Ls = self.indivs[0].Ls
            if not all(i.Ls == self.indivs[0].Ls for i in self.indivs) and not ignore_length_consistency:
                raise ValueError(
                    'Individuals have genomes of different lengths. '
                    'If this is intended, use ignore_length_consistency=True.')
            self.maxLen = max(self.Ls)
        elif fname is not None:
            self.indivs = []
            for name in names:
                try:
                    self.indivs.append(
                        Indiv.from_files(
                            [fname[0] + name + fname[1] + lab + fname[2]
                             for lab in labs],
                            name=name,
                            selectchrom=selectchrom))
                except IndexError:
                    eprint("error reading individuals", name)
                    eprint("fname=", fname, "; labs=", labs, ", selectchrom=", selectchrom)
                    raise IndexError

            self.nind = len(self.indivs)

            assert (ignore_length_consistency or (all(i.Ls == self.indivs[0].Ls for i in self.indivs)))

            self.Ls = self.indivs[0].Ls
            self.maxLen = max(self.Ls)
        else:
            raise ValueError('Population could not be loaded because individuals were not specified.')

    def split_by_props(self, count):
        """ Split this population into groups according to their ancestry
            proportions. The individuals are sorted in ascending order of their
            ancestry named "anc".
        """
        if count == 1:
            return [self]

        return [Population(g)
                for g in _split_indivs(self.indivs, count)]

    def newgen(self):
        """ Build a new generation from this population. """
        return Population([self.new_indiv() for _i in range(self.nind)])

    def new_indiv(self):
        rd = np.random.random_integers(0, self.nind - 1, 2)
        while rd[0] == rd[1]:
            rd = np.random.random_integers(0, self.nind - 1, 2)
        gamete1 = self.indivs[rd[0]].create_gamete()
        gamete2 = self.indivs[rd[1]].create_gamete()
        return Indiv.from_haploids([gamete1, gamete2])

    def save(self):
        file = filedialog.asksaveasfilename(parent=self.win, title='Choose a file')
        self.indivs[self.currentplot].canvas.postscript(file=file)

    def list_chromosome(self, chronum):
        """ Collect the chromosomes with the given number across the whole
            population.
        """
        return [curr_indiv.chroms[chronum] for curr_indiv in self.indivs]

    def ancestry_at_pos(self, select_chrom=0, pos=0, cutoff=.0):
        """ Find ancestry proportion at specific position. The cutoff is used
            to look only at tracts that extend beyond a given position. """
        ancestry = {}
        # keep track of ancestry of long segments
        longancestry = {}
        totlength = {}
        for chropair in self.list_chromosome(select_chrom):
            for ploid in chropair.copies:
                selected_tract = ploid.tracts[ploid.goto(pos)]
                try:
                    if selected_tract.len() > cutoff:
                        ancestry[selected_tract.label] += 1
                        totlength[selected_tract.label] += selected_tract.len()
                except KeyError:
                    ancestry[selected_tract.label] = 0
                    longancestry[selected_tract.label] = 0
                    totlength[selected_tract.label] = 0
                    if selected_tract.len():
                        ancestry[selected_tract.label] += 1
                        totlength[selected_tract.label] += selected_tract.len()

        for key in totlength.keys():
            # prevent division by zero
            if totlength[key] == 0:
                totlength[key] = 0
            else:
                totlength[key] = totlength[key] / float(ancestry[key])

        return ancestry, totlength

    def ancestry_per_pos(self, select_chrom=0, npts=100, cutoff=.0):
        """ Prepare the ancestry per position across chromosome. """
        length = self.indivs[0].chroms[Chrom].len  # Get chromosome length
        plotpts = np.arange(0, length, length / float(npts))  # Get number of points at which to
        # Plot ancestry
        return (plotpts,
                [self.ancestry_at_pos(select_chrom=select_chrom, pos=pt, cutoff=cutoff)
                 for pt in plotpts])

    def applychrom(self, func, indlist=None):
        """ Apply func to chromosomes. If no indlist is supplied, apply to all
            individuals.
        """
        ls = []

        if indlist is None:
            inds = self.indivs
        else:
            inds = indlist

        for ind in inds:
            ls.append(ind.applychrom(func))
        return ls

    def flatpop(self, ls=None):
        """ Returns a flattened version of a population-wide list at the tract
            level, and throws away the start and end information of the tract,
        """
        if ls is not None:
            return list(self.iflatten(ls))
        if self._flats is not None:
            return self._flats
        self._flats = list(self.iflatten(ls))
        return self._flats

    def iflatten(self, indivs=None):
        """ Flatten a list of individuals to the tract level. If the list of
            individuals "indivs" is None, then the complete list of individuals
            contained in this population is flattened.
            The result is a generator.
        """
        if indivs is None:
            indivs = self.indivs
        for i in indivs:
            for _tract in i.iflatten():
                yield _tract

    def merge_ancestries(self, ancestries, newlabel):
        """ Treats ancestries in label list "ancestries" as a single population
            with label "newlabel". Adjacent tracts of the new ancestry are
            merged. """
        f = lambda i: i.merge_ancestries(ancestries, newlabel)
        self.applychrom(f)

    def get_global_tractlengths(self, npts=20, tol=0.01, indlist=None, split_count=1, exclude_tracts_below_cM=0):
        """ tol is the tolerance for full chromosomes: sometimes there are
            small issues at the edges of the chromosomes. If a segment is
            within tol Morgans of the full chromosome, it counts as a full
            chromosome note that we return an extra bin with the complete
            chromosome bin, so that we have one more data point than we have
            bins.
            indlist is the individuals for which we want the tractlength. To
            bootstrap over individuals, provide a bootstrapped list of
            individuals.
        """
        # Figure out whether we're dealing with the set of individuals
        # represented by this population or the one contained in the indlist
        # parameter.
        pop = self if indlist is None else Population(indlist)

        if split_count > 1:
            # If we're doing a split analysis, then break up the population
            # into groups, and just do get_global_tractlengths on the groups.
            ps = pop.split_by_props(split_count)
            bindats = [p.get_global_tractlengths(npts, tol, exclude_tracts_below_cM=exclude_tracts_below_cM) for p in
                       ps]
            bins_list, dats_list = zip(*bindats)
            # the bins will all be the same, so we can throw out the
            # duplicates.
            return bins_list[0], dats_list

        bins = np.arange(exclude_tracts_below_cM * 0.01, self.maxLen * (1 + .5 / npts), float(self.maxLen) / npts)

        bypop = defaultdict(list)

        for indiv in pop:
            for chrom in indiv:
                for copy in chrom:
                    copy._smooth_unknown()
                    for tract in copy:
                        bypop[tract.label].append({
                            'tract': tract,
                            'chromlen': chrom.len
                        })
        dat = {}
        for label, ts in bypop.items():
            # extract full length tracts
            nonfulls = np.array([t['tract'] for t in ts if t['tract'].end - t['tract'].start < t['chromlen'] - tol])

            hdat = np.histogram([n.len() for n in nonfulls], bins=bins)
            dat[label] = list(hdat[0])
            # append the number of fulls
            dat[label].append(len(ts) - len(nonfulls))
        return bins, dat

    def bootinds(self, seed):
        """ Return a bootstrapped list of individuals in the population. Use
            with get_global_tractlength inds=... to get a bootstrapped
            sample.
            """
        np.random.seed(seed=seed)
        return np.random.choice(self.indivs, size=len(self.indivs))

    def get_global_tractlength_table(self, lenbound):
        """ Calculates the fraction of the genome covered by ancestry tracts of
            different lengths, specified by lenbound (which must be sorted). """
        flatdat = self.flatpop()
        bypop = collect_pop(flatdat)

        bins = lenbound
        # np.arange(0,self.maxLen*(1+.5/npts),float(self.maxLen)/npts)

        dat = {}  # np.zeros((len(bypop),len(bins)+1)
        for key, poplen in bypop.items():
            # extract full length tracts
            dat[key] = np.zeros(len(bins) + 1)
            nonfulls = np.array([item
                                 for item in poplen
                                 if (item[0] != item[1])])
            for item in nonfulls:
                pos = bisect.bisect_left(bins, item[0])
                dat[key][pos] += item[0] * 1. / self.nind * 1. / np.sum(self.Ls) / 2.

        return bins, dat

    def get_mean_ancestry_proportions(self, ancestries):
        """ Get the mean ancestry proportion averaged across individuals in
            the population.
        """
        return map(np.mean, zip(*self.get_means(ancestries)))

    def get_means(self, ancestries):
        """ Get the mean ancestry proportion (only among ancestries in
            ancestries) for all individuals. """
        return [ind.ancestryProps(ancestries) for ind in self.indivs]

    def get_meanvar(self, ancestries):
        byind = self.get_means(ancestries)
        return np.mean(byind, axis=0), np.var(byind, axis=0)

    def getMeansByChrom(self, ancestries):
        """ Get the ancestry proportions in each individual of the population
            for each chromosome.
        """
        return [ind.ancestryPropsByChrom(ancestries) for ind in self.indivs]

    # def get_assortment_variance(self,ancestries):
    #     """ancestries is a set of ancestry label. Calculates the assortment variance in
    # ancestry proportions (corresponds to the mean uncertainty about the proportion of
    # genealogical ancestors, given observed ancestry patterns)"""

    # ws=np.array(self.Ls)/np.sum(self.Ls) #the weights, corresponding (approximately) to the inverse variances
    #     arr=np.array(self.getMeansByChrom(ancestries))
    # weighted mean by individual
    # departure from the mean
    #     nchr=arr.shape[2]
    #     vars=[]
    #     for i in range(len(ancestries)):
    #         pl=np.dot(arr[:,i,:], ws )

    #         aroundmean=arr[:,i,:]-np.dot(pl.reshape(self.nind,1),np.ones((1,nchr)))
    #         vars.append((np.mean(aroundmean**2/(1./ws-1),axis=1)).mean())
    # the unbiased estimator for the case where the variance is
    # inversely proportional to the weight. First calculate by
    # individual, then the mean over all individuals.

    #     return vars

    def get_variance(self, ancestries):
        """ Ancestries is a set of ancestry labels. Calculates the total variance
            in ancestry proportions, and the genealogy variance, and the
            assortment variance. (corresponds to the mean uncertainty about the
            proportion of genealogical ancestors, given observed ancestry
            patterns). Note that all ancestries not listed are considered uncalled.
            For example, calling the function with a single ancestry leads to no variance.
            (and some 0/0 errors)"""

        # the weights, corresponding (approximately) to the inverse variances
        ws = np.array(self.Ls) * 1. / np.sum(self.Ls)
        arr = np.array(self.getMeansByChrom(ancestries))
        # weighted mean by individual
        # departure from the mean
        nchr = arr.shape[2]
        assort_vars = []
        tot_vars = []
        gen_vars = []
        for i in range(len(ancestries)):
            pl = np.dot(arr[:, i, :], ws)
            tot_vars.append(np.var(pl))
            aroundmean = arr[:, i, :] - np.dot(
                pl.reshape(self.nind, 1), np.ones((1, nchr)))
            # the unbiased estimator for the case where the variance is
            # inversely proportional to the weight. First calculate by
            # individual, then the mean over all individuals.
            assort_vars.append(
                (np.mean(aroundmean ** 2 / (1. / ws - 1), axis=1)).mean())
            gen_vars.append(tot_vars[-1] - assort_vars[-1])
        return tot_vars, gen_vars, assort_vars

    def plot_next(self):
        self.indivs[self.currentplot].canvas.pack_forget()
        if self.currentplot < self.nind - 1:
            self.currentplot += 1
        return self.plot_indiv()

    def plot_previous(self):
        self.indivs[self.currentplot].canvas.pack_forget()
        if self.currentplot > 0:
            self.currentplot -= 1
        return self.plot_indiv()

    def plot_indiv(self):
        self.win.title("individual %d " % (self.currentplot + 1,))
        self.canv = self.indivs[self.currentplot].plot(
            self.colordict, win=self.win)

    def plot(self, colordict):
        self.colordict = colordict
        self.currentplot = 0
        self.win = tk.Tk()
        printbutton = tk.Button(self.win, text="save to ps", command=self.save)
        printbutton.pack()

        p = tk.Button(self.win, text="Plot previous", command=self.plot_previous)
        p.pack()

        b = tk.Button(self.win, text="Plot next", command=self.plot_next)
        b.pack()
        self.plot_indiv()
        tk.mainloop()

    def plot_chromosome(self, i, colordict, win=None):
        """plot a single chromosome across individuals"""
        self.colordict = colordict
        ls = self.list_chromosome(i)
        if win is None:
            win = tk.Tk()
            win.title("chromosome %d" % (i,))
        self.chro_canvas = tk.Canvas(win, width=250, height=self.nind * 30, bg='white')

        for j in range(len(ls)):
            ls[j].plot(self.chro_canvas, colordict, height=j * .25)

        self.chro_canvas.pack(expand=tk.YES, fill=tk.BOTH)
        tk.mainloop()

    def plot_ancestries(self, chrom=0, npts=100, colordict=None, cutoff=0.0):
        if colordict is None:
            colordict = {"CEU": 'blue', "YRI": 'red'}

        dat = self.ancestry_per_pos(select_chrom=chrom, npts=npts, cutoff=cutoff)
        preprocess_color_dict(colordict=colordict, dat=dat)
        tot = 0
        for pos in dat[1]:
            tot = 0
            for key in colordict.keys():
                tot += pos[0][key]
            for key in colordict.keys():
                if pos[0][key] != 0:
                    eprint(pos[0][key], float(tot)),
                    pos[0][key] /= float(tot)
        for pop, color in colordict.items():
            eprint(tot)
            pylab.figure(1)
            pylab.plot(dat[0], [pos[0][pop] for pos in dat[1]], '.', color=color)
            pylab.title("Chromosome %d" % (chrom + 1,))
            pylab.axis((0, dat[0][-1], 0, 1))
            pylab.figure(2)
            pylab.plot(dat[0], [100 * pos[1][pop] for pos in dat[1]], '.', color=color)
            pylab.title("Chromosome %d" % (chrom + 1,))
            pylab.axis((0, dat[0][-1], 0, 150))

    def plot_all_ancestries(self, npts=100, colordict=None, startfig=0, cutoff=0):
        dat = None
        chrom = None
        pop = None
        color = None
        if colordict is None:
            colordict = {"CEU": 'blue', "YRI": 'red'}
        # TODO: Remove the magic number
        for chrom in range(22):
            dat = self.ancestry_per_pos(select_chrom=chrom, npts=npts, cutoff=cutoff)
            preprocess_color_dict(colordict=colordict, dat=dat)
            for pos in dat[1]:
                tot = 0
                for key in colordict.keys():
                    tot += pos[0][key]
                for key in colordict.keys():
                    if pos[0][key] != 0:
                        pos[0][key] /= float(tot)
            for pop, color in colordict.items():
                pylab.figure(0 + startfig)
                pylab.subplot(6, 4, chrom + 1)
                pylab.plot(dat[0], [pos[0][pop] for pos in dat[1]], '.', color=color)
                pylab.axis([0, dat[0][-1], 0, 1])
                pylab.figure(1 + startfig)
        # TODO: Replace "chrom + 1" with 23?
        pylab.subplot(6, 4, chrom + 1)
        # TODO: What color should be used?
        pylab.plot(dat[0], [100 * pos[1][pop] for pos in dat[1]], '.', color=color)
        pylab.axis([0, dat[0][-1], 0, 150])

    def plot_global_tractlengths(self, colordict, npts=40, legend=True):
        flatdat = self.flatpop()
        bypop = collect_pop(flatdat)
        self.maxLen = max(self.Ls)
        for label, tracts in bypop.items():
            hdat = pylab.histogram([i.len() for i in tracts], npts)
            # note: convert to cM before plotting
            pylab.semilogy(100 * (hdat[1][1:] + hdat[1][:-1]) / 2., hdat[0], 'o', color=colordict[label], label=label)
        pylab.xlabel("length(cM)")
        pylab.ylabel("counts")
        if legend:
            pylab.legend()

    def __iter__(self):
        return self.indivs.__iter__()

    def __getitem__(self, index):
        return self.indivs[index]
