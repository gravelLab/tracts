from __future__ import print_function

import numpy as np
import operator as op
import itertools as it
import os
import pylab
import Tkinter as Tk
import tkFileDialog

from collections import defaultdict

try:
    from scipy.misc.common import factorial
except ImportError:
    from scipy.misc import factorial

from scipy.special import gammainc, gammaln
import scipy.optimize
import sys

eprint = lambda *args, **kwargs: print(*args, file=sys.stderr, **kwargs)

class tract(object):
    """ A tract is the lower-level object of interest. All the remaining
        structure is built on top of lists of tracts. In essence, a tract is
        simply a labelled interval.
    """
    def __init__(self, start, end, label, bpstart=None, bpend=None):
        """ Constructor.

            Arguments:
                start (float):
                    The starting point of this tract, in Morgans.
                end (float):
                    The ending point of this tract, in Morgans.
                label (string):
                    A meaningful identifier for this tract. Generally this
                    marks the ancestry associated with this tract.
                bpstart (int, default: None):
                    The starting point of this tract, in basepairs.
                    Since the rest of Tracts uses Morgans throughout,
                    specifying this parameter is not necessary for Tracts to
                    function correctly.
                bpend (int, default: None):
                    The ending point of this tract, in basepairs.
            """
        self.start = start
        self.end = end
        self.label = label
        self.bpstart = bpstart
        self.bpend = bpend

    def len(self):
        """ Get the length of the tract (in Morgans) """
        return self.end - self.start

    def get_label(self):
        """ Get the label of the tract. """
        return self.label

    def copy(self):
        """ Construct a new tract whose properties are the same as this one.
            """
        return tract(
            self.start, self.end, self.label, self.bpstart, self.bpend)

    def __repr__(self):
        return "tract(%s, %s, %s)" % tuple(
                map(repr, [self.start, self.end, self.label]))

class chrom(object):
    """ A chromosome wraps a list of tracts, which form a paritition on it. The
        chromosome has a finite, immutable length.
        """

    def __init__(self, ls=None, auto=True, label="POP", tracts=None):
        """ Constructor.

            Arguments:
                ls (int, default: None):
                    The length of this chromosome, in Morgans.
                auto (bool, default: True):
                    Whether this chromosome is autosomal.
                label (string, default: "POP"):
                    An identifier categorizing this chromosome.
                tracts (list of tract objects, default: None):
                    The list of tracts that span this chromosome. If None is
                    given, the a single, unlabeled tract is created to span the
                    whole chromosome, according to the length len.
        """
        if tracts is None:
            self.len = ls
            self.auto = auto

            # A single tract spanning the whole chromosome.
            self.tracts = [tract(0, self.len, label)]
            self.start = 0
        else:
            if not tracts:
                raise ValueError("a nonempty list of tracts is required "
                        "for initialization of a chromosome.")
            self.tracts = tracts
            self.auto = auto

            # Set the chromosome's start attribute to the starting point of the
            # first known tract.
            for t in self.tracts:
                self.start = t.start
                if t.label != 'UNKNOWN':
                    break

            # Set the chromosome's end attribute to the ending point of the
            # last known tract.
            for t in self.tracts[-1::-1]:  # Iterate in reverse over tracts
                self.end = t.end
                if t.label != 'UNKNOWN':
                    break

            # consider the length after stripping the UNKNOWN end tracts
            self.len = self.end - self.start

    # initialize a chromosome with a single tract
    def init_unif_tracts(self, label):
        self.tracts = [tract(0, self.len, label)]

    # initiate from a list of tracts
    def init_list_tracts(self, tracts):
        self.tracts = tracts

    def set_sex(self):
        """ Consider this chromosome to be a sex chromosome, in which case it
            is not autosomal. The effect of this method is to set the auto
            property to False.
            """
        self.auto = False

    def len(self):
        """ The length of this chromosome, in Morgans. """
        return self.len

    def goto(self, pos):
        """ Find the first tract containing a given position, in Morgans, and
            return its index in the underlying list. """
        # Use binary search for this, since the tract list is sorted.
        if(pos < 0 or pos > self.len):
            raise ValueError("cannot seek to position outside of chromosome")

        low = 0
        high = len(self.tracts) - 1
        curr = (low + high + 1) / 2

        while high > low:
            if(self.tracts[curr].start < pos):
                low = curr
            else:
                high = curr - 1
            curr = (low + high + 1) / 2

        return low

    # extract a particular segment from a chromosome
    def extract(self, start, end):
        """ Extract a segment from the chromosome.

            Arguments:
                start (int):
                    The starting point of the desired segment to extract.
                end (int):
                    The ending point of the desired segment to extract.

            Returns:
                A list of tract objects that span the desired interval.

            Notes:
                Uses the goto method of this class to identify the starting and
                ending points of the segment, so if those positions are
                invalid, goto will raise a ValueError.
            """
        startpos = self.goto(start)
        endpos = self.goto(end)
        extract = [tract.copy() for tract in self.tracts[startpos:endpos + 1]]
        extract[0].start = start
        extract[-1].end = end
        return extract

    # plot chromosome on the provided canvas
    def plot(self, canvas, colordict, height=0, chrwidth=.1):

        for tract in self.tracts:
            canvas.create_rectangle(
                    100 * tract.start, 100 * height, 100 * tract.end,
                    100 * (height + chrwidth),
                    width=0, disableddash=True, fill=colordict[tract.label])

    def _smooth(self):
        """ Combine adjacent tracts with the same label.
            The side-effect is that the entire list of tracts is copied, so
            unnecessary calls to this method should be avoided.
        """

        if not self.tracts:
            warn("smoothing empty chromosome has no effect")
            return None  # Nothing to smooth since there're no tracts.

        def same_ancestry(my, their):
            return my.label == their.label

        # TODO determine whether copies are really necessary.
        # this could be an avenue for optimization.
        newtracts = [self.tracts[0].copy()]
        for t in self.tracts[1:]:
            if same_ancestry(t, newtracts[-1]):
                # Extend the last tract added to encompass the next one
                newtracts[-1].end = t.end
            else:
                newtracts.append(t.copy())

        self.tracts = newtracts

    def merge_ancestries(self, ancestries, newlabel):
        """ Merge segments that are contiguous and of either the same ancestry,
            or that are labelled as in a given list.

            The label of each tract in the chromosome's inner list is checked
            against the labels listed in `ancestries`. If there is a match,
            then that tract is relabelled to `newlabel`. This batch relabelling
            allows us to consider several technically different ancestries as
            being the same, by relabelling them to actually be the same. Then,
            the resulting list is smoothed, to combine adjacent tracts whose
            labels are the same. This new list replaces the `tracts` list.

            Arguments:
                ancestries (list of strings):
                    The ancestries to merge.
                newlabel (string):
                    The identifier for the new ancesty to assign to the
                    matching tracts.

            Returns:
                Nothing.
            """
        for tract in self.tracts:
            if tract.label in ancestries:
                tract.label = newlabel

        self._smooth()

    def _smooth_unknown(self):
        """ Merge segments that are contiguous and of the same ancestry.  Under
            the hood, what this method does is eliminate medial segments of
            unknown ancestry, inflating the adjacent segments to fill the space
            left by the unknown ancestry.
        """
        i = 0
        while(i < len(self.tracts) - 1):
            if(self.tracts[i].label == 'UNKNOWN'):
                i += 1
                continue
            else:
                j = 0
                while(i + j < len(self.tracts) - 1):
                    j += 1
                    if(self.tracts[i + j].label == "UNKNOWN"):
                         self.tracts.pop(i + j)
                         j -= 1
                    else:
                         midpoint = (self.tracts[i+j].start
                                 + self.tracts[i].end) / 2
                         self.tracts[i+j].start = midpoint
                         self.tracts[i].end = midpoint
                         break
                i += 1
        self._smooth()

    def tractlengths(self):
        """ Gets the distribution of tract lengths. Make sure that proper
            smoothing is implemented.
            """
        self._smooth_unknown()

        return [(t.label, t.end - t.start, self.len) for t in self.tracts]

    def __iter__(self):
        return self.tracts.__iter__()

    def __getitem__(self, index):
        """ Simply wrap the underlying list's __getitem__ method. """
        return self.tracts[index]

    def __repr__(self):
        return "chrom(tracts=%s)" % (repr(self.tracts),)

class chropair(object):
    """ A pair of chromosomes. Using pairs of chromosomes allows us to model
        diploid individuals.
    """
    def __init__(self, chroms=None, len=1, auto=True, label="POP"):
        """ Can instantiate by explictly providing two chromosomes as a tuple
            or an ancestry label, length and autosome status. """
        if(chroms == None):
            self.copies = [chrom(len, auto, label), chrom(len, auto, label)]
            self.len = len
        else:
            if chroms[0].len != chroms[1].len:
                raise ValueError('chromosome pairs of different lengths!')
            self.len = chroms[0].len
            self.copies = chroms

    def recombine(self):
        # decide on the number of recombinations
        n = np.random.poisson(self.len)
        # get recombination points
        unif = (self.len*np.random.random(n)).tolist()
        unif.extend([0, self.len])
        unif.sort()
        # start with a random chromosome
        startchrom = np.random.random_integers(0, 1)
        tractlist = []
        for startpos in range(len(unif)-1):
            tractlist.extend(
                    self.copies[(startchrom+startpos)%2]
                        .extract(unif[startpos],
                    unif[startpos+1]))
        newchrom = chrom(self.copies[0].len, self.copies[0].auto)
        newchrom.init_list_tracts(tractlist)
        return newchrom

    def applychrom(self, func):
        """apply func to chromosomes"""
        ls = []
        for copy in self.copies:
            ls.append(func(copy))
        return ls

    def plot(self, canvas, colordict, height=0):
        self.copies[0].plot(canvas, colordict, height=height+0.1)
        self.copies[1].plot(canvas, colordict, height=height+0.22)

    def __iter__(self):
        return self.copies.__iter__()

    def __getitem__(self, index):
        return self.copies[index]

class indiv(object):
    """ The class of diploid individuals. An individual can hence be though of
        as a list of pairs of chromosomes. Equivalently, a diploid individual
        is a pair of haploid individuals.

        Thus, it is possible to construct instances of this class from a pair
        of instances of the haploid class, as well as directly from a sequence
        of chropair instances.

        The interface for loading individuals from files uses the
        haploid-oriented approach, since individual .bed files describe only
        one haplotype. The loading process is thus the following:

        1. load haploid individuals for each haplotype
        2. combine the haploid individuals into a diploid individual
    """
    @staticmethod
    def from_haploids(haps):
        if len(haps) != 2:
            raise ValueError('more than two haplotypes given to construct '
                    'a diploid individual')

        chroms = [chropair(t)
                for t in it.izip(*[hap.chroms for hap in haps])]

        return indiv(chroms=chroms, Ls=haps[0].Ls)

    @staticmethod
    def from_files(paths, selectchrom=None, name=None):
        """ Construct a diploid individual from two files, which describe the
            individuals haplotypes.
        """
        if len(paths) != 2:
            raise ValueError('more than two paths supplied to construct '
                    'a diploid individual')

        return indiv.from_haploids(
                [haploid.from_file(path, name=name, selectchrom=selectchrom)
                    for path in paths])

    def __init__(self, Ls=None, label="POP", fname=None, labs=("_A", "_B"),
            selectchrom=None, chroms=None, name=None):
        """ Construct a diploid individual. There are several ways to build
            individuals, either from files, from existing data, or
            programmatically.

            The most straightforward way to build an individual is from
            existing data, by supplying only the "Ls" and "chroms" arguments.

            Ls (default: None, type: list of floats):
                The lengths of the chromosomes in the order in which they
                appear in "chroms".
            chroms (default: None, type: list of chropair objects):
                The chromosome pairs that make up this individual. See the
                documentation for "chropair".

            If "Ls" is given, but "chroms" is not, then chromosomes consisting
            each of a single tract will be created with the label "label" and
            lengths drawn from "Ls".

            label (default: "POP", type: string):
                The label to use for building single-tract chromosomes when no
                other data is given to buid this individual.

            (deprecated) If the "fname" argument is given, the constructor will
            perform path manipulation involving the components of "fname" and
            "labs" to generate file names that are commonly used when dealing
            with .bed files.

            fname (default: None, type: 2-tuple of strings):
                Paths are generated by concatenating the first component of
                "fname", each label from "labs" in turn, and the second
                component of "fname".
                > fname[0] + lab + fname[1] for lab in labs

            labs (default: ("_A", "_B"), type: 2-tuple of strings):
                The labels used to identify maternal and paternal haplotypes in
                the paths leading to .bed files.

            selectchrom (default: None, type: list of integers):
                This argument is forwarded as-is to haploid.from_file. It acts
                as a filter on the chromosomes to load. The default value of
                "None" selects all chromosomes.

            Finally, some arguments are very general and are not involved in
            the analysis of the tracts.

            name (default: None, type: string):
                An identifier for this individual.

            The facilities in this constructor for loading individuals from
            files are deprecated. It is recommended to instead use the static
            methods from_files or from_haploids.
        """
        if(fname == None):
            self.Ls = Ls
            if chroms is None:
                self.chroms = [chropair(len=len, label=label) for len in Ls]
            else:
                self.chroms = chroms
        else:
            fnames = [fname[0] + lab + fname[1] for lab in labs]
            i = indiv.from_files(fnames, selectchrom)
            self.name = fname[0].split('/')[-1]
            self.chroms = i.chroms
            self.Ls = i.Ls

    def plot(self, colordict, win=None):
        """Plot an individual. colordict is a dictionary mapping population label to 
        a set of colors. E.g.: 
        colordict = {"CEU":'r',"YRI":b}
        """
        if (win is None):
            win = Tk.Tk()
        self.canvas = Tk.Canvas(
                win, width=250, height=len(self.Ls)*30, bg='white')

        for i in xrange(len(self.chroms)):
            self.chroms[i].plot(self.canvas, colordict, height=i*.3)

        self.canvas.pack(expand=Tk.YES, fill=Tk.BOTH)

        return win

    def create_gamete(self):
        lsc = [chpair.recombine() for chpair in self.chroms]
        return haploid(self.Ls, lsc)

    def applychrom(self, func):
        """ Apply the function `func` to each chromosome of the individual. """
        return map(lambda c: c.applychrom(func), self.chroms)

    def ancestryAmt(self, ancestry):
        """ Calculate the total length of the genome in segments of the given
            ancestry.
            """
        return np.sum(
                t.len()
                for t
                in self.iflatten()
                if t.label == ancestry)

    def ancestryProps(self, ancestries):
        """ Calculate the proportion of the genome represented by the given
            ancestries.
            """

        # We want to compute the sum of all the tract lengths as well as the
        # sum of the tract lengths that match ancestries given in the list
        # "ancestries", so for each tract, in this individual, we compute its
        # length as well as a tuple that represents which ancestry that tract
        # belongs to.
        gen = ((t.len(), [t.len() if t.label == a else 0 for a in ancestries])
                for t
                in self.iflatten()
        )

        all_lengths, all_ancestry_lengths = zip(*gen)
        total_length = float(np.sum(all_lengths))
        ancestry_sums = map(np.sum, zip(*all_ancestry_lengths))

        return [ancestry_sum/total_length for ancestry_sum in ancestry_sums]

    def ancestryPropsByChrom(self, ancestries):
        dat = self.applychrom(chrom.tractlengths)
        dictamt = {}
        nc = len(dat)
        for ancestry in ancestries:
            lsamounts = []
            for chromv in dat:
                lsamounts.append(np.sum([segment[1]
                        for copy in chromv
                        for segment in copy
                        if segment[0] == ancestry]))
            dictamt[ancestry] = lsamounts
        tots = [np.sum(
            [dictamt[ancestry][i]
                for ancestry in ancestries])
            for i in range(nc)]

        return [[dictamt[ancestry][i]*1./tots[i]
            for i in range(nc)]
            for ancestry in ancestries]

    def iflatten(self):
        """ Lazily flatten this individual to the tract level.  """
        for chrom in self.chroms:
            for copy in chrom.copies:
                for tract in copy.tracts:
                    yield tract

    def flat_imap(self, f):
        """ Lazily map a function over the full underlying structure of this
            individual.
            The function must accept 3 parameters:
                chrom: the chromosome pair containing the tract
                copy: the chromosome containing the tract
                tract: the tract itself
        """
        for chrom in self.chroms:
            for copy in chrom.copies:
                for tract in copy.tracts:
                    yield f(chrom, copy, tract)

    def __iter__(self):
        return self.chroms.__iter__()

    def __getitem__(self, index):
        return self.chroms[index]

# haploid individual
class haploid(object):
    @staticmethod
    def from_file(path, name=None, selectchrom=None):
        # TODO move the loading logic from the constructor to this static
        # method. This will facilitate loading logic for future driver
        # scripts.
        chromd = defaultdict(list)

        # Parse the indicated file into a dictionary associating chromosome
        # identifiers (strings) with lists of tract objects.
        with open(path, 'r') as f:
            for line in f:
                fields = line.split()

                # Skip the header, if one is present.
                if fields[0] == 'chrom' or \
                        (fields[0] == 'Chr' and fields[1] == 'Start(bp)'):
                    continue

                chromd[fields[0]].append(
                        tract(
                            .01 * float(fields[4]), .01 * float(fields[5]),
                            fields[3]))

        # Now that the file has been parsed, we need to apply a filtering step,
        # to select only those chromosomes identified by selectchrom.

        # A haploid individual is essentially just a list of chromosomes, so we
        # initialize this list of chromosomes to be ultimately passed to the
        # haploid constructor.
        chroms = []
        labs = []
        Ls = []

        # Construct a function that tells us whether a given chromosome is
        # selected or not.
        if selectchrom is None:
            # selectchrom being None means that all chromosomes are
            # selected, so we manufacture a constant function.
            is_selected = lambda *args: True
        else:
            # Otherwise, we 'normalize' selectchrom by ensuring that it
            # contains only integers. (This is primarily for
            # backwards-compatibility with previous scripts that specified
            # chromosome numbers as strings.) And we make a set out of the
            # resulting normalized list, to speed up lookups later.
            sc = set(map(int, selectchrom))
            # And the function we manufacture simply casts its argument
            # (which is a string since it's read in from a file) to an int,
            # and checks whether its in our set.
            is_selected = lambda c: int(c) in sc

        # Filter the loaded data according to selectchrom using the is_selected
        # function constructed above.
        for chrom_data, tracts in chromd.iteritems():
            chrom_id = chrom_data.split('r')[-1]
            if is_selected(chrom_id):
                c = chrom(tracts=tracts)
                chroms.append(c)
                Ls.append(c.len)
                labs.append(chrom_id)

        # Organize the filtered lists according to the order of their
        # identifiers.
        order = np.argsort(labs)

        chroms = list(
                np.array(chroms)[order])
        Ls = list(
                np.array(Ls)[order])
        labs = list(
                np.array(labs)[order])

        return haploid(Ls=Ls, lschroms=chroms, labs=labs, name=name)

    def __init__(self, Ls=None, lschroms=None, fname=None, selectchrom=None,
            labs=None, name=None):
        if fname is None:
            if Ls is None or lschroms is None:
                raise ValueError(
                        "Ls or lschroms should be defined if file not defined")
            self.Ls = Ls
            self.chroms = lschroms
            self.labs = labs
            self.name = name
        else:
            h = haploid.from_file(fname, selectchrom=selectchrom)
            self.Ls = h.Ls
            self.chroms = h.chroms
            self.labs = h.labs
            self.name = name

    def __repr__(self):
        return "haploid(lschroms=%s, name=%s, Ls=%s)" % tuple(map(repr,
                [self.chroms, self.name, self.Ls]))

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

    s_indivs = sorted(indivs,
            key=lambda i: i.ancestryProps([sort_ancestry])[0])

    n = len(indivs)
    group_frac = 1.0 / count

    groups = [s_indivs[int(n*i*group_frac):int(n*(i+1)*group_frac)]
            for i in xrange(count)]

    return groups

class population(object):
    def __init__(self, list_indivs=None, names=None, fname=None,
            labs=("_A", "_B"), selectchrom=None,
            ignore_length_consistency=False):
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
        if list_indivs is not None:
            self.indivs = list_indivs
            self.nind = len(list_indivs)
            # should probably check that all individuals have same length!
            self.Ls = self.indivs[0].Ls
            assert all(i.Ls == self.indivs[0].Ls for i in self.indivs), \
                    "individuals have genomes of different lengths"
            self.maxLen = max(self.Ls)
        elif fname is not None:
            self.indivs = []
            for name in names:

                pathspec = (fname[0]+name+fname[1], fname[2])

                try:
                    self.indivs.append(
                            indiv.from_files(
                                [fname[0] + name + fname[1] + lab + fname[2]
                                    for lab in labs],
                                name=name,
                                selectchrom=selectchrom))
                except IndexError:
                    eprint("error reading individuals", name)
                    eprint("fname=", fname, \
                            "; labs=", labs, ", selectchrom=", selectchrom)
                    raise IndexError

            self.nind = len(self.indivs)

            assert(ignore_length_consistency or
                    (all(i.Ls == self.indivs[0].Ls for i in self.indivs)))

            self.Ls = self.indivs[0].Ls
            self.maxLen = max(self.Ls)
        else:
            raise

    def split_by_props(self, count):
        """ Split this population into groups according to their ancestry
            proportions. The individuals are sorted in ascending order of their
            ancestry named "anc".
        """
        if count == 1:
            return [self]

        return [population(g)
                for g in _split_indivs(self.indivs, count)]

    def newgen(self):
        """ Build a new generation from this population. """
        return population([self.new_indiv() for i in range(self.nind)])

    def new_indiv(self):
        rd = np.random.random_integers(0, self.nind-1, 2)
        while(rd[0] == rd[1]):
            rd = np.random.random_integers(0, self.nind-1, 2)
        gamete1 = self.indivs[rd[0]].create_gamete()
        gamete2 = self.indivs[rd[1]].create_gamete()
        return indiv.from_haploids(gamete1, gamete2)

    def save(self):
        file = tkFileDialog.asksaveasfilename(parent=self.win,
                title='Choose a file')
        self.indivs[self.currentplot].canvas.postscript(file=file)

    def list_chromosome(self, chronum):
        """ Collect the chromosomes with the given number across the whole
            population.
        """
        return [indiv.chroms[chronum] for indiv in self.indivs]

    def ancestry_at_pos(self, chrom=0, pos=0, cutoff=.0):
        """ Find ancestry proportion at specific position. The cutoff is used
            to look only at tracts that extend beyond a given position. """
        ancestry = {}
        # keep track of ancestry of long segments
        longancestry = {}
        totlength = {}
        for chropair in self.list_chromosome(chrom):
            for chrom in chropair.copies:
                tract = chrom.tracts[chrom.goto(pos)]
                try:
                    if(tract.len() > cutoff):
                        ancestry[tract.label] += 1
                        totlength[tract.label] += tract.len()
                except KeyError:
                    ancestry[tract.label] = 0
                    longancestry[tract.label] = 0
                    totlength[tract.label] = 0
                    if tract.len():
                        ancestry[tract.label] += 1
                        totlength[tract.label] += tract.len()

        for key in totlength.keys():
            # prevent division by zero
            if totlength[key] == 0:
                totlength[key] = 0
            else:
                totlength[key] = totlength[key]/float(ancestry[key])

        return (ancestry, totlength)

    def ancestry_per_pos(self, chrom=0, npts=100, cutoff=.0):
        """ Prepare the ancestry per position across chromosome. """
        len = self.indivs[0].chroms[chrom].len
        plotpts = np.arange(0, len, len/float(npts))
        return (plotpts,
                [self.ancestry_at_pos(chrom=chrom, pos=pt, cutoff=cutoff)
                    for pt in plotpts])

    def applychrom(self,func, indlist=None):
        """ Apply func to chromosomes. If no indlist is supplied, apply to all
            individuals.
        """
        ls=[]

        if indlist is None:
                inds=self.indivs
        else:
                inds=indlist

        for ind in inds:
                ls.append(ind.applychrom(func))
        return ls

    def flatpop(self, ls=None):
        """ Returns a flattened version of a population-wide list at the tract
            level, and throws away the start and end information of the tract,
        """
        if ls is None:
            ls = self.indivs
            try:
                return self._flats
            except AttributeError:
                self._flats = list(self.iflatten(ls))
                return self._flats
        else:
            return list(self.iflatten(ls))

    def iflatten(self, indivs=None):
        """ Flatten a list of individuals to the tract level. If the list of
            individuals "indivs" is None, then the complete list of individuals
            contained in this population is flattened.
            The result is a generator.
        """
        if indivs is None:
            indivs = self.indivs

        for i in indivs:
            for tract in i.iflatten():
                yield tract

    def collectpop(self, flatdat):
        """ Organize a list of tracts into a dictionary keyed on ancestry
            labels.
        """
        dic = defaultdict(list)

        for t in flatdat:
            dic[t.label].append(t)

        return dic

    def merge_ancestries(self, ancestries, newlabel):
        """ Treats ancestries in label list "ancestries" as a single population
            with label "newlabel". Adjacent tracts of the new ancestry are
            merged. """
        f = lambda i: i.merge_ancestries(ancestries, newlabel)
        self.applychrom(f)

    def get_global_tractlengths(self, npts=20, tol=0.01, indlist=None,
            split_count=1):
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
        pop = self if indlist is None else population(indlist)

        if split_count > 1:
            # If we're doing a split analysis, then break up the population
            # into groups, and just do get_global_tractlengths on the groups.
            ps = pop.split_by_props(split_count)
            bindats = [p.get_global_tractlengths(npts, tol) for p in ps]
            bins_list, dats_list = zip(*bindats)
            # the bins will all be the same, so we can throw out the
            # duplicates.
            return bins_list[0], dats_list

        bins = np.arange(0,self.maxLen*(1+.5/npts),float(self.maxLen)/npts)

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

        dat={}
        for label, ts in bypop.iteritems():
            # extract full length tracts
            nonfulls = np.array(
                    [t['tract'] for t in ts
                        if t['tract'].end - t['tract'].start \
                                < t['chromlen'] - tol])

            hdat = np.histogram([n.len() for n in nonfulls], bins=bins)
            dat[label] = list(hdat[0])
            # append the number of fulls
            dat[label].append(len(ts)-len(nonfulls))

        return bins, dat

    def bootinds(self,seed):
        """ Return a bootstrapped list of individuals in the population. Use
            with get_global_tractlength inds=... to get a bootstrapped
            sample.
            """
        np.random.seed(seed=seed)
        return np.random.choice(self.indivs,size=len(self.indivs))

    def get_global_tractlength_table(self, lenbound):
        """ Calculates the fraction of the genome covered by ancestry tracts of
            different lengths, spcified by lenbound (which must be sorted). """
        flatdat = self.flatpop()
        bypop = self.collectpop(flatdat)

        bins = lenbound
        # np.arange(0,self.maxLen*(1+.5/npts),float(self.maxLen)/npts)

        import bisect
        dat = {} # np.zeros((len(bypop),len(bins)+1)
        for key, poplen in bypop.iteritems():
            # extract full length tracts
            dat[key] = np.zeros(len(bins)+1)
            nonfulls = np.array([item
                        for item in poplen
                        if (item[0] != item[1])])
            for item in nonfulls:
                pos = bisect.bisect_left(bins, item[0])
                dat[key][pos] += item[0]/self.nind/np.sum(self.Ls)/2.

        return (bins, dat)

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
    #     ""ancestries is a set of ancestry label. Calculates the assortment variance in ancestry proportions (corresponds to the mean uncertainty about the proportion of genealogical ancestors, given observed ancestry patterns)""

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
        """ Ancestries is a set of ancestry label. Calculates the total variance
            in ancestry proportions, and the genealogy variance, and the
            assortment variance. (corresponds to the mean uncertainty about the
            proportion of genealogical ancestors, given observed ancestry
            patterns). """

        # the weights, corresponding (approximately) to the inverse variances
        ws = np.array(self.Ls)/np.sum(self.Ls)
        arr = np.array(self.getMeansByChrom(ancestries))
        # weighted mean by individual
        # departure from the mean
        nchr = arr.shape[2]
        assort_vars = []
        tot_vars = []
        gen_vars = []
        for i in range(len(ancestries)):
            pl = np.dot(arr[:, i, :], ws )
            tot_vars.append(np.var(pl))
            aroundmean = arr[:, i, :] - np.dot(
                    pl.reshape(self.nind, 1), np.ones((1, nchr)))
            # the unbiased estimator for the case where the variance is
            # inversely proportional to the weight. First calculate by
            # individual, then the mean over all individuals.
            assort_vars.append(
                    (np.mean(aroundmean**2/(1./ws-1), axis=1)).mean())
            gen_vars.append(tot_vars[-1]-assort_vars[-1])
        return tot_vars, gen_vars, assort_vars

    def plot_next(self):
        self.indivs[self.currentplot].canvas.pack_forget()
        if(self.currentplot < self.nind-1):
            self.currentplot += 1
        return self.plot_indiv()

    def plot_previous(self):
        self.indivs[self.currentplot].canvas.pack_forget()
        if(self.currentplot > 0):
            self.currentplot -= 1
        return self.plot_indiv()

    def plot_indiv(self):
        self.win.title("individual %d " % (self.currentplot+1,))
        self.canv = self.indivs[self.currentplot].plot(
                self.colordict, win=self.win)

    def plot(self, colordict):
        self.colordict = colordict
        self.currentplot = 0
        self.win = Tk.Tk()#self.indivs[self.currentplot].plot(self.colordict)
        printbutton = Tk.Button(self.win, text="save to ps", command=self.save)
        printbutton.pack()

        p = Tk.Button(self.win, text="Plot previous",
                command=self.plot_previous)
        p.pack()

        b = Tk.Button(self.win, text="Plot next", command=self.plot_next)
        b.pack()
        self.plot_indiv()
        Tk.mainloop()

    def plot_chromosome(self, i, colordict, win=None):
        """plot a single chromosome across individuals"""
        self.colordict = colordict
        ls = self.list_chromosome(i)
        if (win is None):
            win = Tk.Tk()
            win.title("chromosome %d" % (i,))
        self.chro_canvas = Tk.Canvas(win, width=250, height=self.nind*30,
                bg='white')

        for j in xrange(len(ls)):
            ls[j].plot(self.chro_canvas, colordict, height=j*.25)

        self.chro_canvas.pack(expand=Tk.YES, fill=Tk.BOTH)
        Tk.mainloop()

    def plot_ancestries(self, chrom=0, npts=100,
            colordict={"CEU": 'blue', "YRI": 'red'}, cutoff=.0):
        dat = self.ancestry_per_pos(chrom=chrom, npts=npts, cutoff=cutoff)
        for pop, color in colordict.iteritems():
            for pos in dat[1]:
                try:
                    pos[0][pop]
                except KeyError:
                    pos[0][pop] = 0
                    pos[1][pop] = 0
        for pos in dat[1]:
            tot = 0
            for key in colordict.keys():
                tot += pos[0][key]
            for key in colordict.keys():
                if(pos[0][key] != 0):
                    eprint(pos[0][key], float(tot)),
                    pos[0][key] /= float(tot)
        for pop, color in colordict.iteritems():
            eprint(tot)
            pylab.figure(1)
            pylab.plot(dat[0], [pos[0][pop] for pos in dat[1]],
                    '.', color=color)
            pylab.title("Chromosome %d" % (chrom+1,))
            pylab.axis([0, dat[0][-1], 0, 1])
            pylab.figure(2)
            pylab.plot(dat[0], [100*pos[1][pop] for pos in dat[1]],
                    '.', color=color)
            pylab.title("Chromosome %d" % (chrom+1,))
            pylab.axis([0, dat[0][-1], 0, 150])

    def plot_all_ancestries(self, npts=100,
            colordict={"CEU": 'blue', "YRI": 'red'}, startfig=0, cutoff=0):
        for chrom in range(22):
            dat = self.ancestry_per_pos(chrom=chrom, npts=npts, cutoff=cutoff)

            for pop, color in colordict.iteritems():
                for pos in dat[1]:
                    try:
                        pos[0][pop]
                    except KeyError:
                        pos[0][pop] = 0
                        pos[1][pop] = 0
            for pos in dat[1]:
                tot = 0
                for key in colordict.keys():
                    tot += pos[0][key]
                for key in colordict.keys():
                    if(pos[0][key] != 0):
                        pos[0][key] /= float(tot)
            for pop, color in colordict.iteritems():
                pylab.figure(0+startfig)
                pylab.subplot(6, 4, chrom+1)
                pylab.plot(dat[0], [pos[0][pop] for pos in dat[1]], '.',
                        color=color)
                # pylab.title("Chromosome %d" % (chrom+1,))
                pylab.axis([0, dat[0][-1], 0, 1])
                pylab.figure(1+startfig)
        pylab.subplot(6,4,chrom+1)
        pylab.plot(dat[0],[100*pos[1][pop] for pos in dat[1]],'.',color=color)
        # pylab.title("Chromosome %d" % (chrom+1,))
        pylab.axis([0,dat[0][-1],0,150])

    def plot_global_tractlengths(self, colordict, npts=40, legend=True):
        flatdat = self.flatpop()
        bypop = self.collectpop(flatdat)
        self.maxLen = max(self.Ls)
        for label, tracts in bypop.iteritems():
            hdat = pylab.histogram([i.len() for i in tracts], npts)
            # note: convert to cM before plotting
            pylab.semilogy(100*(hdat[1][1:]+hdat[1][:-1])/2., hdat[0], 'o',
                    color=colordict[label], label=label)
        pylab.xlabel("length(cM)")
        pylab.ylabel("counts")
        if legend:
            pylab.legend()

    def __iter__(self):
        return self.indivs.__iter__()

    def __getitem__(self, index):
        return self.indivs[index]

class demographic_model(object):
    def __init__(self, mig, max_remaining_tracts=1e-5, max_morgans=100):
        """ Migratory model takes as an input a vector containing the migration
            proportions over the last generations. Each row is a time, each
            column is a population. row zero corresponds to the current
            generation. The migration rate at the last generation (time $T$) is
            the "founding generation" and should sum up to 1. Assume that
            non-admixed individuals have been removed.

            max_remaining_tracts is the proportion of tracts that are allowed
            to be incomplete after cutoff Lambda
            (Appendix 2 in Gravel: doi: 10.1534/genetics.112.139808)
            cutoff=1-\sum(b_i)

            max_morgans is used to impose a cutoff to the number of Markov
            transitions.  If the simulated morgan lengths of tracts in an
            infinite genome is more than max_morgans, issue a warning and stop
            generating new transitions
        """
        small=1e-10
        self.mig = mig
        (self.ngen, self.npop) = mig.shape
        self.max_remaining_tracts=max_remaining_tracts
        self.max_morgans=max_morgans
        # the total migration per generation
        self.totmig = mig.sum(axis=1)

        if abs(self.totmig[-1] - 1) > small:
            eprint("founding migration should sum up to 1. Now:", mig[-1, :],
                    "sum up to ", self.totmig[-1])
            raise ValueError("founding migration sum is not 1")

        if self.totmig[0] > small:
            eprint("migrants at last generation should be removed from",
                    "sample!")
            eprint("currently", self.totmig[0])
            raise ValueError("migrants from last generation are not removed")

        self.totmig[0] = 0

        if self.totmig[1] > small:
            eprint("migrants at penultimate generation should be removed from",
                    "sample!")
            eprint("currently", self.totmig[1])
            raise ValueError(
                    "migrants from penultimate generation are not removed")

        if ((self.totmig > 1).any() or (mig < 0).any()):
            eprint("migration rates should be between 0 and 1")
            eprint("currently", mig)
            raise ValueError("mig")
        if (mig[:-1] == 1).any():
            eprint("warning: population was completely replaced after",
                    "founding event")

        # identify states where migration occurred as these are the relevant
        # states in our Markov model. Each state is a tuple of the form:
        # (generation, population)

        self.states = map(tuple, np.array(mig.nonzero()).transpose())
        self.nstates = len(self.states)
        self.npops = mig.shape[1]

        # get the equilibrium distribution in each state
        self.equil = np.zeros(self.nstates)
        self.stateINpop = [[] for pop in range(self.npops)]
        self.stateOUTpop = [[] for pop in range(self.npops)]

        for i, state in enumerate(self.states):
            self.stateINpop[state[1]].append(i)
            for other in range(1, self.npops+1):
                self.stateOUTpop[(state[1]+other)%self.npops].append(i)
            self.equil[i] = mig[state]*(1-self.totmig)[1:state[0]].prod()

        self.equil /= self.equil.sum()

        # calculate the ancestry proportions as a function of time
        self.proportions = np.zeros(mig.shape)

        # could be optimized using array operations and precomputing survivals

        for pop in range(self.npop):
            for time in range(self.ngen):
                for g in range(time, self.ngen):
                    self.proportions[time, pop] += \
                        mig[g, pop]*(1-self.totmig)[time:g].prod()

        # calculate the transition matrix

        self.dicTpopTau = {}

        # we could precompute prod
        for (t, pop) in self.states:
            for tau in range(t):
                prod = (1-self.totmig)[tau+1:t].prod()
                self.dicTpopTau[(t, pop, tau)] = mig[t, pop]*prod

        # for t in range(self.T):
        #     for tau in range(t)
        #         prod=(1-self.totmig)[tau:t].prod()
        #         for pop in range(self.npop):
        #             self.dicTpopTau[(t,pop,tau)]=mig[t,pop]*prod

        # self.mat=np.zeros(((self.T-1)*self.npop,(self.T-1)*self.npop))
        # we do not consider last-generation migrants! We could trim one row
        # and column from the transition matrix.
        # for popp in range(self.npop):
        #     for t in range(1,self.T):
        #         for tp in range(1,self.T):
        #             tot=0
        #             for tau in range(min(t,tp)):
        #                 tot+=self.dicTpopTau[(tp,popp,tau)]
        #             for pop in range(self.npop):
        #                 self.mat[self.tpToPos(t - 1, pop),
        #                         self.tpToPos(tp - 1, popp)] = tot

        self.mat = np.zeros((len(self.states), len(self.states)))
        for nump, (tp, popp) in enumerate(self.states):
            for num, (t, pop) in enumerate(self.states):
                tot = 0
                for tau in range(1, min(t, tp)):
                    tot += self.dicTpopTau[(tp, popp, tau)]
                self.mat[num, nump] = tot

        # note that the matrix could be uniformized in a population-specific
        # way, for optimization purposes
        self.uniformizemat()
        self.ndists = []
        for i in range(self.npops):
            self.ndists.append(self.popNdist(i)) #the distribution of the number of steps
            #required to reach either the length of the genome, or equilibrium
        self.switchdensity()

    def gen_variance(self, popnum):
        """ 1. Calculate the expected genealogy variance in the model. Need to
            double-check +-1s.
            2. Calculate the e(d)
            3. Generations go from 0 to self.ngen-1.
            """
        legterm = [self.proportions[self.ngen - d, popnum]**2 * \
                np.prod(1 - self.totmig[:(self.ngen - d)])
                for d in range(1, self.ngen)]
        trunkterm = [np.sum([self.mig[u, popnum] * \
                np.prod(1 - self.totmig[:u])
                for u in range(self.ngen-d)])
                for d in range(1, self.ngen)]

        # Now calculate the actual variance.
        return np.sum([2**(d-self.ngen) * (legterm[d-1]+trunkterm[d-1])\
            for d in range(1, self.ngen)]) +\
                    self.proportions[0, popnum] *\
                    (1/2.**(self.ngen-1)-self.proportions[0, popnum])

    def uniformizemat(self):
        """ Uniformize the transition matrix so that each state has the same
            total transition rate. """
        self.unifmat = self.mat.copy()
        lmat = len(self.mat)
        # identify the highest non-self total transition rate
        outgoing = (self.mat - np.diag(self.mat.diagonal())).sum(axis=1)

        self.maxrate = outgoing.max() #max outgoing rate
        #reset the self-transition rate. We forget about the prior self-transition rates,
        #as they do not matter for the trajectories.
        for i in range(lmat):
            self.unifmat[i, i] = self.maxrate-outgoing[i]
        self.unifmat /= self.maxrate

    def popNdist(self, pop):
        """ Calculate the distribution of number of steps before exiting
            population. """
        if len(self.stateINpop[pop]) == 0:
            return []
        # get the equilibrium distribution in tracts OUTSIDE pop.
        tempequil = self.equil.copy()
        tempequil[self.stateINpop[pop]] = 0
        # Apply one evolution step
        new = np.dot(tempequil, self.unifmat)
        # select states in relevant population

        newrest = new[self.stateINpop[pop]]
        newrest /= newrest.sum() #normalize probabilities
        # reduce the matrix to apply only to states of current population
        shortmat = self.unifmat[
                np.meshgrid(self.stateINpop[pop],
                    self.stateINpop[pop])].transpose()

        # calculate the amount that fall out of the state
        escapes = 1 - shortmat.sum(axis=1)
        # decide on the number of iterations



        #if the expected length of tracts becomes more than maxmorgans,
        #bail our and issue warning.
        nit = int(self.max_morgans* self.maxrate)

        nDistribution = []
        for i in range(nit): #nit is the max number of iterations.
            #will exit loop earlier if tracts are all complete.
            nDistribution.append(np.dot(escapes, newrest))
            newrest = np.dot(newrest, shortmat)
            if newrest.sum()<self.max_remaining_tracts:#we stop when there are at most
            #we request that the proportions of tracts that are still
            #incomplete be at most self.cutoff tracts left
                break
        if newrest.sum()>self.max_remaining_tracts:
            print("Warning: After %d time steps, %f of tracts are incomplete" %
                    (nit,newrest.sum()))
            print("This can happen when one population has really long",
                    "tracts.")


        nDistribution.append(newrest.sum())
        return nDistribution

    def Erlang(self, i, x, T):
        if i > 10:
            lg = i*np.log(T)+(i-1)*np.log(x)-T*x-gammaln(i)
            return np.exp(lg)
        return T**i*x**(i - 1)*np.exp(- T*x)/factorial(i - 1)

    def inners(self, L, x, pop):
        """ Calculate the length distribution of tract lengths not hitting a
            chromosome edge. """
        if(x > L):
            return 0
        else:
            return np.sum(
                    self.ndists[pop][i] *
                        (L-x) *
                        self.Erlang(i+1, x, self.maxrate)
                        for i in range(len(self.ndists[pop])))

    def outers(self, L, x, pop):
        """ Calculate the length distribution of tract lengths hitting a single
            chromosome edge. """
        if(x > L):
            return 0
        else:
            nd = self.ndists[pop]
            mx = self.maxrate * x
            return 2 * np.sum(
                    nd[i] * (1 - gammainc(i+1, mx))
                    for i in xrange(
                        len(nd))
            ) + 2 * (1-np.sum(nd))

    def full(self, L, pop):
        """ The expected fraction of full-chromosome tracts, p. 63 May 24,
            2011. """
        return np.sum(
                self.ndists[pop][i] * (((i+1) / float(self.maxrate) - L) +
                    L * gammainc(i + 1, self.maxrate * L) -
                    float(i+1) / self.maxrate * gammainc(i+2, self.maxrate*L))
                for i in xrange(len(self.ndists[pop]))
        ) + (1 - np.sum(self.ndists[pop])) * \
                (len(self.ndists[pop])/self.maxrate - L)

    def Z(self, L, pop):
        """the normalizing factor, to ensure that the tract density is 1."""
        return L + np.sum(
                self.ndists[pop][i]*(i+1)/self.maxrate
                for i in xrange(len(self.ndists[pop]))
        ) + (1 - np.sum([self.ndists[pop]])) * \
                len(self.ndists[pop])/self.maxrate

    def switchdensity(self):
        """ Calculate the density of ancestry switchpoints per morgan in our
            model. """
        self.switchdensities = np.zeros((self.npops, self.npops))
        # could optimize by precomputing survivals earlier
        self.survivals = [(1 - self.totmig[:i]).prod()
                for i in range(self.ngen)]
        for pop1 in range(self.npops):
            for pop2 in range(pop1):
                self.switchdensities[pop1, pop2] = \
                        np.sum(
                                [2 * self.proportions[i+1, pop1] * \
                                        self.proportions[i+1, pop2]* \
                                        self.survivals[i+1]
                                    for i in range(1, self.ngen-1)])
                self.switchdensities[pop2, pop1] = \
                        self.switchdensities[pop1, pop2]

        self.totSwitchDens = self.switchdensities.sum(axis=1)

    def expectperbin(self, Ls, pop, bins):
        """ The expected number of tracts per bin for a diploid individual with
            distribution of chromosome lengths given by Ls. The bin should be a
            list with n+1 breakpoints for n bins. We will always add an extra
            value for the full chromosomes as an extra bin at the end. The last
            bin should not go beyond the end of the longest chromosome. For
            now, perform poor man's integral by using the bin midpoint value
            times width. """
        self.totalPerInd = \
                [L*self.totSwitchDens[pop]+2.*self.proportions[0, pop]
                        for L in Ls]
        self.totalfull = \
                np.sum(
                        [(L*self.totSwitchDens[pop]+2. * \
                                self.proportions[0, pop]) * \
                                self.full(L, pop)/self.Z(L, pop)
                            for L in Ls])
        lsval = []
        for binNum in range(len(bins) - 1):
            mid = (bins[binNum] + bins[binNum+1]) / 2.
            val = np.sum(
                    [(L*self.totSwitchDens[pop] + \
                            2. * self.proportions[0, pop]) * \
                            (self.inners(L, mid, pop) + \
                                self.outers(L, mid, pop)) / self.Z(L, pop)
                        for L in Ls]) \
                    * (bins[binNum+1] - bins[binNum])
            lsval.append(max(val, 1e-17))

        lsval.append(max(self.totalfull, 1e-17))
        return lsval

    def random_realization(self, Ls, bins, nind):
        expect = []
        for pop in range(self.npops):
            expect.append(
                    np.random.poisson(
                        nind * np.array(self.expectperbin(Ls, pop, bins))))
        return expect

    def loglik(self, bins, Ls, data, nsamp, cutoff=0):
        """ Calculate the maximum-likelihood in a Poisson Random Field. Last
            bin of data is the number of whole-chromosome. """
        self.maxLen = max(Ls)
        # define bins that contain all possible values
        # bins=np.arange(0,self.maxLen+1./2./float(npts),self.maxLen/float(npts))
        ll = 0
        if np.sum(data)>1./self.max_remaining_tracts:
            eprint("warning: the convergence criterion max_remining_tracts",
                    "may be too high, tracts calculates the distribution",
                    "of tract lengths from the shortest to the longest,",
                    "and uses approximations after a fraction",
                    "1-max_remining_tracts of all tracts have been",
                    "accounted for. Since we have a total of",
                    np.sum(data), "we'd be underestimating the length of",
                    "the longest ",
                    np.sum(data) * self.max_remaining_tracts, " tracts.")

        for pop in range(self.npops):
            models = self.expectperbin(Ls, pop, bins)
            for binnum in range(cutoff, len(bins)-1):
                dat = data[pop][binnum]
                ll += -nsamp*models[binnum] + \
                        dat*np.log(nsamp*models[binnum]) - \
                        gammaln(dat + 1.)
        return ll

    def loglik_biascorrect(self, bins, Ls, data, nsamp, cutoff=0,
            biascorrect=True):
        """ Calculates the maximum-likelihood in a Poisson Random Field. Last
            bin of data is the number of whole-chromosome. Compares the model
            to the first bins, and simulates the addition (or removal) of the
            corresponding tracts.
            """
        self.maxLen = max(Ls)

        mods = []
        for pop in range(self.npops):
            mods.append(nsamp*np.array(self.expectperbin(Ls, pop, bins)))

        if biascorrect:
            if self.npops != 2:
                eprint("bias correction not implemented for more than 2",
                        "populations")
                sys.exit()
            cbypop = []
            for pop in range(self.npops):
                mod = mods[pop]
                corr = []
                for binnum in range(cutoff):
                    diff = mod[binnum]-data[pop][binnum]
                    lg = ((bins[binnum]+bins[binnum+1]))/2;

                    corr.append((lg, diff))
                eprint(corr)
                cbypop.append(corr)
            for pop in range(self.npops):
                # total length in tracts
                tot = np.sum([bins[i]*data[pop][i]
                    for i in range(cutoff, len(bins))])
                # probability that a given tract is hit by a given "extra short
                # tracts"
                probs = [bins[i]/tot for i in range(cutoff, len(bins))]
                eprint("tot", tot)
                eprint("probs", probs)
                for shortbin in range(cutoff):
                    transfermat = np.zeros(
                            (len(bins)-cutoff, len(bins)-cutoff))
                    corr = cbypop[1-pop][shortbin]
                    if corr[1] > 0:
                        eprint("correction for lack of short tracts not",
                                "implemented!")
                        sys.exit()
                    for lbin in range(len(bins)-cutoff):
                        eprint("corr[1]", corr[1])
                        transfermat[lbin, lbin] = 1+corr[1]*probs[lbin-cutoff]
                eprint(transfermat)

                # count the number of missing bits in each population.
                eprint("population ", pop, " ", cbypop[pop])

        # define bins that contain all possible values
        # bins=np.arange(0,self.maxLen+1./2./float(npts),self.maxLen/float(npts))
        ll = 0
        for pop in range(self.npops):
            models = mods[pop]
            for binnum in range(cutoff, len(bins)-1):
                dat = data[pop][binnum]
                ll += -nsamp*models[binnum] + \
                        dat*np.log(nsamp*models[binnum]) - \
                        gammaln(dat + 1.)
        return ll

    def plot_model_data(self, Ls, bins, data, nsamp, pop, colordict):
        # plot the migration model with the data
        pop.plot_global_tractlengths(colordict)
        for pop in range(len(data)):
            pylab.plot(
                    100*np.array(bins),
                    nsamp*np.array(self.expectperbin(Ls, 0, bins)))

class composite_demographic_model(object):
    """ The class of demographic models that account for variance in the number
        of ancestors of individuals of the underlying population.

        Specifically, this is the demographic model constructed by the
        "multifracs" family of optimization routines.

        The expected tract counts per bin in the composite demographic model is
        simply a component-wise sum of the expected tract counts per bin across
        the component demographic models.

        The log-likelihood of the composite demographic model is the computed
        based on the combined expected tract counts per bin.
    """
    def __init__(self, model_function, parameters, proportions_list):
        """ Construct a composite demographic model, in which we consider split
            groups of individuals.

            Arguments;
                model_function (callable):
                    A function that produces a migration matrix given some
                    model parameters and fixed ancestry proportions.
                parameters:
                    The parameters given to the model function when the
                    component demographic models are built.
                proportions_list:
                    The lists of ancestry proportions used to construc each
                    component demographic model.
        """
        self.model_function = model_function
        self.parameters = parameters
        self.proportions_list = proportions_list

        # build the component models
        self.models = [
                demographic_model(model_function(parameters, props))
                for props in proportions_list]

        self.npops = self.models[0].npops

    def loglik(self, bins, Ls, data_list, nsamp_list, cutoff=0):
        """ Evaluate the log-likelihood of the composite demographic model.

            To compute the log-likelihood, we combine the expected count of
            tracts per bin in each of the component demographic models into the
            composite expected counts per bin. The expected counts per bin are
            compared with the sum across subgroups of the actual counts per
            bin. This gives a likelihood that is directly comparable with the
            likelihoods of the component demographic models.

            See demographic_model.loglik for more information about the
            specifics of the log-likelihood calculation.
        """
        maxlen = max(Ls)
        data = sum(np.array(d) for d in data_list)

        s = 0

        for i in xrange(self.npops):
            expects = self.expectperbin(Ls, i, bins, nsamp_list=nsamp_list)
            for j in xrange(cutoff, len(bins) - 1):
                dat = data[i][j]
                s += -expects[j] + dat * np.log(expects[j]) - gammaln(dat + 1.)

        return s

    def expectperbin(self, Ls, pop, bins, nsamp_list=None):
        """ A wrapper for demographic_model.expectperbin that yields a
            component-wise sum of the counts per bin in the underlying
            demographic models.
            Since the counts given by the demographic_model.expectperbin are
            normalized, performing a simple sum of the counts is not
            particularly meaningful; it throws away some of the structure
            that we have gained by using a composite model.
            Hence, the nsamp_list parameter allows for specifying the
            count of individuals in each of the groups represented by this
            composite_demographic_model, which is then used to rescale the
            counts reported by the expectperbin of the compoennt demographic
            models.
        """
        if nsamp_list is None:
            nsamp_list = [1 for _ in xrange(len(self.proportions_list[0]))]

        return sum(
                nsamp * np.array(mod.expectperbin(Ls, pop, bins))
                for nsamp, mod in zip(nsamp_list, self.models))

    def migs(self):
        """ Get the list of migration matrices of the component demographic
            models.
            This method merely projects the "mig" attribute from the component
            models.
        """
        return [m.mig for m in self.models]


def plotmig(mig,
        colordict={'CEU': 'red', 'NAH': 'orange', 'NAT': 'orange',
            'UNKNOWN': 'gray', 'YRI': 'blue'},
        order=['CEU', 'NAT', 'YRI']):
    pylab.figure()
    axes = pylab.axes()
    shape = mig.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            c = pylab.Circle(
                    (j, i),
                    radius=np.sqrt(mig[i, j]) / 1.7,
                    color=colordict[order[j]])
            axes.add_patch(c)
    pylab.axis('scaled')
    pylab.ylabel("generations from present")

def optimize(p0, bins, Ls, data, nsamp, model_func, outofbounds_fun=None,
        cutoff=0, verbose=0, flush_delay=0.5, epsilon=1e-3, gtol=1e-5,
        maxiter=None, full_output=True, func_args=[], fixed_params=None,
        ll_scale=1):
    """
    Optimize params to fit model to data using the BFGS method.

    This optimization method works well when we start reasonably close to the
    optimum. It is best at burrowing down a single minimum.

    It should also perform better when parameters range over scales.

    p0:
        Initial parameters.
    data:
        Spectrum with data.
    model_function:
        Function to evaluate model spectrum. Should take arguments (params,
        pts)
    out_of_bounds_fun:
        A funtion evaluating to True if the current parameters are in a
        forbidden region.
    cutoff:
        the number of bins to drop at the beginning of the array. This could be
        achieved with masks.
    verbose:
        If greater than zero, print optimization status every <verbose> steps.
    flush_delay:
        Standard output will be flushed once every <flush_delay> minutes. This
        is useful to avoid overloading I/O on clusters.
    epsilon:
        Step-size to use for finite-difference derivatives.
    gtol:
        Convergence criterion for optimization. For more info, see
                 help(scipy.optimize.fmin_bfgs)
    maxiter:
        Maximum iterations to run for.
    full_output:
        If True, return full outputs as described in help.
        (scipy.optimize.fmin_bfgs)
    func_args:
        Additional arguments to model_func. It is assumed that model_func's
        first argument is an array of parameters to optimize, that its second
        argument is an array of sample sizes for the sfs, and that its last
        argument is the list of grid points to use in evaluation.
    fixed_params:
        (Not yet implemented)
        If not None, should be a list used to fix model parameters at
        particular values. For example, if the model parameters are
        (nu1,nu2,T,m), then fixed_params = [0.5,None,None,2] will hold nu1=0.5
        and m=2. The optimizer will only change T and m. Note that the bounds
        lists must include all parameters. Optimization will fail if the fixed
        values lie outside their bounds. A full-length p0 should be passed in;
        values corresponding to fixed parameters are ignored.
    ll_scale:
        The bfgs algorithm may fail if your initial log-likelihood is too
        large. (This appears to be a flaw in the scipy implementation.) To
        overcome this, pass ll_scale > 1, which will simply reduce the
        magnitude of the log-likelihood. Once in a region of reasonable
        likelihood, you'll probably want to re-optimize with ll_scale=1.
    """
    args = ( bins, Ls, data, nsamp, model_func,
                 outofbounds_fun, cutoff,
                 verbose, flush_delay, func_args)

    if fixed_params is not None:
        raise ValueError("fixed parameters not implemented in optimize_bfgs")

    outputs = scipy.optimize.fmin_bfgs(_object_func,
            p0, epsilon=epsilon,
            args = args, gtol=gtol,
            full_output=full_output,
            disp=False,
            maxiter=maxiter)
    xopt, fopt, gopt, Bopt, func_calls, grad_calls, warnflag = outputs

    if not full_output:
        return xopt
    else:
        return xopt, fopt, gopt, Bopt, func_calls, grad_calls, warnflag

optimize_bfgs = optimize

def optimize_cob(p0, bins, Ls, data, nsamp, model_func, outofbounds_fun=None,
        cutoff=0, verbose=0, flush_delay=0.5, epsilon=1e-3, gtol=1e-5,
        maxiter=None, full_output=True, func_args=[], fixed_params=None,
        ll_scale=1):
    """
    Optimize params to fit model to data using the cobyla method.

    This optimization method works well when we start reasonably close to the
    optimum. It is best at burrowing down a single minimum.

    It should also perform better when parameters range over scales.

    p0:
        Initial parameters.
    data:
        Spectrum with data.
    model_function:
        Function to evaluate model spectrum. Should take arguments (params,
        pts)
    out_of_bounds_fun:
        A funtion evaluating to True if the current parameters are in a
        forbidden region.
    cutoff:
        the number of bins to drop at the beginning of the array. This could be
        achieved with masks.
    verbose:
        If > 0, print optimization status every <verbose> steps.
    flush_delay:
        Standard output will be flushed once every <flush_delay> minutes. This
        is useful to avoid overloading I/O on clusters.
    epsilon:
        Step-size to use for finite-difference derivatives.
    gtol:
        Convergence criterion for optimization. For more info, see
                 help(scipy.optimize.fmin_bfgs)
    maxiter:
        Maximum iterations to run for.
    full_output:
        If True, return full outputs as in described in
        help(scipy.optimize.fmin_bfgs)
    func_args:
        Additional arguments to model_func. It is assumed that model_func's
        first argument is an array of parameters to optimize, that its second
        argument is an array of sample sizes for the sfs, and that its last
        argument is the list of grid points to use in evaluation.
    fixed_params:
        If not None, should be a list used to fix model parameters at
        particular values. For example, if the model parameters are
        (nu1,nu2,T,m), then fixed_params = [0.5,None,None,2] will hold nu1=0.5
        and m=2. The optimizer will only change T and m. Note that the bounds
        lists must include all parameters. Optimization will fail if the fixed
        values lie outside their bounds. A full-length p0 should be passed in;
        values corresponding to fixed parameters are ignored.
    ll_scale:
        The bfgs algorithm may fail if your initial log-likelihood is too
        large. (This appears to be a flaw in the scipy implementation.) To
        overcome this, pass ll_scale > 1, which will simply reduce the
        magnitude of the log-likelihood. Once in a region of reasonable
        likelihood, you'll probably want to re-optimize with ll_scale=1.
    """
    fun = lambda x: _object_func(x, bins, Ls, data, nsamp, model_func,
                outofbounds_fun=outofbounds_fun, cutoff=cutoff,
                verbose=verbose, flush_delay=flush_delay,
                func_args=func_args)


    outputs = scipy.optimize.fmin_cobyla(
            fun, p0, outofbounds_fun, rhobeg=.01, rhoend=.0001, maxfun=maxiter)

    return outputs

def optimize_slsqp(p0, bins, Ls, data, nsamp, model_func, outofbounds_fun=None,
        cutoff=0, bounds=[], verbose=0, flush_delay=0.5, epsilon=1e-3,
        gtol=1e-5, maxiter=None, full_output=True, func_args=[],
        fixed_params=None, ll_scale=1):
    """
    Optimize params to fit model to data using the slsq method.

    This optimization method works well when we start reasonably close to the
    optimum. It is best at burrowing down a single minimum.

    It should also perform better when parameters range over scales.

    p0:
        Initial parameters.
    data:
        Spectrum with data.
    model_function:
        Function to evaluate model spectrum. Should take arguments (params,
        pts)
    out_of_bounds_fun:
        A funtion evaluating to True if the current parameters are in a
        forbidden region.
    cutoff:
        the number of bins to drop at the beginning of the array. This could be
        achieved with masks.
    verbose:
        If > 0, print optimization status every <verbose> steps.
    flush_delay:
        Standard output will be flushed once every <flush_delay> minutes. This
        is useful to avoid overloading I/O on clusters.
    epsilon:
        Step-size to use for finite-difference derivatives.
    gtol:
        Convergence criterion for optimization. For more info, see
                 help(scipy.optimize.fmin_bfgs)
    maxiter:
        Maximum iterations to run for.
    full_output:
        If True, return full outputs as in described in
        help(scipy.optimize.fmin_bfgs)
    func_args:
        Additional arguments to model_func. It is assumed that model_func's
        first argument is an array of parameters to optimize, that its second
        argument is an array of sample sizes for the sfs, and that its last
        argument is the list of grid points to use in evaluation.
    fixed_params:
        If not None, should be a list used to fix model parameters at
        particular values. For example, if the model parameters are
        (nu1,nu2,T,m), then fixed_params = [0.5,None,None,2] will hold nu1=0.5
        and m=2. The optimizer will only change T and m. Note that the bounds
        lists must include all parameters. Optimization will fail if the fixed
        values lie outside their bounds. A full-length p0 should be passed in;
        values corresponding to fixed parameters are ignored.
    ll_scale:
        The bfgs algorithm may fail if your initial log-likelihood is too
        large. (This appears to be a flaw in the scipy implementation.) To
        overcome this, pass ll_scale > 1, which will simply reduce the
        magnitude of the log-likelihood. Once in a region of reasonable
        likelihood, you'll probably want to re-optimize with ll_scale=1.
    """
    args = ( bins, Ls, data, nsamp, model_func,
                outofbounds_fun, cutoff,
                verbose, flush_delay, func_args)

    def onearg(a, *args):
        return outofbounds_fun(a)

    if maxiter is None:
        maxiter = 100

    outputs = scipy.optimize.fmin_slsqp(_object_func,
            p0, ieqcons=[onearg], bounds=bounds,
            args = args,
            iter=maxiter, acc=1e-4, epsilon=1e-4)

    return outputs
    # xopt, fopt, gopt, Bopt, func_calls, grad_calls, warnflag = outputs
    # xopt = _project_params_up(np.exp(xopt), fixed_params)
    #
    # if not full_output:
    #    return xopt
    # else:
    #    return xopt, fopt, gopt, Bopt, func_calls, grad_calls, warnflag

def _project_params_down(pin, fixed_params):
    """ Eliminate fixed parameters from pin. Copied from Dadi (Gutenkunst et
        al., PLoS Genetics, 2009). """
    if fixed_params is None:
        return pin

    if len(pin) != len(fixed_params):
        raise ValueError('fixed_params list must have same length as input '
                        'parameter array.')

    pout = []
    for ii, (curr_val, fixed_val) in enumerate(zip(pin, fixed_params)):
        if fixed_val is None:
            pout.append(curr_val)

    return np.array(pout)

def _project_params_up(pin, fixed_params):
    """ Fold fixed parameters into pin. Copied from Dadi (Gutenkunst et al.,
        PLoS Genetics, 2009). """
    if fixed_params is None:
        return pin

    pout = np.zeros(len(fixed_params))
    orig_ii = 0
    for out_ii, val in enumerate(fixed_params):
        if val is None:
            pout[out_ii] = pin[orig_ii]
            orig_ii += 1
        else:
            pout[out_ii] = fixed_params[out_ii]
    return pout

#: Counts calls to object_func
_counter = 0
# calculate the log-likelihood value for tract length data.
def _object_func(params, bins, Ls, data, nsamp, model_func,
        outofbounds_fun=None, cutoff=0, verbose=0, flush_delay=0,
        func_args=[]):
    _out_of_bounds_val = -1e32
    global _counter
    _counter += 1

    if outofbounds_fun is not None:
        # outofbounds can return either True or a negative valueto signify out-of-boundedness.
        ooa = outofbounds_fun(params)
        if ooa < 0:
            result = -(ooa-1)*_out_of_bounds_val
        else:
            mod = demographic_model(model_func(params))
            result = mod.loglik(bins, Ls, data, nsamp, cutoff=cutoff)
    else:
        eprint("No bound function defined")
        mod = demographic_model(model_func(params))
        result = mod.loglik(bins, Ls, data, nsamp, cutoff=cutoff)

    if True:#(verbose > 0) and (_counter % verbose == 0):
        param_str = 'array([%s])' % (', '.join(['%- 12g'%v for v in params]))
        eprint('%-8i, %-12g, %s' % (_counter, result, param_str))
        # Misc.delayed_flush(delay=flush_delay)

    return -result

# define the optimization routine for when the final ancestry proportions are
# specified.
def optimize_cob_fracs(p0, bins, Ls, data, nsamp, model_func, fracs,
        outofbounds_fun=None, cutoff=0, verbose=0, flush_delay=0.5,
        epsilon=1e-3, gtol=1e-5, maxiter=None, full_output=True, func_args=[],
        fixed_params=None, ll_scale=1):
    """
    Optimize params to fit model to data using the COBYLA method.

    This optimization method works well when we start reasonably close to the
    optimum. It is best at burrowing down a single minimum.

    It should also perform better when parameters range over scales.

    p0: Initial parameters.
    data: Spectrum with data.
    model_function: Function to evaluate model spectrum. Should take arguments
                    (params, pts)
    out_of_bounds_fun: A funtion evaluating to True if the current parameters are in a forbidden region.
    cutoff: the number of bins to drop at the beginning of the array. This could be achieved with masks.

    verbose: If > 0, print optimization status every <verbose> steps.
    flush_delay: Standard output will be flushed once every <flush_delay>
                 minutes. This is useful to avoid overloading I/O on clusters.
    epsilon: Step-size to use for finite-difference derivatives.
    gtol: Convergence criterion for optimization. For more info,
          see help(scipy.optimize.fmin_bfgs)

    maxiter: Maximum iterations to run for.
    full_output: If True, return full outputs as in described in
                 help(scipy.optimize.fmin_bfgs)
    func_args: Additional arguments to model_func. It is assumed that
               model_func's first argument is an array of parameters to
               optimize, that its second argument is an array of sample sizes
               for the sfs, and that its last argument is the list of grid
               points to use in evaluation.
    fixed_params: If not None, should be a list used to fix model parameters at
                  particular values. For example, if the model parameters
                  are (nu1,nu2,T,m), then fixed_params = [0.5,None,None,2]
                  will hold nu1=0.5 and m=2. The optimizer will only change
                  T and m. Note that the bounds lists must include all
                  parameters. Optimization will fail if the fixed values
                  lie outside their bounds. A full-length p0 should be passed
                  in; values corresponding to fixed parameters are ignored.
    ll_scale: The bfgs algorithm may fail if your initial log-likelihood is
              too large. (This appears to be a flaw in the scipy
              implementation.) To overcome this, pass ll_scale > 1, which will
              simply reduce the magnitude of the log-likelihood. Once in a
              region of reasonable likelihood, you'll probably want to
              re-optimize with ll_scale=1.
    """
    args = ( bins, Ls, data, nsamp, model_func, fracs,
                outofbounds_fun, cutoff,
                verbose, flush_delay, func_args)


    outfun = lambda x:outofbounds_fun(x, fracs)


    outputs = scipy.optimize.fmin_cobyla(_object_func_fracs,
            p0, outfun, rhobeg=.01, rhoend=.001,
            args = args,
            maxfun=maxiter)

    return outputs

def optimize_cob_fracs2(p0, bins, Ls, data, nsamp, model_func, fracs,
        outofbounds_fun=None, cutoff=0, verbose=0, flush_delay=0.5,
        epsilon=1e-3, gtol=1e-5, maxiter=None, full_output=True, func_args=[],
        fixed_params=None, ll_scale=1):
    """
    Optimize params to fit model to data using the cobyla method.

    This optimization method works well when we start reasonably close to the
    optimum. It is best at burrowing down a single minimum.


    It should also perform better when parameters range over scales.

    p0: Initial parameters.
    data: Spectrum with data.
    model_function: Function to evaluate model spectrum. Should take arguments
                    (params, pts)
    out_of_bounds_fun: A funtion evaluating to True if the current parameters are in a forbidden region.
    cutoff: the number of bins to drop at the beginning of the array. This could be achieved with masks.

    verbose: If > 0, print optimization status every <verbose> steps.
    flush_delay: Standard output will be flushed once every <flush_delay>
                 minutes. This is useful to avoid overloading I/O on clusters.
    epsilon: Step-size to use for finite-difference derivatives.
    gtol: Convergence criterion for optimization. For more info,
          see help(scipy.optimize.fmin_bfgs)

    maxiter: Maximum iterations to run for.
    full_output: If True, return full outputs as in described in
                 help(scipy.optimize.fmin_bfgs)
    func_args: Additional arguments to model_func. It is assumed that
               model_func's first argument is an array of parameters to
               optimize, that its second argument is an array of sample sizes
               for the sfs, and that its last argument is the list of grid
               points to use in evaluation.
    fixed_params: If not None, should be a list used to fix model parameters at
                  particular values. For example, if the model parameters
                  are (nu1,nu2,T,m), then fixed_params = [0.5,None,None,2]
                  will hold nu1=0.5 and m=2. The optimizer will only change
                  T and m. Note that the bounds lists must include all
                  parameters. Optimization will fail if the fixed values
                  lie outside their bounds. A full-length p0 should be passed
                  in; values corresponding to fixed parameters are ignored.
    ll_scale: The bfgs algorithm may fail if your initial log-likelihood is
              too large. (This appears to be a flaw in the scipy
              implementation.) To overcome this, pass ll_scale > 1, which will
              simply reduce the magnitude of the log-likelihood. Once in a
              region of reasonable likelihood, you'll probably want to
              re-optimize with ll_scale=1.
    """
    def outfun(p0,verbose=False):
        # cobyla uses the constraint function and feeds it the reduced
        # parameters. Hence we have to project back up first
        x0 = _project_params_up(p0, fixed_params)
        if verbose:
            eprint("p0", p0)
            eprint("x0", x0)
            eprint("fracs", fracs)
            eprint("res", outofbounds_fun(p0, fracs))

        return outofbounds_fun(x0, fracs)

    modstrip = lambda x: model_func(x, fracs)

    fun = lambda x: _object_func_fracs2(x, bins, Ls, data, nsamp, modstrip,
            outofbounds_fun=outfun, cutoff=cutoff,
            verbose=verbose, flush_delay=flush_delay,
            func_args=func_args, fixed_params=fixed_params)

    p0 = _project_params_down(p0, fixed_params)
    outputs = scipy.optimize.fmin_cobyla(fun, p0, outfun, rhobeg=.01,
            rhoend=.001, maxfun=maxiter)
    xopt = _project_params_up(outputs, fixed_params)

    return xopt

def optimize_cob_multifracs(
        p0, bins, Ls, data_list, nsamp_list, model_func, fracs_list,
        outofbounds_fun=None, cutoff=0, verbose=0, flush_delay=0.5,
        epsilon=1e-3, gtol=1e-5, maxiter=None, full_output=True, func_args=[],
        fixed_params=None, ll_scale=1):
    """
    Optimize params to fit model to data using the cobyla method.

    This optimization method works well when we start reasonably close to the
    optimum. It is best at burrowing down a single minimum.


    It should also perform better when parameters range over scales.

    p0: Initial parameters.
    data: Spectrum with data.
    model_function: Function to evaluate model spectrum. Should take arguments
                    (params, pts)
    out_of_bounds_fun: A funtion evaluating to True if the current parameters are in a forbidden region.
    cutoff: the number of bins to drop at the beginning of the array. This could be achieved with masks.

    verbose: If > 0, print optimization status every <verbose> steps.
    flush_delay: Standard output will be flushed once every <flush_delay>
                 minutes. This is useful to avoid overloading I/O on clusters.
    epsilon: Step-size to use for finite-difference derivatives.
    gtol: Convergence criterion for optimization. For more info,
          see help(scipy.optimize.fmin_bfgs)

    maxiter: Maximum iterations to run for.
    full_output: If True, return full outputs as in described in
                 help(scipy.optimize.fmin_bfgs)
    func_args: Additional arguments to model_func. It is assumed that
               model_func's first argument is an array of parameters to
               optimize, that its second argument is an array of sample sizes
               for the sfs, and that its last argument is the list of grid
               points to use in evaluation.
    fixed_params: If not None, should be a list used to fix model parameters at
                  particular values. For example, if the model parameters
                  are (nu1,nu2,T,m), then fixed_params = [0.5,None,None,2]
                  will hold nu1=0.5 and m=2. The optimizer will only change
                  T and m. Note that the bounds lists must include all
                  parameters. Optimization will fail if the fixed values
                  lie outside their bounds. A full-length p0 should be passed
                  in; values corresponding to fixed parameters are ignored.
    ll_scale: The bfgs algorithm may fail if your initial log-likelihood is
              too large. (This appears to be a flaw in the scipy
              implementation.) To overcome this, pass ll_scale > 1, which will
              simply reduce the magnitude of the log-likelihood. Once in a
              region of reasonable likelihood, you'll probably want to
              re-optimize with ll_scale=1.
    """
    # Now we iterate over each set of ancestry proportions in the list, and
    # construct the outofbounds functions and the model functions, storing
    # each into the empty lists defined above.
    # construct the out of bounds function.
    def outfun(p0, fracs, verbose=False):
        # cobyla uses the constraint function and feeds it the reduced
        # parameters. Hence we have to project back up first
        x0 = _project_params_up(p0, fixed_params)
        if verbose:
            eprint("p0", p0)
            eprint("x0", x0)
            eprint("fracs", fracs)
            eprint("res", outofbounds_fun(p0, fracs))

        return outofbounds_fun(x0, fracs)

    # construct the objective function. The input x is wrapped in the
    # function r constructed above.
    objfun = lambda x: _object_func_multifracs(
            x, bins, Ls, data_list, nsamp_list, model_func, fracs_list,
            outofbounds_fun=outfun, cutoff=cutoff, verbose=verbose,
            flush_delay=flush_delay, func_args=func_args,
            fixed_params=fixed_params)

    composite_outfun = lambda x: min(
            outfun(x, frac) for frac in fracs_list)

    p0 = _project_params_down(p0, fixed_params)
    outputs = scipy.optimize.fmin_cobyla(
            objfun, p0, composite_outfun, rhobeg=.01, rhoend=.001,
            maxfun=maxiter)
    xopt = _project_params_up(outputs, fixed_params)

    return xopt

def optimize_brute_fracs2(bins, Ls, data, nsamp, model_func, fracs,
        searchvalues, outofbounds_fun=None, cutoff=0, verbose=0,
        flush_delay=0.5,  full_output=True, func_args=[], fixed_params=None,
        ll_scale=1):
    """
    Optimize params to fit model to data using the brute force method.

    This optimization method works well when we start reasonably close to the
    optimum. It is best at burrowing down a single minimum.


    It should also perform better when parameters range over scales.

    p0: Initial parameters.
    data: Spectrum with data.
    model_function: Function to evaluate model spectrum. Should take arguments
                    (params, pts)
    out_of_bounds_fun: A funtion evaluating to True if the current parameters are in a forbidden region.
    cutoff: the number of bins to drop at the beginning of the array. This could be achieved with masks.

    verbose: If > 0, print optimization status every <verbose> steps.
    flush_delay: Standard output will be flushed once every <flush_delay>
                 minutes. This is useful to avoid overloading I/O on clusters.
    epsilon: Step-size to use for finite-difference derivatives.
    gtol: Convergence criterion for optimization. For more info,
          see help(scipy.optimize.fmin_bfgs)


    full_output: If True, return full outputs as in described in
                 help(scipy.optimize.fmin_bfgs)
    func_args: Additional arguments to model_func. It is assumed that
               model_func's first argument is an array of parameters to
               optimize, that its second argument is an array of sample sizes
               for the sfs, and that its last argument is the list of grid
               points to use in evaluation.
    fixed_params: If not None, should be a list used to fix model parameters at
                  particular values. For example, if the model parameters
                  are (nu1,nu2,T,m), then fixed_params = [0.5,None,None,2]
                  will hold nu1=0.5 and m=2. The optimizer will only change
                  T and m. Note that the bounds lists must include all
                  parameters. Optimization will fail if the fixed values
                  lie outside their bounds. A full-length p0 should be passed
                  in; values corresponding to fixed parameters are ignored.
    ll_scale: The bfgs algorithm may fail if your initial log-likelihood is
              too large. (This appears to be a flaw in the scipy
              implementation.) To overcome this, pass ll_scale > 1, which will
              simply reduce the magnitude of the log-likelihood. Once in a
              region of reasonable likelihood, you'll probably want to
              re-optimize with ll_scale=1.
    """
    def outfun(p0,verbose=False):
        # cobyla uses the constraint function and feeds it the reduced
        # parameters. Hence we have to project back up first
        x0 = _project_params_up(p0, fixed_params)
        if verbose:
            eprint("p0", p0)
            eprint("x0", x0)
            eprint("fracs", fracs)
            eprint("res", outofbounds_fun(p0, fracs))

        return outofbounds_fun(x0, fracs)

    modstrip = lambda x: model_func(x, fracs)

    fun = lambda x: _object_func_fracs2(x, bins, Ls, data, nsamp, modstrip,
            outofbounds_fun=outfun, cutoff=cutoff, verbose=verbose,
            flush_delay=flush_delay, func_args=func_args,
            fixed_params=fixed_params)

    if len(searchvalues) == 1:
        def fun2(x):
            return fun((float(x),))
    else:
        fun2 = fun

    outputs = scipy.optimize.brute(fun2, searchvalues, full_output=full_output)
    xopt = _project_params_up(outputs[0], fixed_params)

    return xopt, outputs[1:]

def optimize_brute_multifracs(
        bins, Ls, data_list, nsamp_list, model_func, fracs_list, searchvalues,
        outofbounds_fun=None, cutoff=0, verbose=0, flush_delay=0.5,
        full_output=True, func_args=[], fixed_params=None, ll_scale=1):
    """
    Optimize params to fit model to data using the brute force method.

    This optimization method works well when we start reasonably close to the
    optimum. It is best at burrowing down a single minimum.


    It should also perform better when parameters range over scales.

    p0: Initial parameters.
    data: Spectrum with data.
    model_function: Function to evaluate model spectrum. Should take arguments
                    (params, pts)
    out_of_bounds_fun: A funtion evaluating to True if the current parameters are in a forbidden region.
    cutoff: the number of bins to drop at the beginning of the array. This could be achieved with masks.

    verbose: If > 0, print optimization status every <verbose> steps.
    flush_delay: Standard output will be flushed once every <flush_delay>
                 minutes. This is useful to avoid overloading I/O on clusters.
    epsilon: Step-size to use for finite-difference derivatives.
    gtol: Convergence criterion for optimization. For more info,
          see help(scipy.optimize.fmin_bfgs)


    full_output: If True, return full outputs as in described in
                 help(scipy.optimize.fmin_bfgs)
    func_args: Additional arguments to model_func. It is assumed that
               model_func's first argument is an array of parameters to
               optimize, that its second argument is an array of sample sizes
               for the sfs, and that its last argument is the list of grid
               points to use in evaluation.
    fixed_params: If not None, should be a list used to fix model parameters at
                  particular values. For example, if the model parameters
                  are (nu1,nu2,T,m), then fixed_params = [0.5,None,None,2]
                  will hold nu1=0.5 and m=2. The optimizer will only change
                  T and m. Note that the bounds lists must include all
                  parameters. Optimization will fail if the fixed values
                  lie outside their bounds. A full-length p0 should be passed
                  in; values corresponding to fixed parameters are ignored.
    ll_scale: The bfgs algorithm may fail if your initial log-likelihood is
              too large. (This appears to be a flaw in the scipy
              implementation.) To overcome this, pass ll_scale > 1, which will
              simply reduce the magnitude of the log-likelihood. Once in a
              region of reasonable likelihood, you'll probably want to
              re-optimize with ll_scale=1.
    """
    # construct the out of bounds function.
    def outfun(p0, fracs, verbose=False):
        # cobyla uses the constraint function and feeds it the reduced
        # parameters. Hence we have to project back up first
        x0 = _project_params_up(p0, fixed_params)
        if verbose:
            eprint("p0", p0)
            eprint("x0", x0)
            eprint("fracs", fracs)
            eprint("res", outofbounds_fun(p0, fracs))

        return outofbounds_fun(x0, fracs)

    # construct a wrapper function that will tuple up its argument in the case
    # where searchvalues has length 1; in that case, the optimizer expects a
    # tuple (it always wants tuples), but the input will be a single float.
    # Hence why we need to tuple it up.
    # The wrapper function is called on the x given as input to
    # _object_func_multifracs
    r = (lambda x: x) \
            if len(searchvalues) > 1 else \
            (lambda x: (float(x),))

    # construct the objective function. The input x is wrapped in the
    # function r constructed above.
    objfun = lambda x: _object_func_multifracs(r(x), bins, Ls, data_list,
            nsamp_list, model_func, fracs_list, outofbounds_fun=outfun,
            cutoff=cutoff, verbose=verbose, flush_delay=flush_delay,
            func_args=func_args, fixed_params=fixed_params)

    outputs = scipy.optimize.brute(objfun, searchvalues, full_output=full_output)
    xopt = _project_params_up(outputs[0], fixed_params)

    return xopt, outputs[1:]

#: Counts calls to object_func
_counter = 0
# define the objective function for when the ancestry porportions are specified.
def _object_func_fracs(params, bins, Ls, data, nsamp, model_func, fracs,
        outofbounds_fun=None, cutoff=0, verbose=0, flush_delay=0,
                 func_args=[]):
    _out_of_bounds_val = -1e32
    global _counter
    _counter += 1

    if outofbounds_fun is not None:
        # outofbounds can return either True or a negative valueto signify out-of-boundedness.
        oob = outofbounds_fun(params,fracs)
        if oob < 0:
            result = -(oob-1)*_out_of_bounds_val
        else:
            mod = demographic_model(model_func(params, fracs))
            result = mod.loglik(bins, Ls, data, nsamp, cutoff=cutoff)
    else:
        eprint("No bound function defined")
        mod = demographic_model(model_func(params))
        result = mod.loglik(bins, Ls, data, nsamp, cutoff=cutoff)

    if verbose > 0 and _counter % verbose == 0:
        param_str = 'array([%s])' % (', '.join(['%- 12g'%v for v in params]))
        eprint( '%-8i, %-12g, %s' % (_counter, result, param_str))
        # Misc.delayed_flush(delay=flush_delay)

    return -result

def _object_func_fracs2(params, bins, Ls, data, nsamp, model_func,
        outofbounds_fun=None, cutoff=0, verbose=0, flush_delay=0,
        func_args=[],fixed_params=None):
    # this function will be minimized. We first calculate likelihoods (to be
    # maximized), and return minus that.
    eprint("evaluating at params", params)
    _out_of_bounds_val = -1e32
    global _counter
    _counter += 1

    # Deal with fixed parameters
    params_up = _project_params_up(params, fixed_params)

    if outofbounds_fun is not None:
        # outofbounds returns  a negative value to signify out-of-boundedness.
        oob = outofbounds_fun(params)

        if oob < 0:
            # we want bad functions to give very low likelihoods, and worse
            # likelihoods when the function is further out of bounds.
            mresult = - (oob - 1) * _out_of_bounds_val
            # challenge: if outofbounds is very close to 0, this can return a
            # reasonable likelihood. When oob is negative, we take away an
            # extra 1 to make sure this cancellation does not happen.
        else:
            mod = demographic_model(model_func(params_up))

            sys.stdout.flush()
            mresult=mod.loglik(bins,Ls,data,nsamp,cutoff=cutoff)
    else:
        eprint("No bound function defined")
        mod=demographic_model(model_func(params_up))
        mresult=mod.loglik(bins,Ls,data,nsamp,cutoff=cutoff)

    if True:#(verbose > 0) and (_counter % verbose == 0):
        param_str = 'array([%s])' % (', '.join(['%- 12g'%v for v in params_up]))
        eprint('%-8i, %-12g, %s' % (_counter, mresult, param_str))
        # Misc.delayed_flush(delay=flush_delay)

    return -mresult

#: Counts calls to object_func
_counter = 0
# define the objective function for when the ancestry porportions are
# specified.
def _object_func_multifracs(
        params, bins, Ls, data_list, nsamp_list, model_func, fracs_list,
        outofbounds_fun=None, cutoff=0, verbose=0, flush_delay=0,
        func_args=[],fixed_params=None):
    # this function will be minimized. We first calculate likelihoods (to be
    # maximized), and return minus that.
    eprint("evaluating at params", params)
    _out_of_bounds_val = -1e32
    global _counter
    _counter += 1

    # Deal with fixed parameters
    params_up = _project_params_up(params, fixed_params)

    mkmodel = lambda: composite_demographic_model(
            model_func, params, fracs_list)

    if outofbounds_fun is not None:
        # outofbounds returns  a negative value to signify out-of-boundedness.
        # Compute the out of bounds function for each fraction and take the
        # minimum as the overall out of bounds value.
        oob = min(outofbounds_fun(params, fracs)
                for fracs in fracs_list)

        if oob < 0:
            # we want bad functions to give very low likelihoods, and worse
            # likelihoods when the function is further out of bounds.
            mresult = -(oob-1)*_out_of_bounds_val
            # challenge: if outofbounds is very close to 0, this can return a
            # reasonable likelihood. When oob is negative, we take away an
            # extra 1 to make sure this cancellation does not happen.
        else:
            comp_model = mkmodel()

            sys.stdout.flush()

            mresult = comp_model.loglik(bins, Ls, data_list, nsamp_list,
                cutoff=cutoff)
    else:
        eprint("No bound function defined")
        comp_model = mkmodel()

        mresult = comp_model.loglik(bins, Ls, data_list, nsamp_list)

    if True:#(verbose > 0) and (_counter % verbose == 0):
        param_str = 'array([%s])' % (', '.join(['%- 12g'%v for v in params_up]))
        eprint('%-8i, %-12g, %s' % (_counter, mresult, param_str))
        # Misc.delayed_flush(delay=flush_delay)

    return -mresult
