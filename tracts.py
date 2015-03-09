import numpy
import pylab
import Tkinter as Tk
import tkFileDialog

try:
    from scipy.misc.common import factorial
except ImportError:
    from scipy.misc import factorial

from scipy.special import gammainc, gammaln
import scipy.optimize
import sys

# tracts are our lower-level objects. They are single intervals with uniform
# labels, typically a population name A tract has a start, and end, a label,
# and a next_tract
class tract:
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

class chrom:
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

    def get_len(self):
        """ Synonym for the len function. """
        return self.len

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
        # TODO verify that my reimplementation of this method is consistent
        # with the older behaviour.

        if not self.tracts:
            warn("smoothing empty chromosome has no effect")
            return None  # Nothing to smooth since there're no tracts.

        def same_ancestry(my, their):
            return my.label == their.label

        # TODO determine whether copies are really necessary.
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
        """ Merge segments that are contiguous and of the same ancestry. """
        # TODO understand what this method is for. Clearly, it does some kind
        # of tract merging, apparently only if the tract is not unknown.
        # TODO rewrite this method to use a similar strategy as in the rewrite
        # of merge_ancestries.
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
        self._smooth_unknown() # TODO figure out why this is called here.

        # TODO why not use t.len oslt ?
        return [(t.label, t.end - t.start, self.len) for t in self.tracts]

    # TODO use this instead of directly getting at the tracts attribute of the
    # chromosome.
    def __getitem__(self, index):
        """ Simply wrap the underlying list's __getitem__ method. """
        return self.tracts[index]

class chropair:
    """ A pair of chromosomes. """ # TODO better description.
    def __init__(self, chroms=None, len=1, auto=True, label="POP"):
        """ Can instantiate by explictly providing two chromosomes as a tuple
            or an ancestry label, length and autosome status. """
        if(chroms == None):
            self.copies = [chrom(len, auto, label), chrom(len, auto, label)]
            self.len = len
        else:
            assert chroms[0].get_len() == chroms[1].get_len(), \
                    Exception("chromosome pairs of different lengths!")
            self.len = chroms[0].get_len()
            self.copies = chroms

    def recombine(self):
        # decide on the number of recombinations
        n = numpy.random.poisson(self.len)
        # get recombination points
        unif = (self.len*numpy.random.random(n)).tolist()
        unif.extend([0, self.len])
        unif.sort()
        # start with a random chromosome
        startchrom = numpy.random.random_integers(0, 1)
        tractlist = []
        for startpos in range(len(unif)-1):
            tractlist.extend(
                    self.copies[(startchrom+startpos)%2]
                        .extract(unif[startpos],
                    unif[startpos+1]))
        newchrom = chrom(self.copies[0].len, self.copies[0].auto)
        newchrom.init_list_tracts(tractlist)
        return newchrom

    def plot(self, canvas, colordict, height=0):
        self.copies[0].plot(canvas, colordict, height=height+0.1)
        self.copies[1].plot(canvas, colordict, height=height+0.22)
    def applychrom(self, func):
        """apply func to chromosomes"""
        ls = []
        for copy in self.copies:
            ls.append(func(copy))
        return ls

# individual
class indiv:
    def __init__(self, Ls=None, label="POP", fname=None, labs=("_A", "_B"),
            selectchrom=None):
        """ If reading from a file, fname should be a tuple with the start and
            end of the file names. Otherwise, provide list of chromosome
            lengths.  Distinguishing labels for maternal and paternal
            chromosomes are given in lab. """
        if(fname == None):
            self.Ls = Ls
            self.chroms = [chropair(len=len, label=label) for len in Ls]
        else:
            fnames = [fname[0] + lab + fname[1] for lab in labs]
            f1, f2 = [haploid(fname=f, selectchrom=selectchrom)
                    for f in fnames]
            self.name = fname[0].split('/')[-1]
            try:
                self.from_haploids(f1, f2)
            except AssertionError:
                print "".join(["error in individual ", fname[0], labs[0],
                    fname[1], " or ", fname[0], labs[1], fname[1]])
                raise

    def plot(self, colordict, win=None):
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

    def from_haploids(self, hap1, hap2):
        self.chroms = [chropair(chroms=(hap1.chroms[i], hap2.chroms[i]))
                for i in xrange(len(hap1.Ls))]
        self.Ls = hap1.Ls

    def applychrom(self, func):
        """ Apply the function `func` to each chromosome of the individual. """
        return map(lambda c: c.applychrom(func), self.chroms)

    def ancestryAmt(self, ancestry):
        """ Calculate the total length of the genome in segments of the given
            ancestry.
            """
        dat = self.applychrom(chrom.tractlengths)
        return numpy.sum([segment[1]
            for chromv in dat
            for copy in chromv
            for segment in copy
            if segment[0] == ancestry])

    def ancestryProps(self, ancestries):
        """ Calculate the proportion of the genome represented by the given
            ancestries.
            """
        # TODO make this more efficient by switching ancestryAmt and this, or
        # by making a helper function.
        amts = [self.ancestryAmt(anc) for anc in ancestries]
        tot = numpy.sum(amts)
        return [amt*1./tot for amt in amts]

    def ancestryPropsByChrom(self, ancestries):
        dat = self.applychrom(chrom.tractlengths)
        dictamt = {}
        nc = len(dat)
        for ancestry in ancestries:
            lsamounts = []
            for chromv in dat:
                lsamounts.append(numpy.sum([segment[1]
                        for copy in chromv
                        for segment in copy
                        if segment[0] == ancestry]))
            dictamt[ancestry] = lsamounts
        tots = [numpy.sum(
            [dictamt[ancestry][i]
                for ancestry in ancestries])
            for i in range(nc)]

        return [[dictamt[ancestry][i]*1./tots[i]
            for i in range(nc)]
            for ancestry in ancestries]

# haploid individual
class haploid:
    def __init__(self, Ls=None, lschroms=None, fname=None, selectchrom=None):
        if fname is None:
            if Ls is None or  lschroms is None:
                raise ValueError(
                        "Ls or lschroms should be defined if file not defined")
            self.Ls = Ls
            self.chroms = lschroms
        else:
            dic = {}
            f = open(fname, 'r')
            lines = f.readlines()
            for line in lines:
                lsp = line.split()
                if lsp[0] == "chrom" or \
                        (lsp[0] == "Chr" and lsp[1] == "Start(bp)"):
                    continue
                try:
                    dic[lsp[0]].append(
                            tract(
                                .01*float(lsp[4]), .01*float(lsp[5]), lsp[3]))
                except KeyError:
                    try:
                        dic[lsp[0]] = \
                                [tract(.01*float(lsp[4]),
                                    .01*float(lsp[5]), lsp[3])]
                    except IndexError:
                        print "error defining haploid"
                        print "line to parse:", line
                        print "IOerror in: "\
                            "dic[lsp[0]]=[tract(.01*float(lsp[4]),"\
                            ".01*float(lsp[5]),lsp[3])]"
                        print "Local ancestry file may not have enough columns"
                        raise IndexError

            self.chroms = []
            self.labs = []
            self.Ls = []

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

            for num, vals in dic.iteritems():
                if selectchrom is None or is_selected(num.split('r')[-1]):
                    self.chroms.append(chrom(tracts=vals))
                    self.Ls.append(self.chroms[-1].get_len())
                    self.labs.append(num.split('r')[-1])
            self.chroms = list(
                    numpy.array(self.chroms)[numpy.argsort(self.labs)])
            self.Ls = list(
                    numpy.array(self.Ls)[numpy.argsort(self.labs)])
            self.labs = list(
                    numpy.array(self.labs)[numpy.argsort(self.labs)])

class population:
    def __init__(self, list_indivs=None, names=None, fname=None,
            labs=("_A", "_B"), selectchrom=None):
        """ The population can be initialized by providing it with a list of
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
            for ind in self.indivs:
                if ind.Ls != self.Ls:
                    print "error: individuals have genomes of different "\
                            "lengths!"
                    sys.exit(1)
            self.maxLen = max(self.Ls)
        elif fname is not None:
            self.indivs = []
            for name in names:
                pathspec = (os.path.join(fname[0], name+fname[1]), fname[2])
                try:
                    self.indivs.append(
                            indiv(fname=pathspec, labs=labs,
                                selectchrom=selectchrom))
                except IndexError:
                    print "error reading individuals", name
                    print "fname=", fname, \
                            "; labs=", labs, ", selectchrom=", selectchrom
                    #TODO why do we do this again? esp. when an error occurs
                    self.indivs.append(
                            indiv(fname=fname, labs=labs,
                                selectchrom=selectchrom))
                    raise IndexError
            self.nind = len(self.indivs)
            # should probably check that all individuals have same length!
            self.Ls = self.indivs[0].Ls
            self.maxLen = max(self.Ls)
        else:
            raise

    def newgen(self):
        return population([self.new_indiv() for i in range(self.nind)])

    def new_indiv(self):
        rd = numpy.random.random_integers(0, self.nind-1, 2)
        while(rd[0] == rd[1]):
            rd = numpy.random.random_integers(0, self.nind-1, 2)
        gamete1 = self.indivs[rd[0]].create_gamete()
        gamete2 = self.indivs[rd[1]].create_gamete()
        new = indiv(gamete1.Ls)
        new.from_haploids(gamete1, gamete2)
        return new

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

    def save(self):
        file = tkFileDialog.asksaveasfilename(parent=self.win,
                title='Choose a file')
        self.indivs[self.currentplot].canvas.postscript(file=file)

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

    def list_chromosome(self, chronum):
        return [indiv.chroms[chronum] for indiv in self.indivs]

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
                    # TODO why is this commented out?
                    # if (tract.end-pos)>.1 and (chrom.end-pos)>.2:
                    #    longancestry[tract.label]+=1
                    # if (pos-tract.start)>.2:
                    #    totlength[tract.label]+=1
                    # if (pos-tract.start)>.1 and (pos-chrom.start)>.2:
                    #    longancestry[tract.label]+=1
                except KeyError:
                    ancestry[tract.label] = 0
                    longancestry[tract.label] = 0
                    totlength[tract.label] = 0
                    if tract.len():
                        ancestry[tract.label] += 1
                        totlength[tract.label] += tract.len()
                    # if (tract.end-pos)>.2:
                    #    totlength[tract.label]+=1
                    # if (tract.end-pos)>.1 and (chrom.end-pos)>.2:
                    #    longancestry[tract.label]+=1
                    # if (pos-tract.start)>.2:
                    #    totlength[tract.label]+=1
                    # if (pos-tract.start)>.1 and (pos-chrom.start)>.2:
                    #    longancestry[tract.label]+=1


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
        plotpts = numpy.arange(0, len, len/float(npts))
        return (plotpts,
                [self.ancestry_at_pos(chrom=chrom, pos=pt, cutoff=cutoff)
                    for pt in plotpts])

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
                    print pos[0][key], float(tot)
                    pos[0][key] /= float(tot)
        for pop, color in colordict.iteritems():
            print tot
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

    def applychrom(self,func, indlist=None):
            """apply func to chromosomes. If no indlist is supplied, apply to all individuals"""
            ls=[]

            if indlist is None:
                    inds=self.indivs
            else:
                    inds=indlist

            for ind in inds:
                    ls.append(ind.applychrom(func))
            return ls

    def flatpop(self,ls):
            """returns a flattened version of a population-wide list at the tract level"""
            flatls=[]
            for indiv in ls:
                for chrom in indiv:
                    for copy in chrom:
                        flatls.extend(copy)
            return flatls

    # TODO Why is this named using the double-underscore notation?
    def __collectpop__(self, flatdat):
        """ Returns a dictionary sorted by the first item in a list. Used in
            plot_tractlength. """
        dic = {}
        for datum in flatdat:
            try:
                dic[datum[0]].append(datum[1:])
            except KeyError:
                dic[datum[0]] = [datum[1:]]
        for key in dic.keys():
            dic[key] = numpy.array(dic[key])
        return dic

    def merge_ancestries(self, ancestries, newlabel):
        """ Treats ancestries in label list "ancestries" as a single population
            with label "newlabel". Adjacent tracts of the new ancestry are
            merged. """
        f = lambda i: i.merge_ancestries(ancestries, newlabel)
        self.applychrom(f)

    def plot_global_tractlengths(self, colordict, npts=40, legend=True):
        dat = self.applychrom(chrom.tractlengths)
        flatdat = self.flatpop(dat)
        bypop = self.__collectpop__(flatdat)
        self.maxLen = max(self.Ls)
        for key, item in bypop.iteritems():
            hdat = pylab.histogram(item[:, 0], npts)
            # note: convert to cM before plotting
            pylab.semilogy(100*(hdat[1][1:]+hdat[1][:-1])/2., hdat[0], 'o',
                    color=colordict[key], label=key)
        pylab.xlabel("length(cM)")
        pylab.ylabel("counts")
        if legend:
            pylab.legend()

    def get_global_tractlengths(self,npts=20,tol=0.01,indlist=None):
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
        if indlist is None:
            inds=self.indivs
        else:
            inds=indlist

        dat=self.applychrom(chrom.tractlengths, indlist=indlist)
        flatdat=self.flatpop(dat)
        bypop=self.__collectpop__(flatdat)

        bins=numpy.arange(0,self.maxLen*(1+.5/npts),float(self.maxLen)/npts)
        dat={}
        for key, poplen in bypop.iteritems():
            # extract full length tracts
            nonfulls = numpy.array(
                    [item
                        for item in poplen
                        if (item[0] < item[1]-tol)])

            hdat = pylab.histogram(nonfulls[:, 0], bins=bins)
            dat[key] = list(hdat[0])
            # append the number of fulls
            dat[key].append(len(poplen)-len(nonfulls))

        return (bins,dat)

    def bootinds(self,seed):
        """ Return a bootstrapped list of individuals in the population. Use
            with get_global_tractlength inds=... to get a bootstrapped
            sample.
            """
        numpy.random.seed(seed=seed)
        return numpy.random.choice(self.indivs,size=len(self.indivs))

    def get_global_tractlength_table(self, lenbound):
        """ Calculates the fraction of the genome covered by ancestry tracts of
            different lengths, spcified by lenbound (which must be sorted). """
        dat = self.applychrom(chrom.tractlengths)
        flatdat = self.flatpop(dat)
        bypop = self.__collectpop__(flatdat)

        bins = lenbound
        # numpy.arange(0,self.maxLen*(1+.5/npts),float(self.maxLen)/npts)

        import bisect
        dat = {} # numpy.zeros((len(bypop),len(bins)+1)
        for key, poplen in bypop.iteritems():
            # extract full length tracts
            dat[key] = numpy.zeros(len(bins)+1)
            nonfulls = numpy.array([item
                        for item in poplen
                        if (item[0] != item[1])])
            for item in nonfulls:
                pos = bisect.bisect_left(bins, item[0])
                dat[key][pos] += item[0]/self.nind/numpy.sum(self.Ls)/2.

        return (bins, dat)

    def get_means(self, ancestries):
        """ Get the mean ancestry proportion (only among ancestries in
            ancestries) for all individuals. """
        return [ind.ancestryProps(ancestries) for ind in self.indivs]

    def get_meanvar(self, ancestries):
        byind = self.get_means(ancestries)
        return numpy.mean(byind, axis=0), numpy.var(byind, axis=0)

    def getMeansByChrom(self, ancestries):
        return [ind.ancestryPropsByChrom(ancestries) for ind in self.indivs]

    # def get_assortment_variance(self,ancestries):
    #     ""ancestries is a set of ancestry label. Calculates the assortment variance in ancestry proportions (corresponds to the mean uncertainty about the proportion of genealogical ancestors, given observed ancestry patterns)""

    # ws=numpy.array(self.Ls)/numpy.sum(self.Ls) #the weights, corresponding (approximately) to the inverse variances
    #     arr=numpy.array(self.getMeansByChrom(ancestries))
    # weighted mean by individual
    # departure from the mean
    #     nchr=arr.shape[2]
    #     vars=[]
    #     for i in range(len(ancestries)):
    #         pl=numpy.dot(arr[:,i,:], ws )

    #         aroundmean=arr[:,i,:]-numpy.dot(pl.reshape(self.nind,1),numpy.ones((1,nchr)))
    #         vars.append((numpy.mean(aroundmean**2/(1./ws-1),axis=1)).mean())
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
        ws = numpy.array(self.Ls)/numpy.sum(self.Ls)
        arr = numpy.array(self.getMeansByChrom(ancestries))
        # weighted mean by individual
        # departure from the mean
        nchr = arr.shape[2]
        assort_vars = []
        tot_vars = []
        gen_vars = []
        for i in range(len(ancestries)):
            pl = numpy.dot(arr[:, i, :], ws )
            tot_vars.append(numpy.var(pl))
            aroundmean = arr[:, i, :] - numpy.dot(
                    pl.reshape(self.nind, 1), numpy.ones((1, nchr)))
            # the unbiased estimator for the case where the variance is
            # inversely proportional to the weight. First calculate by
            # individual, then the mean over all individuals.
            assort_vars.append(
                    (numpy.mean(aroundmean**2/(1./ws-1), axis=1)).mean())
            gen_vars.append(tot_vars[-1]-assort_vars[-1])
        return tot_vars, gen_vars, assort_vars

class demographic_model():
    def __init__(self, mig):
        """ Migratory model takes as an input a vector containing the migration
            proportions over the last generations. Each row is a time, each
            column is a population. row zero corresponds to the current
            generation. The migration rate at the last generation (time $T$) is
            the "founding generation" and should sum up to 1. Assume that
            non-admixed individuals have been removed. """
        small=1e-10
        self.mig = mig
        (self.ngen, self.npop) = mig.shape

        # the total migration per generation
        self.totmig = mig.sum(axis=1)
        if abs(self.totmig[-1] - 1) > small:
            print "founding migration should sum up to 1. Now:", mig[-1, :], \
                    "sum up to ", self.totmig[-1]
            raise ValueError("founding migration sum is not 1")
        if self.totmig[0] > small:
            print "migrants at last generation should be removed from sample!"
            print "currently", self.totmig[0]
            raise ValueError("migrants from last generation are not removed")
        self.totmig[0] = 0
        if self.totmig[1] > small:
            print "migrants at penultimate generation should be removed from "\
                    "sample!"
            print "currently", self.totmig[1]
            raise ValueError(
                    "migrants from penultimate generation are not removed")

        if ((self.totmig > 1).any() or (mig < 0).any()):
            print "migration rates should be between 0 and 1"
            print "currently", mig
            raise ValueError("mig")
        if (mig[:-1] == 1).any():
            print "warning: population was completely replaced after "\
                    "founding event"
        # identify states where migration occurred as these are the relevant
        # states in our Markov model. Each state is a tuple of the form:
        # (generation, population)

        self.states = map(tuple, numpy.array(mig.nonzero()).transpose())
        self.nstates = len(self.states)
        self.npops = mig.shape[1]

        # get the equilibrium distribution in each state
        # print self.nstates, " states"
        self.equil = numpy.zeros(self.nstates)
        self.stateINpop = [[] for pop in range(self.npops)]
        self.stateOUTpop = [[] for pop in range(self.npops)]

        for i, state in enumerate(self.states):
            self.stateINpop[state[1]].append(i)
            for other in range(1, self.npops+1):
                self.stateOUTpop[(state[1]+other)%self.npops].append(i)
            self.equil[i] = mig[state]*(1-self.totmig)[1:state[0]].prod()

        # print "equilibrium states sum up to ", self.equil.sum(),\
        # "normalizing"
        self.equil /= self.equil.sum()

        # calculate the ancestry proportions as a function of time
        self.proportions = numpy.zeros(mig.shape)

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

        # self.mat=numpy.zeros(((self.T-1)*self.npop,(self.T-1)*self.npop))
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

        self.mat = numpy.zeros((len(self.states), len(self.states)))
        for nump, (tp, popp) in enumerate(self.states):
            for num, (t, pop) in enumerate(self.states):
                tot = 0
                for tau in range(1, min(t, tp)):
                    tot += self.dicTpopTau[(tp, popp, tau)]
                for pop in range(self.npop):
                    self.mat[num, nump] = tot

        # note that the matrix could be uniformized in a population-specific
        # way, for optimization purposes
        self.__uniformizemat__()
        self.ndists = []
        for i in range(self.npops):
            self.ndists.append(self.popNdist(i))
        self.switchdensity()

    def gen_variance(self, popnum):
        """ 1. Calculate the expected genealogy variance in the model. Need to
            double-check +-1s.
            2. Calculate the e(d)
            3. Generations go from 0 to self.ngen-1.
            """
        legterm = [self.proportions[self.ngen - d, popnum]**2 * \
                numpy.prod(1 - self.totmig[:(self.ngen - d)])
                for d in range(1, self.ngen)]
        trunkterm = [numpy.sum([self.mig[u, popnum] * \
                numpy.prod(1 - self.totmig[:u])
                for u in range(self.ngen-d)])
                for d in range(1, self.ngen)]

        # Now calculate the actual variance.
        return numpy.sum([2**(d-self.ngen) * (legterm[d-1]+trunkterm[d-1])\
            for d in range(1, self.ngen)]) +\
                    self.proportions[0, popnum] *\
                    (1/2.**(self.ngen-1)-self.proportions[0, popnum])

    def __uniformizemat__(self): # TODO why __function__ ?
        """ Uniformize the transition matrix so that each state has the same
            total transition rate. """
        self.unifmat = self.mat.copy()
        lmat = len(self.mat)
        # identify the highest non-self transition rate
        maxes = (self.mat - numpy.diag(self.mat.diagonal())).sum(axis=1)

        self.maxrate = maxes.max()
        for i in range(lmat):
            self.unifmat[i, i] = self.maxrate-maxes[i]
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
        new = numpy.dot(tempequil, self.unifmat)
        # select states in relevant population

        newrest = new[self.stateINpop[pop]]
        newrest /= newrest.sum()
        # print newrest
        # reduce the matrix to apply only to states of current population
        shortmat = self.unifmat[
                numpy.meshgrid(self.stateINpop[pop],
                    self.stateINpop[pop])].transpose()
        # calculate the amount that fall out of the state
        escapes = 1 - shortmat.sum(axis=1)
        # decide on the number of itertaions
        nit = int(6 * self.maxrate)

        nDistribution = []
        for i in range(nit):
            nDistribution.append(numpy.dot(escapes, newrest))
            newrest = numpy.dot(newrest, shortmat)
            # print newrest

        # print newrest.sum(), "remaining tracts at cutoff."
        nDistribution.append(newrest.sum())
        return nDistribution

    def Erlang(self, i, x, T):
        if i > 10:
            lg = i*numpy.log(T)+(i-1)*numpy.log(x)-T*x-gammaln(i)
            return numpy.exp(lg)
        return T**i*x**(i - 1)*numpy.exp(- T*x)/factorial(i - 1)

    def inners(self, L, x, pop):
        """ Calculate the length distribution of tract lengths not hitting a
            chromosome edge. """
        if(x > L):
            return 0
        else:
            return numpy.sum(
                    [self.ndists[pop][i] *
                        (L-x) *
                        self.Erlang(i+1, x, self.maxrate)
                        for i in range(len(self.ndists[pop]))])

    def outers(self, L, x, pop):
        """ Calculate the length distribution of tract lengths hitting a single
            chromosome edge. """
        if(x > L):
            return 0
        else:
            return 2 * numpy.sum(
                    [self.ndists[pop][i]*(1-gammainc(i+1, self.maxrate*x))
                        for i in xrange(
                            len(self.ndists[pop]))]) + \
                2 * (1-numpy.sum(self.ndists[pop]))


        # 2*Sum[distr[[i]]* Gamma[i, T x]/((i - 1)!), {i, 1, Length[distr]}] +
         # 2 (1 - Sum[distr[[i]], {i, 1, Length[distr]}])

    def full(self, L, pop):
        """ The expected fraction of full-chromosome tracts, p. 63 May 24,
            2011. """
        return numpy.sum(
            [self.ndists[pop][i] * (((i+1) / float(self.maxrate) - L) + \
                    L * gammainc(i + 1, self.maxrate * L) - \
                    float(i+1) / self.maxrate * gammainc(i+2, self.maxrate*L))\
                    for i in xrange(len(self.ndists[pop]))]) + \
            (1 - numpy.sum(self.ndists[pop])) * \
            (len(self.ndists[pop])/self.maxrate - L)

    def Z(self, L, pop):
        """the normalizing factor, to ensure that the tract density is 1."""
        return L + numpy.sum([self.ndists[pop][i]*(i+1)/self.maxrate
            for i in xrange(len(self.ndists[pop]))]) + \
            (1 - numpy.sum([self.ndists[pop]])) * \
            len(self.ndists[pop])/self.maxrate

    def switchdensity(self):
        """ Calculate the density of ancestry switchpoints per morgan in our
            model. """
        self.switchdensities = numpy.zeros((self.npops, self.npops))
        # could optimize by precomputing survivals earlier
        self.survivals = [(1 - self.totmig[:i]).prod()
                for i in range(self.ngen)]
        for pop1 in range(self.npops):
            for pop2 in range(pop1):
                self.switchdensities[pop1, pop2] = \
                        numpy.sum(
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
                numpy.sum(
                        [(L*self.totSwitchDens[pop]+2. * \
                                self.proportions[0, pop]) * \
                                self.full(L, pop)/self.Z(L, pop)
                            for L in Ls])
        lsval = []
        for binNum in range(len(bins) - 1):
            mid = (bins[binNum] + bins[binNum+1]) / 2.
            val = numpy.sum(
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
                    numpy.random.poisson(
                        nind * numpy.array(self.expectperbin(Ls, pop, bins))))
        return expect

    def loglik(self, bins, Ls, data, nsamp, cutoff=0):
        """ Calculate the maximum-likelihood in a Poisson Random Field. Last
            bin of data is the number of whole-chromosome. """
        self.maxLen = max(Ls)
        # define bins that contain all possible values
        # bins=numpy.arange(0,self.maxLen+1./2./float(npts),self.maxLen/float(npts))
        ll = 0
        for pop in range(self.npops):
            models = self.expectperbin(Ls, pop, bins)
            for binnum in range(cutoff, len(bins)-1):
                dat = data[pop][binnum]
                ll += -nsamp*models[binnum] + \
                        dat*numpy.log(nsamp*models[binnum]) - \
                        gammaln(dat + 1.)
        return ll

    def add_random(self, numtoadd, length, pop, bins, data):
        """ Add a number of tracts of specified length, taking tracts in data and
            breaking them down. """
        # TODO What is data doing here ?!? This function seems unused. Is it
        # incomplete? Grepping for it shows no hits besides its definition.
        for lpop in range(self.npops):
            if lpop == pop:
                continue
            bins*2

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
            mods.append(nsamp*numpy.array(self.expectperbin(Ls, pop, bins)))

        if biascorrect:
            if self.npops != 2:
                print "bias correction not implemented for more than 2 "\
                        "populations"
                sys.exit()
            cbypop = []
            for pop in range(self.npops):
                mod = mods[pop]
                corr = []
                for binnum in range(cutoff):
                    diff = mod[binnum]-data[pop][binnum]
                    lg = ((bins[binnum]+bins[binnum+1]))/2;

                    corr.append((lg, diff))
                print corr
                cbypop.append(corr)
            for pop in range(self.npops):
                # total length in tracts
                tot = numpy.sum([bins[i]*data[pop][i]
                    for i in range(cutoff, len(bins))])
                # probability that a given tract is hit by a given "extra short
                # tracts"
                probs = [bins[i]/tot for i in range(cutoff, len(bins))]
                print "tot", tot
                print "probs", probs
                for shortbin in range(cutoff):
                    transfermat = numpy.zeros(
                            (len(bins)-cutoff, len(bins)-cutoff))
                    corr = cbypop[1-pop][shortbin]
                    if corr[1] > 0:
                        print "correction for lack of short tracts not "\
                                "implemented!"
                        sys.exit()
                    for lbin in range(len(bins)-cutoff):
                        print "corr[1]", corr[1]
                        transfermat[lbin, lbin] = 1+corr[1]*probs[lbin-cutoff]
                print transfermat

                # count the number of missing bits in each population.
                print "population ", pop, " ", cbypop[pop]

        # define bins that contain all possible values
        # bins=numpy.arange(0,self.maxLen+1./2./float(npts),self.maxLen/float(npts))
        ll = 0
        for pop in range(self.npops):
            models = mods[pop]
            for binnum in range(cutoff, len(bins)-1):
                dat = data[pop][binnum]
                ll += -nsamp*models[binnum] + \
                        dat*numpy.log(nsamp*models[binnum]) - \
                        gammaln(dat + 1.)
        return ll

    def plot_model_data(self, Ls, bins, data, nsamp, pop, colordict):
        # plot the migration model with the data
        pop.plot_global_tractlengths(colordict)
        for pop in range(len(data)):
            pylab.plot(
                    100*numpy.array(bins),
                    nsamp*numpy.array(self.expectperbin(Ls, 0, bins)))

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
                    radius=numpy.sqrt(mig[i, j]) / 1.7,
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
        If True, return full outputs as in described in help.
        (scipy.optimize.fmin_bfgs)
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

    # TODO this does not correspond with the description of `fixed_params` in
    # the docstring.
    if fixed_params is not None:
        print "error: fixed parameters not implemented in optimize"
        raise

    # TODO make full_output passed to scipy.optimize actually depend on
    # full_output passed to this function.

    # p0 = _project_params_down(p0, fixed_params)
    outputs = scipy.optimize.fmin_bfgs(_object_func,
            p0, epsilon=epsilon,
            args = args, gtol=gtol,
            full_output=True,
            disp=False,
            maxiter=maxiter)
    xopt, fopt, gopt, Bopt, func_calls, grad_calls, warnflag = outputs

    # xopt = _project_params_up(xopt, fixed_params)

    if not full_output:
        return xopt
    else:
        return xopt, fopt, gopt, Bopt, func_calls, grad_calls, warnflag

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
    # print bounds
    outputs = scipy.optimize.fmin_slsqp(_object_func,
            p0, ieqcons=[onearg], bounds=bounds,
            args = args,
            iter=maxiter, acc=1e-4, epsilon=1e-4)

    return outputs

    # xopt, fopt, gopt, Bopt, func_calls, grad_calls, warnflag = outputs
    # xopt = _project_params_up(numpy.exp(xopt), fixed_params)
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

    return numpy.array(pout)

def _project_params_up(pin, fixed_params):
    """ Fold fixed parameters into pin. Copied from Dadi (Gutenkunst et al.,
        PLoS Genetics, 2009). """
    if fixed_params is None:
        return pin

    pout = numpy.zeros(len(fixed_params))
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
    # print "in objective function","\n"
    if outofbounds_fun is not None:
        # outofbounds can return either True or a negative valueto signify out-of-boundedness.
        # print "out_of_bound is ",outofbounds_fun(params)
        ooa = outofbounds_fun(params)
        if ooa < 0:
            # print "out_of_bounds value " , outofbounds_fun(params),"\n"
            result = -(ooa-1)*_out_of_bounds_val
        else:
            # print "out_of_bounds value " , outofbounds_fun(params),"\n"
            mod = demographic_model(model_func(params))
            result = mod.loglik(bins, Ls, data, nsamp, cutoff=cutoff)
    else:
        print "No bound function defined"
        mod = demographic_model(model_func(params))
        result = mod.loglik(bins, Ls, data, nsamp, cutoff=cutoff)

    if True:#(verbose > 0) and (_counter % verbose == 0):
        param_str = 'array([%s])' % (', '.join(['%- 12g'%v for v in params]))
        print '%-8i, %-12g, %s' % (_counter, result, param_str)
        # Misc.delayed_flush(delay=flush_delay)

    return -result

# define the optimization routine for when the final ancestry proportions are
# specified.
def optimize_cob_fracs(p0, bins, Ls, data, nsamp, model_func, fracs,
        outofbounds_fun=None, cutoff=0, verbose=0, flush_delay=0.5,
        epsilon=1e-3, gtol=1e-5, maxiter=None, full_output=True, func_args=[],
        fixed_params=None, ll_scale=1):
    # TODO this docstring doesn't actually match the function's behaviour ?
    """
    Optimize params to fit model to data using the BFGS method.

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
            print "p0", p0
            print "x0", x0
            print "fracs", fracs
            print "res", outofbounds_fun(p0, fracs)

        return outofbounds_fun(x0, fracs)

    # print outfun(p0)
    modstrip = lambda x:model_func(x, fracs)


    fun = lambda x: _object_func_fracs2(x, bins, Ls, data, nsamp, modstrip,
            outofbounds_fun=outfun, cutoff=cutoff,
            verbose=verbose, flush_delay=flush_delay,
            func_args=func_args, fixed_params=fixed_params)


    p0 = _project_params_down(p0, fixed_params)
    # print "p0",p0
    outputs = scipy.optimize.fmin_cobyla(fun, p0, outfun, rhobeg=.01,
            rhoend=.001, maxfun=maxiter)
    # print "outputs", outputs
    xopt = _project_params_up(outputs, fixed_params)
    # print "xopt",xopt

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
            print "p0", p0
            print "x0", x0
            print "fracs", fracs
            print "res", outofbounds_fun(p0, fracs)

        return outofbounds_fun(x0, fracs)

    # print outfun(p0)
    modstrip = lambda x: model_func(x, fracs)

    fun = lambda x: _object_func_fracs2(x, bins, Ls, data, nsamp, modstrip,
            outofbounds_fun=outfun, cutoff=cutoff, verbose=verbose,
            flush_delay=flush_delay, func_args=func_args,
            fixed_params=fixed_params)


    # p0 = _project_params_down(p0, fixed_params)
    # print "p0",p0

    if len(searchvalues) == 1:
        def fun2(x):
            return fun((float(x),))
    else:
        fun2 = fun
    print "foutput", full_output
    print "searchvalues", searchvalues
    outputs = scipy.optimize.brute(fun2, searchvalues, full_output=full_output)
    print "outputs", outputs
    xopt = _project_params_up(outputs[0], fixed_params)
    # print "xopt",xopt
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

    # print "in objective function","\n"
    if outofbounds_fun is not None:
        # outofbounds can return either True or a negative valueto signify out-of-boundedness.
        # print "out_of_bound is ",outofbounds_fun(params)
        oob = outofbounds_fun(params,fracs)
        if oob < 0:
            # print "out_of_bounds value " , outofbounds_fun(params),"\n"
            result = -(oob-1)*_out_of_bounds_val
        else:
            # print "out_of_bounds value " , outofbounds_fun(params),"\n"
            mod = demographic_model(model_func(params, fracs))
            result = mod.loglik(bins, Ls, data, nsamp, cutoff=cutoff)
    else:
        print "No bound function defined"
        mod = demographic_model(model_func(params))
        result = mod.loglik(bins, Ls, data, nsamp, cutoff=cutoff)

    # TODO why is this not using a real condition?
    if True:#(verbose > 0) and (_counter % verbose == 0):
        param_str = 'array([%s])' % (', '.join(['%- 12g'%v for v in params]))
        print '%-8i, %-12g, %s' % (_counter, result, param_str)
        # Misc.delayed_flush(delay=flush_delay)

    return -result

#: Counts calls to object_func
_counter = 0
# define the objective function for when the ancestry porportions are
# specified.
def _object_func_fracs2(params, bins, Ls, data, nsamp, model_func,
        outofbounds_fun=None, cutoff=0, verbose=0, flush_delay=0,
        func_args=[],fixed_params=None):
    # this function will be minimized. We first calculate likelihoods (to be maximized), and return minus that.
    print "evaluating at params",params
    _out_of_bounds_val = -1e32
    global _counter
    _counter += 1

    # Deal with fixed parameters
    params_up = _project_params_up(params, fixed_params)

    if outofbounds_fun is not None:
        # outofbounds returns  a negative value to signify out-of-boundedness.
        oob = outofbounds_fun(params)

        if oob < 0:
            mresult = -(oob-1)*_out_of_bounds_val     #we want bad functions to give very low likelihoods, and worse likelihoods when the function is further out of bounds.
            # challenge: if outofbounds is very close to 0, this can return a reasonable likelihood. When oob is negative, we take away an extra 1 to make sure this cancellation does not happen.
        else:
            try:
                mod = demographic_model(model_func(params_up))
            except ValueError:
                print "valueError for params ", params

                print "res was", outofbounds_fun(params,verbose=True)
                print "mig was" , model_func(params)
                # mresult=min(-outofbounds_fun(params)*_out_of_bounds_val,-1e-8)
                raise ValueError

            sys.stdout.flush()
            mresult=mod.loglik(bins,Ls,data,nsamp,cutoff=cutoff)
    else:
        print "No bound function defined"
        mod=demographic_model(model_func(params_up))
        mresult=mod.loglik(bins,Ls,data,nsamp,cutoff=cutoff)

    if True:#(verbose > 0) and (_counter % verbose == 0):
        param_str = 'array([%s])' % (', '.join(['%- 12g'%v for v in params_up]))
        print '%-8i, %-12g, %s' % (_counter, mresult, param_str)
        # Misc.delayed_flush(delay=flush_delay)

    return -mresult
