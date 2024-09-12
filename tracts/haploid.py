from collections import defaultdict
from tracts.tract import Tract
from tracts.chrom import Chrom
import numpy as np


class Haploid:
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
                    Tract(
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
            # selected, so the selection function always returns True.
            # TODO Decide what to do with this dummy function
            def is_selected(*args):
                return True
        else:
            # Otherwise, we 'normalize' selectchrom by ensuring that it
            # contains only integers. (This is primarily for
            # backwards-compatibility with previous scripts that specified
            # chromosome numbers as strings.) And we make a set out of the
            # resulting normalized list, to speed up lookups later.
            sc = set(map(int, selectchrom))

            # And the function that tests for inclusion simply casts its argument
            # (which is a string since it's read in from a file) to an int,
            # and checks whether its in our set.

            def is_selected(chrom_label):
                return int(chrom_label) in sc

        # Filter the loaded data according to selectchrom using the is_selected
        # function constructed above.
        for chrom_data, tracts in chromd.items():
            chrom_id = chrom_data.split('r')[-1]
            if is_selected(chrom_id):
                c = Chrom(tracts=tracts)
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

        return Haploid(Ls=Ls, lschroms=chroms, labs=labs, name=name)

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
            h = Haploid.from_file(fname, selectchrom=selectchrom)
            self.Ls = h.Ls
            self.chroms = h.chroms
            self.labs = h.labs
            self.name = name

    def __repr__(self):
        return "haploid(lschroms=%s, name=%s, Ls=%s)" % tuple(map(repr, [self.chroms, self.name, self.Ls]))
