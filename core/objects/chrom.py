from __future__ import print_function

import numpy as np
import operator as op
import itertools as it
import os
import pylab
import Tkinter as Tk
import tkFileDialog

from collections import defaultdict
from core import tracts as tracts
from core.objects import tract as tract

try:
    from scipy.misc.common import factorial
except ImportError:
    from scipy.misc import factorial

from scipy.special import gammainc, gammaln
import scipy.optimize
import sys

eprint = lambda *args, **kwargs: print(*args, file=sys.stderr, **kwargs)

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
