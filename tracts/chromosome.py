from tracts.tract import Tract
from tracts.util import eprint
import numpy as np
import tkinter as tk

class Chrom:
    """ 
    A chromosome wraps a list of tracts, which form a partition on it. The
    chromosome has a finite, immutable length.

    Attributes
    ----------
    tracts: list of tract objects
        The list of tracts that span this chromosome.
    len: int
        The length of this chromosome, in Morgans.
    start: int
        The starting point of this chromosome, in Morgans. This is set to the starting point of the first known tract, which may be greater than zero if the chromosome starts with a segment of unknown ancestry.
    end: int
        The ending point of this chromosome, in Morgans. This is set to the ending point of the last known tract, which may be less than the chromosome's length if it ends with a segment of unknown ancestry.
    unknown_labels: set of strings
        The set of labels that are considered to correspond to unknown ancestry. This is used by the :func:`~tracts.chromosome.Chrom.smooth_unknown` method to identify which segments to remove.
    """

    def __init__(self, ls: int=None, label: str="POP", tracts: list[Tract]=None):
        """ 
        Constructor.

        Parameters
        ----------
        ls: int, default: None
            The length of this chromosome, in Morgans.
        label: string, default: "POP"
            An identifier categorizing this chromosome.
        tracts: list of tract objects, default: None
            The list of tracts that span this chromosome. If None is
            given, then a single, unlabeled tract is created to span the
            whole chromosome, according to the length len.
        """
        if tracts is None: # A single tract spanning the whole chromosome.
            self.len = ls
            self.tracts = [Tract(0, self.len, label)]
            self.start = 0
        else:
            if not tracts:
                raise ValueError("A non-empty list of tracts is required to initialize a chromosome.")
            self.tracts = tracts

            for t in self.tracts: # Set the chromosome's start attribute to the starting point of the first known tract.
                self.start = t.start
                if t.label != 'UNKNOWN':
                    break

            for t in self.tracts[-1::-1]:  # Set the chromosome's end attribute to the ending point of the last known tract. Iterate in reverse over tracts.
                self.end = t.end
                if t.label != 'UNKNOWN':
                    break

            # consider the length after stripping the UNKNOWN end tracts
            self.len = self.end - self.start

    def len(self) -> int:
        """ 
        Gets the length of this chromosome, in Morgans. 
        
        Returns
        -------
        int
            The length of this chromosome, in Morgans.
        """
        return self.len

    def goto(self, pos: int) -> int:
        """ 
        Finds the first tract containing a given position, in Morgans, and
        returns its index in the underlying list.

        Parameters
        ----------
        pos: int
            The position, in Morgans, to find.
        
        Returns
        -------
        int
            The index of the first tract containing the given position.        
        """
        
        if pos < 0 or pos > self.len: # Use binary search for this, since the tract list is sorted.
            raise ValueError("cannot seek to position outside of chromosome")

        low = 0
        high = len(self.tracts) - 1
        curr = (low + high + 1) // 2

        while high > low:
            if self.tracts[curr].start < pos:
                low = curr
            else:
                high = curr - 1
            curr = (low + high + 1) // 2

        return low

    def extract(self, start: int, end: int):
        """ 
        Extracts a segment from the chromosome.

        Parameters
        ----------
        start: int
            The starting point of the desired segment to extract.
        end: int
            The ending point of the desired segment to extract.

        Returns
        -------
        list
            A list of tract objects that span the desired interval.

        Notes
        -----
        Uses the :func:`~tracts.chromosome.Chrom.goto` method of this class to identify the starting and
        ending points of the segment, so if those positions are invalid,
        :func:`~tracts.chromosome.Chrom.goto` will raise a ValueError.
        """
        startpos = self.goto(start)
        endpos = self.goto(end)
        extract = [_tract.copy() for _tract in self.tracts[startpos:endpos + 1]]
        extract[0].start = start
        extract[-1].end = end
        return extract

    def plot(self, canvas: tk.Canvas, colordict: dict, height: float=0, chrwidth: float=.1):
        """
        Plots this chromosome on the provided canvas.

        Parameters
        ----------
        canvas: tk.Canvas
            The canvas to plot on.
        colordict: dict
            A dictionary mapping tract labels to colors, used to determine the color of each tract when plotting.
        height: float, default: 0
            The height at which to plot this chromosome. This is used to stack multiple chromosomes on top of each other when plotting a population.
        chrwidth: float, default: 0.1
            The width of the chromosome when plotting. This is used to stack the two copies of a chromosome pair on top of each other when plotting a population.
        """

        for current_tract in self.tracts:
            canvas.create_rectangle(
                100 * current_tract.start, 100 * height, 100 * current_tract.end,
                100 * (height + chrwidth),
                width=0, disableddash=True, fill=colordict[current_tract.label])

    def _smooth(self):
        """ 
        Combines adjacent tracts with the same label. The side effect is that the entire list of tracts is copied, so
        unnecessary calls to this method should be avoided.
        """

        if not self.tracts:
            eprint("Warning: smoothing empty chromosome has no effect")
            return None  # Nothing to smooth since there are no tracts.

        def same_ancestry(my, their):
            return my.label == their.label

        # TODO: determine whether copies are really necessary. This could be an avenue for optimization.
        newtracts = [self.tracts[0].copy()]
        for t in self.tracts[1:]:
            if same_ancestry(t, newtracts[-1]): # Extend the last tract added to encompass the next one.
                
                newtracts[-1].end = t.end
            else:
                newtracts.append(t.copy())

        self.tracts = newtracts

    def merge_ancestries(self, ancestries: list, newlabel: str):
        """ 
        Merges segments that are contiguous and either have the same ancestry or are labeled as belonging to a specified list.
        The label of each tract in the chromosome’s inner list is checked against the labels listed in `ancestries`. 
        If a match is found, the tract is relabeled to `newlabel`. This batch relabeling allows several technically
        different ancestries to be treated as equivalent by assigning them the same label. The resulting list is then smoothed to combine adjacent tracts with identical labels.
        This new list replaces the original `tracts` list.

        Parameters
        ----------
        ancestries: list of strings
            The ancestries to merge.
        newlabel: string    
            The identifier for the new ancestry to assign to the matching tracts.
        """
        for _tract in self.tracts:
            if _tract.label in ancestries:
                _tract.label = newlabel

        self._smooth()

    def smooth_unknown(self):
        """ 
        Removes segments of unknown ancestry. Unknown segments at begining and end of chromosomes are removed.
        Internal unknwon segments are removed, extending the neighboring
        segments to occupy the space previously assigned to the unknown segments.
        """
        while self.tracts and self.tracts[0].label in self.unknown_labels:
            self.tracts.pop(0)
        while self.tracts and self.tracts[-1].label in self.unknown_labels:
            self.tracts.pop()
        
        if not self.tracts:
            return        
        i = 0
        while i < len(self.tracts) - 1: # Find the first non-unknown after i
            
            j = i + 1 
            while j < len(self.tracts) and self.tracts[j].label in self.unknown_labels:
                j += 1

            # At this point, i<j<=len(self.tracts)
            # Collapse unknowns (possibly zero-length slice)
            left  = self.tracts[i]
            right = self.tracts[j]

            mid = (left.end + right.start) / 2.0
            left.end = mid
            right.start = mid

            del self.tracts[i+1:j] # Remove any unknowns

            i += 1
        self._smooth()

    def tractlengths(self):
        """ 
        Gets the list of tract lengths. Make sure that proper smoothing is implemented. 

        Returns
        -------
        list of tuples
            A list of tuples, where each tuple contains the ancestry label of a tract, the length of the tract, and the length of the chromosome.
        """
        self.smooth_unknown()
        return [(t.label, t.end - t.start, self.len) for t in self.tracts]

    def __iter__(self):
        """
        Returns an iterator over the tracts in this chromosome.

        Returns
        -------
        iterator
            An iterator over the tracts in this chromosome.
        """
        return self.tracts.__iter__()

    def __getitem__(self, index: int) -> Tract:
        """
        Simply wraps the underlying list's `__getitem__` method.

        Parameters
        ----------
        index: int
            The index of the tract to retrieve.

        Returns
        -------
        Tract
            The tract at the specified index.
        """
        return self.tracts[index]

    def __repr__(self):
        """
        Returns a string representation of the chromosome.

        Returns
        -------
        str
            A string representation of the chromosome, showing its tracts.
        """
        return "chrom(tracts=%s)" % (repr(self.tracts),)

    def is_equal(self, chrom) -> bool:
        """ 
        Check if two chromosomes are equal, in terms of their tracts.
        
        Parameters
        ----------
        chrom: Chrom
            The chromosome to compare to.

        Returns
        -------
        bool
            `True` if the two chromosomes have the same tracts, `False` otherwise.
        """
        if len(chrom.tracts) != len(self.tracts):
            return False
        for i in range(len(self.tracts)):
            if not self.tracts[i].is_equal(chrom.tracts[i]):
                return False
        return True
class Chropair:
    """ 
    A pair of chromosomes. Using chromosome pairs allows modeling of diploid individuals.

    Attributes
    ----------
    copies: list of Chrom
        The two copies of this chromosome pair. Each copy is a :class:`~tracts.chromsome.Chrom` object.
    len: int
        The length of this chromosome pair, in Morgans. This is set to the length of the first chromosome copy, and the second copy is checked to have the same length.    
    """

    def __init__(self, chroms: list[Chrom] | tuple[Chrom] = None, chropair_len: int = 1, auto: bool = True, label: str = "POP"):
        """
        Can be instantiated either by explicitly providing two chromosomes as a tuple, or
        by specifying an ancestry label, length, and autosome status.
        
        Parameters
        ----------
        chroms: list of Chrom or tuple of Chrom, default: None
            The two chromosomes to form this chromosome pair. If None is given, then two identical chromosomes are created according to the other parameters.
        chropair_len: int, default: 1
            The length of this chromosome pair, in Morgans. This is used if `chroms` is None to create two identical chromosomes of the specified length.
        auto: bool, default: True
            Whether this chromosome pair is an autosome. This is used if `chroms` is None to create two identical chromosomes with the specified autosome status.
        label: str, default: "POP"
            An identifier categorizing this chromosome pair. This is used if `chroms` is None to create two identical chromosomes with the specified label.        
        """

        if chroms is None:
            self.copies = [Chrom(chropair_len, auto, label), Chrom(chropair_len, auto, label)]
            self.len = chropair_len
        else:
            if chroms[0].len != chroms[1].len:
                raise ValueError('Chromosome pairs of different lengths!')
            self.len = chroms[0].len
            self.copies = chroms

    def recombine(self) -> Chrom:
        """
        Recombine this chromosome pair.

        Returns
        -------
        Chrom
            A new chromosome resulting from recombination of the two copies of this chromosome pair.
        
        Notes
        -----
        Recombination is modeled as a Poisson process along the chromosome, with the number of recombinations drawn from a Poisson
        distribution with mean equal to the chromosome's length in Morgans. The positions of the recombinations are drawn
        uniformly at random along the chromosome. The resulting chromosome is formed by alternating between
        the two copies of this chromosome pair at each recombination point, starting with a randomly chosen copy.
        """
        
        n = np.random.poisson(self.len) # Decide on the number of recombinations
        unif = (self.len * np.random.random(n)).tolist() # Get recombination points
        unif.extend([0, self.len])
        unif.sort()
        startchrom = np.random.random_integers(0, 1) # Start with a random chromosome
        tractlist = []
        for startpos in range(len(unif) - 1):
            tractlist.extend(
                self.copies[(startchrom + startpos) % 2]
                .extract(unif[startpos],
                         unif[startpos + 1]))
        newchrom = Chrom(self.copies[0].len, self.copies[0].auto)
        newchrom.init_list_tracts(tractlist)
        return newchrom

    def applychrom(self, func: callable):
        """
        Apply `func` to chromosomes.
        
        Parameters
        ----------
        func: callable
            A function that takes a chromosome as input and returns some output. This function is applied to each copy of this chromosome pair, and the results are returned in a list.
        
        Returns
        -------
        list
            A list containing the results of applying `func` to each copy of this chromosome pair.
        """
        ls = []
        for copy in self.copies:
            ls.append(func(copy))
        return ls

    def plot(self, canvas: tk.Canvas, colordict: dict, height=0):
        """
        Plot the chromosome pair on a tkinter canvas.

        Parameters
        ----------
        canvas : tk.Canvas
            The tkinter canvas on which to draw the chromosome pair.
        colordict: dict
            A dictionary mapping chromosome types to colors.
        height: float
            The vertical position at which to plot the chromosome pair.
        """
        self.copies[0].plot(canvas, colordict, height=height + 0.1)
        self.copies[1].plot(canvas, colordict, height=height + 0.22)

    def __iter__(self):
        """
        Simply wraps the underlying list's `__iter__` method.

        Returns
        -------
        iterator
            An iterator over the two copies of this chromosome pair.
        """
        return self.copies.__iter__()

    def __getitem__(self, index: int) -> Chrom:
        """
        Get the copy of the chromosome at the specified index.

        Parameters
        ----------
        index: int
            The index of the copy to retrieve.

        Returns
        -------
        Chrom
            The copy of the chromosome at the specified index.
        """
        return self.copies[index]
    
        
