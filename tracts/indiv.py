from tracts.haploid import Haploid
from tracts.chromosome import Chropair, Chrom
import tkinter as tk
import numpy as np
import logging
logger=logging.getLogger(__name__)
class Indiv:
    """ 
    The class of diploid individuals. An individual is thought of as a list of pairs of chromosomes. Equivalently, a diploid individual
    is a pair of haploid individuals. Thus, it is possible to construct instances of this class from a pair
    of instances of the haploid class, as well as directly from a sequence of chropair instances.

    The interface for loading individuals from files uses the haploid-oriented approach, since individual .bed files describe only
    one haplotype. The loading process is the following (i) load haploid individuals for each haplotype and (ii) combine the haploid
    individuals into a diploid individual.
    
    Attributes
    ----------
    Ls: list of float
        The lengths of the chromosomes.
    chroms: list of :class:`tracts.chromosome.Chropair`   
        The chromosome pairs that make up this individual. See the documentation for :class:`~tracts.chromosome.Chropair`.
    name: str
        An optional name for the individual.
    allosomes: dict
        A dictionary mapping chromosome labels to chromosome objects, for chromosomes that are treated as allosomes.        
    """

    @staticmethod
    def from_haploids(haps: list[Haploid], name: str = None, allosome_labels: list[str] | None = None):
        """
        Construct a diploid individual from a list of two haploid individuals.

        Parameters
        ----------
        haps: list of :class:`tracts.haploid.Haploid`
            A list of two haploid individuals to combine into a diploid individual.
        name: str, optional
            An optional name for the individual. If not provided, the name will be set to the name of the first haploid individual.
        allosome_labels: list of str, optional
            An optional list of chromosome labels that should be treated as allosomes. If not provided, no chromosomes will be treated as allosomes. Chromosome labels should be strings corresponding to the chromosome identifiers in the haploid individuals (e.g., "X" for the X chromosome). Chromosome labels that are not present in the haploid individuals will be ignored, and no chromosomes will be treated as allosomes.
        
        Returns
        -------
        Individual
            An instance of the :class:`Indiv` class representing the combined diploid individual.        
        """
        
        allosome_labels = [] if allosome_labels is None else allosome_labels
        
        if len(haps) != 2:
            raise ValueError('Two haplotypes must given to construct a diploid individual.')
        name = haps[0].name if name is None else name

        chroms = [Chropair(t) for t in zip(*[hap.chroms for hap in haps])]
        allosomes={}
        if allosome_labels is not None:
            for key in allosome_labels:
                allosomes[key] = [hap.allosomes[key] for hap in haps if key in hap.allosomes]
                if not allosomes[key]:
                    logger.warning(f"Allosome {key} was not found when reading individual {name} from file.")

        return Indiv(chroms=chroms, Ls=haps[0].Ls, allosomes=allosomes, name=name)

    @staticmethod
    def from_files(paths: list[str], selectchrom: str | None = None, name: str | None = None, allosomes: list[str] | None = None):
        """ 
        Constructs a diploid individual from two files, which describe the individuals haplotypes.

        Parameters
        ----------
        paths: list of str
            A list of two file paths, each describing one haplotype of the individual. The files should be tab-delimited text files with columns: chrom, start, end, label, and optionally others. The first line may be a header, which will be automatically skipped if it contains the expected column names.
        selectchrom: str, optional
            An optional string specifying which chromosomes to select from the files. If not provided, all chromosomes will be selected. The string should contain chromosome identifiers separated by commas (e.g., "1,2,3"). Chromosome identifiers should be integers corresponding to the chromosome numbers in the files (e.g., "1" for chromosome 1). Chromosome identifiers that cannot be converted to integers will be ignored, and the corresponding chromosomes will not be selected.
        name: str, optional
            An optional name for the individual. If not provided, the name will be set to the name of the first haploid individual loaded from the files.
        allosomes: list of str, optional
            An optional list of chromosome labels that should be treated as allosomes. If not provided, no chromosomes will be treated as allosomes. Chromosome labels should be strings corresponding to the chromosome identifiers in the files (e.g., "X" for the X chromosome). Chromosome labels that are not present in the files will be ignored, and no chromosomes will be treated as allosomes.
        
        Returns
        -------
        Individual
            An instance of the :class:`Indiv` class representing the combined diploid individual loaded from the files.        
        """

        allosomes = [] if allosomes is None else allosomes
        
        if len(paths) != 2:
            raise ValueError('More than two paths supplied to construct a diploid individual.')

        return Indiv.from_haploids(haps = [Haploid.from_file(path=path,
                                                            name=name,
                                                            selectchrom=selectchrom,
                                                            allosome_labels=allosomes)
                                                            for path in paths],
                                    name = name,
                                    allosome_labels=allosomes)

    def __init__(self, Ls:list[float]|None=None, label:str="POP", fname:str|None=None, 
                 labs: tuple[str, str]=("_A", "_B"), selectchrom: list[int]|None=None, chroms: list[Chropair] | None = None, 
                 allosomes: dict[str, list[Chrom]]=None, name:str|None=None):
        """ 
        Constructs a diploid individual. There are several ways to build individuals, either from files, from existing data, or
        programmatically. The most straightforward way to build an individual is from existing data, by supplying
        only the ``Ls`` and ``chroms`` arguments.

        Parameters
        ----------
    	Ls : list of floats
         	Default is ``None``. The lengths of the chromosomes in the order in which they appear in `chroms`.
        chroms : list of chropair objects
            Default is ``None``. The chromosome pairs that make up this individual. See the documentation for :class:`~tracts.chromosome.Chropair`.
        label : string
            Default is ``POP``. The label to use for building single-tract chromosomes when no other data is given to buid this individual.
        fname : 2-tuple of str
            Default is ``None``. Paths are generated by concatenating the first component of ``fname``, each label from ``labs`` in turn, and the second
            component of ``fname``.
        labs : 2-tuple of str
            Default is ``("_A", "_B")``. The labels used to identify maternal and paternal haplotypes in the paths leading to .bed files.
        selectchrom : list[int] | None
            Default is ``None``. Selects which chromosomes to load. The default value of ``None`` selects all chromosomes.
        name : string
            Default is ``None``. An identifier for this individual.

        
        Notes
        -----
	    If ``Ls`` is given, but ``chroms`` is not, then chromosomes consisting each of a single tract will be created with the label ``label`` and lengths drawn from ``Ls``. If the ``fname`` argument is given, the constructor will perform path manipulation involving the components of `fname` and `labs` to generate file names that are commonly used when dealing with .bed files. The facilities in this constructor for loading individuals from files are deprecated. It is recommended to instead use the static methods :func:`~tracts.indiv.Indiv.from_files` or :func:`~tracts.indiv.Indiv.from_haploids`.
        """

        if fname is None:
            self.Ls = Ls
            if name:    
                self.name = name
            else:
                self.name = fname[0].split('/')[-1]  
            if chroms is None:
                self.chroms = [Chropair(chropair_len=length, label=label) for length in Ls]
            else:
                self.chroms = chroms
            self.allosomes=allosomes
        else:
            fnames = [fname[0] + lab + fname[1] for lab in labs]
            i = Indiv.from_files(fnames, selectchrom)
            if name:    
                self.name = name
            else:
                self.name = fname[0].split('/')[-1]  
            self.chroms = i.chroms
            self.Ls = i.Ls
            self.allosomes=i.allosomes
        self.canvas = None

    def plot(self, colordict: dict, win: tk.Tk=None):
        """
        Plots an individual. 
        
        Parameters
        ----------
        colordict: dict
            A dictionary mapping population labels to colors, used to determine the color of each tract when plotting. E.g.: ``colordict = {"CEU":'r',"YRI":b}``.
        win: tk.Tk, optional
            An optional Tkinter window to plot on. If not provided, a new window will be created for this plot. If provided, the plot will be drawn on the given window instead of creating a new one. This can be used to plot multiple individuals on the same window, or to integrate the plot of this individual into a larger Tkinter application.
        
        Returns
        -------
        tk.Tk
            The Tkinter window on which the plot was drawn.
        """
        if win is None:
            win = tk.Tk()
        self.canvas = tk.Canvas(win, width=250, height=len(self.Ls) * 30, bg='white')

        for i in range(len(self.chroms)):
            self.chroms[i].plot(self.canvas, colordict, height=int(i * .3))

        self.canvas.pack(expand=tk.YES, fill=tk.BOTH)

        return win

    def create_gamete(self):
        """
        Creates a haploid gamete from the individual.

        Returns
        -------
        Haploid
            A haploid genome representing a gamete produced by this individual. The gamete is generated by recombining the chromosome pairs of this individual, and then taking one chromosome from each pair.  
        """
        lsc = [chpair.recombine() for chpair in self.chroms]
        return Haploid(self.Ls, lsc)

    def applychrom(self, func:callable):
        """ 
        Apply the function `func` to each chromosome of the individual.
        
        Parameters
        ----------
        func: callable
            A function that takes a chromosome as input and returns a value. This function will be applied to each chromosome of the individual, and the results will be collected into a list.

        Returns
        -------
        list
            A list containing the results of applying `func` to each chromosome of the individual. The order of the results corresponds to the order of the chromosomes in the individual's `chroms` attribute.
        """
        return map(lambda c: c.applychrom(func), self.chroms)

    def ancestryAmt(self, ancestry: str):
        """
        Calculates the total length of the genome in segments of the given ancestry.
        
        Parameters
        ----------
        ancestry: str
            The ancestry for which to calculate the total length.

        Returns
        -------
        float
            The total length of the genome in segments of the given ancestry.
        """
        return np.sum(
            t.len()
            for t
            in self.iflatten()
            if t.label == ancestry)

    def ancestryProps(self, ancestries: list, allosome_label: bool = False, cutoff: float = 0.0):
        """
        Calculates the proportion of the genome represented by the given ancestries.
        
        Parameters
        ----------
        ancestries: list of str
            A list of ancestries for which to calculate the proportions.
        allosome_label: str or bool, optional
            An optional label for the allosome chromosome to consider when calculating ancestry proportions. If `False` (the default), allosomes will not be considered when calculating ancestry proportions. If a string is provided, only the chromosome with the corresponding label in `self.allosomes` will be considered when calculating ancestry proportions. If the specified label is not present in `self.allosomes`, a warning will be logged and no chromosomes will be considered as allosomes.
        cutoff: float, optional
            An optional cutoff value for tract lengths. Only tracts with lengths greater than this cutoff will be considered when calculating ancestry proportions. The default value is 0, meaning that all tracts will be considered regardless of their length.  
        
        Returns
        -------
        list of float
            A list of proportions corresponding to the input list of ancestries, where each proportion represents the fraction of the genome that is represented by segments of the corresponding ancestry. The order of the proportions corresponds to the order of the input ancestries.
        """
        
        gen = ((t.len(), [t.len() if (t.label == a  and t.len() > cutoff) else 0 for a in ancestries])
               for t in self.iflatten(allosome_label = allosome_label))

        all_lengths, all_ancestry_lengths = zip(*gen)
        total_length = float(np.sum(all_lengths))
        ancestry_sums = map(np.sum, zip(*all_ancestry_lengths))

        return [ancestry_sum * 1. / total_length for ancestry_sum in ancestry_sums]

    def ancestryPropsByChrom(self, ancestries: list[str]):
        """
        Calculates the proportion of the genome represented by the given ancestries, separately for each chromosome.

        Parameters
        ----------
        ancestries: list of str
            A list of ancestries for which to calculate the proportions.
        
        Returns
        -------
        list of list of float
            A list of lists of proportions, where the outer list corresponds to the input list of ancestries, and the inner lists correspond to the chromosomes of the individual. Each inner list contains the proportions of the corresponding chromosome that are represented by segments of the corresponding ancestry. The order of the outer list corresponds to the order of the input ancestries, and the order of the inner lists corresponds to the order of the chromosomes in the individual's `chroms` attribute.
        """
        dat = self.applychrom(Chrom.tractlengths)
        dictamt = {}
        nc = len(list(dat))
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

        return [[dictamt[ancestry][i] * 1. / tots[i]
                 for i in range(nc)]
                for ancestry in ancestries]

    def iflatten(self, allosome_label: str | bool | None = False):
        """
        Lazily flatten this individual to the tract level. 
        
        Parameters
        ----------
        allosome_label: str or bool, optional
            An optional label for the allosome chromosome to consider when flattening. If `False` (the default), allosomes will not be considered when flattening. If a string is provided, only the chromosome with the corresponding label in `self.allosomes` will be considered when flattening. If the specified label is not present in `self.allosomes`, a warning will be logged and no chromosomes will be considered as allosomes.
        """
        if allosome_label:
            chromosome_considered = [self.allosomes[allosome_label]]
            for _chrom in chromosome_considered:
                for _copy in _chrom:
                    for _tract in _copy:
                        yield _tract
        else:
            chromosome_considered = self.chroms 
            for _chrom in chromosome_considered:
                for _copy in _chrom.copies:
                    for _tract in _copy.tracts:
                        yield _tract

    def flat_imap(self, f:callable):
        """ 
        Lazily map a function over the full underlying structure of this individual. 

        Parameters
        ----------
        f: callable
            A function that takes three parameters: `chrom`, the chromosome pair containing the tract, `copy`, the chromosome containing the tract and `tract`, the tract itself. This function will be applied to each tract in the individual's genome, and the results will be collected into a list. The order of the results corresponds to the order of the tracts in the individual's genome, as determined by iterating through the chromosomes, then the copies within each chromosome, and then the tracts within each copy.

        Returns
        -------
        list
            A list containing the results of applying `f` to each tract in the individual's genome. 
        """
        for _chrom in self.chroms:
            for _copy in _chrom.copies:
                for _tract in _copy.tracts:
                    yield f(_chrom, _copy, _tract)

    def __iter__(self):
        """
        Iterate over the chromosomes in the individual.

        Returns
        -------
        iterator
            An iterator over the chromosome pairs in the individual's `chroms` attribute. The order of the chromosome pairs corresponds to the order of the chromosomes in the individual's genome. 
        """
        return self.chroms.__iter__()

    def __getitem__(self, index: int):
        """
        Get the chromosome pair at the specified index.

        Parameters
        ----------
        index: int
            The index of the chromosome pair to retrieve. The index should be an integer corresponding to the position of the chromosome pair in the individual's `chroms` attribute (e.g., 0 for the first chromosome pair, 1 for the second chromosome pair, etc.). If the index is out of bounds (i.e., less than 0 or greater than or equal to the number of chromosome pairs), an `IndexError` will be raised.
        
        Returns
        -------
        Chropair
            The chromosome pair at the specified index in the individual's `chroms` attribute.
        """
        return self.chroms[index]
