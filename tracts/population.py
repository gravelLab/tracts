from tracts.indiv import Indiv
from tracts.util import eprint
from tracts.chromosome import Chrom, Tract
from tracts.demography import SexType
import numpy as np
import tkinter as tk
from matplotlib import pylab
from tkinter import filedialog
import bisect
from collections import defaultdict
from tracts.logs import get_current_func_info
import logging
logger = logging.getLogger(__name__)

#------------ Helper functions ------------

def collect_pop(flatdat: list[Tract]) -> dict[str, list[Tract]]:
    """
    Organizes a list of tracts into a dictionary keyed on ancestry labels.

    Parameters
    ----------
    flatdat: list[Tract]
        A list of tracts, where each tract has a "label" attribute corresponding to its ancestry label.
    
    Returns
    -------
    dict[str, list[Tract]]
        A dictionary where the keys are ancestry labels and the values are lists of tracts with that ancestry label.    
    """
    dic = defaultdict(list)
    for t in flatdat:
        dic[t.label].append(t)
    return dic


def preprocess_color_dict(colordict: dict[str, str], dat: tuple[dict[str, float], list[dict[str, float]]]) -> None:
    """
    Preprocesses the color dictionary to ensure that all ancestry labels present in the data have an entry in the color dictionary.
    If an ancestry label is missing from the color dictionary, it is added with a default color of 'black'.

    Parameters
    ----------
    colordict: dict[str, str]
        A dictionary mapping ancestry labels to color strings.
    dat: tuple[dict[str, float], list[dict[str, float]]]
        A tuple where the first element is a dictionary of ancestry proportions at different positions, and the second element is a list of dictionaries of ancestry proportions at different positions.
        Each dictionary in the list corresponds to a position and has ancestry labels as keys and their corresponding proportions as values.
    """
    for pop, color in colordict.items():
        for pos in dat[1]:
            try:
                pos[0][pop]
            except KeyError:
                pos[0][pop] = 0
                pos[1][pop] = 0


def _split_indivs(indivs: list[Indiv], count: int, sort_ancestry: str | None=None) -> list[list[Indiv]]:
    """
    Internal function used to split a list of individuals into equally
    sized groups after sorting the individuals according to the proportion
    of the ancestry identified by `sort_ancestry`. When no `sort_ancestry`
    is provided, the ancestry of the first tract of the first chromosome
    copy of the first chromosome of the first individual in the list is used.

    Notes
    -----
    Not documented.
    """
    if sort_ancestry is None:
        sort_ancestry = indivs[0].chroms[0].copies[0].tracts[0].label

    s_indivs = sorted(indivs, key=lambda i: i.ancestryProps([sort_ancestry])[0])

    n = len(indivs)
    group_frac = 1.0 / count

    groups = [s_indivs[int(n * i * group_frac):int(n * (i + 1) * group_frac)] for i in range(count)]

    return groups

# ------------ Population class ------------
class Population:
    """
    A class representing a population of diploid individuals. A :class:`~tracts.population.Population` is
    a list of :class:`~tracts.indiv.Indiv` objects.

    Attributes
    ----------
    currentplot: int
        The index of the currently plotted individual.
    win: tk.Tk
        The Tkinter window used for plotting.
    canv: tk.Canvas
        The Tkinter canvas used for plotting the current individual. This is stored as an attribute to allow for updating the plot when navigating between individuals.
    chro_canvas: tk.Canvas
        The Tkinter canvas used for plotting chromosomes.
    colordict: dict[str, str]
        A dictionary mapping ancestry labels to color strings.
    _flats: list[Tract] | None
        A cached flattened list of tracts for the population. If None, the flattened list has not been computed yet.
    allosome_labels: list[str]
        A list of labels for the allosomes in the population.
    allosome_lengths: dict[str, float]
        A dictionary mapping allosome labels to their lengths.
    indivs: list[Indiv]
        A list of individuals in the population.
    male_list: list[str]
        A list of labels for male individuals in the population. This is used to determine the sex of individuals when the sex cannot be inferred from the data. If None, the sex of individuals will be inferred from the data by checking the number of X chromosomes.
    nind: int
        The number of individuals in the population.
    Ls: list[float]
        A list of chromosome lengths for the individuals in the population. It is assumed that all individuals have the same chromosome lengths.
    maxLen: float
        The maximum chromosome length among the individuals in the population.
    num_males: int
        The number of male individuals in the population.
    num_females: int
        The number of female individuals in the population.
    """

    def __init__(self, list_indivs: list[Indiv]=None, names:list[str]=None, fname:str=None,
                labs:list[str]=("_A", "_B"), selectchrom:list[str]=None, allosomes:list[str]=[], 
                ignore_length_consistency:bool=False, filenames_by_individual:list[str]=None, male_list: list[str] | str = None):
        """
        Initializes the :class:`~tracts.population.Population` class.

        Parameters
        ----------
        list_indivs: list[Indiv]
            A list of :class:`~tracts.indiv.Indiv` objects representing the individuals in the population. If provided, this will be used to initialize the population directly.
        names: list[str]
            A list of names for the individuals in the population. This is used when initializing the population from a file format.
        fname: str
            A tuple with the start, middle and end of the filenames for loading individuals from files. The individual files should be specified in the format `start--Indiv--Middle--_A--End`.
        labs: list[str]
            A list of labels for the chromosome copies. This is used when initializing the population from a file format.
        selectchrom: list[str]
            A list of chromosome labels to select when initializing the population from a file format. If None, all chromosomes will be selected.
        allosomes: list[str]
            A list of labels for the allosomes in the population. This is used when initializing the population from a file format to identify which chromosomes are allosomes.
        ignore_length_consistency: bool
            A flag indicating whether to ignore consistency in chromosome lengths across individuals when initializing the population from a file format. If False, an error will be raised if individuals have different chromosome lengths. If True, the population will be initialized even if individuals have different chromosome lengths.
        filenames_by_individual: dict[str, list[str]]
            A dictionary mapping individual names to lists of filenames for loading individuals from files. The individual files should be specified in the format `start--Indiv--Middle--_A--End`. This is an alternative to using `fname` and `names` for loading individuals from files, and allows for more flexibility in specifying the filenames for each individual.
        male_list: list[str] | str
            A list of labels for male individuals in the population. 
              
        Notes
        -----
        There are two ways to build populations, either from a dataset stored in files or from a list of individuals. The facilities for
        loading populations from files present in this constructor are deprecated. It is advised to instead load a list of individuals,
        using :func:`tracts.indiv.Indiv.from_files`, and to then pass that list to this constructor.

        The population can be initialized by providing it with a list `list_indivs` of :class:`~tracts.indiv.Indiv` objects, 
        or a file format `fname` and a list `names` of names. If reading from a file, `fname` should be a tuple with the start,
        middle and end of the filenames, where an individual file is specified by `start--Indiv--Middle--_A--End`. Otherwise, provide list of individuals.
        """
        
        self.currentplot = None
        self.win = None
        self.chro_canvas = None
        self.colordict = None
        self._flats = None
        self.canv = None
        self.allosome_labels=allosomes
        self.allosome_lengths: dict[str, float]={}
        self.indivs: list[Indiv] = []
        if male_list is None:
            self.male_list: list[str] = [] # Will be populated by labels of male individuals.
        else:
            self.male_list = male_list 
        if list_indivs is not None:
            self.indivs: list[Indiv] = list_indivs
            self.nind = len(list_indivs)
            self.Ls = self.indivs[0].Ls
            assert all(i.Ls == self.indivs[0].Ls for i in self.indivs), "Individuals have genomes of different lengths."
            self.maxLen = max(self.Ls)
        elif filenames_by_individual is not None:
            for name, files in filenames_by_individual.items():
                try:
                    self.indivs.append(
                        Indiv.from_files(paths=files,
                                        name=name,
                                        selectchrom=selectchrom,
                                        allosomes=allosomes))
                except Exception as e:
                    raise IndexError(f'Files for individual {name} ({files}) could not be found.') from e

            self.nind = len(self.indivs)
            self.Ls = self.indivs[0].Ls
            if not all(i.Ls == self.indivs[0].Ls for i in self.indivs) and not ignore_length_consistency: # Check that all individuals have the same length.
                raise ValueError('Individuals have genomes of different lengths. If this is intended, set ignore_length_consistency to True.')
            self.maxLen = max(self.Ls)
        elif fname is not None:
            for name in names:
                try:
                    self.indivs.append(
                        Indiv.from_files(paths=[fname[0] + name + fname[1] + lab + fname[2] for lab in labs],
                                        name=name,
                                        selectchrom=selectchrom,
                                        allosomes=allosomes))
                except IndexError:
                    eprint("Error reading individual", name)
                    eprint("fname=", fname, "; labs=", labs, ", selectchrom=", selectchrom)
                    raise IndexError

            self.nind = len(self.indivs)

            assert (ignore_length_consistency or (all(i.Ls == self.indivs[0].Ls for i in self.indivs)))

            self.Ls = self.indivs[0].Ls
            self.maxLen = max(self.Ls)
        else:
            raise ValueError('Population could not be loaded because individuals were not specified.')
        
        self.allosome_lengths=self.calculate_allosome_lengths(self.indivs, self.allosome_labels)
        if male_list is None:
            self.num_males, self.num_females = self.calculate_num_sexes(self.indivs, self.allosome_labels)
        else:
            self.num_males = len(male_list) 
            self.num_females = self.nind -  self.num_males

    @staticmethod
    def calculate_allosome_lengths(indivs: list[Indiv], allosome_labels: list[str]):
        """
        Calculate the lengths of allosomes across individuals.

        Parameters
        ----------
        indivs: list[Indiv]
            A list of individuals in the population.
        allosome_labels: list[str]
            A list of labels for the allosomes in the population. This is used to identify which chromosomes are allosomes.
        
        Returns
        -------
        dict[str, float]
            A dictionary mapping allosome labels to their lengths. The length of an allosome is determined by the length of the chromosome with that label in the first individual that has that allosome. It is assumed that all individuals have the same lengths for their allosomes, and an error is raised if this is not the case.
        """

        allosome_lengths={}
        for indiv in indivs:
            for allosome_label in allosome_labels:
                if allosome_label in indiv.allosomes:
                    for chrom in indiv.allosomes[allosome_label]:
                        if allosome_label not in allosome_lengths:
                            allosome_lengths[allosome_label] = chrom.len
                            continue
                        assert (allosome_lengths[allosome_label] == chrom.len), f"Allosome {allosome_label} has inconsistent length across individuals ({indiv.name}.)"
        return allosome_lengths


    @staticmethod
    def calculate_num_sexes(indivs: list[Indiv], allosome_labels: list[str]):
        """
        Calculate the number of males and females in the population based on their allosome composition.
        If the allosome labels do not include 'X', a warning is raised and the number of males and females
        is recorded as zero.

        Parameters
        ----------
        indivs: list[Indiv]
            A list of individuals in the population.
        allosome_labels: list[str]
            A list of labels for the allosomes in the population. This is used to identify which chromosomes are allosomes and to determine the sex of individuals based on their allosome composition.
        
        Returns
        -------
        tuple[int, int]
            A tuple containing the number of males and females in the population.

        Notes
        -----
        Currently, the function only checks for the presence of 'X' in the allosome labels. 
        """
        file_name, func_name, line_number = get_current_func_info()
        if 'X' not in allosome_labels:
            logger.warning("X is not in the allosomes of this population. The number of males and females will be recorded as 0.")
            logger.warning(f"If using different sex chromosomes please change this function: {func_name} in {file_name} at line {line_number}.")
            return 0, 0
        
        num_males = 0
        num_females = 0
        for indiv in indivs:
            if 'X' in indiv.allosomes:
                if len(indiv.allosomes['X'])==1:
                    num_males+=1
                else:
                    num_females+=1
        return num_males, num_females


    def split_by_props(self, count: int):
        """ 
        Splits this population into groups according to their ancestry proportions. 
        The individuals are sorted in ascending order of their ancestry named `anc`.

        Parameters
        ----------
        count: int
            The number of groups to split the population into. If `count` is 1, the function returns a list containing this population without splitting.
        
        Returns
        -------
        list[Population]
            A list of `count` Population objects, each containing a group of individuals from the original population. 
        """
        if count == 1:
            return [self]

        return [Population(g) for g in _split_indivs(self.indivs, count)]

    def newgen(self):
        """
        Build a new generation from this population.
        
        Returns
        -------
        Population
            A new Population object representing the next generation.
        """
        return Population([self.new_indiv() for _i in range(self.nind)])

    def new_indiv(self):
        """
        Creates a new individual by randomly selecting two parents from the population, creating gametes from each parent, and combining those gametes to form a new individual.

        Returns
        -------
        Indiv
            A new Indiv object representing the offspring of the two randomly selected parents.
        """
        rd = np.random.random_integers(0, self.nind - 1, 2)
        while rd[0] == rd[1]:
            rd = np.random.random_integers(0, self.nind - 1, 2)
        gamete1 = self.indivs[rd[0]].create_gamete()
        gamete2 = self.indivs[rd[1]].create_gamete()
        return Indiv.from_haploids([gamete1, gamete2])

    def save(self):
        """
        Saves the current plot of the population to a file. The user is prompted to choose a file location and name for saving the plot.
        """
        file = filedialog.asksaveasfilename(parent=self.win, title='Choose a file')
        self.indivs[self.currentplot].canvas.postscript(file=file)

    def list_chromosome(self, chronum: int):
        """
        Collects the chromosomes with the given number across the whole population.

        Parameters
        ----------
        chronum: int
            The index of the chromosome to collect across the population. It is assumed that all individuals have the same number of chromosomes and that the chromosome with index `chronum` corresponds to the same chromosome across individuals.
        
        Returns
        -------
        list[Chrom]
            A list of Chrom objects corresponding to the chromosome with index `chronum` across all individuals in the population.
        """
        return [curr_indiv.chroms[chronum] for curr_indiv in self.indivs]

    def ancestry_at_pos(self, select_chrom: int = 0, pos: int = 0, cutoff: float = 0.0):
        """ 
        Finds ancestry proportion at specific position. The cutoff is used to look only at tracts that extend beyond a given position.
        
        Parameters
        ----------
        select_chrom: int, default 0
            The index of the chromosome to analyze. It is assumed that all individuals have the same number of chromosomes and that the chromosome with index `select_chrom` corresponds to the same chromosome across individuals.
        pos: int, default 0
            The position along the chromosome at which to calculate ancestry proportions. It is assumed that all individuals have the same chromosome lengths and that the position `pos` corresponds to the same location along the chromosome across individuals.
        cutoff: float, default 0.0
            A threshold for the length of ancestry tracts to consider when calculating ancestry proportions. Only tracts that extend beyond the position `pos` by at least `cutoff` will be included in the calculation of ancestry proportions.
        
        Returns
        -------
        tuple[dict[str, int], dict[str, float]]
            A tuple containing two dictionaries. The first dictionary maps ancestry labels to the count of tracts of that ancestry that extend beyond the position `pos` by at least `cutoff`. The second dictionary maps ancestry labels to the average length of tracts of that ancestry that extend beyond the position `pos` by at least `cutoff`.
        """
        ancestry = {}
        longancestry = {} # Keep track of ancestry of long segments
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
            if totlength[key] == 0: # Prevent division by zero
                totlength[key] = 0
            else:
                totlength[key] = totlength[key] / float(ancestry[key])

        return ancestry, totlength

    def ancestry_per_pos(self, select_chrom: int = 0, npts: int = 50, cutoff: float = 0.0):
        """
        Prepare the ancestry per position across chromosome.
        
        Parameters
        ----------
        select_chrom: int, default 0
            The index of the chromosome to analyze. It is assumed that all individuals have the same number of chromosomes and that the chromosome with index `select_chrom` corresponds to the same chromosome across individuals.
        npts: int, default 50
            The number of positions along the chromosome at which to calculate ancestry proportions. The positions will be evenly spaced along the chromosome, starting from position 0 and ending at the length of the chromosome.
        cutoff: float, default 0.0
            A threshold for the length of ancestry tracts to consider when calculating ancestry proportions. Only tracts that extend beyond the position `pos` by at least `cutoff` will be included in the calculation of ancestry proportions.
        
        Returns
        -------
        np.ndarray
            An array of positions along the chromosome at which ancestry proportions were calculated. The positions are evenly spaced along the chromosome, starting from position 0 and ending at the length of the chromosome.
        list[tuple[dict[str, int], dict[str, float]]]
            A list of tuples, where each tuple corresponds to a position along the chromosome and contains two dictionaries. The first dictionary maps ancestry labels to the count of tracts of that ancestry that extend beyond the corresponding position by at least `cutoff`. The second dictionary maps ancestry labels to the average length of tracts of that ancestry that extend beyond the corresponding position by at least `cutoff`.
        """
        length = self.indivs[0].chroms[Chrom].len  # Get chromosome length
        plotpts = np.arange(0, length, length / float(npts))  # Get number of points at which to plot ancestry
        return plotpts, [self.ancestry_at_pos(select_chrom=select_chrom, pos=pt, cutoff=cutoff) for pt in plotpts]

    def applychrom(self, func:callable, indlist:list=None):
        """ 
        Apply a function to chromosomes.

        Parameters
        ----------
        func: callable
            A function that takes a Chrom object as input and returns a value. This function will be applied to each chromosome in the population.
        indlist: list, default None
            A list of individuals to which the function should be applied. If None, the function will be applied to all individuals in the population.
        
        Returns
        -------
        list
            A list of the results of applying the function `func` to each chromosome in the population (or to the chromosomes of the individuals in `indlist` if it is not None).
        """
        ls = []

        if indlist is None:
            inds = self.indivs
        else:
            inds = indlist

        for ind in inds:
            ls.append(ind.applychrom(func))
        return ls

    def flatpop(self, ls:list|None=None):
        """ 
        Returns a flattened version of a population-wide list at the tract level,
        and throws away the start and end information of the tract.

        Parameters
        ----------
        ls: list | None, default None
            A list of tracts to flatten. If None, the function will flatten the complete list of tracts contained in this population. If a list is provided, the function will flatten that list instead of the complete list of tracts in the population.
        
        Returns
        -------
        list[Tract]
            A list of Tract objects representing the flattened version of the input list of tracts (or the complete list of tracts in the population if `ls` is None). The start and end information of the tracts is discarded in the returned list.
        """
        if ls is not None:
            return list(self.iflatten(ls))
        if self._flats is not None:
            return self._flats
        self._flats = list(self.iflatten(ls))
        return self._flats

    def iflatten(self, indivs:list|None=None):
        """ 
        Flattens a list of individuals to the tract level. 

        Parameters
        ----------
        indivs: list | None, default None
            A list of individuals to flatten. If None, the function will flatten the complete list of individuals contained in this population. If a list is provided, the function will flatten that list of individuals instead of the complete list of individuals in the population.
        
        Returns
        generator
            A generator that yields Tract objects representing the flattened version of the input list of individuals (or the complete list of individuals in the population if `indivs` is None). The start and end information of the tracts is preserved in the yielded Tract objects.
        """
        if indivs is None:
            indivs = self.indivs
        for i in indivs:
            for _tract in i.iflatten():
                yield _tract

    def merge_ancestries(self, ancestries: list[str], newlabel: str):
        """ 
        Treats ancestries in label list `ancestries` as a single population
        with label `newlabel`. Adjacent tracts of the new ancestry are merged.
        
        Parameters
        ----------
        ancestries: list[str]
            A list of ancestry labels to merge into a single population. The function will treat all tracts with labels in this list as belonging to the same population and will merge adjacent tracts of these ancestries into a single tract with the label `newlabel`.
        newlabel: str
            The label to assign to the merged ancestry.  
        """
        f = lambda i: i.merge_ancestries(ancestries, newlabel)
        self.applychrom(f)

    def get_global_tractlengths(self, npts: int = 50, tol: float = 0.01, indlist: list = None, split_count: int = 1, exclude_tracts_below_cM: float = 0):
        """ 
        Parameters
        ----------
        tol: float, default 0.01
            The tolerance for full chromosomes. 
        npts: int, default 50
            The number of bins for the histogram.
        indlist: list, default None
            The individuals for which we want the tractlength. To bootstrap over individuals, provide a bootstrapped list individuals.
        split_count: int, default 1
            If greater than 1, the population is split into `split_count` groups according to their ancestry proportions, and the tractlength histogram is computed separately for each group. 
        exclude_tracts_below_cM: float, default 0
            Exclude tracts below this length in cM.
            
        Returns
        -------
        np.ndarray
            The bins for the histogram
        dict[str, np.ndarray]
            A dictionary with ancestry labels as keys and a histogram of tract lengths as values.
           
        Notes
        -----
        Sometimes there are small issues at the edges of the chromosomes. If a segment is within tol Morgans of the full chromosome, it counts as a full
        chromosome note that we return an extra bin with the complete chromosome bin, so that we have one more data point than we have bins.
        """
        # NOTE: Figure out whether we're dealing with the set of individuals represented by this population or the one contained in the indlist parameter.
        if indlist is None:
            pop = self  
        else:
            pop = Population(indlist)
            pop.unknown_labels = self.unknown_labels

        if split_count > 1: # If we're doing a split analysis, then break up the population into groups, and just do get_global_tractlengths on the groups.
            ps = pop.split_by_props(split_count)
            bindats = [p.get_global_tractlengths(npts, tol, exclude_tracts_below_cM=exclude_tracts_below_cM) for p in
                       ps]
            bins_list, dats_list = zip(*bindats)  # The bins will all be the same, so we can throw out the duplicates.
            return bins_list[0], dats_list

        bypop: dict[str, list[tuple[Tract, float]]] = defaultdict(list)
        for indiv in pop:
            for chrom in indiv:
                for copy in chrom:
                    copy.unknown_labels = pop.unknown_labels if hasattr(pop, 'unknown_labels') else []
                    copy.smooth_unknown()
                    for tract in copy:
                        bypop[tract.label].append((
                            tract,
                            chrom.len
                        ))
        return self.tractlength_histogram(tracts_by_population=bypop,
                                        npts=npts,
                                        tol=tol,
                                        exclude_tracts_below_cM=exclude_tracts_below_cM)

    def tractlength_histogram(self, tracts_by_population:dict[str, list[tuple[Tract, float]]], npts: int = 50, tol: float = 0.01, exclude_tracts_below_cM: float = 0, maxLen: float | None = None):
        """
        Helper function for get_global_tractlengths that takes in a dictionary of tracts organized by population and returns the histogram of tract lengths for each population.
        
        Parameters
        ----------
        tracts_by_population: dict[str, list[tuple[Tract, float]]]
            A dictionary where the keys are ancestry labels and the values are lists of tuples, where each tuple contains a Tract object and the length of the chromosome that tract is on. This dictionary is used to compute the histogram of tract lengths for each ancestry label.
        npts: int, default 50
            The number of bins for the histogram.
        tol: float, default 0.01
            The tolerance for full chromosomes. Sometimes there are small issues at the edges of the chromosomes. If a segment is within tol Morgans of the full chromosome, it counts as a full chromosome note that we return an extra bin with the complete chromosome bin, so that we have one more data point than we have bins.
        exclude_tracts_below_cM: float, default 0
            Exclude tracts below this length in centiMorgans from the histogram.
        
        Returns
        -------
        np.ndarray
            The bins for the histogram.
        dict[str, np.ndarray]
            A dictionary with ancestry labels as keys and a histogram of tract lengths as values.
        """
        if maxLen==None:
            maxLen=self.maxLen
        bins = np.linspace(exclude_tracts_below_cM * 0.01, maxLen + tol, npts + 1)
        
        dat: dict[str, list[np.ndarray]] = {}
        for label, ts in tracts_by_population.items():
            
            corrected_lengths = np.array([tract.len() if tract.len() < chrom_length - tol else chrom_length for tract, chrom_length in ts])
            hdat = np.histogram(corrected_lengths, bins=bins)
            dat[label] = hdat[0]

        return bins, dat

 
    def set_males(self, male_list: list[str] | str, allosome_label: str='X'):
        """
        Sets the list of males for each individual.
        
        Parameters
        ----------
        male_list: list[str] | str
            A list of labels for male individuals in the population. 
        allosome_label: str, default 'X'
            The label for the allosome to use for determining the sex of individuals when the sex cannot be inferred from the data. 
        """
        num_males_processed = 0;
        if male_list != "auto":
            
            for indiv in self:
                if indiv.name in self.male_list:
                    indiv.is_male = True
                else:
                    indiv.is_male = False
                
                if indiv.is_male and  len(indiv.allosomes[allosome_label]) == 2:
                    logger.info(f"Individual {indiv.name} is listed as male but has two X chromosomes. Selecting first of the two.")
                    assert indiv.allosomes[allosome_label][0].is_equal(indiv.allosomes[allosome_label][1]), f"Male individual {indiv} has two different X chromosomes." 
                    indiv.allosomes[allosome_label] = [indiv.allosomes[allosome_label][0]]
                indiv.is_male = (len(indiv.allosomes[allosome_label]) == 1) # Males are now either individuals labeled as males, or individuals who have a single X chromsome in data file. 
                num_males_processed += indiv.is_male
            
            if len(self.male_list) not in [0, num_males_processed]:
                raise logger.warning(f"A male list of length {len(self.male_list)} is provided, but we have identified"+ 
                                    "{num_males_processed} males") 
        else: 
            for indiv in self:
                if len(indiv.allosomes[allosome_label]) == 2:
                    indiv.is_male = False
                elif len(indiv.allosomes[allosome_label]) == 1:
                    indiv.is_male = True
                
                else:
                    raise ValueError("There should be one or two allosome copies")
                num_males_processed += indiv.is_male
            print(f"Identified {num_males_processed} males from allosomal data")
        self.males_set = True

    def smooth_unknowns(self, allosome_labels: list[str]='X'):
        """
        Smooths the unknown labels for each individual in the population.

        Parameters
        ----------
        allosome_labels: list[str], default 'X'
            A list of labels for the allosomes in the population.
        """
       
        for indiv in self:
            for allosome_label in allosome_labels:             
                if allosome_label not in indiv.allosomes.keys():
                    raise logger.warning(f"Data for chromosome {allosome_label} does not exist on individual {indiv.name}.")
                for chrom in indiv.allosomes[allosome_label]:
                    chrom.unknown_labels= self.unknown_labels if hasattr(self, 'unknown_labels') else []
                    chrom.smooth_unknown()
            for chrom in indiv:
                for copy in chrom:
                    copy.unknown_labels = self.unknown_labels if hasattr(self, 'unknown_labels') else []
                    copy.smooth_unknown()

    def get_global_allosome_tractlengths(self, allosome:str, npts: int = 50, tol: float = 0.01, indlist: list = None, exclude_tracts_below_cM: float = 0):
        """
        Returns the allosomal tractlength histogram in males and the allosomal tractlength histogram in females.

        Parameters
        ----------
        allosome: str
            The label for the allosome to analyze.
        npts: int, default 50
            The number of bins for the histogram.
        tol: float, default 0.01
            The tolerance for full chromosomes.
        indlist: list, default None
            The individuals for which we want the tractlength
        exclude_tracts_below_cM: float, default 0
            The minimum length of tracts to include in the histogram.
        
        Returns
        -------
        np.ndarray
            The bins for the histogram.
        dict[SexType, dict[str, np.ndarray]]
            A dictionary with keys `SexType.MALE` and `SexType.FEMALE`, where the value for each key is a dictionary with ancestry labels as
            keys and a histogram of tract lengths as values for each ancestry.
        """
        if allosome not in self.allosome_labels:
            raise KeyError(f"Data for chromosome {allosome} was never initialized for this population.")
        
        if indlist is None:
            pop = self  
        else:
            pop = Population(indlist)
            pop.unknown_labels = self.unknown_labels
        assert(self.males_set), "Males should have been set using set_males before calling get_global_allosome_tractlengths."

        bypop_male: dict[str, list[tuple[Tract, float]]] = defaultdict(list)
        bypop_female: dict[str, list[tuple[Tract, float]]] = defaultdict(list)
        
        for indiv in pop:
            tracts_added = False
            
            if allosome not in indiv.allosomes:
                raise logger.warning(f"Data for chromosome {allosome} does not exist on individual {indiv.name}.")
            
            for chrom in indiv.allosomes[allosome]:
                chrom.unknown_labels= pop.unknown_labels if hasattr(pop, 'unknown_labels') else []
                chrom.smooth_unknown()
                for tract in chrom:
                    tracts_added = True
                    if indiv.is_male:
                        bypop_male[tract.label].append((tract, chrom.len))
                    else:
                        bypop_female[tract.label].append((tract, chrom.len))
            if tracts_added is False:
                raise logger.warning(f"Data for chromosome {allosome} does not exist on individual {indiv.name}.")
        
        if len(bypop_male)==0 and len(bypop_female)==0:
            raise ValueError(f"Data for chromosome {allosome} does not exist on any individuals of this population.")
         
        L=self.allosome_lengths[allosome]
        bins, male_data = self.tractlength_histogram(tracts_by_population=bypop_male,
                                                    npts=npts,
                                                    tol=tol,
                                                    exclude_tracts_below_cM=exclude_tracts_below_cM,
                                                    maxLen=L)
        _, female_data = self.tractlength_histogram(tracts_by_population=bypop_female,
                                                   npts=npts,
                                                   tol=tol,
                                                   exclude_tracts_below_cM=exclude_tracts_below_cM,
                                                   maxLen=L)
        return bins, {SexType.MALE: male_data, SexType.FEMALE: female_data}
                

    def bootinds(self, seed: int = 0):
        """ 
        Returns a bootstrapped list of individuals in the population. Set this function as the `indlist`
        parameter of :func:`~tracts.population.Population.get_global_tractlength` to get a bootstrapped sample.
        
        Parameters
        ----------
        seed: int, default 0
            The random seed to use for bootstrapping. Setting the seed allows for reproducibility of the bootstrapped samples.
        """
        np.random.seed(seed=seed)
        return np.random.choice(self.indivs, size=len(self.indivs))

    def get_global_tractlength_table(self, lenbound: list[float]):
        """
        Calculates the fraction of the genome covered by ancestry tracts of different lengths, specified by `lenbound` (which must be sorted).
        
        Parameters
        ----------
        lenbound: list[float]
            A sorted list of length boundaries for categorizing ancestry tracts. The function will calculate the fraction of the genome covered by ancestry tracts that fall into each of the length categories defined by these boundaries.
        
        Returns
        -------
        list[float]
            The length boundaries for categorizing ancestry tracts, as specified by the input `lenbound`.
        dict[str, np.ndarray]
            A dictionary with ancestry labels as keys and an array of the fraction of the genome covered by ancestry tracts of different lengths as values.
        """
        flatdat = self.flatpop()
        bypop = collect_pop(flatdat)

        bins = lenbound
        dat = {} 
        for key, poplen in bypop.items():
            dat[key] = np.zeros(len(bins) + 1) # Extract full length tracts
            nonfulls = np.array([item
                                 for item in poplen
                                 if (item[0] != item[1])])
            for item in nonfulls:
                pos = bisect.bisect_left(bins, item[0])
                dat[key][pos] += item[0] * 1. / self.nind * 1. / np.sum(self.Ls) / 2.

        return bins, dat

    def get_mean_ancestry_proportions(self, ancestries: list[str]):
        """
        Gets the mean ancestry proportion averaged across individuals in the population.

        Parameters
        ----------
        ancestries: list[str]
            A list of ancestry labels for which to calculate the mean ancestry proportion. The function will calculate the mean ancestry proportion for each ancestry label in this list, averaged across all individuals in the population.
        
        Returns
        -------
        list[float]
            A list of mean ancestry proportions corresponding to the ancestry labels in the input `ancestries` list.
        """
        return list(map(np.mean, zip(*self.get_means(ancestries))))

    def calculate_ancestry_proportions(self, population_labels: list[str], cutoff:float = 0.0):
        """
        Calculates the mean ancestry proportion across individuals in the population using only autosomal data.

        Parameters
        ----------
        population_labels: list[str]
            A list of ancestry labels for which to calculate the ancestry proportions. The function will calculate the ancestry proportion for each ancestry label in this list for each individual in the population.
        cutoff: float, default 0.0
            A threshold for the length of ancestry tracts to consider when calculating ancestry proportions.
        
        Returns
        -------
        list[float]
            A list of ancestry proportions corresponding to the ancestry labels in the input `population_labels` list, averaged across all individuals in the population.

        
        """
        bypopfrac = [[] for _ in range(len(population_labels))]
        for ind in self.indivs:
            for i, population_label in enumerate(population_labels):
                bypopfrac[i].append(ind.ancestryProps([population_label], cutoff = cutoff))
        return np.mean(bypopfrac, axis=1).flatten()

    def calculate_allosome_proportions(self, population_labels: list[str], allosome_label: str, cutoff: float = 0.0):
        """
        Calculates the mean ancestry proportion across individuals in the population using only data from a specified allosome.

        Parameters
        ----------
        population_labels: list[str]
            A list of ancestry labels for which to calculate the ancestry proportions. 
        allosome_label: str
            The label for the allosome to use for calculating ancestry proportions. It is assumed that all individuals have the same allosomes and that the allosome with label `allosome_label` corresponds to the same chromosome across individuals.
        cutoff: float, default 0.0
            A threshold for the length of ancestry tracts to consider when calculating ancestry proportions.
        
        Returns
        -------
        list[float]
            A list of ancestry proportions corresponding to the ancestry labels in the input `population_labels` list, averaged across all individuals in the population, calculated using only data from the specified allosome.

        Notes
        -----
        IDE warnings may appear and can be ignored.
        """
        bypopfrac = [[] for _ in range(len(population_labels))]
        weights = []
        for ind in self.indivs:
            for i, population_label in enumerate(population_labels):
                bypopfrac[i].append(ind.ancestryProps([population_label], allosome_label=allosome_label, cutoff = cutoff))
            weights.append(1 if ind.is_male else 2)
        return np.average(bypopfrac, weights = weights, axis=1).flatten()

    def get_means(self, ancestries: list[str]):
        """
        Gets the mean ancestry proportion (only among ancestries in ancestries) for all individuals.
        
        Parameters
        ----------
        ancestries: list[str]
            A list of ancestry labels for which to calculate the mean ancestry proportion for each individual.
        
        Returns
        -------
        list[list[float]]
            A list of lists, where each inner list contains the mean ancestry proportions for the ancestry labels in the input `ancestries` list for a single individual in the population. The outer list contains one inner list for each individual in the population. 
        """
        return [ind.ancestryProps(ancestries) for ind in self.indivs]

    def get_meanvar(self, ancestries: list[str]):
        """
        Gets the mean and variance of ancestry proportions across individuals in the population, for ancestries in `ancestries`.

        Parameters
        ----------
        ancestries: list[str]
            A list of ancestry labels for which to calculate the mean and variance of ancestry proportions across individuals in the population. 

        Returns
        -------
        list[float]
            A list of mean ancestry proportions corresponding to the ancestry labels in the input `ancestries` list, averaged across all individuals in the population.
        list[float]
            A list of variances of ancestry proportions corresponding to the ancestry labels in the input `ancestries` list, calculated across all individuals in the population.
        """
        byind = self.get_means(ancestries)
        return np.mean(byind, axis=0), np.var(byind, axis=0)

    def getMeansByChrom(self, ancestries: list[str]):
        """ 
        Gets the ancestry proportions in each individual of the population for each chromosome.

        Parameters
        ----------
        ancestries: list[str]
            A list of ancestry labels for which to calculate the ancestry proportions for each chromosome. The function will calculate the ancestry proportions for each ancestry label in this list for each chromosome in each individual in the population.
        
        Returns
        -------
        list[list[list[float]]]
            A list of lists of lists, where the outer list contains one inner list for each individual in the population, the middle list contains one inner list for each ancestry label in the input `ancestries` list, and the innermost list contains the ancestry proportions for each chromosome for that ancestry label for that individual.
        """
        return [ind.ancestryPropsByChrom(ancestries) for ind in self.indivs]

    def get_variance(self, ancestries: list[str]):
        """
        Calculates the total variance in ancestry proportions, the genealogy variance, and the
        assortment variance, that corresponds to the mean uncertainty about the proportion of genealogical ancestors, given observed ancestry
        patterns.
        
        Parameters
        ----------
        ancestries: list[str]
            A list of ancestry labels for which to calculate the variance in ancestry proportions. The function will calculate the variance in ancestry proportions for each ancestry label in this list across all individuals in the population.
        
        Returns
        -------
        list[float]
            A list of total variances in ancestry proportions corresponding to the ancestry labels in the input `ancestries` list, calculated across all individuals in the population.
        list[float]
            A list of genealogy variances corresponding to the ancestry labels in the input `ancestries` list, calculated across all individuals in the population. 
        list[float]
            A list of assortment variances corresponding to the ancestry labels in the input `ancestries` list, calculated across all individuals in the population.
        
        Notes
        -----
        All unlisted ancestries are considered uncalled. For example, calling the function with a single ancestry leads to no variance (and some 0/0 errors).
        """

        ws = np.array(self.Ls) * 1. / np.sum(self.Ls) # The weights, corresponding (approximately) to the inverse variances
        arr = np.array(self.getMeansByChrom(ancestries)) # Weighted mean by individual. Departure from the mean.
        nchr = arr.shape[2]
        assort_vars = []
        tot_vars = []
        gen_vars = []
        for i in range(len(ancestries)):
            pl = np.dot(arr[:, i, :], ws)
            tot_vars.append(np.var(pl))
            aroundmean = arr[:, i, :] - np.dot(
                pl.reshape(self.nind, 1), np.ones((1, nchr)))
            assort_vars.append( # The unbiased estimator for the case where the variance is inversely proportional to the weight. First calculate by individual, then the mean over all individuals.
                (np.mean(aroundmean ** 2 / (1. / ws - 1), axis=1)).mean())
            gen_vars.append(tot_vars[-1] - assort_vars[-1])
        return tot_vars, gen_vars, assort_vars

    def plot_next(self):
        """
        Plots the next individual in the population.

        Returns
        -------
        tk.Tk
            A visual representation of the individual's ancestry tracts. See :func:`~tracts.individual.Individual.plot` for details on the visual representation.
        """
        self.indivs[self.currentplot].canvas.pack_forget()
        if self.currentplot < self.nind - 1:
            self.currentplot += 1
        return self.plot_indiv()

    def plot_previous(self):
        """
        Plots the previous individual in the population.

        Returns
        -------
        tk.Tk
            A visual representation of the individual's ancestry tracts. See :func:`~tracts.individual.Individual.plot` for details on the visual representation.
        """
        self.indivs[self.currentplot].canvas.pack_forget()
        if self.currentplot > 0:
            self.currentplot -= 1
        return self.plot_indiv()

    def plot_indiv(self):
        """
        Plots the individual at the current plot index and stores it in `self.canv`.
        """
        self.win.title("Individual %d " % (self.currentplot + 1,))
        self.canv = self.indivs[self.currentplot].plot(
            self.colordict, win=self.win)

    def plot(self, colordict: dict[str, str]):
        """
        Plots the individuals in the population using a color dictionary that maps ancestry labels to colors.

        Parameters
        ----------
        colordict: dict[str, str]
            A dictionary that maps ancestry labels (as strings) to color codes (also as strings) that can be used in plotting.
        """
        self.colordict = colordict
        self.currentplot = 0
        self.win = tk.Tk()
        printbutton = tk.Button(self.win, text="Save to ps", command=self.save)
        printbutton.pack()

        p = tk.Button(self.win, text="Plot previous", command=self.plot_previous)
        p.pack()

        b = tk.Button(self.win, text="Plot next", command=self.plot_next)
        b.pack()
        self.plot_indiv()
        tk.mainloop()

    def plot_chromosome(self, i: int, colordict: dict[str, str], win: tk.Tk=None):
        """
        Plot a single chromosome across individuals in the population using a color dictionary that maps ancestry labels to colors.

        Parameters
        ----------
        i: int
            The index of the chromosome to plot. It is assumed that all individuals have the same number of chromosomes and that the chromosome with index `i` corresponds to the same chromosome across individuals.
        colordict: dict[str, str]
            A dictionary that maps ancestry labels (as strings) to color codes (also as strings) that can be used in plotting.
        win: tk.Tk, default None
            A Tkinter window in which to plot the chromosome. If None, a new window will be created for the plot. If a window is provided, the chromosome will be plotted in that window instead of creating a new one.
        """
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

    def plot_ancestries(self, chrom: int = 0, npts: int = 50, colordict: dict[str, str] = None, cutoff: float = 0.0):
        """
        Plots the ancestry proportions along a chromosome across individuals in the population using a color dictionary that maps ancestry labels to colors.

        Parameters
        ----------
        chrom: int, default 0
            The index of the chromosome to plot. It is assumed that all individuals have the same number of chromosomes and that the chromosome with index `chrom` corresponds to the same chromosome across individuals.
        npts: int, default 50
            The number of points along the chromosome at which to plot the ancestry proportions.
        colordict: dict[str, str], default None
            A dictionary that maps ancestry labels (as strings) to color codes (also as strings) that can be used in plotting. If None, a default color dictionary will be used that maps "CEU" to 'blue' and "YRI" to 'red'.
        cutoff: float, default 0.0
            A threshold for the length of ancestry tracts to consider when calculating ancestry proportions at each point along the chromosome. Only tracts that are longer than this threshold will be considered when calculating the ancestry proportions at each point.
        """
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

    def plot_all_ancestries(self, npts: int = 50, colordict: dict[str, str] = None, startfig: int = 0, cutoff: float = 0):
        """
        Plots the ancestry proportions along all chromosomes across individuals in the population using a color dictionary that maps ancestry labels to colors.

        Parameters
        ----------
        npts: int, default 50
            The number of points along each chromosome at which to plot the ancestry proportions.
        colordict: dict[str, str], default None
            A dictionary that maps ancestry labels (as strings) to color codes (also as strings) that can be used in plotting. If None, a default color dictionary will be used that maps "CEU" to 'blue' and "YRI" to 'red'.
        startfig: int, default 0
            The starting figure number for plotting. The function will plot the ancestry proportions for each chromosome in a separate subplot, and the figure numbers for these subplots will start from this value.
        cutoff: float, default 0
            A threshold for the length of ancestry tracts to consider when calculating ancestry proportions at each point along the chromosomes. Only tracts that are longer than this threshold will be considered when calculating the ancestry proportions at each point.
        """
        dat = None
        chrom = None
        pop = None
        color = None
        if colordict is None:
            colordict = {"CEU": 'blue', "YRI": 'red'}
        
        for chrom in range(22): # TODO: Remove the magic number
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
                pylab.subplot(6, 4, chrom + 1) # TODO: Replace "chrom + 1" with 23?
                pylab.plot(dat[0], [pos[0][pop] for pos in dat[1]], '.', color=color)
                pylab.axis([0, dat[0][-1], 0, 1])
                pylab.figure(1 + startfig)
        
        pylab.subplot(6, 4, chrom + 1) # TODO: What color should be used?
        pylab.plot(dat[0], [100 * pos[1][pop] for pos in dat[1]], '.', color=color)
        pylab.axis([0, dat[0][-1], 0, 150])

    def plot_global_tractlengths(self, colordict: dict[str, str], npts: int=50, legend: bool=True):
        """
        Plot the distribution of global tract lengths for each population.

        Parameters
        ----------
        colordict: dict[str, str]
            A dictionary that maps ancestry labels (as strings) to color codes (also as strings) that can be used in plotting. The function will plot the distribution of global tract lengths for each ancestry label in this dictionary using the corresponding color.
        npts: int, default 50
            The number of bins for the histogram of tract lengths. The function will use this number of bins when plotting the distribution of global tract lengths for each ancestry label.
        legend: bool, default
            Whether to include a legend in the plot. If True, a legend will be included that maps ancestry labels to colors. If False, no legend will be included in the plot.
        """
        flatdat = self.flatpop()
        bypop = collect_pop(flatdat)
        self.maxLen = max(self.Ls)
        for label, tracts in bypop.items():
            hdat = pylab.histogram([i.len() for i in tracts], npts)
            pylab.semilogy(100 * (hdat[1][1:] + hdat[1][:-1]) / 2., hdat[0], 'o', color=colordict[label], label=label)
        pylab.xlabel("Length (cM)")
        pylab.ylabel("Counts")
        if legend:
            pylab.legend()

    def __iter__(self):
        """
        Returns an iterator over the individuals in the population.

        Returns
        -------
        Iterator[Indiv]
            An iterator that yields the individuals in the population one at a time.
        """
        return self.indivs.__iter__()

    def __getitem__(self, index: int) -> Indiv:
        """
        Returns the individual at the specified index.

        Parameters
        ----------
        index: int
            The index of the individual to retrieve. It is assumed that the individuals in the population are stored in a list and that the index corresponds to the position of the individual in this list.
        
        Returns
        -------
        Indiv
            The individual at the specified index in the population.
        """
        return self.indivs[index]
