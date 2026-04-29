from collections import defaultdict
from typing import Any
from tracts.tract import Tract
from tracts.chromosome import Chrom
import numpy as np
import logging
logger=logging.getLogger(__name__)
class Haploid:
    """
    A class representing a haploid genome, composed of a set of chromosomes,
    each of which consists of a list of tracts.

    Attributes
    ----------
    Ls: list of float
      	The lengths of the chromosomes.
    chroms: list of :class:`tracts.chromosome.Chrom`
      	The chromosome objects.
    labs: list of str
      	Labels identifying the chromosomes.
    name: str
       	An optional name for the haploid.
    allosomes: dict
        A dictionary mapping chromosome labels to chromosome objects, for chromosomes that are treated as allosomes.
    """

    @staticmethod
    def from_file(path: str, name: str = None, selectchrom: list[int | str] | None = None, allosome_labels: set | None = None):
        """
        Load a haploid genome from a file.

        Parameters
        ----------
        path: str
            The path to the file containing the haploid genome data. The file should be a tab-delimited text file with columns: chrom, start, end, label, and optionally others. The first line may be a header, which will be automatically skipped if it contains the expected column names.
        name: str, optional
            An optional name for the haploid genome. If not provided, the name will be set to None.
        selectchrom: list[int | str] | None
            An optional list of chromosome identifiers specifying which chromosomes to select from the file. If not provided, all chromosomes will be selected. Elements are converted to integers when possible. Non-numeric values are ignored with a warning.
        allosome_labels: set | None, optional
            An optional set of chromosome labels that should be treated as allosomes. If not provided, no chromosomes will be treated as allosomes. Chromosome labels should be strings corresponding to the chromosome identifiers in the file (e.g., "X" for the X chromosome). Chromosome labels that are not present in the file will be ignored, and no chromosomes will be treated as allosomes.
        
        Returns
        -------
        Haploid
            A Haploid object representing the loaded haploid genome.        
        """

        if allosome_labels is None:
            allosome_labels = set()

        # TODO move the loading logic from the constructor to this static method. This will facilitate loading logic for future driver scripts.
        chromd = defaultdict(list)

        with open(path, 'r') as f: # Parse the indicated file into a dictionary associating chromosome identifiers (strings) with lists of tract objects.
            header_mode = True
            for line in f:
                fields = line.split()
                if len(fields) == 0:
                    continue

                if fields[0] == 'chrom' or \
                        (fields[0] == 'Chr' and fields[1] == 'Start(bp)'): # Skip the header, if one is present.
                    continue
                
                try:
                    chromd[fields[0]].append(
                        Tract(
                            .01 * float(fields[4]), .01 * float(fields[5]),
                        fields[3]))
                    header_mode = False
                    
                except Exception as e: # To catch (different) headers in data files
                    if header_mode:
                        continue
                    else:
                        raise e # ValueError from e 

        # Now that the file has been parsed, we need to apply a filtering step,
        # to select only those chromosomes identified by selectchrom.
        # A haploid individual is essentially just a list of chromosomes, so we
        # initialize this list of chromosomes to be ultimately passed to the
        # haploid constructor.

        chroms: list[Chrom] = []
        labs = []
        Ls: list[int] = []
        allosomes: dict[Any, Chrom]={}
            
        # Construct a function that tells us whether a given chromosome is selected or not.
        if selectchrom is None:
            # selectchrom being None means that all chromosomes are selected, so the selection function always returns True.
            def is_selected(*args): # TODO Decide what to do with this dummy function.
                return True
        else:
            # Otherwise, we 'normalize' selectchrom by ensuring that it
            # contains only integers. (This is primarily for
            # backwards-compatibility with previous scripts that specified
            # chromosome numbers as strings.) And we make a set out of the
            # resulting normalized list, to speed up lookups later.
            valid_chroms = set()
            invalid = []
            for x in selectchrom:
                try:
                    valid_chroms.add(int(x))
                except (ValueError, TypeError):
                    invalid.append(x)
            if invalid:
                logger.warning(f"Ignoring non-numeric chromosome identifiers: {invalid}. Please ensure that all chromosome identifiers in selectchrom can be converted to integers.")
            sc = set(valid_chroms)

            # The function that tests for inclusion simply casts its argument
            # (which is a string since it's read in from a file) to an int, and checks whether its in our set.
            def is_selected(chrom_label):
                try:
                    return int(chrom_label) in sc
                except:
                    return False

        # Filter the loaded data according to selectchrom using the is_selected function constructed above.
        for chrom_data, tracts in chromd.items():
            chrom_id = chrom_data.split('r')[-1]
            if is_selected(chrom_id):
                c = Chrom(tracts=tracts)
                chroms.append(c)
                Ls.append(c.len)
                labs.append(chrom_id)
            if chrom_id in allosome_labels:
                #print(f'{chrom_id} in {path}')
                allosomes[chrom_id] = Chrom(tracts=tracts)


        # Organize the filtered lists according to the order of their identifiers.
        order = np.argsort(labs)

        chroms = list(
            np.array(chroms)[order])
        Ls = list(
            np.array(Ls)[order])
        labs = list(
            np.array(labs)[order])

        return Haploid(Ls=Ls, lschroms=chroms, labs=labs, name=name, allosomes=allosomes)

    def __init__(self, Ls: list = None, lschroms: list = None, fname: str = None, selectchrom: list[int | str] | None = None,
                 labs: list = None, name: str = None, allosomes: dict[Any, Chrom] = None):
        """
        Initialize a Haploid object.

        Parameters
        ----------
        Ls: list of float, optional
            The lengths of the chromosomes. If not provided, the lengths will be inferred from the chromosome objects provided in lschroms. If neither Ls nor lschroms are provided, an error will be raised.
        lschroms: list of :class:`tracts.chromosome.Chrom`, optional
            The chromosome objects. If not provided, the chromosome objects will be loaded from the file specified by fname. If neither lschroms nor fname are provided, an error will be raised.
        fname: str, optional
            The path to the file containing the haploid genome data. If not provided, the chromosome objects will be provided directly in lschroms. If neither fname nor lschroms are provided, an error will be raised. The file should be a tab-delimited text file with columns: chrom, start, end, label, and optionally others. The first line may be a header, which will be automatically skipped if it contains the expected column names.
        selectchrom: list[int | str] | None, optional
            An optional list of chromosome identifiers specifying which chromosomes to select from the file. If not provided, all chromosomes will be selected. The list should contain chromosome identifiers as strings or integers corresponding to the chromosome numbers in the file (e.g., ["1", "2", "3"]). Chromosome identifiers that cannot be converted to integers will be ignored, and the corresponding chromosomes will not be selected.
        labs: list of str, optional
            A list of labels for the chromosomes. If not provided, no labels will be assigned.
        name: str, optional
            An optional name for the haploid genome. If not provided, the name will be set to None.
        allosomes: dict, optional
            An optional dictionary mapping chromosome labels to chromosome objects, for chromosomes that are treated as allosomes. If not provided, no chromosomes will be treated as allosomes. Chromosome labels should be strings corresponding to the chromosome identifiers in the file (e.g., "X" for the X chromosome). Chromosome labels that are not present in the file will be ignored, and no chromosomes will be treated as allosomes.
        """

        if fname is None:
            if Ls is None or lschroms is None:
                raise ValueError(
                    "Ls or lschroms should be defined if file not defined.")
            self.Ls = Ls
            self.chroms = lschroms
            self.labs = labs
            self.name = name
            self.allosomes=allosomes if allosomes else {}
        else:
            h = Haploid.from_file(path=fname,
                                selectchrom=selectchrom)
            self.Ls = h.Ls
            self.chroms = h.chroms
            self.labs = h.labs
            self.name = name
            self.allosomes=h.allosomes

    def __repr__(self):
        """
        Returns a string representation of the haploid genome.

        Returns
        -------
        str
            A string representation of the haploid genome, showing its chromosomes, name, and lengths.
        """
        return "haploid(lschroms=%s, name=%s, Ls=%s)" % tuple(map(repr, [self.chroms, self.name, self.Ls]))
