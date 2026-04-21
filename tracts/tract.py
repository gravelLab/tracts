class Tract:
    """
    Represent a labeled genomic interval.

    A tract is the basic object manipulated by Tracts. Higher-level
    structures are built from collections of tracts. Each tract is
    defined by an interval and an associated label, typically
    corresponding to an ancestry.

    Parameters
    ----------
    start : float
        Starting position of the tract, in Morgans.
    end : float
        Ending position of the tract, in Morgans.
    label : str
        Identifier associated with the tract. In most applications,
        this corresponds to the ancestry carried by the tract.
    bpstart : int, optional
        Starting position of the tract, in base pairs. This attribute
        is optional, since Tracts primarily uses Morgans internally.
        Default is ``None``.
    bpend : int, optional
        Ending position of the tract, in base pairs. This attribute
        is optional, since Tracts primarily uses Morgans internally.
        Default is ``None``.
    """

    def __init__(self, start, end, label, bpstart=None, bpend=None):
        """
        Initialize a :class:`Tract` instance.

        Parameters
        ----------
        start : float
            Starting position of the tract, in Morgans.
        end : float
            Ending position of the tract, in Morgans.
        label : str
            Identifier associated with the tract. In most applications,
            this corresponds to the ancestry carried by the tract.
        bpstart : int, optional
            Starting position of the tract, in base pairs. This attribute
            is optional, since Tracts primarily uses Morgans internally.
            Default is ``None``.
        bpend : int, optional
            Ending position of the tract, in base pairs. This attribute
            is optional, since Tracts primarily uses Morgans internally.
            Default is ``None``.
        """
        self.start = start
        self.end = end
        self.label = label
        self.bpstart = bpstart
        self.bpend = bpend

    def len(self):
        """
        Return the length of the tract in Morgans.

        Returns
        -------
        float
            Length of the tract, computed as ``end - start``.
        """
        return self.end - self.start

    def get_label(self):
        """
        Return the label of the tract.

        Returns
        -------
        str
            Label associated with the tract.
        """
        return self.label

    def copy(self):
        """
        Return a copy of the tract.

        Returns
        -------
        Tract
            New :class:`Tract` instance with the same attributes as the
            current tract.
        """
        return Tract(
            self.start, self.end, self.label, self.bpstart, self.bpend
        )

    def __repr__(self):
        return "tract(%s, %s, %s)" % tuple(
            map(repr, [self.start, self.end, self.label])
        )

    def is_equal(self, other):
        """
        Check whether two tracts have the same start, end, and label.

        Parameters
        ----------
        other : Tract
            Tract to compare with the current instance.

        Returns
        -------
        bool
            ``True`` if both tracts have identical ``start``, ``end``,
            and ``label`` attributes, and ``False`` otherwise.
        """
        return (
            self.start == other.start
            and self.end == other.end
            and self.label == other.label
        )