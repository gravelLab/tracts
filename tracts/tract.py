class Tract:
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
        return Tract(
            self.start, self.end, self.label, self.bpstart, self.bpend)

    def __repr__(self):
        return "tract(%s, %s, %s)" % tuple(
            map(repr, [self.start, self.end, self.label]))
